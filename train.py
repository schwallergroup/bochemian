from bochemian.data.module import BaseDataModule
from bochemian.bo.optimizer import BotorchOptimizer
import torch
from pytorch_lightning import seed_everything
import wandb
from tqdm import tqdm
from bochemian.utils import flatten_namespace

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)
from bochemian.utils import instantiate_class


from bochemian.gprotorch.metrics import (
    negative_log_predictive_density,
    mean_standardized_log_loss,
    quantile_coverage_error,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bochemian.plotting.bo_plotting import BOPlotter
from botorch.acquisition import AcquisitionFunction
from bochemian.surrogate_models.gp import SimpleGP, SurrogateModel
import psutil
import os
import logging
import time
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def calculate_data_metrics(data_module):
    target_stat_max = torch.max(data_module.y)
    target_stat_mean = torch.mean(data_module.y)
    target_stat_std = torch.std(data_module.y)
    target_stat_var = torch.var(data_module.y)
    input_stat_feature_dimension = data_module.x.shape[-1]
    input_stat_n_points = data_module.x.shape[0]

    target_q75 = torch.quantile(data_module.y, 0.75)
    target_q90 = torch.quantile(data_module.y, 0.9)
    target_q95 = torch.quantile(data_module.y, 0.95)
    target_q99 = torch.quantile(data_module.y, 0.99)

    top_3_values, _ = torch.topk(data_module.y, 3, dim=0)
    top_5_values, _ = torch.topk(data_module.y, 5, dim=0)
    top_10_values, _ = torch.topk(data_module.y, 10, dim=0)

    top_1 = torch.max(data_module.y)
    top_3 = top_3_values[-1]
    top_5 = top_5_values[-1]
    top_10 = top_10_values[-1]

    return {
        "target_stat_max": target_stat_max,
        "target_stat_mean": target_stat_mean,
        "target_stat_std": target_stat_std,
        "target_stat_var": target_stat_var,
        "input_stat_feature_dimension": input_stat_feature_dimension,
        "input_stat_n_points": input_stat_n_points,
        "target_q75": target_q75,
        "target_q90": target_q90,
        "target_q95": target_q95,
        "target_q99": target_q99,
        "top_1": top_1,
        "top_3": top_3,
        "top_5": top_5,
        "top_10": top_10,
    }


def log_data_metrics(data_metrics, train_y, epoch=0):
    log_top_n_counts(data_metrics, train_y, epoch)
    log_quantile_counts(data_metrics, train_y, epoch)


def log_data_stats(data_metrics):
    for key, value in data_metrics.items():
        if "stat" in key:
            wandb.summary[key] = value.item() if torch.is_tensor(value) else value


def log_top_n_counts(data_metrics, train_y, epoch=0):
    for n in [1, 3, 5, 10]:
        threshold = data_metrics[f"top_{n}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"top_{n}_count": count, "epoch": epoch})


def log_quantile_counts(data_metrics, train_y, epoch=0):
    for q in [0.75, 0.9, 0.95, 0.99]:
        threshold = data_metrics[f"target_q{int(q * 100)}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"quantile_{int(q * 100)}_count": count, "epoch": epoch})


def log_model_parameters(model, epoch=0):
    for name, param in model.named_hyperparameters():
        transformed_name = name.replace("raw_", "")
        attr = model
        for part in transformed_name.split("."):
            attr = getattr(attr, part)
        value = attr.cpu().detach().numpy()

        wandb.log({transformed_name: value, "epoch": epoch})


def calculate_model_fit_metrics(
    model,
    train_x,
    train_y,
    heldout_x,
    heldout_y,
    enable_logging_images=True,
    epoch=0,
):
    predictions_valid, var_valid = model.predict(
        heldout_x, observation_noise=True, return_var=True
    )
    predictions_train, var_train = model.predict(
        train_x, observation_noise=True, return_var=True
    )
    if enable_logging_images:
        plotting_utils = BOPlotter()
        pred_vs_gt_fig_train = plotting_utils.plot_predicted_vs_actual(
            predictions_train,
            train_y,
            var_train.sqrt(),
        )
        pred_vs_gt_fig_valid = plotting_utils.plot_predicted_vs_actual(
            predictions_valid,
            heldout_y,
            var_valid.sqrt(),
        )
        residuals_figure = plotting_utils.plot_residuals(predictions_valid, heldout_y)
        wandb.log(
            {"train/pred-vs-gt": wandb.Image(pred_vs_gt_fig_train), "epoch": epoch}
        )
        wandb.log(
            {"valid/pred-vs-gt": wandb.Image(pred_vs_gt_fig_valid), "epoch": epoch}
        )
        wandb.log({"residuals": wandb.Image(residuals_figure), "epoch": epoch})

    mse_valid = mean_squared_error(heldout_y, predictions_valid)
    r2_valid = r2_score(heldout_y, predictions_valid)
    mae_valid = mean_absolute_error(heldout_y, predictions_valid)

    mse_train = mean_squared_error(train_y, predictions_train)
    r2_train = r2_score(train_y, predictions_train)
    mae_train = mean_absolute_error(train_y, predictions_train)

    pred_dist_valid = model.posterior(heldout_x, observation_noise=True)
    pred_dist_train = model.posterior(train_x, observation_noise=True)

    # Compute GP-specific uncertainty metrics
    nlpd_valid = negative_log_predictive_density(pred_dist_valid, heldout_y)
    msll_valid = mean_standardized_log_loss(pred_dist_valid, heldout_y)
    qce_valid = quantile_coverage_error(pred_dist_valid, heldout_y)

    nlpd_train = negative_log_predictive_density(pred_dist_train, train_y)
    msll_train = mean_standardized_log_loss(pred_dist_train, train_y)
    qce_train = quantile_coverage_error(pred_dist_train, train_y)

    wandb.log(
        {
            "train/mse": mse_train,
            "train/mae": mae_train,
            "train/r2": r2_train,
            "valid/mse": mse_valid,
            "valid/mae": mae_valid,
            "valid/r2": r2_valid,
            "train/nlpd": nlpd_train,
            "train/msll": msll_train,
            "train/qce": qce_train,
            "valid/nlpd": nlpd_valid,
            "valid/msll": msll_valid,
            "valid/qce": qce_valid,
            "epoch": epoch,
        }
    )


def calculate_top_n_metrics(y, discovered_values, quantile=0.99):
    top_n_threshold = torch.quantile(y, quantile)
    found_top_n = torch.sum(discovered_values >= top_n_threshold)
    return found_top_n


def train(config):
    dm = instantiate_class(config["data"])
    data_metrics = calculate_data_metrics(dm)
    log_data_stats(data_metrics)
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    bo = BotorchOptimizer(
        train_x=dm.train_x,
        train_y=dm.train_y,
        heldout_x=dm.heldout_x,
        heldout_y=dm.heldout_y,
        acq_function_config=acquisition_config,
        surrogate_model_config=surrogate_model_config,
        batch_size=bo_config["batch_size"],
        batch_strategy=bo_config["batch_strategy"],
    )

    max_values = []
    current_max = torch.max(bo.train_y)

    # Start the training loop
    for i in tqdm(range(config["n_iters"])):
        wandb.log({"train/best_so_far": current_max, "epoch": i})
        # Call the 'ask' method to determine the next point(s) to evaluate
        x_next = bo.ask(n_points=bo_config["batch_size"])

        x_next = torch.stack(x_next)
        matches = (bo.heldout_x.unsqueeze(0) == x_next).all(dim=-1)
        indices = matches.nonzero(as_tuple=True)[1]

        if not torch.all(matches.sum(dim=-1) == 1):
            print("Unable to find a unique match for some x_next in the dataset.")

        # Using the indices found, directly index into self.heldout_y to get y_next
        y_next = bo.heldout_y[indices]

        # Ensure that y_next and x_next have the correct shape
        x_next = x_next.squeeze(1)
        # Get the number of top n% points found in the current batch
        found_top_n = calculate_top_n_metrics(dm.y, bo.train_y, quantile=0.99)
        wandb.log({"train/found_top_n": found_top_n, "epoch": i})
        log_data_metrics(data_metrics, bo.train_y, epoch=i)

        # Update the current maximum if a larger value was observed
        current_max = max(current_max, torch.max(y_next))
        max_values.append(current_max.item())  # Store the current maximum

        calculate_model_fit_metrics(
            bo.surrogate_model,
            bo.train_x,
            bo.train_y,
            bo.heldout_x,
            bo.heldout_y,
            epoch=i,
        )
        log_model_parameters(bo.surrogate_model, epoch=i)
        # Call the 'tell' method to update the surrogate model with the new data point(s)
        bo.tell(x_next, y_next)

    wandb.finish()


def main():
    # Initialize the parser with a description
    parser = ArgumentParser(
        description="Training script",
        default_config_files=["config.yaml"],
        error_handler=None,
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")  # nargs="+",
    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(SurrogateModel, "surrogate_model", instantiate=False)
    parser.add_subclass_arguments(
        AcquisitionFunction,
        "acquisition",
        instantiate=False,
        skip=["model", "best_f"],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    # Parse arguments
    args = parser.parse_args()

    seed_everything(args["seed"])
    parser.link_arguments(
        "data.train_x", "bo.init_args.train_x", apply_on="instantiate"
    )
    parser.link_arguments(
        "data.train_y", "bo.init_args.train_y", apply_on="instantiate"
    )
    parser.link_arguments(
        "data.heldout_x", "bo.init_args.heldout_x", apply_on="instantiate"
    )
    parser.link_arguments(
        "data.heldout_y", "bo.init_args.heldout_y", apply_on="instantiate"
    )
    parser.link_arguments(
        "seed", "data.init_args.initializer.init_args.seed", apply_on="parse"
    )

    parser.link_arguments(
        "surrogate_model",
        "bo.init_args.surrogate_model_config",
        apply_on="parse",
    )
    parser.link_arguments(
        "acquisition",
        "bo.init_args.acq_function_config",
        apply_on="parse",
    )
    wandb_config = flatten_namespace(args)
    if (
        wandb_config["acquisition.class_path"]
        == "botorch.acquisition.ExpectedImprovement"
    ):
        print("yes")
        parser.link_arguments(
            "data.train_y",
            "acquisition.init_args.best_f",
            apply_on="instantiate",
            compute_fn=torch.max,
        )
    run = wandb.init(project="bochemian_paper", config=wandb_config)
    train(args.as_dict())


if __name__ == "__main__":
    main()
