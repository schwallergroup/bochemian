from bochemian.data.module import BaseDataModule, Featurizer
from bochemian.initialization.initializers import BOInitializer
from bochemian.bo.optimizer import BotorchOptimizer
import torch
from pytorch_lightning import seed_everything
import wandb
from tqdm import tqdm
from bochemian.utils import flatten, flatten_namespace

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
)

import os
import random
import numpy as np

from bochemian.gprotorch.metrics import (
    negative_log_predictive_density,
    mean_standardized_log_loss,
    quantile_coverage_error,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bochemian.plotting.bo_plotting import BOPlotter


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_model_parameters(model):
    for name, param in model.named_hyperparameters():
        transformed_name = name.replace("raw_", "")
        attr = model
        for part in transformed_name.split("."):
            attr = getattr(attr, part)
        value = attr.cpu().detach().numpy()

        wandb.log({transformed_name: value})


def calculate_model_fit_metrics(
    model, train_x, train_y, heldout_x, heldout_y, enable_logging_images=True
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
        wandb.log({"train/pred-vs-gt": wandb.Image(pred_vs_gt_fig_train)})
        wandb.log({"valid/pred-vs-gt": wandb.Image(pred_vs_gt_fig_valid)})
        wandb.log({"residuals": wandb.Image(residuals_figure)})

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
        }
    )


def calculate_top_n_metrics(y, discovered_values, quantile=0.99):
    top_n_threshold = torch.quantile(y, quantile)
    found_top_n = torch.sum(discovered_values >= top_n_threshold)
    return found_top_n


def train(config):
    bo = config["bo"]
    dm = config["data"]
    max_values = []
    current_max = torch.max(
        bo.train_y
    )  # Start with the maximum value in the initial training set

    ys = []
    # Start the training loop
    for i in tqdm(range(config["n_iters"])):
        wandb.log({"train/best_so_far": current_max})
        # Call the 'ask' method to determine the next point(s) to evaluate
        x_next = bo.ask(n_points=5)
        y_next = []
        for x in x_next:
            index = (bo.heldout_x == x).all(dim=1).nonzero().squeeze(-1)
            if len(index) != 1:
                print(
                    f"Unable to find a unique match for x_next in the dataset. Found {len(index)} matches."
                )
            y_next.append(bo.heldout_y[index[0]])

        # Get the number of top n% points found in the current batch
        # Get the number of top n% points found in the current batch
        found_top_n = calculate_top_n_metrics(dm.y, bo.train_y, quantile=0.99)

        wandb.log({"train/found_top_n": found_top_n})

        y_next = torch.stack(y_next)
        x_next = torch.stack(x_next).squeeze(1)

        # Update the current maximum if a larger value was observed
        current_max = max(current_max, torch.max(y_next))
        max_values.append(current_max.item())  # Store the current maximum

        # Ensure that we found a match
        if len(index) != 1:
            raise ValueError(
                f"Unable to find a unique match for x_next in the dataset. Found {len(index)} matches."
            )

        calculate_model_fit_metrics(
            bo.surrogate_model, bo.train_x, bo.train_y, bo.heldout_x, bo.heldout_y
        )
        log_model_parameters(bo.surrogate_model)
        # Call the 'tell' method to update the surrogate model with the new data point(s)
        bo.tell(x_next, y_next)
    wandb.finish()


if __name__ == "__main__":
    # Initialize the parser with a description
    parser = ArgumentParser(description="Training script")

    # Adding the path to the configuration file
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")  # nargs="+",
    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument("--data", type=BaseDataModule, help="Data module")
    parser.add_argument("--bo", type=BotorchOptimizer, help="BO optimizer")

    # Parse the arguments
    args = parser.parse_args()
    set_seeds(args["seed"])

    wandb_config = flatten_namespace(args)
    run = wandb.init(project="bochemian", config=wandb_config)

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
    if (
        wandb_config["bo.init_args.acq_function_config.class_path"]
        == "botorch.acquisition.ExpectedImprovement"
    ):
        parser.link_arguments(
            "data.train_y",
            "bo.init_args.acq_function_config.init_args.best_f",
            apply_on="instantiate",
            compute_fn=torch.max,
        )
    cfg_instantiated = parser.instantiate_classes(args)
    # # Call the training function with the configuration dictionary
    train(cfg_instantiated)
