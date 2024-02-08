from bochemian.gprotorch.metrics import (
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
)
from bochemian.plotting.bo_plotting import BOPlotter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import wandb


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
