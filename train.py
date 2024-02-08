from bochemian.data.module import BaseDataModule
from bochemian.bo.optimizer import BotorchOptimizer
from bochemian.data.utils import torch_delete_rows
from bochemian.finetuning.training import beta_scheduler
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
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


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
        if "target" in key:
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


def log_embeddings_with_yields(
    embeddings, yields, additional_info, name, predictions=None
):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()
    if isinstance(yields, torch.Tensor):
        yields = yields.cpu().detach().numpy()

    additional_info_yields = additional_info["yield"].to_numpy().reshape(-1, 1)
    assert np.allclose(
        additional_info_yields, yields
    ), "Yield values in additional_info do not match the concatenated yields."

    combined_data = np.hstack(
        (
            embeddings,
            yields.reshape(-1, 1),
            additional_info,
        )
    )

    embedding_columns = [f"Dim_{i}" for i in range(embeddings.shape[1])]
    additional_info_columns = list(additional_info.columns)
    columns = embedding_columns + ["Yield"] + additional_info_columns
    if predictions is not None:
        predictions_np = (
            predictions.cpu().numpy().reshape(-1, 1)
            if isinstance(predictions, torch.Tensor)
            else predictions.reshape(-1, 1)
        )
        combined_data = np.hstack((combined_data, predictions_np))
        columns.append("Prediction")

    combined_data_list = combined_data.tolist()

    table = wandb.Table(data=combined_data_list, columns=columns)
    wandb.log({name: table})


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


import torch
from torch.utils.data import Dataset, DataLoader


# alpha = 1
# beta = 0

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_to_wandb(
    epoch,
    original_embeddings,
    fine_tuned_embeddings,
    finefine_tuned_embeddings,
    yields,
    finetuning_model,
    surrogate_model,
):
    # Calculate t-SNE for the original and fine-tuned embeddings
    original_tsne = TSNE(n_components=2, random_state=42, init="pca").fit_transform(
        original_embeddings.numpy()
    )
    fine_tuned_tsne = TSNE(n_components=2, random_state=42, init="pca").fit_transform(
        fine_tuned_embeddings.detach().numpy()
    )
    finefine_tuned_tsne = TSNE(
        n_components=2, random_state=42, init="pca"
    ).fit_transform(finefine_tuned_embeddings.detach().numpy())

    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 6))

    # Original Embeddings t-SNE plot
    scatter1 = ax1.scatter(
        original_tsne[:, 0], original_tsne[:, 1], c=yields, cmap="viridis"
    )
    ax1.set_title("Original Embeddings t-SNE")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # Fine-Tuned Embeddings t-SNE plot
    scatter2 = ax2.scatter(
        fine_tuned_tsne[:, 0], fine_tuned_tsne[:, 1], c=yields, cmap="viridis"
    )
    ax2.set_title("Fine-Tuned Embeddings t-SNE")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    # Fine-Tuned Embeddings with predicted yields t-SNE plot
    with torch.no_grad():
        predicted_yields = finetuning_model(original_embeddings)
        scatter3 = ax3.scatter(
            fine_tuned_tsne[:, 0],
            fine_tuned_tsne[:, 1],
            c=predicted_yields,
            cmap="viridis",
        )

        scatter5 = ax5.scatter(
            finefine_tuned_tsne[:, 0],
            finefine_tuned_tsne[:, 1],
            # HERE AS WELL
            c=yields,
            cmap="viridis",
        )

    ax5.set_title("Just finetuned with actual yield")
    ax5.set_xlabel("Component 1")
    ax5.set_ylabel("Component 2")

    ax3.set_title("Finetuning predictions")
    ax3.set_xlabel("Component 1")
    ax3.set_ylabel("Component 2")

    # Fine-Tuned Embeddings with predicted yields t-SNE plot
    with torch.no_grad():
        predicted_yields = surrogate_model.predict(
            fine_tuned_embeddings, return_var=False
        )
        scatter4 = ax4.scatter(
            fine_tuned_tsne[:, 0],
            fine_tuned_tsne[:, 1],
            c=predicted_yields.squeeze().numpy(),
            cmap="viridis",
        )
    ax4.set_title("GP predictions t-SNE")
    ax4.set_xlabel("Component 1")
    ax4.set_ylabel("Component 2")

    # Colorbar for Fine-Tuned Embeddings t-SNE plot
    cbar = fig.colorbar(scatter2, ax=ax2, orientation="vertical")
    cbar.set_label("Yield")

    # Log the plot to wandb
    wandb.log({"valid/t-SNE-orig-vs-finetune": wandb.Image(plt), "epoch": epoch})
    plt.close()


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def calculate_distances(
    embeddings, scores, high_score_threshold=70, low_score_threshold=20
):
    """
    Plots histograms of L2 distances between points in latent space.

    Args:
    - embeddings (np.array): The embeddings in the latent space.
    - scores (np.array): The scores corresponding to each embedding.
    - high_score_threshold (float): The threshold to consider a score as high.
    - low_score_threshold (float): The threshold to consider a score as low.
    """
    # Determine high and low scores
    high_scores = (scores >= high_score_threshold).squeeze()
    low_scores = (scores < low_score_threshold).squeeze()

    # Calculate pairwise L2 distances
    distances = squareform(pdist(embeddings, "euclidean"))

    # Extract distances based on scores
    high_to_high_distances = distances[np.ix_(high_scores, high_scores)].flatten()
    high_to_low_distances = distances[np.ix_(high_scores, low_scores)].flatten()
    low_to_low_distances = distances[np.ix_(low_scores, low_scores)].flatten()

    return high_to_high_distances, high_to_low_distances, low_to_low_distances


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


def plot_distance_histograms(
    high_to_high_distances, high_to_low_distances, low_to_low_distances, bins=50
):
    """
    Plots histograms of L2 distances between points in latent space.

    Args:
    - bins (int): Number of bins for the histogram.
    """

    # Plot histograms
    fig = plt.figure(figsize=(20, 5))

    sns.histplot(
        high_to_high_distances,
        bins=bins,
        kde=True,
        color="blue",
        label="High-to-High",
        stat="density",
        element="step",
        fill=True,
        alpha=0.5,
    )
    sns.histplot(
        high_to_low_distances,
        bins=bins,
        kde=True,
        color="orange",
        label="High-to-Low",
        stat="density",
        element="step",
        fill=True,
        alpha=0.5,
    )
    sns.histplot(
        low_to_low_distances,
        bins=bins,
        kde=True,
        color="green",
        label="Low-to-Low",
        stat="density",
        element="step",
        fill=True,
        alpha=0.5,
    )

    plt.xlabel("L2 distance")
    plt.ylabel("Normalized count")
    plt.legend()
    plt.tight_layout()

    return fig


def retrieve_indices(heldout_x, x_next):
    indices = []
    for point in x_next:
        # Find the index in heldout_x that matches point
        index = (heldout_x == point.unsqueeze(0)).all(dim=1).nonzero(as_tuple=True)[0]
        indices.append(index.item())
    return indices


def train(config):
    dm = instantiate_class(config["data"])
    data_metrics = calculate_data_metrics(dm)
    log_data_stats(data_metrics)
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    finetune = bo_config["finetuning"]

    bo = BotorchOptimizer(
        original_train_x=dm.train_x.clone(),
        original_heldout_x=dm.heldout_x.clone(),
        train_x=dm.train_x.clone(),  # model.adapter(dm.train_x).detach(), #dm.train_x,
        train_y=dm.train_y.clone(),
        heldout_x=dm.heldout_x.clone(),  # dm.heldout_x,
        heldout_y=dm.heldout_y.clone(),
        acq_function_config=acquisition_config,
        surrogate_model_config=surrogate_model_config,
        batch_size=bo_config["batch_size"],
        batch_strategy=bo_config["batch_strategy"],
        finetuning=bo_config["finetuning"],
        emb_criterion=bo_config["emb_criterion"],
        threshold=bo_config["threshold"],
        margin=bo_config["margin"],
        concat=bo_config["concat"],
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

        # TODO: this is not correct update
        if not torch.all(matches.sum(dim=-1) == 1):
            print("Unable to find a unique match for some x_next in the dataset.")

        # Using the indices found, directly index into self.heldout_y to get y_next
        y_next = bo.heldout_y[indices]

        wandb.log(
            {"evaluated_suggestions": wandb.Histogram(y_next.numpy()), "epoch": i}
        )

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
        # beta = beta_scheduler(i, config["n_iters"], 5, 0, 1)
        bo.tell(x_next, y_next, beta=1)  # x_next
        # Update indices tracking
        evaluated_original_indices = dm.heldout_indices[indices]
        dm.train_indexes = np.append(dm.train_indexes, evaluated_original_indices)
        dm.heldout_indices = np.delete(dm.heldout_indices, indices)

        assert len(np.unique(dm.train_indexes)) == len(
            dm.train_indexes
        ), "Duplicates found in dm.train_indexes"
        assert len(np.unique(dm.heldout_indices)) == len(
            dm.heldout_indices
        ), "Duplicates found in dm.heldout_indices"

        # Check for any common indices between dm.train_indexes and dm.heldout_indices
        common_indices = np.intersect1d(dm.train_indexes, dm.heldout_indices)
        assert (
            len(common_indices) == 0
        ), f"Common indices found between train and heldout: {common_indices}"
        total_indices = len(dm.train_indexes) + len(dm.heldout_indices)
        assert total_indices == len(dm.data), "Mismatch in the total number of indices"

        if finetune:
            if (
                i == 1 or i == config["n_iters"] - 1
            ):  # plot the first and last iteration
                plot_tsne_to_wandb(
                    i,
                    bo.original_heldout_x,
                    bo.heldout_x,
                    bo.finetuned_embeddings,
                    bo.heldout_y,
                    bo.finetuning_model,
                    bo.surrogate_model,
                )

                (
                    high_to_high_distances_finetuned,
                    high_to_low_distances_finetuned,
                    low_to_low_distances_finetuned,
                ) = calculate_distances(
                    torch.cat(
                        [bo.finetuned_train_embeddings, bo.finetuned_embeddings], dim=0
                    ),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                )

                (
                    high_to_high_distances_original,
                    high_to_low_distances_original,
                    low_to_low_distances_original,
                ) = calculate_distances(
                    torch.cat([bo.original_train_x, bo.original_heldout_x], dim=0),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                )

                (
                    high_to_high_distances_concat,
                    high_to_low_distances_concat,
                    low_to_low_distances_concat,
                ) = calculate_distances(
                    torch.cat([bo.train_x, bo.heldout_x], dim=0),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                )

                data_high_to_high_distances_concat = [
                    [d] for d in high_to_high_distances_concat
                ]
                data_high_to_low_distances_concat = [
                    [d] for d in high_to_low_distances_concat
                ]
                data_low_to_low_distances_concat = [
                    [d] for d in low_to_low_distances_concat
                ]
                data_high_to_high_distances_finetuned = [
                    [d] for d in high_to_high_distances_finetuned
                ]
                data_high_to_low_distances_finetuned = [
                    [d] for d in high_to_low_distances_finetuned
                ]
                data_low_to_low_distances_finetuned = [
                    [d] for d in low_to_low_distances_finetuned
                ]
                data_high_to_high_distances_original = [
                    [d] for d in high_to_high_distances_original
                ]
                data_high_to_low_distances_original = [
                    [d] for d in high_to_low_distances_original
                ]
                data_low_to_low_distances_original = [
                    [d] for d in low_to_low_distances_original
                ]

                table_high_to_high_distances_concat = wandb.Table(
                    data=data_high_to_high_distances_concat, columns=["distances"]
                )
                table_low_to_low_distances_concat = wandb.Table(
                    data=data_low_to_low_distances_concat, columns=["distances"]
                )
                table_high_to_low_distances_concat = wandb.Table(
                    data=data_high_to_low_distances_concat, columns=["distances"]
                )
                table_high_to_high_distances_finetuned = wandb.Table(
                    data=data_high_to_high_distances_finetuned, columns=["distances"]
                )
                table_low_to_low_distances_finetuned = wandb.Table(
                    data=data_low_to_low_distances_finetuned, columns=["distances"]
                )
                table_high_to_low_distances_finetuned = wandb.Table(
                    data=data_high_to_low_distances_finetuned, columns=["distances"]
                )
                table_high_to_high_distances_original = wandb.Table(
                    data=data_high_to_high_distances_original, columns=["distances"]
                )
                table_low_to_low_distances_original = wandb.Table(
                    data=data_low_to_low_distances_original, columns=["distances"]
                )
                table_high_to_low_distances_original = wandb.Table(
                    data=data_high_to_low_distances_original, columns=["distances"]
                )

                distances_plot_finetuned = plot_distance_histograms(
                    high_to_high_distances_finetuned,
                    high_to_low_distances_finetuned,
                    low_to_low_distances_finetuned,
                )
                distances_plot_concat = plot_distance_histograms(
                    high_to_high_distances_concat,
                    high_to_low_distances_concat,
                    low_to_low_distances_concat,
                )
                distances_plot_original = plot_distance_histograms(
                    high_to_high_distances_original,
                    high_to_low_distances_original,
                    low_to_low_distances_original,
                )

                wandb.log(
                    {
                        "valid/distances_concat": wandb.Image(distances_plot_concat),
                        "valid/distances_finetuned": wandb.Image(
                            distances_plot_finetuned
                        ),
                        "valid/distances_original": wandb.Image(
                            distances_plot_original
                        ),
                        "epoch": i,
                    }
                )
                wandb.log(
                    {
                        "original_low_to_low": table_low_to_low_distances_original,
                        "original_high_to_low": table_high_to_low_distances_original,
                        "original_high_to_high": table_high_to_high_distances_original,
                        "finetuned_low_to_low": table_low_to_low_distances_finetuned,
                        "finetuned_high_to_low": table_high_to_low_distances_finetuned,
                        "finetuned_high_to_high": table_high_to_high_distances_finetuned,
                        "concat_low_to_low": table_low_to_low_distances_concat,
                        "concat_high_to_low": table_high_to_low_distances_concat,
                        "concat_high_to_high": table_high_to_high_distances_concat,
                        "epoch": i,
                    }
                )
                dm.data["Set"] = "Valid"
                dm.data.loc[dm.train_indexes, "Set"] = "Train"
                additional_info_df = dm.data.iloc[
                    np.concatenate([dm.train_indexes, dm.heldout_indices])
                ][["additive", "base", "ligand", "aryl halide", "yield"]]

                with torch.no_grad():
                    gp_preds = bo.surrogate_model.predict(
                        torch.cat([bo.train_x, bo.heldout_x], dim=0), return_var=False
                    )
                    finetuning_model_preds = bo.finetuning_model(
                        torch.cat([bo.original_train_x, bo.original_heldout_x], dim=0)
                    )

                log_embeddings_with_yields(
                    torch.cat(
                        [bo.finetuned_train_embeddings, bo.finetuned_embeddings], dim=0
                    ),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                    additional_info_df,
                    "finetuned_embeddings",
                )
                log_embeddings_with_yields(
                    torch.cat([bo.train_x, bo.heldout_x], dim=0),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                    additional_info_df,
                    "concatenated_embeddings",
                    gp_preds,
                )

                log_embeddings_with_yields(
                    torch.cat([bo.original_train_x, bo.original_heldout_x], dim=0),
                    torch.cat([bo.train_y, bo.heldout_y], dim=0),
                    additional_info_df,
                    "original_embeddings",
                    finetuning_model_preds,
                )

    wandb.finish()


def main():
    # Initialize the parser with a description
    parser = ArgumentParser(
        description="Training script",
        default_config_files=["config.yaml"],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")  # nargs="+",
    parser.add_argument("--n_iters", type=int, help="How many iterations to run")

    parser.add_argument("--group", type=str, help="Wandb group runs")

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
        parser.link_arguments(
            "data.train_y",
            "acquisition.init_args.best_f",
            apply_on="instantiate",
            compute_fn=torch.max,
        )
    run = wandb.init(
        project="bochemian_paper", config=wandb_config, group=args["group"]
    )
    train(args.as_dict())


if __name__ == "__main__":
    main()
