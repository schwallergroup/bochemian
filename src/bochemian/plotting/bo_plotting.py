import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import matplotlib.cm as cm


class BOPlotter:
    def __init__(self, enable_plotting=False):
        sns.set(style="whitegrid")
        self.enable_plotting = enable_plotting

    def plot_predicted_vs_actual(
        self,
        predicted,
        actual,
        variance,
        title="Predicted vs Actual",
    ):
        # Convert tensors to numpy arrays if needed
        predicted = (
            predicted.cpu().numpy()
            if isinstance(predicted, torch.Tensor)
            else predicted
        )
        actual = actual.cpu().numpy() if isinstance(actual, torch.Tensor) else actual
        # Define colormap
        cmap = cm.get_cmap("viridis")

        # Plot training data
        plt.figure()
        plt.scatter(
            predicted,
            actual,
            c=variance,
            cmap=cmap,
            label="Train",
            edgecolors="blue",
        )
        plt.colorbar(label="Variance")

        min_value = min(
            np.min(predicted), np.min(actual), np.min(predicted), np.min(actual)
        )
        max_value = max(
            np.max(predicted),
            np.max(actual),
            np.max(predicted),
            np.max(actual),
        )

        # Set the same range for both the x and y axes
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            "b--",
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        # plt.legend()
        plt.ticklabel_format(style="plain", axis="both", useOffset=False)
        plot_image = plt.gcf()  # Get the current figure
        plt.close()  # Close the current figure so it doesn't interfere with the next one
        return plot_image

    # def plot_predicted_vs_actual(
    #     self, predicted, actual, variance=None, title="Predicted vs Actual"
    # ):
    #     predicted = (
    #         predicted.cpu().numpy()
    #         if isinstance(predicted, torch.Tensor)
    #         else predicted
    #     )
    #     actual = actual.cpu().numpy() if isinstance(actual, torch.Tensor) else actual
    #     plt.figure()
    #     plt.scatter(predicted, actual, label="Predicted")
    #     plt.plot([min(predicted), max(predicted)], [min(actual), max(actual)], "r--")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title(title)
    #     plt.ticklabel_format(style="plain", axis="both", useOffset=False)
    #     return plt

    def plot_best_so_far(self, best_values, title="Best So Far"):
        plt.plot(best_values)
        plt.xlabel("Iteration")
        plt.ylabel("Best Value")
        plt.title(title)
        plt.show()

    def plot_acquisition_function(self, acq_values, title="Acquisition Function"):
        plt.plot(acq_values)
        plt.xlabel("Iteration")
        plt.ylabel("Acquisition Value")
        plt.title(title)
        plt.show()

    def plot_top_n_count(self, top_n_count, top_n_values, title="Top-N Count"):
        plt.bar(top_n_values, top_n_count)
        plt.xlabel("Top-N")
        plt.ylabel("Count")
        plt.title(title)
        plt.show()

    def plot_residuals(self, predicted, actual, title="Residuals Plot"):
        residuals = predicted - actual
        plt.figure()
        plt.scatter(predicted, residuals)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(title)
        plot_image = plt.gcf()  # Get the current figure
        plt.close()  # Close the current figure so it doesn't interfere with the next one
        return plot_image
