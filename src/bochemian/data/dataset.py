from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import torch


class SingleSampleDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.samples = ((x, y),)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.samples[idx]


class DynamicSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def reset(self, dataset):
        self.dataset = dataset


def weight_func(targets, epsilon=1e-8):
    """
    Calculate weights for each target based on its value.
    Higher targets get higher weights, with a logarithmic transformation
    to reduce the impact of very high yields.

    Args:
        targets (Tensor): The target yield values.
        epsilon (float): Small value to avoid log(0)

    Returns:
        Tensor: Weights for each target.
    """
    if torch.isnan(targets).any():
        raise ValueError("Targets contain NaN values")

    # Apply a logarithmic transformation with an epsilon to avoid log(0)
    log_targets = torch.log(targets + epsilon)

    # Normalize log-transformed targets to have a minimum of zero
    normalized_log_targets = log_targets - log_targets.min()

    # Avoid division by zero by adding a small epsilon where mean is zero
    mean_log_targets = torch.mean(normalized_log_targets)
    if mean_log_targets == 0:
        mean_log_targets += epsilon

    weights = normalized_log_targets / mean_log_targets
    return weights


class FineTuningDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.weights = data_y / torch.mean(data_y)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.weights[idx]


class TripletFineTuningDataset(Dataset):
    def __init__(self, data_x, data_y, margin=10):
        self.data_x = data_x
        self.data_y = data_y
        # self.margin = margin  # Margin defines how different the yields should be for negative examples

        # Sort data by yield
        sorted_indices = np.argsort(data_y)
        self.sorted_x = data_x[sorted_indices]
        self.sorted_y = data_y[sorted_indices]

        # Precompute positive and negative examples
        # self.positive_examples = []
        # self.negative_examples = []
        # for idx in range(len(self.data_x)):
        #     margin = self.margin
        #     positive_indices = [
        #         i
        #         for i, y in enumerate(self.data_y)
        #         if i != idx and abs(y - self.data_y[idx]) <= self.margin
        #     ]
        #     while not positive_indices and margin <= max(self.data_y):
        #         margin += 1
        #         positive_indices = [
        #             i
        #             for i, y in enumerate(self.data_y)
        #             if i != idx and abs(y - self.data_y[idx]) <= margin
        #         ]

        #     self.positive_examples.append(positive_indices)

        #     margin = self.margin
        #     negative_indices = [
        #         i
        #         for i, y in enumerate(self.data_y)
        #         if i != idx and abs(y - self.data_y[idx]) > margin
        #     ]

        #     # If no negative examples are found, incrementally decrease the margin
        #     while not negative_indices and margin >= 0:
        #         margin -= 1
        #         negative_indices = [
        #             i
        #             for i, y in enumerate(self.data_y)
        #             if i != idx and abs(y - self.data_y[idx]) > margin
        #         ]

        #     self.negative_examples.append(negative_indices)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        # anchor_x = self.data_x[idx]
        # anchor_y = self.data_y[idx]

        # positive_idx = random.choice(self.positive_examples[idx])
        # positive_x = self.data_x[positive_idx]

        # negative_idx = random.choice(self.negative_examples[idx])
        # negative_x = self.data_x[negative_idx]

        # return (anchor_x, positive_x, negative_x), anchor_y

        # Anchor example
        anchor_x = self.sorted_x[idx]
        anchor_y = self.sorted_y[idx]

        # Positive example (next highest yield)
        positive_indices = [
            i for i in range(idx + 1, len(self.sorted_y)) if self.sorted_y[i] > anchor_y
        ]
        positive_idx = positive_indices[0] if positive_indices else idx
        positive_x = self.sorted_x[positive_idx]

        # Negative example (next lowest yield)
        negative_indices = [i for i in range(0, idx) if self.sorted_y[i] < anchor_y]
        negative_idx = negative_indices[-1] if negative_indices else idx
        negative_x = self.sorted_x[negative_idx]

        return (anchor_x, positive_x, negative_x), (
            anchor_y,
            self.sorted_y[positive_idx],
            self.sorted_y[negative_idx],
        )
