from typing import Any, Dict, Optional
from bochemian.acquisition.function import calculate_density
from bochemian.data.utils import torch_delete_rows
import torch
import warnings
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from bochemian.utils import instantiate_class
from torch import Tensor

import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn import MSELoss
from bochemian.data.dataset import FineTuningDataset
from bochemian.finetuning.adapter import FineTuningModel
from bochemian.finetuning.losses import (
    ContrastiveLossTorch,
    LogRatioLossTorch,
    TripletLossTorch,
)
from bochemian.finetuning.training import (
    train_finetuning_model,
)
from bochemian.finetuning.embeddings import update_embeddings
from typing import Optional


class BotorchOptimizer:
    def __init__(
        self,
        design_space: Optional[Tensor] = None,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        original_train_x: Optional[Tensor] = None,
        original_heldout_x: Optional[Tensor] = None,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
        heldout_x: Optional[Tensor] = None,
        heldout_y: Optional[Tensor] = None,
        batch_strategy: str = "cl_min",
        batch_size: int = 1,
        fixed_features: Optional[Dict[int, float]] = None,
        tkwargs: Optional[Dict[str, Any]] = {
            "device": torch.device("cpu"),
            "dtype": torch.double,
        },
        finetuning: bool = False,
        emb_criterion: str = "contrastive",
        threshold: float = 0.1,
        margin: float = 0.2,
        concat: bool = True,
    ):
        self.finetuning = finetuning
        self.concat = concat
        if emb_criterion == "contrastive":
            self.emb_criterion = ContrastiveLossTorch(threshold=threshold)
        elif emb_criterion == "logratio":
            self.emb_criterion = LogRatioLossTorch()
        elif emb_criterion == "triplet":
            self.emb_criterion = TripletLossTorch(threshold=threshold, margin=margin)
        self.design_space = design_space
        self.surrogate_model_config = (
            surrogate_model_config or BotorchOptimizer.default_surrogate_model_config()
        )
        self.acq_function_config = (
            acq_function_config or BotorchOptimizer.default_acq_function_config()
        )

        self.pending_x = []

        self.train_x = None
        self.train_y = None
        self.heldout_x = heldout_x.to(**tkwargs) if heldout_x is not None else None
        if self.heldout_x is not None:
            if heldout_y is not None:
                self.heldout_y = heldout_y.to(**tkwargs)
            else:
                self.heldout_y = torch.full(
                    (len(self.heldout_x), 1),
                    fill_value=torch.nan,
                    dtype=tkwargs["dtype"],
                    device=tkwargs["device"],
                )
        else:
            self.heldout_y = None
        self.batch_strategy = batch_strategy
        self.batch_size = batch_size
        self.fixed_features = fixed_features

        train_x = train_x.to(**tkwargs) if train_x is not None else None
        train_y = train_y.to(**tkwargs) if train_y is not None else None

        self.original_train_x = original_train_x
        self.original_heldout_x = original_heldout_x

        if train_x is not None and train_y is not None:
            self.tell(train_x, train_y)

        self.tkwargs = tkwargs

    def define_design_space(
        self, train_x: Optional[torch.Tensor] = None, raw_samples: int = 10000
    ) -> torch.Tensor:
        if train_x is not None:
            lower_bounds = train_x.amin(dim=0)
            upper_bounds = train_x.amax(dim=0)
            # TODO: Define expansion factor based on the problem and user input
            expansion_factor = 0.1
            lower_bounds = lower_bounds - expansion_factor * (
                upper_bounds - lower_bounds
            )
            upper_bounds = upper_bounds + expansion_factor * (
                upper_bounds - lower_bounds
            )
            bounds = torch.stack([lower_bounds, upper_bounds])

        else:
            # TODO: Define design space based on the problem and user input
            print(
                "Please define the design space bounds (lower and upper bounds for each dimension):"
            )
            lower_bounds = torch.tensor([0.0, 0.0])
            upper_bounds = torch.tensor([1.0, 1.0])
            bounds = torch.stack([lower_bounds, upper_bounds])
        # TODO use optimize_acqf to generate the heldout set gotta see how that would work
        self.heldout_x = (
            draw_sobol_samples(bounds=bounds, n=raw_samples, q=1)
            .to(**self.tkwargs)
            .squeeze()
        )
        return self.heldout_x

    def lie_to_me(self, candidate, train_y, strategy="kriging"):
        supported_strategies = ["cl_min", "cl_mean", "cl_max", "kriging"]
        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of "
                + str(supported_strategies)
                + ", "
                + "got %s" % strategy
            )

        if strategy == "cl_min":
            y_lie = (
                torch.min(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-min lie
        elif strategy == "cl_mean":
            y_lie = (
                torch.mean(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-mean lie
        elif strategy == "cl_max":
            y_lie = (
                torch.max(train_y).view(-1, 1) if train_y.numel() > 0 else 0.0
            )  # CL-max lie
        else:
            y_lie, _ = self.surrogate_model.predict(candidate)
        return y_lie

    def ask(self, n_points=None, strategy="cl_min"):
        if n_points is None:
            n_points = self.batch_size

        if self.surrogate_model is None:
            warnings.warn(
                "Surrogate model has not been initialized. Call 'tell' with initial data before calling 'ask'."
            )
            return self.pending_x

        # Save a copy of the pre_batch heldout_x and heldout_y
        pre_batch_heldout_x = self.heldout_x.clone()
        pre_batch_heldout_y = self.heldout_y.clone()

        candidates = []
        candidate_indices = []
        for i in range(n_points):
            candidate, idx = self.next_evaluations(
                self.acq_function,
                # self.design_space,
                # num_restarts=50,
                # raw_samples=10000,
                # q=1,
            )
            candidates.append(candidate)
            candidate_indices.append(idx.item())
            # TODO this is a mistake because return cadidates will exit the function?
            if n_points == 1:
                return candidates
            y_lie = self.lie_to_me(
                candidate, self.train_y, strategy=self.batch_strategy
            )

            self.tell(candidate, y_lie, lie=True)
            self.heldout_x = torch_delete_rows(self.heldout_x, idx)
            self.heldout_y = torch_delete_rows(self.heldout_y, idx)

        self.candidate_indices = candidate_indices
        self.train_x = self.train_x[:-n_points]
        self.train_y = self.train_y[:-n_points]
        self.pending_x.extend(candidates)

        self.heldout_x = pre_batch_heldout_x
        self.heldout_y = pre_batch_heldout_y

        return candidates

    def tell(self, x, y, lie=False, beta=0):
        self.train_x = torch.cat([self.train_x, x]) if self.train_x is not None else x
        self.train_y = torch.cat([self.train_y, y]) if self.train_y is not None else y

        # TODO: would have to be changed because the experiment might not be exactly the same
        if len(self.pending_x) > 0 and not lie:
            pending_x_stack = torch.stack(self.pending_x)
            eq_mask = (pending_x_stack.unsqueeze(0) == x.unsqueeze(1)).all(dim=-1)
            indices_to_keep = ~torch.any(eq_mask, dim=0)
            self.pending_x = [
                self.pending_x[i]
                for i in range(len(self.pending_x))
                if indices_to_keep[i]
            ]

        # If heldout_x and heldout_y are not None, remove the points in x from heldout_x and heldout_y
        if self.heldout_x is not None and self.heldout_y is not None and not lie:
            eq_mask = torch.eq(self.heldout_x.unsqueeze(0), x.unsqueeze(1)).all(dim=-1)
            # Finding indices to delete
            matches = (self.heldout_x.unsqueeze(0) == x.unsqueeze(1)).all(dim=-1)
            indices_to_delete = matches.nonzero(as_tuple=True)[1]
            if indices_to_delete.numel() > 0:
                self.heldout_x = torch_delete_rows(self.heldout_x, indices_to_delete)
                self.heldout_y = torch_delete_rows(self.heldout_y, indices_to_delete)

                # Update original data
                original_x_to_move = torch.index_select(
                    self.original_heldout_x, 0, indices_to_delete
                )
                self.original_train_x = torch.cat(
                    [self.original_train_x, original_x_to_move]
                )
                self.original_heldout_x = torch_delete_rows(
                    self.original_heldout_x, indices_to_delete
                )

            if self.finetuning:
                embedding_dim = self.original_train_x.shape[1]
                self.finetuning_model = FineTuningModel(embedding_dim, 1)
                self.finetuning_model.to(self.train_x.dtype)
                finetuning_train_dataset = FineTuningDataset(
                    self.original_train_x.clone(), self.train_y.clone()
                )
                finetuning_valid_dataset = FineTuningDataset(
                    self.original_heldout_x.clone(), self.heldout_y.clone()
                )

                # finetuning_dataset = TripletFineTuningDataset(
                #     self.original_train_x.clone(), self.train_y.clone()
                # )
                train_loader = torch.utils.data.DataLoader(
                    finetuning_train_dataset, batch_size=8, shuffle=True, drop_last=True
                )
                # valid_loader = torch.utils.data.DataLoader(
                #     finetuning_valid_dataset, batch_size=8, shuffle=True, drop_last=True
                # )

                mse_criterion = MSELoss(reduction="none")
                optimizer = optim.AdamW(self.finetuning_model.parameters(), lr=0.001)
                train_finetuning_model(
                    self.finetuning_model,
                    train_loader,
                    self.emb_criterion,
                    mse_criterion,
                    optimizer,
                    5 * int(self.original_train_x.shape[0] / 5),
                    beta=beta,
                )
                # evaluate_model(self.finetuning_model, valid_loader, mse_criterion)
                with torch.no_grad():
                    self.train_x = update_embeddings(
                        self.finetuning_model,
                        self.original_train_x.clone(),
                        concatenation=self.concat,
                    )
                    self.heldout_x = update_embeddings(
                        self.finetuning_model,
                        self.original_heldout_x.clone(),
                        concatenation=self.concat,
                    )

                    self.finetuned_embeddings = self.finetuning_model.fc1(
                        self.finetuning_model.adapter(self.original_heldout_x.clone())
                    )
                    self.finetuned_train_embeddings = self.finetuning_model.fc1(
                        self.finetuning_model.adapter(self.original_train_x.clone())
                    )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config,
                train_x=self.train_x,
                train_y=self.train_y,
            )
            self.surrogate_model.fit(self.train_x, self.train_y)

            additional_acq_function_params = self.update_acquisition_function_params()
            self.acq_function = instantiate_class(
                self.acq_function_config, **additional_acq_function_params
            )

    def next_evaluations(self, acq_function):
        X = self.heldout_x.unsqueeze(1)
        self.surrogate_model.eval()
        acq_values = acq_function(X).squeeze()

        # Get the indices of the best points based on acquisition values
        best_indices = acq_values.topk(1)[1]
        best_point = X[best_indices].squeeze(1)
        return best_point, best_indices

    @staticmethod
    def default_surrogate_model_config():
        # Return default surrogate model config
        return {
            "class_path": "bochemian.surrogate_models.gp.SimpleGP",
            "init_args": {
                "covar_module": {
                    "class_path": "gpytorch.kernels.ScaleKernel",
                    "init_args": {
                        "base_kernel": {
                            "class_path": "gpytorch.kernels.MaternKernel",
                            "init_args": {"nu": 2.5},
                        }
                    },
                },
                "likelihood": {
                    "class_path": "gpytorch.likelihoods.GaussianLikelihood",
                    "init_args": {"noise": 1e-4},
                },
                "normalize": False,
                "initial_noise_val": 1.0e-4,
                "noise_constraint": 1.0e-05,
                "initial_outputscale_val": 2.0,
                "initial_lengthscale_val": 0.5,
            },
        }

    @staticmethod
    def default_acq_function_config():
        # Return default acquisition function config
        return {
            "class_path": "botorch.acquisition.UpperConfidenceBound",
            "init_args": {"beta": 2.0, "maximize": True},
        }

    def update_acquisition_function_params(self):
        params = {"model": self.surrogate_model}

        if "ExpectedImprovement" in self.acq_function_config["class_path"]:
            params["best_f"] = self.train_y.max().item()
            if "CustomExpectedImprovement" in self.acq_function_config["class_path"]:
                params["yield_model"] = self.finetuning_model
            if "EmbeddingInformed" in self.acq_function_config["class_path"]:
                params["train_x"] = self.train_x
                params["train_y"] = self.train_y

        return params
