from typing import Any, Dict, Optional
from bochemian.data.utils import torch_delete_rows
import torch
import warnings
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from bochemian.utils import instantiate_class
from torch import Tensor


class BotorchOptimizer:
    def __init__(
        self,
        design_space: Optional[Tensor] = None,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
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
    ):
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

        if train_x is not None and train_y is not None:
            self.tell(train_x, train_y)
        self.tkwargs = tkwargs

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

        candidates = []
        for i in range(n_points):
            candidate, _ = self.next_evaluations(
                self.acq_function,
                self.design_space,
                num_restarts=50,
                raw_samples=10000,
                q=1,
            )
            candidates.append(candidate)
            if n_points == 1:
                return candidates
            y_lie = self.lie_to_me(
                candidate, self.train_y, strategy=self.batch_strategy
            )

            self.tell(candidate, y_lie, lie=True)

        self.train_x = self.train_x[:-n_points]
        self.train_y = self.train_y[:-n_points]
        self.pending_x.extend(candidates)
        return candidates

    def tell(self, x, y, lie=False):
        self.train_x = torch.cat([self.train_x, x]) if self.train_x is not None else x
        self.train_y = torch.cat([self.train_y, y]) if self.train_y is not None else y

        if len(self.pending_x) > 0 and not lie:
            pending_x_stack = torch.stack(self.pending_x)
            eq_mask = (pending_x_stack.unsqueeze(0) == x.unsqueeze(1)).all(dim=-1)
            indices_to_keep = ~torch.any(eq_mask, dim=0)
            self.pending_x = [
                self.pending_x[i]
                for i in range(len(self.pending_x))
                if indices_to_keep[i]
            ]

        # would have to be changed because the experiment might not be exactly the same

        # If heldout_x and heldout_y are not None, remove the points in x from heldout_x and heldout_y
        if self.heldout_x is not None and self.heldout_y is not None and not lie:
            eq_mask = torch.eq(self.heldout_x.unsqueeze(0), x.unsqueeze(1)).all(dim=-1)
            # Finding indices to delete
            indices_to_delete = torch.any(eq_mask, dim=0).nonzero(as_tuple=True)[0]
            if indices_to_delete.numel() > 0:
                self.heldout_x = torch_delete_rows(self.heldout_x, indices_to_delete)
                self.heldout_y = torch_delete_rows(self.heldout_y, indices_to_delete)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate_model = instantiate_class(
                self.surrogate_model_config, train_x=self.train_x, train_y=self.train_y
            )
            self.surrogate_model.fit(self.train_x, self.train_y)

            additional_acq_function_params = self.update_acquisition_function_params()
            self.acq_function = instantiate_class(
                self.acq_function_config, **additional_acq_function_params
            )

    def next_evaluations(
        self,
        acq_function,
        bounds=None,
        num_restarts=50,
        raw_samples=10000,
        categorical_mask=None,
        q=1,
    ):
        # Determine the set of points to use
        if self.heldout_x is not None:
            # Convert heldout set to the appropriate tensor format
            X = self.heldout_x.unsqueeze(1)
            self.surrogate_model.eval()
            # Get the acquisition function values at these points
            acq_values = acq_function(X).squeeze()

            # If using a heldout set, return all the points directly without further optimization
            best_index = acq_values.topk(1)[1].item()
            best_point = X[best_index].squeeze(1)
            best_acq_value = acq_values[best_index]
            return best_point, best_acq_value

        else:
            # Sample a large number of points from the design space
            X = draw_sobol_samples(bounds=bounds, n=raw_samples, q=q).to(**self.tkwargs)
            if categorical_mask is not None:
                # If there are categorical variables, round them to the nearest integer
                X[..., categorical_mask] = X[..., categorical_mask].round()

            # Get the acquisition function values at these points
            acq_values = acq_function(X).squeeze()

            # Get the best points based on acquisition values
            best_points = X[acq_values.topk(num_restarts)[1]][0].squeeze(1)

            # If not using a heldout set, optimize the acquisition function starting from the best points
            candidate, acq_val = optimize_acqf(
                acq_function,
                bounds=bounds,
                q=1,  # we're getting one point at a time
                num_restarts=num_restarts,
                batch_initial_conditions=best_points,  # start optimization from the best sampled points
                fixed_features=self.fixed_features,
            )

            return candidate, acq_val

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

        return params
