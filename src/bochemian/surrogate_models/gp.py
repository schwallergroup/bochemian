# from __future__ import annotations
import gpytorch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL, SingleTaskGP
from botorch.models.transforms.input import Normalize, Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.priors import GammaPrior, LogNormalPrior
from torch import Tensor


from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from abc import ABC, abstractmethod

from typing import Optional
import gpytorch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL, SingleTaskGP
from botorch.models.transforms.input import InputTransform, Normalize, Warp
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from gpytorch.priors import GammaPrior, LogNormalPrior
from torch import Tensor
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, List
import torch
from botorch.settings import debug
import numpy as np
from gpytorch.kernels import AdditiveKernel
from jsonargparse import ArgumentParser, ActionConfigFile


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class SimpleGP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 2.0,
        initial_lengthscale_val: float = 0.5,
    ) -> None:
        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        # Check existing parameters in the model
        existing_parameters = {name for name, _ in self.named_parameters()}

        # Only initialize parameters that exist in the model and have a non-None value
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }

        # Apply the initialization
        self.initialize(**hypers_to_use)
        self.standardize = standardize
        self.normalize = normalize
        self.initial_noise_val = initial_noise_val
        self.noise_constraint = noise_constraint

    def fit(self, train_X, train_Y):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        try:
            with debug(True):
                fit_gpytorch_mll(mll)

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(self, x, observation_noise=False, return_var=True):
        self.eval()  # set the model to evaluation mode
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (posterior.mean, posterior.variance) if return_var else posterior.mean


from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts


class SaaSGP(SurrogateModel, SaasFullyBayesianSingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        standardize: bool = True,
        normalize: bool = False,
        warmup_steps: int = 512,
        num_samples: int = 256,
        thinning: int = 16,
    ):
        super().__init__(
            train_x,
            train_y,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning

    def fit(self, train_X, train_Y):
        self.train()
        try:
            with debug(True):
                fit_fully_bayesian_model_nuts(
                    self,
                    warmup_steps=self.warmup_steps,
                    num_samples=self.num_samples,
                    thinning=self.thinning,
                    disable_progbar=True,
                )

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(self, x, observation_noise=False, return_var=True):
        self.eval()  # set the model to evaluation mode
        with torch.no_grad():
            posterior = self.posterior(x)
            mixture_mean = posterior.mixture_mean
            mixture_variance = posterior.mixture_variance
        return (mixture_mean, mixture_variance) if return_var else mixture_mean
