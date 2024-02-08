from botorch.acquisition import ExpectedImprovement
import torch


def calculate_density(embeddings, bandwidth=0.1):
    """
    Calculate the density of points in the embedding space using KDE.

    Args:
        embeddings (numpy.ndarray): The embeddings of the points.
        bandwidth (float): The bandwidth for the KDE. Smaller values will lead to a denser estimation.

    Returns:
        A function that takes new points and returns their density scores.
    """
    kde = gaussian_kde(embeddings.T, bw_method=bandwidth)

    def density_function(X):
        """
        Calculate the density scores for new points.

        Args:
            X (torch.Tensor): New points for which to calculate density.

        Returns:
            torch.Tensor: The density scores for the points in X.
        """
        X_np = X.detach().numpy()
        density_scores = kde.evaluate(X_np.T)
        return torch.from_numpy(density_scores).float()

    return density_function


# class CustomExpectedImprovement(ExpectedImprovement):
#     def __init__(self, model, best_f, maximize=True, density_function=None):
#         super().__init__(model, best_f, maximize=maximize)
#         self.density_function = density_function

#     def forward(self, X):
#         # Standard EI calculation
#         standard_ei = super().forward(X)

#         if self.density_function is not None:
#             # Calculate the density scores for each point
#             density_scores = self.density_function(X)
#             # Adjust the EI values based on density scores
#             adjusted_ei = standard_ei * density_scores
#             return adjusted_ei

#         return standard_ei


class CustomExpectedImprovement(ExpectedImprovement):
    def __init__(self, model, best_f, maximize=True, yield_model=None):
        super().__init__(model, best_f, maximize=maximize)
        self.yield_model = yield_model

    def forward(self, X):
        standard_ei = super().forward(X)
        if self.yield_model is not None:
            yield_predictions = self.yield_model(X.squeeze(1)).squeeze(-1)
            adjusted_ei = (
                standard_ei * yield_predictions / 100
            )  # or any other combination
            return adjusted_ei
        return standard_ei


from botorch.acquisition import ExpectedImprovement


class EmbeddingInformedExpectedImprovement(ExpectedImprovement):
    def __init__(self, model, best_f, train_x=None, train_y=None, maximize=True):
        super().__init__(model=model, best_f=best_f, maximize=maximize)
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, X):
        # Compute the standard EI
        standard_ei = super().forward(X)

        # Compute distances in the embedding space to the known high-yield points
        distances = self.compute_distances(
            X, self.train_x, self.train_y, high_yield_threshold=70
        )

        # Incorporate the distances into the acquisition values
        # Here, we assume that closer distances are better, so we use a negative sign
        # This needs to be adjusted based on the actual relationship in your space
        adjusted_ei = standard_ei / (1 + distances)

        return adjusted_ei

    @staticmethod
    def compute_distances(X, train_x, train_y, high_yield_threshold):
        # Filter train_x based on train_y to consider only high-yield points
        high_yield_embeddings = train_x[(train_y >= high_yield_threshold).squeeze(1)]

        # Calculate Euclidean distances from points in X to high-yield embeddings
        distances = torch.cdist(X, high_yield_embeddings, p=2)

        # Take the minimum distance to the closest high-yield point for each point in X
        min_distances = distances.squeeze(1).min(dim=1).values

        return min_distances
