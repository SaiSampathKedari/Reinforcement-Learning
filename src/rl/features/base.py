"""Feature maps x(s) for linear function approximation.

A FeatureMap is the *representation*: it turns a state into a feature vector
x(s), and a linear model predicts v_hat(s) = w . x(s). This base class plays the
same role for representations that `TabularEnv` plays for environments -- every
feature map (state aggregation, tile coding, polynomial / Fourier bases)
subclasses it and exposes one interface, so the learning algorithms work against
any representation unchanged.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class FeatureMap(ABC):
    """Map a state to a feature vector x(s) of length `n_features`."""

    n_features: int

    @abstractmethod
    def __call__(self, s: int) -> np.ndarray:
        """Return the feature vector x(s), shape (n_features,)."""

    def matrix(self, n_states: int) -> np.ndarray:
        """Stack x(s) for s = 0..n_states-1 into X, shape (n_states, n_features).

        For finite state spaces this lets you get all values at once as `X @ w`.
        """
        return np.stack([self(s) for s in range(n_states)])
