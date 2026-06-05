"""Feature maps for linear function approximation.

A feature map is the *representation*: it turns an input into a feature vector
that a linear model dots with its weights. Two sibling interfaces, parallel to
the two approximators:

    FeatureMap            -> x(s)     for v_hat(s)   = w . x(s)   (prediction)
    StateActionFeatureMap -> x(s, a)  for q_hat(s,a) = w . x(s,a) (control)

You pick the one matching the value you're approximating, just as you pick
`ValueApproximator` vs `ActionValueApproximator`. They share no base beyond this
docstring -- a common ancestor would only carry `n_features`. (A state map can be
reused for control by per-action blocking, but that's a composition adapter built
on demand, not a shared base.)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

State = Any  # int for a tabular env, np.ndarray([...]) for a continuous one


class FeatureMap(ABC):
    """Map a state to a feature vector x(s) of length `n_features`."""

    n_features: int

    @abstractmethod
    def __call__(self, s: State) -> np.ndarray:
        """Return the feature vector x(s), shape (n_features,)."""

    def matrix(self, n_states: int) -> np.ndarray:
        """Stack x(s) for s = 0..n_states-1 into X, shape (n_states, n_features).

        For finite state spaces this lets you get all values at once as `X @ w`.
        """
        return np.stack([self(s) for s in range(n_states)])


class StateActionFeatureMap(ABC):
    """Map a (state, action) pair to a feature vector x(s, a) of length `n_features`.

    The action is a first-class, required argument (used for q_hat in control).
    """

    n_features: int

    @abstractmethod
    def __call__(self, s: State, a: int) -> np.ndarray:
        """Return the feature vector x(s, a), shape (n_features,)."""
