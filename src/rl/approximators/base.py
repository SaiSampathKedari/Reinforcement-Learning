"""Base class for value-function approximators v_hat(s) = f(s; w).

The standard model/learner split: the *model* (subclass) holds the parameters
`w`, predicts `value(s)`, and reports its gradient `grad(s) = d v_hat / d w`; the
*learner* (e.g. gradient MC) owns the update, typically `w += alpha * delta *
grad(s)`. Linear, tabular, and neural approximators all implement this one
interface, so the learners are written against it -- not against a concrete type.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ValueApproximator(ABC):
    """Parameterized state-value function with a parameter vector `w`."""

    w: np.ndarray

    @abstractmethod
    def value(self, s: int) -> float:
        """v_hat(s)."""

    @abstractmethod
    def grad(self, s: int) -> np.ndarray:
        """Gradient of v_hat(s) with respect to the parameters w."""

    @abstractmethod
    def all_values(self, n_states: int) -> np.ndarray:
        """v_hat for s = 0..n_states-1 (vectorized)."""
