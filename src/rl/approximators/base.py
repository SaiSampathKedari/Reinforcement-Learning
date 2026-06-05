"""Base classes for value-function approximators.

The standard model/learner split: the *model* (subclass) holds the parameters
`w`, predicts the value, and reports its gradient `d value / d w`; the *learner*
(e.g. gradient MC, semi-gradient Sarsa) owns the update, typically
`w += alpha * delta * grad`. Linear, tabular, and neural approximators all
implement this one interface, so the learners are written against it -- not
against a concrete type.

`ValueApproximator` models v_hat(s); `ActionValueApproximator` models q_hat(s, a)
for control. The state `s` is whatever the feature map consumes -- an int for a
tabular env, a real-valued vector for a continuous one -- so it is typed `State`,
not `int`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np

State = Any  # int for a tabular env, np.ndarray([...]) for a continuous one


class ValueApproximator(ABC):
    """Parameterized state-value function with a parameter vector `w`."""

    w: np.ndarray

    @abstractmethod
    def value(self, s: int) -> float:
        """v_hat(s, w)."""

    @abstractmethod
    def grad(self, s: int) -> np.ndarray:
        """Gradient of v_hat(s, w) with respect to the parameters w."""

    @abstractmethod
    def all_values(self, n_states: int) -> np.ndarray:
        """v_hat for s = 0..n_states-1 (vectorized)."""
        
class ActionValueApproximator(ABC):
    """Parameterized action-value function q_hat(s, a) with parameters `w`.

    For *discrete, enumerable* actions: control picks `argmax_a q_hat(s, a)` over
    the candidate set. (Huge/continuous action spaces need policy approximation
    instead, not this interface.)
    """

    w: np.ndarray

    @abstractmethod
    def value(self, s: State, a: int) -> float:
        """q_hat(s, a, w)."""

    @abstractmethod
    def grad(self, s: State, a: int) -> np.ndarray:
        """Gradient of q_hat(s, a, w) with respect to the parameters w."""

    @abstractmethod
    def greedy(self, s: State, actions: Sequence[int], rng: np.random.Generator) -> int:
        """argmax_a q_hat(s, a) over `actions`, ties broken uniformly via `rng`."""