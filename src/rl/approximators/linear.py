"""Linear approximators: v_hat(s) = w . x(s) and q_hat(s,a) = w . x(s,a)."""

from __future__ import annotations
import numpy as np
from typing import Sequence

from rl.features.base import FeatureMap, StateActionFeatureMap
from rl.approximators.base import ValueApproximator, ActionValueApproximator, State


class LinearValue(ValueApproximator):
    """Linear state-value approximator over a feature map (holds the weights w).

    Standard split (the same one neural nets use): the *model* only predicts and
    reports its gradient; the *learner* owns the weight update. For a linear model
    the gradient of v_hat(s) = w . x(s) w.r.t. w is just x(s). When this becomes a
    neural net later, `value` becomes a forward pass and `grad` is replaced by
    autograd, but the learner still calls the same interface.
    """

    def __init__(self, features: FeatureMap):
        self.features = features
        self.w = np.zeros(features.n_features)

    def value(self, s: int) -> float:
        """v_hat(s) = w . x(s)."""
        return float(self.w @ self.features(s))

    def grad(self, s: int) -> np.ndarray:
        """Gradient of v_hat(s) w.r.t. w -- for a linear model, just x(s)."""
        return self.features(s)

    def all_values(self, n_states: int) -> np.ndarray:
        """v_hat for s = 0..n_states-1, vectorized as X @ w (terminals come out 0)."""
        return self.features.matrix(n_states) @ self.w
    

class LinearActionValue(ActionValueApproximator):
    """Linear action-value approximator q_hat(s, a) = w . x(s, a).

    One flat weight vector `w` over a `StateActionFeatureMap`. The action lives in
    the features x(s, a) (for tile coding, as a hashed-in tile coordinate), so the
    per-action regions of `w` are disjoint -- the q-surfaces share one `w` but
    never overlap. Same `w . x` form as `LinearValue`, just with the action
    threaded into the features.
    """

    def __init__(self, features: StateActionFeatureMap):
        self.features = features
        self.w = np.zeros(features.n_features)

    def value(self, s: State, a: int) -> float:
        """q_hat(s, a) = w . x(s, a)."""
        return float(self.w @ self.features(s, a))

    def grad(self, s: State, a: int) -> np.ndarray:
        """Gradient of q_hat(s, a) w.r.t. w -- for a linear model, just x(s, a)."""
        return self.features(s, a)

    def greedy(self, s: State, actions: Sequence[int], rng: np.random.Generator) -> int:
        """argmax_a q_hat(s, a) over `actions`, ties broken uniformly via `rng`."""
        q = np.array([self.value(s, a) for a in actions])
        best = np.flatnonzero(q == q.max())          # all argmax actions
        return int(actions[rng.choice(best)])         # uniform tie-break
