"""State aggregation: the simplest linear feature map (S&B 9.3)."""

from __future__ import annotations
import numpy as np

from rl.features.base import FeatureMap


class StateAggregation(FeatureMap):
    """Partition a set of states into `n_groups`; x(s) is one-hot over the group.

    So v_hat(s) = w[group(s)] -- one shared value per group, i.e. a step
    function. States outside the partition (e.g. terminals) map to the zero
    vector, so their value is 0. For the 1000-state walk: pass states 1..1000 and
    n_groups=10 -> 100 states per group.
    """

    def __init__(self, states, n_groups: int):
        self._states = list(states)
        self.n_groups = n_groups
        self.n_features = n_groups
        m = len(self._states)
        # contiguous groups: the i-th aggregated state falls in group i*n_groups//m
        self._group = {s: (i * n_groups) // m for i, s in enumerate(self._states)}

    def __call__(self, s: int) -> np.ndarray:
        x = np.zeros(self.n_features)
        g = self._group.get(s)
        if g is not None:                 # states outside the partition (terminals) stay zero
            x[g] = 1.0
        return x
