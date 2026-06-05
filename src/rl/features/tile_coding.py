"""Tile coding over continuous inputs (S&B 9.5.4).

Thin wrapper over Sutton's reference tile coder (`tiles3.py`, vendored verbatim).
It maps a continuous state -- or state + discrete action -- to a sparse set of
active tile indices in a hashed feature space (an IHT), then exposes that as the
`FeatureMap` interface so `LinearValue` and the gradient learners use it
unchanged.

Two uses:
    - state value v_hat(s):   `tc(state)`              (e.g. Fig 9.10)
    - action value q_hat(s,a): `tc(state, ints=(a,))`  (e.g. Mountain Car, Ch.10)

For action values, the action enters as an integer coordinate so each action gets
its own tiles within the shared IHT -- the standard tiles3 idiom for q(s,a).

Note: unlike the discrete feature maps, this is over *continuous* inputs, so the
`matrix(n_states)` helper (which enumerates integer states) does not apply.
"""

from __future__ import annotations
import numpy as np

from rl.features.base import FeatureMap
from rl.features import tiles3


class TileCoding(FeatureMap):
    """N-dimensional grid tile coding via an index-hash table (IHT).

    Args:
        n_tilings: number of overlapping tilings (e.g. 8). Each contributes one
            active tile per query, so exactly `n_tilings` features are active.
        tiles_per_dim: tiles spanning each dimension's range, so a tile covers
            1/tiles_per_dim of that range (the resolution).
        dim_bounds: list of (low, high) per input dimension, used to scale each
            input so its range spans `tiles_per_dim` units (what tiles3 expects).
        iht_size: size of the hash table = `n_features`. Must exceed the number
            of distinct tiles actually visited or hashing collides (mountain car
            uses 4096).
    """

    def __init__(self, n_tilings, tiles_per_dim, dim_bounds, iht_size=4096):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.dim_bounds = list(dim_bounds)
        self.iht = tiles3.IHT(iht_size)
        self.n_features = iht_size
        # Pre-fold the affine scaling: scaled_i = (tiles_per_dim/(hi-lo)) * (x_i - lo).
        self._scale = [tiles_per_dim / (hi - lo) for (lo, hi) in self.dim_bounds]
        self._low = [lo for (lo, _) in self.dim_bounds]

    def active(self, state, ints=()) -> list[int]:
        """Active tile indices for `state` (length `n_tilings`).

        `ints` are extra integer coordinates (e.g. a discrete action) that key
        distinct tiles within the IHT. Use this in the learner to read/update
        only the active weights -- the sparse, fast path.
        """
        scaled = [sc * (x - lo) for x, sc, lo in zip(state, self._scale, self._low)]
        return tiles3.tiles(self.iht, self.n_tilings, scaled, list(ints))

    def __call__(self, state, ints=()) -> np.ndarray:
        """Dense multi-hot feature vector x(state) of length `n_features`.

        Conforms to `FeatureMap` so `LinearValue` works directly; for speed in
        the inner loop prefer `active()` (the active weights sum to the value).
        """
        x = np.zeros(self.n_features)
        x[self.active(state, ints)] = 1.0
        return x

    def matrix(self, n_states: int) -> np.ndarray:
        raise NotImplementedError(
            "TileCoding is over continuous inputs; matrix(n_states) does not apply."
        )
