"""Tile coding for action-value features (S&B 9.5.4).

Thin wrapper over Sutton's reference tile coder (`tiles3.py`, vendored verbatim).
It maps a continuous state plus a discrete action to a binary feature vector
x(s, a) -- a `StateActionFeatureMap` -- so `LinearActionValue` and the control
learners use it through the standard interface.

The action enters as an integer tile coordinate, so each action gets its own
tiles within the shared IHT (the standard tiles3 idiom for q(s, a)): one flat
weight vector, three disjoint per-action regions.
"""

from __future__ import annotations
import numpy as np

from rl.features.base import StateActionFeatureMap, State
from rl.features import tiles3


class TileCoding(StateActionFeatureMap):
    """N-dimensional grid tile coding of (state, action) via an index-hash table.

    Args:
        n_tilings: number of overlapping tilings (e.g. 8). Each contributes one
            active tile per query, so exactly `n_tilings` features are active.
        tiles_per_dim: tiles spanning each dimension's range, so a tile covers
            1/tiles_per_dim of that range (the resolution).
        dim_bounds: list of (low, high) per state dimension, used to scale each
            input so its range spans `tiles_per_dim` units (what tiles3 expects).
        iht_size: size of the hash table = `n_features`. Must exceed the number
            of distinct (tile, action) keys actually visited or hashing collides
            (mountain car uses 4096).
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

    def __call__(self, s: State, a: int) -> np.ndarray:
        """Dense binary feature vector x(s, a) of length `n_features`.

        Exactly `n_tilings` entries are 1 (the active tiles for this (s, a)); the
        action is hashed in as an extra tile coordinate, so distinct actions never
        share tiles.
        """
        scaled = [sc * (x - lo) for x, sc, lo in zip(s, self._scale, self._low)]
        active = tiles3.tiles(self.iht, self.n_tilings, scaled, [a])
        x = np.zeros(self.n_features)
        x[active] = 1.0
        return x
