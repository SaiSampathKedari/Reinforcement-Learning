"""Mountain Car (S&B Example 10.1, p.245).

An underpowered car in a valley must reach the goal at the top of the right hill.
Gravity is stronger than the engine, so the only solution is to first reverse up
the left slope and build momentum -- a continuous control task where things must
get worse (move away from the goal) before they can get better.

Continuous state `(position, velocity)`; three discrete throttle actions; reward
-1 every step until the goal terminates the episode. The car follows the
simplified physics of eq. (10.something): velocity is nudged by the throttle and
pulled by gravity `-0.0025 * cos(3 * position)`, then position follows velocity,
both clipped to their bounds. Hitting the left wall resets velocity to 0;
reaching the right wall ends the episode.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import Env


class MountainCar(Env):
    """Continuous-state Mountain Car; subclasses `Env` (no enumerable model)."""

    # --- problem constants (the task definition) ---
    POS_MIN, POS_MAX = -1.2, 0.5
    VEL_MIN, VEL_MAX = -0.07, 0.07
    START_MIN, START_MAX = -0.6, -0.4     # initial position range; start velocity 0
    GOAL = POS_MAX                         # reach the right wall -> episode ends

    n_actions = 3                          # 0, 1, 2 -> throttle -1, 0, +1 (fixed)

    def __init__(self, gamma: float = 1.0):
        """`gamma` defaults to 1.0 (undiscounted episodic, as in the book)."""
        self.gamma = gamma

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Start state: position ~ Uniform[START_MIN, START_MAX), velocity 0."""
        position = rng.uniform(self.START_MIN, self.START_MAX)
        return np.array([position, 0.0])

    def step(
        self,
        s: np.ndarray,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, bool]:
        """One step of the simplified physics. Returns (next_state, reward, terminated).

        Velocity is updated first (gravity uses the *current* position), then
        position from the *new* velocity; both are clipped. The left wall resets
        velocity to 0; reaching the goal terminates with the usual -1 reward.
        """
        position, velocity = s
        throttle = a - 1                                  # {0,1,2} -> {-1,0,+1}

        velocity += 0.001 * throttle - 0.0025 * np.cos(3 * position)
        velocity = np.clip(velocity, self.VEL_MIN, self.VEL_MAX)

        position += velocity
        position = np.clip(position, self.POS_MIN, self.POS_MAX)
        if position == self.POS_MIN:                      # left-wall reset
            velocity = 0.0

        terminated = position >= self.GOAL
        return np.array([position, velocity]), -1.0, bool(terminated)
