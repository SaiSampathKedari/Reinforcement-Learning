"""Episode rollout helper. Returns SARS' trajectories for any tabular method."""

from __future__ import annotations
from typing import Callable
import numpy as np
from rl.envs.base import TabularEnv

Policy = Callable[[int, np.random.Generator], int]  # (state, rng) -> action


def generate_episode(
    env: TabularEnv,
    policy: Policy,
    rng: np.random.Generator,
) -> list[tuple[int, int, float, int, bool]]:
    """Play one episode under `policy`. Returns the SARS' trajectory.

    Trajectory format:
        [(s_t, a_t, r_{t+1}, s_{t+1}, terminated)  for t = 0, ..., T-1]

    Conventions:
        1. Indexing follows S&B: action a_t in state s_t yields reward r_{t+1}.
        2. Tuple layout matches Gymnasium's per-step return
           (obs, action, reward, next_obs, terminated).
        3. Terminal state s_T is `trajectory[-1][3]`.
        4. Sufficient for MC, TD, n-step, and offline eligibility-trace methods
           operating on stored trajectories.
    """
    trajectory = []
    s_t = env.reset(rng)  # S_0 -- sample initial state
    terminated = False
    while not terminated:
        a_t = policy(s_t, rng)
        s_next, r, terminated = env.step(s_t, a_t, rng)
        trajectory.append((s_t, a_t, r, s_next, terminated))
        s_t = s_next
    return trajectory
