"""Episode rollout helper. Returns SARS' trajectories for any tabular method."""

from __future__ import annotations
from typing import Callable
import numpy as np
from rl.envs.base import TabularEnv

Policy = Callable[[int, np.random.Generator], int]  # (state, rng) -> action
EpisodeCallback = Callable[[int, np.ndarray], None]  # (episode_index, Q) -> None
"""Optional hook fired by control algorithms at each episode boundary, receiving
the current episode index and the live Q array. Copy Q if you store it (it is
mutated in place). Used to record learning dynamics (Q / greedy-policy evolution)
without changing the algorithm's return value."""

Trajectory = list[tuple[int, int, float, int, bool]] 
"""Trajectory format:
        [(s_t, a_t, r_{t+1}, s_{t+1}, terminated)  for t = 0, ..., T-1]
"""

def generate_episode(
    env: TabularEnv,
    policy: Policy,
    rng: np.random.Generator,
) -> Trajectory:
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

def sample_uniform_start(
    env     :   TabularEnv,
    rng     :   np.random.Generator,
) -> tuple[int, int]:
    """Sample `(s_0, a_0)` uniformly. `s_0` from non-terminal states,
    `a_0` from `[0, env.n_actions)`. Used by MC with Exploring Starts.
    """
    terminals = env.terminal_states()
    non_terminals = [s for s in range(env.n_states) if s not in terminals]
    s_0 = int(rng.choice(non_terminals))
    a_0 = int(rng.integers(env.n_actions))
    return s_0, a_0

def generate_episode_from(
  env       :   TabularEnv,
  policy    :   Policy,
  s_0       :   int,
  a_0       :   int,
  rng       :   np.random.Generator,
) -> Trajectory:
    """Play one episode starting from `(s_0, a_0)`, then following `policy`.

    First step uses the forced action `a_0`. Subsequent actions are chosen
    via `policy(s_t, rng)`. Used by MC with Exploring Starts.
    """
    trajectory: Trajectory = []
    s_t, a_t = s_0, a_0
    terminated = False
    while not terminated:
        s_next, r, terminated = env.step(s_t, a_t, rng)
        trajectory.append((s_t, a_t, r, s_next, terminated))
        s_t = s_next
        if not terminated:
            a_t = policy(s_t, rng)
    return trajectory
