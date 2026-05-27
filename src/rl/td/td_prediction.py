"""Tabular TD(0) Prediction (S&B 2020, §6.1, p.120).

One-step temporal-difference learning: updates V(S_t) after every transition
using the bootstrapped target R_{t+1} + gamma * V(S_{t+1}). Online — no need
to wait for the episode to end (unlike Monte Carlo).
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy


def td_prediction(
    env         :   TabularEnv,
    policy      :   Policy,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
) -> np.ndarray:
    """TD(0) prediction of V^pi.

    Online, per-step updates. Each step does:
        1. Take action a_t ~ policy(s_t), observe r, s_{t+1}, terminated.
        2. Compute TD error: `delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)`.
        3. Update: `V(S_t) <- V(S_t) + alpha * delta_t`.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        policy: callable `(s, rng) -> a`.
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate (step size) in (0, 1].

    Returns:
        V: estimated state values of shape `(n_states,)`.
    """
    gamma = env.gamma
    V = np.zeros(env.n_states)

    for _ in range(n_episodes):
        s_t = env.reset(rng)  # S_0
        terminated = False

        while not terminated:
            # Step.
            a_t = policy(s_t, rng)
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # TD error: delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t).
            delta = r + gamma * V[s_next] * (1 - int(terminated)) - V[s_t]

            # Value update.
            V[s_t] += alpha * delta

            # Advance.
            s_t = s_next

    return V
