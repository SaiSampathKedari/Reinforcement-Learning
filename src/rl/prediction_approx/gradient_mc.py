"""Gradient Monte Carlo prediction with function approximation (S&B 9.3)."""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy, generate_episode
from rl.approximators.base import ValueApproximator


def gradient_mc(
    env: TabularEnv,
    policy: Policy,
    value_fn: ValueApproximator,
    rng: np.random.Generator,
    n_episodes: int,
    alpha: float,
) -> ValueApproximator:
    """Train `value_fn` by gradient Monte Carlo; returns the same (trained) object.

    Each episode: roll out under `policy`, compute the returns G_t, then update
    every visited state toward its return,

        w <- w + alpha * (G_t - v_hat(S_t)) * grad(S_t).

    G_t is the full Monte Carlo return (no bootstrap), so the target does not
    depend on w -- this is the true gradient, unlike semi-gradient TD which
    bootstraps on w. The learner (this function) owns the weight update; the model
    only supplies `value` and `grad`.
    """
    gamma = env.gamma
    for _ in range(n_episodes):
        trajectory = generate_episode(env, policy, rng)   # [(s, a, r, s_next, terminated), ...]
        G = 0.0
        for s, _a, r, _s_next, _terminated in reversed(trajectory):
            G = r + gamma * G                              # return from state s
            value_fn.w += alpha * (G - value_fn.value(s)) * value_fn.grad(s)
    return value_fn
