"""Uniform random-walk policy for the Random Walk env (S&B 6.2 / 7.1 / 9.1)."""

from __future__ import annotations
import numpy as np

from rl.envs.random_walk import RandomWalk
from rl.utils.rollout import Policy


def uniform_jump_policy(env: RandomWalk, max_step: int) -> np.ndarray:
    """pi(a|s) for the uniform +/-max_step walk, shape (n_states, n_actions).

    Off-edge neighbors fold onto the terminal actions; terminal rows stay zero.
    """
    pi = np.zeros((env.n_states, env.n_actions))
    prob = 1.0 / (2 * max_step)
    for s in range(env.LEFT_TERMINAL + 1, env.RIGHT_TERMINAL):
        for d in (*range(-max_step, 0), *range(1, max_step + 1)):
            target = s + d
            if target >= env.RIGHT_TERMINAL:
                a = env.RIGHT_TERMINAL
            elif target <= env.LEFT_TERMINAL:
                a = env.LEFT_TERMINAL
            else:
                a = target
            pi[s, a] += prob
    return pi


def policy_sampler(pi: np.ndarray) -> Policy:
    """Turn a policy matrix pi(a|s) into a sampler (s, rng) -> action."""
    n_actions = pi.shape[1]

    def policy(s: int, rng: np.random.Generator) -> int:
        return int(rng.choice(n_actions, p=pi[s]))

    return policy
