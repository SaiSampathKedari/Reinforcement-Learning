"""First-visit (and every-visit) Monte Carlo prediction of V^pi
(S&B 2020, §5.1, p.92).
"""

from __future__ import annotations
import numpy as np
from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy, generate_episode


def mc_prediction(
    env: TabularEnv,
    policy: Policy,
    n_episodes: int,
    rng: np.random.Generator,
    first_visit: bool = True,
) -> np.ndarray:
    """Estimate V^pi by sample-averaging returns over `n_episodes` episodes.

    Algorithm (S&B §5.1, p.92):
        1. Generate an episode under `policy`:
               S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T.
        2. Compute returns backward:
               G_t = R_{t+1} + gamma * G_{t+1},   G_T = 0.
        3. Update V at each visited state by incremental average:
               V(S_t) <- V(S_t) + (G_t - V(S_t)) / N(S_t).

    Visit conventions:
        - `first_visit=True` (default): only the earliest occurrence of each
          state in an episode contributes -- S&B's First-Visit MC.
        - `first_visit=False`: every occurrence contributes -- Every-Visit MC.

    Args:
        env: any `TabularEnv`. `env.gamma` is used as the discount factor.
        policy: callable `(s, rng) -> a` (see `rl.utils.rollout.Policy`).
        n_episodes: number of episodes to sample.
        rng: numpy Generator, threaded through rollouts and the policy.
        first_visit: see "Visit conventions" above.

    Returns:
        V: np.ndarray of shape `(env.n_states,)` -- estimated state values
        under `policy`. States never visited remain at their initial value (0).
    """
    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states, dtype=np.int64)
    gamma = env.gamma

    for _ in range(n_episodes):
        trajectory = generate_episode(env, policy, rng)
        T = len(trajectory)

        # Backward pass: G_t = R_{t+1} + gamma * G_{t+1}, G_T = 0.
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            r = trajectory[t][2]  # R_{t+1}
            G = gamma * G + r
            returns[t] = G

        # Forward pass: update V at first-/every-visit states.
        seen: set[int] = set()
        for t in range(T):
            s_t = trajectory[t][0]
            if first_visit:
                if s_t in seen:
                    continue
                seen.add(s_t)
            N[s_t] += 1
            V[s_t] += (returns[t] - V[s_t]) / N[s_t]

    return V
