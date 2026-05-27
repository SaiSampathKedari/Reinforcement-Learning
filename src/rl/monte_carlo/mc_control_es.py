"""Monte Carlo Control with Exploring Starts (S&B 2020, §5.3, p.99).

Generalized Policy Iteration on tabular Q: each episode alternates Monte
Carlo policy evaluation (update Q from sample returns) and greedy policy
improvement (`pi(s) <- argmax_a Q(s, a)`). Exploration is guaranteed by
sampling a uniformly random `(s_0, a_0)` per episode; the policy itself
is deterministic.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy, Trajectory, generate_episode_from, sample_uniform_start
from rl.utils.policies import greedy_action


def mc_control_es(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int = 50_000,
    first_visit :   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo Control with Exploring Starts.

    Generalized Policy Iteration on tabular Q. Each episode does:
        1. Sample exploring start `(s_0, a_0)` uniformly.
        2. Generate the episode following the current greedy policy after step 0.
        3. Policy evaluation: update `Q(s, a)` via incremental average of returns.
        4. Policy improvement: `pi(s) <- argmax_a Q(s, a)`, interleaved with (3).

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator (threaded through sampling, env, and policy).
        n_episodes: number of episodes. Default 50_000; S&B Fig 5.2 uses 500_000.
        first_visit: if True (default), first-visit MC; else every-visit.

    Returns:
        Q: action-value estimates of shape `(n_states, n_actions)`.
        pi: greedy policy derived from Q, shape `(n_states,)`, dtype int.
    """
    gamma = env.gamma
    Q  = np.zeros((env.n_states, env.n_actions))                  # Q(s, a)
    N  = np.zeros((env.n_states, env.n_actions), dtype=np.int64)  # visit counts
    pi = np.zeros(env.n_states, dtype=np.int64)                   # arbitrary initial policy

    # Policy callable wrapping pi. Captures pi by reference, so it tracks updates.
    # The Policy signature requires (s, rng); the deterministic policy ignores rng.
    policy_fn: Policy = lambda s, _: int(pi[s])

    for _ in range(n_episodes):
        # Exploring start: sample (s_0, a_0) uniformly.
        s_0, a_0 = sample_uniform_start(env, rng)
        trajectory: Trajectory = generate_episode_from(env, policy_fn, s_0, a_0, rng)
        T = len(trajectory)

        # Backward pass: G_t = r_{t+1} + gamma * G_{t+1}, with G_T = 0.
        G = 0.0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            r = trajectory[t][2]
            G = r + gamma * G
            returns[t] = G

        # Forward pass: policy evaluation (Q update) + greedy policy improvement.
        seen: set[tuple[int, int]] = set()
        for t in range(T):
            s_t = trajectory[t][0]
            a_t = trajectory[t][1]
            if first_visit:
                if (s_t, a_t) in seen:
                    continue
                seen.add((s_t, a_t))
            N[s_t, a_t] += 1
            Q[s_t, a_t] += (returns[t] - Q[s_t, a_t]) / N[s_t, a_t]   # policy evaluation
            pi[s_t] = greedy_action(Q[s_t])                           # greedy policy improvement

    return Q, pi