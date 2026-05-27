"""On-policy Monte Carlo Control with epsilon-soft policies (S&B 2020, §5.4, p.101).

Generalized Policy Iteration on tabular Q: each episode is generated under an
epsilon-greedy behavior policy derived from Q, then Q is updated from sample
returns. No exploring starts needed -- exploration comes from the epsilon-greedy
action selection itself. With `decay_epsilon=True`, implements GLIE (Greedy in
the Limit with Infinite Exploration) which converges to Q*.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy, Trajectory, generate_episode
from rl.utils.policies import epsilon_greedy_action, greedy_action


def mc_control_eps_soft(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int = 500_000,
    epsilon     :   float = 0.1,
    decay_epsilon:  bool = False,
    first_visit :   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """On-policy MC Control with epsilon-greedy exploration.

    Generalized Policy Iteration on tabular Q. Each episode does:
        1. Generate the episode under the current epsilon-greedy policy from Q.
        2. Compute returns G_t backward.
        3. Policy evaluation: update `Q(s, a)` via incremental average of returns.
        4. Policy tracking: `pi(s) <- argmax_a Q(s, a)` (greedy snapshot of Q).

    The behavior policy is implicitly epsilon-greedy w.r.t. Q -- no separate
    policy array drives episode generation. `pi` is a diagnostic that tracks
    the greedy policy as Q evolves.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator (threaded through episode generation and policy).
        n_episodes: number of episodes. Default 500_000.
        epsilon: exploration probability for epsilon-greedy. Default 0.1.
        decay_epsilon: if True, use GLIE schedule `epsilon = 1/k` instead of
            the fixed value. Converges to Q* (true optimal).
        first_visit: if True (default), first-visit MC; else every-visit.

    Returns:
        Q: action-value estimates of shape `(n_states, n_actions)`.
        pi: greedy policy derived from Q, shape `(n_states,)`, dtype int.
    """
    gamma = env.gamma
    Q  = np.zeros((env.n_states, env.n_actions))                  # Q(s, a)
    N  = np.zeros((env.n_states, env.n_actions), dtype=np.int64)  # visit counts
    pi = np.zeros(env.n_states, dtype=np.int64)                   # greedy policy tracker

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon   # GLIE vs fixed

        # epsilon-greedy behavior policy derived from current Q.
        policy_fn: Policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)
        trajectory: Trajectory = generate_episode(env, policy_fn, rng)
        T = len(trajectory)

        # Backward pass: G_t = r_{t+1} + gamma * G_{t+1}, with G_T = 0.
        G = 0.0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            r = trajectory[t][2]
            G = r + gamma * G
            returns[t] = G

        # Forward pass: policy evaluation (Q update) + track greedy policy.
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
            pi[s_t] = greedy_action(Q[s_t])                            # track greedy policy

    return Q, pi
