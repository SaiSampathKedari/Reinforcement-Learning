"""SARSA(0): on-policy TD control (S&B 2020, §6.4, p.130).

Online Generalized Policy Iteration on tabular Q. The behavior policy is
epsilon-greedy w.r.t. Q; the same action A_{t+1} sampled from it is used both
as the bootstrap in the target and as the next action executed (the on-policy
property). With `decay_epsilon=True`, epsilon = 1/episode gives GLIE, which
converges to q*; with fixed epsilon it converges to the best epsilon-soft policy.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy, EpisodeCallback
from rl.utils.policies import epsilon_greedy_action, greedy_action


def sarsa(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    epsilon     :   float = 0.1,
    decay_epsilon:  bool = False,
    on_episode_end: EpisodeCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SARSA(0) on-policy TD control.

    Each episode (S&B p.130 pseudocode):
        1. Initialize s_t; choose a_t epsilon-greedily from Q.
        2. Each step: take a_t, observe r, s_{t+1}.
           - Terminal: target = r (no next action, no bootstrap).
           - Otherwise: choose a_{t+1} epsilon-greedily;
                        target = r + gamma * Q(s_{t+1}, a_{t+1}).
           Update Q(s_t, a_t) toward the target, then advance.

    The next action a_{t+1} serves three roles (on-policy): the bootstrap in
    the target, the next environment action, and a_t of the next iteration.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator (threaded through env, policy, action selection).
        n_episodes: number of episodes to run.
        alpha: learning rate (step size) in (0, 1].
        epsilon: exploration probability for the epsilon-greedy policy.
        decay_epsilon: if True, epsilon = 1/episode (GLIE, converges to q*);
            if False, fixed epsilon (converges to best epsilon-soft policy).
        on_episode_end: optional hook called as `on_episode_end(episode, Q)`
            after each episode, for recording learning dynamics. Copy Q to store
            it (mutated in place). None (default) leaves behaviour unchanged.

    Returns:
        Q: action-value estimates of shape `(n_states, n_actions)`.
        pi: greedy policy derived from Q, shape `(n_states,)`, dtype int.
    """
    gamma = env.gamma
    Q  = np.zeros((env.n_states, env.n_actions))   # Q(s, a)
    pi = np.zeros(env.n_states, dtype=np.int64)    # greedy policy tracker

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon   # GLIE vs fixed

        # epsilon-greedy behavior policy derived from current Q.
        policy_fn: Policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)

        # Initialize state and first action.
        s_t = env.reset(rng)
        a_t = policy_fn(s_t, rng)
        terminated = False

        while not terminated:
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # Terminal transition: no next action, no bootstrap. Update and stop.
            if terminated:
                delta = r - Q[s_t, a_t]
                Q[s_t, a_t] += alpha * delta
                pi[s_t] = greedy_action(Q[s_t])
                break

            # Non-terminal: sample a_{t+1}, bootstrap, update, advance.
            a_next = policy_fn(s_next, rng)
            target = r + gamma * Q[s_next, a_next]   # SARSA(0) target
            delta = target - Q[s_t, a_t]
            Q[s_t, a_t] += alpha * delta             # policy evaluation
            pi[s_t] = greedy_action(Q[s_t])          # track greedy policy

            s_t, a_t = s_next, a_next                 # advance

        if on_episode_end is not None:
            on_episode_end(k, Q)                      # expose Q at the episode boundary

    return Q, pi
