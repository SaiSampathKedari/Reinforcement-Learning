"""Q-learning: off-policy TD control (S&B 2020, §6.5, p.131).

Online Generalized Policy Iteration on tabular Q. The TD target is
    R_{t+1} + gamma * max_{a'} Q(S_{t+1}, a'),
which references no policy -- this is the off-policy mechanism. Q-learning
converges to q* directly, regardless of the behavior policy (provided every
state-action pair is visited infinitely often).

The behavior policy is epsilon-greedy w.r.t. Q (the universal default in
tabular and deep Q-learning). It is decoupled from the target: any policy
with full support would converge to the same q*. With `decay_epsilon=True`,
epsilon = 1/episode; convergence to q* does NOT require GLIE (unlike SARSA),
because the target already evaluates the greedy policy.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy
from rl.utils.policies import epsilon_greedy_action, greedy_action


def q_learning(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    epsilon     :   float = 0.1,
    decay_epsilon:  bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Q-learning off-policy TD control.

    Each step (S&B p.131 pseudocode):
        1. Choose a_t epsilon-greedily from Q (behavior policy).
        2. Take a_t, observe r, s_{t+1}.
           - Terminal: target = r (no bootstrap).
           - Otherwise: target = r + gamma * max_{a'} Q(s_{t+1}, a').
           Update Q(s_t, a_t) toward the target, then advance.

    Unlike SARSA, no next action is sampled for the bootstrap -- the target
    uses max_{a'} Q, the greedy value at s_{t+1}. The update uses the
    quadruple (S_t, A_t, R_{t+1}, S_{t+1}), not a quintuple.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator (threaded through env and action selection).
        n_episodes: number of episodes to run.
        alpha: learning rate (step size) in (0, 1].
        epsilon: exploration probability for the epsilon-greedy behavior policy.
        decay_epsilon: if True, epsilon = 1/episode; if False, fixed epsilon.
            Either way Q-learning converges to q* (no GLIE requirement).

    Returns:
        Q: action-value estimates of shape `(n_states, n_actions)`.
        pi: greedy policy derived from Q, shape `(n_states,)`, dtype int.
    """
    gamma = env.gamma
    Q  = np.zeros((env.n_states, env.n_actions))   # Q(s, a)
    pi = np.zeros(env.n_states, dtype=np.int64)    # greedy policy tracker

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon   # GLIE not required, but optional

        # epsilon-greedy behavior policy derived from current Q.
        policy_fn: Policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)

        s_t = env.reset(rng)
        terminated = False

        while not terminated:
            # Behavior policy selects a_t (decoupled from target).
            a_t = policy_fn(s_t, rng)
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # Terminal transition: no bootstrap.
            if terminated:
                delta = r - Q[s_t, a_t]
                Q[s_t, a_t] += alpha * delta
                pi[s_t] = greedy_action(Q[s_t])
                break

            # Off-policy target: max over actions at s_next, no a_next sampled.
            target = r + gamma * Q[s_next].max()
            delta = target - Q[s_t, a_t]
            Q[s_t, a_t] += alpha * delta              # policy evaluation
            pi[s_t] = greedy_action(Q[s_t])           # track greedy policy

            s_t = s_next                              # advance (no a_t carry-over)

    return Q, pi
