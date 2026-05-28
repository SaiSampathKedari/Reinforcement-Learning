"""SARSA(lambda): on-policy TD control with eligibility traces (S&B 2020, §12.7, p.303).

The control analogue of backward-view TD(lambda): traces z(s, a) accumulate over
state-action pairs, and each TD error is distributed to all recently-visited
pairs proportional to their trace. lambda=0 recovers SARSA(0); lambda=1
approaches Monte Carlo control. Behavior policy is epsilon-greedy w.r.t. Q.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy
from rl.utils.policies import greedy_action, epsilon_greedy_action


def sarsa_lambda(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    lam         :   float,
    epsilon     :   float = 0.1,
    decay_epsilon:  bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """SARSA(lambda) on-policy TD control with accumulating eligibility traces.

    Each step (S&B p.303):
        1. Take a_t, observe r, s_{t+1}; choose a_{t+1} epsilon-greedily
           (unless terminal).
        2. TD error: delta = r + gamma*Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
           (no bootstrap at terminal).
        3. Traces: z = gamma*lambda*z (decay all); z(s_t, a_t) += 1 (bump).
        4. Q += alpha*delta*z (update all state-action pairs by their trace).

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate in (0, 1].
        lam: trace decay in [0, 1]. lambda=0 -> SARSA(0); lambda=1 -> MC.
        epsilon: exploration probability for the epsilon-greedy policy.
        decay_epsilon: if True, epsilon = 1/episode (GLIE); else fixed.

    Returns:
        Q: action-value estimates of shape `(n_states, n_actions)`.
        pi: greedy policy from final Q, shape `(n_states,)`, dtype int.
    """
    gamma = env.gamma
    Q = np.zeros((env.n_states, env.n_actions))   # Q(s, a)

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon   # GLIE vs fixed

        z = np.zeros((env.n_states, env.n_actions))   # eligibility traces, reset per episode
        policy_fn: Policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)

        s_t = env.reset(rng)
        a_t = policy_fn(s_t, rng)
        terminated = False

        while not terminated:
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # Terminal transition: no next action, no bootstrap.
            if terminated:
                delta = r - Q[s_t, a_t]
                z *= gamma * lam            # decay all traces
                z[s_t, a_t] += 1            # bump current pair
                Q += alpha * delta * z      # update all (s, a) by trace
                break

            # Non-terminal: SARSA TD error uses the next action a_{t+1}.
            a_next = policy_fn(s_next, rng)
            delta = r + gamma * Q[s_next, a_next] - Q[s_t, a_t]
            z *= gamma * lam                # decay all traces
            z[s_t, a_t] += 1                # bump current pair
            Q += alpha * delta * z          # update all (s, a) by trace

            s_t, a_t = s_next, a_next

    # Derive the greedy policy from the final Q.
    pi = np.array([greedy_action(Q[s]) for s in range(env.n_states)])
    return Q, pi
