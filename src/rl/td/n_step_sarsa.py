"""n-step SARSA: on-policy TD control (S&B 2020, §7.2, p.147).

Control analogue of n-step TD prediction: the n-step return uses the first n
observed rewards and bootstraps from Q(S_{t+n}, A_{t+n}) -- the action actually
taken n steps ahead (on-policy). n=1 recovers SARSA(0); n >= T is Monte Carlo
control. Online with n-step delay. Behavior policy is epsilon-greedy w.r.t. Q.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy
from rl.utils.policies import greedy_action, epsilon_greedy_action


def n_step_sarsa(
    env         :   TabularEnv,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    n           :   int,
    epsilon     :   float = 0.1,
    decay_epsilon:  bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """n-step SARSA on-policy TD control (online with n-step delay).

    Each episode has two phases (mirrors n-step TD prediction):
        Phase 1 (while episode runs): step + update Q(S_tau, A_tau) once tau >= 0.
            G_{tau:tau+n} = R_{tau+1} + ... + gamma^{n-1}*R_{tau+n}
                            + gamma^n * Q(S_{tau+n}, A_{tau+n}).
            The bootstrap term is included only while S_{tau+n} is non-terminal
            (tau+n < T); at the terminal step the return is truncated to the
            rewards alone (there is no A_T to bootstrap from).
        Phase 2 (after terminal): flush remaining n-1 pairs with truncated
            returns (no bootstrap -- past terminal).

    The bootstrap uses A_{tau+n}, the action actually taken (on-policy SARSA),
    not max_a Q (which would be off-policy Q-learning).

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate in (0, 1].
        n: number of steps. n=1 is SARSA(0); n >= episode length is MC control.
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
        policy_fn: Policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)

        s_0 = env.reset(rng)
        states  = [s_0]                  # grows: [S_0, S_1, ...]
        actions = [policy_fn(s_0, rng)]  # grows: [A_0, A_1, ...]
        rewards = []                      # grows: [R_1, R_2, ...]
        terminated = False

        # Phase 1: step through episode, update with n-step delay.
        while not terminated:
            # Take A_t, observe R_{t+1}, S_{t+1}; choose A_{t+1} if not terminal.
            s_next, r, terminated = env.step(states[-1], actions[-1], rng)
            states.append(s_next)
            rewards.append(r)
            if not terminated:
                actions.append(policy_fn(s_next, rng))

            # Update Q(S_tau, A_tau) once enough steps buffered.
            tau = len(rewards) - n
            if tau >= 0:
                # n-step return: discounted rewards + bootstrap on (S_{tau+n}, A_{tau+n}).
                G = 0.0
                for j in range(n):
                    G += gamma**j * rewards[tau + j]
                if not terminated:   # bootstrap only while S_{tau+n} is non-terminal (tau+n < T);
                    # at the terminal step there is no A_T, and the return is truncated (no bootstrap).
                    G += gamma**n * Q[states[tau + n], actions[tau + n]]

                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

        # Phase 2: flush the last n-1 pairs (truncated returns, no bootstrap).
        T = len(rewards)
        for tau in range(max(0, T - n + 1), T):
            G = 0.0
            for j in range(T - tau):
                G += gamma**j * rewards[tau + j]

            s_tau, a_tau = states[tau], actions[tau]
            Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

    # Derive the greedy policy from the final Q.
    pi = np.array([greedy_action(Q[s]) for s in range(env.n_states)])
    return Q, pi
