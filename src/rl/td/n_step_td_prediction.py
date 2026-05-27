"""n-step TD Prediction (S&B 2020, §7.1, p.144).

Bridges TD(0) and Monte Carlo: the n-step return uses the first n observed
rewards and bootstraps from V(S_{t+n}) for the remaining tail. n=1 recovers
TD(0); n >= T recovers MC. Online with n-step delay.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy


def n_step_td_prediction(
    env         :   TabularEnv,
    policy      :   Policy,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    n           :   int,
) -> np.ndarray:
    """n-step TD prediction of V^pi (online with n-step delay).

    Each episode has two phases:
        Phase 1 (while episode runs): step + update V(S_tau) once tau >= 0.
            G_{tau:tau+n} = R_{tau+1} + gamma*R_{tau+2} + ... + gamma^{n-1}*R_{tau+n}
                            + gamma^n * V(S_{tau+n}).
        Phase 2 (after terminal): flush remaining n-1 states with truncated
            returns (no bootstrap — past terminal).

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        policy: callable `(s, rng) -> a`.
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate in (0, 1].
        n: number of steps. n=1 is TD(0); n >= episode length is MC.

    Returns:
        V: estimated state values of shape `(n_states,)`.
    """
    gamma = env.gamma
    V = np.zeros(env.n_states)

    for _ in range(n_episodes):
        states  = [env.reset(rng)]  # grows: [S_0, S_1, ...]
        rewards = []                 # grows: [R_1, R_2, ...]
        terminated = False

        # Phase 1: step through episode, update with n-step delay.
        while not terminated:
            # Step.
            a_t = policy(states[-1], rng)
            s_next, r, terminated = env.step(states[-1], a_t, rng)
            states.append(s_next)
            rewards.append(r)

            # Update V(S_tau) if enough steps have been buffered.
            tau = len(rewards) - n
            if tau >= 0:
                # n-step return: G_{tau:tau+n} = sum of discounted rewards + bootstrap.
                G = 0.0
                for k in range(n):
                    G += gamma**k * rewards[tau + k]
                G += gamma**n * V[states[tau + n]]

                # Value update.
                V[states[tau]] += alpha * (G - V[states[tau]])

        # Phase 2: flush the last n-1 states (truncated returns, no bootstrap).
        T = len(rewards)
        for tau in range(max(0, T - n + 1), T):
            # Truncated return: fewer than n rewards, no bootstrap (past terminal).
            G = 0.0
            for k in range(T - tau):
                G += gamma**k * rewards[tau + k]

            # Value update.
            V[states[tau]] += alpha * (G - V[states[tau]])

    return V
