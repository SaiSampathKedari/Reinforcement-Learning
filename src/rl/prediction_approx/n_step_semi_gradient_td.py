"""n-step semi-gradient TD prediction with function approximation (S&B 9.3).

The function-approximation analog of tabular n-step TD (Chapter 7): the n-step
return uses the first n observed rewards and bootstraps from v_hat(S_{t+n}) for
the tail. n=1 recovers semi-gradient TD(0); n >= episode length recovers gradient
MC. Online with n-step delay. Like TD(0), the bootstrap target depends on w but
we do not differentiate through it -- hence "semi-gradient".
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy
from rl.approximators.base import ValueApproximator


def n_step_semi_gradient_td(
    env         :   TabularEnv,
    policy      :   Policy,
    value_fn    :   ValueApproximator,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    n           :   int,
) -> ValueApproximator:
    """Train `value_fn` by n-step semi-gradient TD; returns the same (trained) object.

    Each episode has two phases (mirroring tabular n-step TD):
        Phase 1 (while episode runs): step + update S_tau once tau >= 0, toward
            G_{tau:tau+n} = R_{tau+1} + gamma*R_{tau+2} + ... + gamma^{n-1}*R_{tau+n}
                            + gamma^n * v_hat(S_{tau+n}),
            using w <- w + alpha * (G - v_hat(S_tau)) * grad(S_tau).
        Phase 2 (after terminal): flush the last n-1 states with truncated
            returns (no bootstrap -- past terminal).

    The bootstrap is dropped when S_{tau+n} is terminal, encoding
    v_hat(terminal) = 0 explicitly rather than relying on the feature map to zero
    terminal states. The learner owns the update; the model only supplies `value`
    and `grad`.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        policy: callable `(s, rng) -> a`.
        value_fn: the approximator to train (mutated in place).
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate (step size).
        n: number of steps. n=1 is semi-gradient TD(0); n >= episode length is MC.
    """
    gamma = env.gamma
    terminals = env.terminal_states()

    for _ in range(n_episodes):
        states  = [env.reset(rng)]   # grows: [S_0, S_1, ...]
        rewards = []                 # grows: [R_1, R_2, ...]
        terminated = False

        # Phase 1: step through episode, update with n-step delay.
        while not terminated:
            # Step.
            a_t = policy(states[-1], rng)
            s_next, r, terminated = env.step(states[-1], a_t, rng)
            states.append(s_next)
            rewards.append(r)

            # Update S_tau once enough steps have been buffered.
            tau = len(rewards) - n
            if tau >= 0:
                # n-step return: discounted rewards + bootstrap (dropped at terminal).
                G = 0.0
                for k in range(n):
                    G += gamma**k * rewards[tau + k]
                s_boot = states[tau + n]
                G += gamma**n * value_fn.value(s_boot) * (s_boot not in terminals)

                # Semi-gradient weight update.
                value_fn.w += alpha * (G - value_fn.value(states[tau])) * value_fn.grad(states[tau])

        # Phase 2: flush the last n-1 states (truncated returns, no bootstrap).
        T = len(rewards)
        for tau in range(max(0, T - n + 1), T):
            # Truncated return: fewer than n rewards, no bootstrap (past terminal).
            G = 0.0
            for k in range(T - tau):
                G += gamma**k * rewards[tau + k]

            # Semi-gradient weight update.
            value_fn.w += alpha * (G - value_fn.value(states[tau])) * value_fn.grad(states[tau])

    return value_fn