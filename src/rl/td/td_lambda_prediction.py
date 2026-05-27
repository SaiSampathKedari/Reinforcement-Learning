"""Backward-view TD(lambda) Prediction with eligibility traces (S&B 2020, §12.2, p.294).

Online, per-step updates using eligibility traces. Equivalent to the forward-
view lambda-return but computable incrementally: each TD error delta_t is
distributed to all previously visited states proportional to their trace.
lambda=0 recovers TD(0); lambda=1 approaches Monte Carlo.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.utils.rollout import Policy


def td_lambda_prediction(
    env         :   TabularEnv,
    policy      :   Policy,
    rng         :   np.random.Generator,
    n_episodes  :   int,
    alpha       :   float,
    lam         :   float,
) -> np.ndarray:
    """TD(lambda) prediction of V^pi using backward-view eligibility traces.

    At each step t (S&B p.294 pseudocode):
        1. Take action a_t, observe r, s_{t+1}, terminated.
        2. Compute TD error:  delta_t = r + gamma * V(s_{t+1}) - V(s_t).
        3. Update traces:    z(s) = gamma * lambda * z(s)  for all s,
                             z(s_t) += 1                   (accumulating trace).
        4. Update values:    V(s) = V(s) + alpha * delta_t * z(s)  for all s.

    Args:
        env: any `TabularEnv`; `env.gamma` is used as the discount.
        policy: callable `(s, rng) -> a`.
        rng: numpy Generator.
        n_episodes: number of episodes to run.
        alpha: learning rate in (0, 1].
        lam: trace decay parameter in [0, 1].
            lambda=0 -> TD(0);  lambda=1 -> approaches MC.

    Returns:
        V: estimated state values of shape `(n_states,)`.
    """
    gamma = env.gamma
    V = np.zeros(env.n_states)

    for _ in range(n_episodes):
        s_t = env.reset(rng)  # S_0
        z = np.zeros(env.n_states)  # eligibility trace vector, reset each episode
        terminated = False

        while not terminated:
            # Step.
            a_t = policy(s_t, rng)
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # TD error.
            delta = r + gamma * V[s_next] * (1 - int(terminated)) - V[s_t]

            # Trace update: decay all, then bump current state (accumulating).
            z *= gamma * lam
            z[s_t] += 1

            # Value update: distribute delta to all states proportional to trace.
            V += alpha * delta * z

            # Advance.
            s_t = s_next

    return V
