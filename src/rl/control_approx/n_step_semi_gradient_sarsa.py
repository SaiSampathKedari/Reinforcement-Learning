"""Episodic n-step semi-gradient Sarsa control (S&B 10.2, p.247).

n-step on-policy control with function approximation: the n-step return uses the
first n observed rewards and bootstraps from q_hat(S_{tau+n}, A_{tau+n}) -- the
action actually taken n steps ahead (on-policy). n=1 recovers one-step
semi-gradient Sarsa; n >= episode length is Monte Carlo control. Online with an
n-step delay; the behavior policy is epsilon-greedy w.r.t. q_hat.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import Env
from rl.approximators.base import ActionValueApproximator
from rl.control_approx.semi_gradient_sarsa import ApproxEpisodeCallback
from rl.utils.policies import epsilon_greedy_action


def n_step_semi_gradient_sarsa(
    env: Env,
    action_value_fn: ActionValueApproximator,
    rng: np.random.Generator,
    n_episodes: int,
    alpha: float,
    n: int,
    epsilon: float = 0.0,
    decay_epsilon: bool = False,
    on_episode_end: ApproxEpisodeCallback | None = None,
) -> ActionValueApproximator:
    """Train `action_value_fn` by n-step semi-gradient Sarsa; returns it (trained).

    The action-value, function-approximation analog of tabular n-step Sarsa. Each
    episode runs in two phases:
        Phase 1 (while the episode runs): update (S_tau, A_tau) once tau >= 0
            toward the n-step return
              G = R_{tau+1} + ... + gamma^{n-1} R_{tau+n}
                  + gamma^n q_hat(S_{tau+n}, A_{tau+n}),
            via w <- w + alpha * (G - q_hat(S_tau, A_tau)) * grad(S_tau, A_tau).
            The bootstrap is dropped once S_{tau+n} is terminal.
        Phase 2 (after terminal): flush the last n-1 pairs with truncated returns
            (no bootstrap past the terminal state).

    The bootstrap uses A_{tau+n}, the action actually taken (on-policy Sarsa), not
    max_a q_hat (which would be off-policy). "Semi-gradient": the target depends on
    w but we do not differentiate through it. n=1 recovers one-step Sarsa.

    `epsilon` defaults to 0 (greedy) -- with optimistic initial weights this still
    explores (Mountain Car). `decay_epsilon` uses epsilon = 1/k on episode k.
    `on_episode_end(k, action_value_fn)` fires after each episode k (1-based).
    """
    gamma = env.gamma

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon

        # Behavior policy: epsilon-greedy on the q_hat(s, .) row over all actions.
        def policy_fn(s, rng) -> int:
            q_row = np.array([action_value_fn.value(s, a) for a in range(env.n_actions)])
            return epsilon_greedy_action(q_row, eps, rng)

        s_0 = env.reset(rng)
        states = [s_0]                       # grows: [S_0, S_1, ...]
        actions = [policy_fn(s_0, rng)]      # grows: [A_0, A_1, ...]
        rewards = []                         # grows: [R_1, R_2, ...]
        terminated = False

        # Phase 1: step through the episode, updating with an n-step delay.
        while not terminated:
            # Take A_t, observe R_{t+1}, S_{t+1}; choose A_{t+1} if not terminal.
            s_next, r, terminated = env.step(states[-1], actions[-1], rng)
            states.append(s_next)
            rewards.append(r)
            if not terminated:
                actions.append(policy_fn(s_next, rng))

            # Update (S_tau, A_tau) once enough steps are buffered.
            tau = len(rewards) - n
            if tau >= 0:
                # n-step return: discounted rewards + bootstrap on (S_{tau+n}, A_{tau+n}).
                G = 0.0
                for j in range(n):
                    G += gamma**j * rewards[tau + j]
                if not terminated:           # skip bootstrap when S_{tau+n} is terminal
                    G += gamma**n * action_value_fn.value(states[tau + n], actions[tau + n])

                s_tau, a_tau = states[tau], actions[tau]
                delta = G - action_value_fn.value(s_tau, a_tau)
                action_value_fn.w += alpha * delta * action_value_fn.grad(s_tau, a_tau)

        # Phase 2: flush the last n-1 pairs (truncated returns, no bootstrap).
        T = len(rewards)
        for tau in range(max(0, T - n + 1), T):
            G = 0.0
            for j in range(T - tau):
                G += gamma**j * rewards[tau + j]

            s_tau, a_tau = states[tau], actions[tau]
            delta = G - action_value_fn.value(s_tau, a_tau)
            action_value_fn.w += alpha * delta * action_value_fn.grad(s_tau, a_tau)

        if on_episode_end is not None:
            on_episode_end(k, action_value_fn)

    return action_value_fn
