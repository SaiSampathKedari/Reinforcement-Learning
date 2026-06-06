"""Episodic semi-gradient one-step Sarsa control (S&B 10.1, p.244)."""

from __future__ import annotations
from typing import Callable
import numpy as np

from rl.envs.base import Env
from rl.approximators.base import ActionValueApproximator
from rl.utils.policies import epsilon_greedy_action

# Fired at each episode boundary with (episode_index, trained q_hat). The
# approximator's `w` is mutated in place, so snapshot it (copy `w`, or evaluate
# the value surface) inside the callback if you store it -- e.g. the cost-to-go
# -max_a q_hat(s, a) for Figure 10.1.
ApproxEpisodeCallback = Callable[[int, ActionValueApproximator], None]


def semi_gradient_sarsa(
    env: Env,
    action_value_fn: ActionValueApproximator,
    rng: np.random.Generator,
    n_episodes: int,
    alpha: float,
    epsilon: float = 0.0,
    decay_epsilon: bool = False,
    on_episode_end: ApproxEpisodeCallback | None = None,
) -> ActionValueApproximator:
    """Train `action_value_fn` by one-step semi-gradient Sarsa; returns it (trained).

    On-policy control with function approximation: act epsilon-greedy w.r.t. the
    current q_hat and update online, each step, toward the bootstrapped target

        w <- w + alpha * (R + gamma * q_hat(S', A') - q_hat(S, A)) * grad(S, A),

    with the bootstrap dropped at termination (target is just R). "Semi-gradient":
    the target depends on w but we do not differentiate through it. The learner
    owns the update *and* the policy; the model only supplies `value` and `grad`.

    `epsilon` defaults to 0 (greedy) -- with optimistic initial weights this still
    explores, as in the Mountain Car example. Set `decay_epsilon` to use
    epsilon = 1/k on the k-th episode instead of a fixed `epsilon`.

    `on_episode_end(k, action_value_fn)` fires after each episode k (1-based);
    use it to record learning dynamics (e.g. the cost-to-go surface for Fig 10.1).
    """
    gamma = env.gamma

    for k in range(1, n_episodes + 1):
        eps = 1.0 / k if decay_epsilon else epsilon

        # Behavior policy: epsilon-greedy on the q_hat(s, .) row over all actions.
        def policy_fn(s, rng) -> int:
            q_row = np.array([action_value_fn.value(s, a) for a in range(env.n_actions)])
            return epsilon_greedy_action(q_row, eps, rng)

        s_t = env.reset(rng)
        a_t = policy_fn(s_t, rng)
        terminated = False
        while not terminated:
            # Step.
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # Terminal transition: no bootstrap (q_hat(terminal) = 0), then end.
            if terminated:
                delta = r - action_value_fn.value(s_t, a_t)
                action_value_fn.w += alpha * delta * action_value_fn.grad(s_t, a_t)
                break

            # Sarsa: choose A' on-policy, bootstrap on q_hat(S', A').
            a_next = policy_fn(s_next, rng)
            delta = r + gamma * action_value_fn.value(s_next, a_next) - action_value_fn.value(s_t, a_t)
            action_value_fn.w += alpha * delta * action_value_fn.grad(s_t, a_t)

            s_t, a_t = s_next, a_next

        if on_episode_end is not None:
            on_episode_end(k, action_value_fn)

    return action_value_fn
