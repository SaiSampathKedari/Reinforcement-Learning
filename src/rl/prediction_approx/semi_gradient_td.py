"""Semi-gradient TD(0) prediction with function approximation (S&B 9.3)."""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv
from rl.approximators.base import ValueApproximator
from rl.utils.rollout import Policy


def semi_gradient_td(
    env     :   TabularEnv,
    policy  :   Policy,
    value_fn:   ValueApproximator,
    rng     :   np.random.Generator,
    n_episodes: int,
    alpha   :   float,
) -> ValueApproximator:
    """Train `value_fn` by semi-gradient TD(0); returns the same (trained) object.

    Online, per-step update with a bootstrapped target,

        w <- w + alpha * (R + gamma * v_hat(S') - v_hat(S)) * grad(S).

    Unlike gradient MC, the target R + gamma * v_hat(S') depends on w, but we do
    *not* differentiate through it -- hence "semi-gradient". The learner owns the
    update; the model only supplies `value` and `grad`.

    The bootstrap is masked by `(1 - terminated)` so the target at a terminal
    transition is just R, encoding v_hat(terminal) = 0 explicitly rather than
    relying on the feature map to zero terminal states.
    """
    gamma = env.gamma
    for _ in range(n_episodes):
        s_t = env.reset(rng)                              # S_0
        terminated = False
        while not terminated:
            # Step: take a_t ~ policy, observe R, S', and whether S' is terminal.
            a_t = policy(s_t, rng)
            s_next, r, terminated = env.step(s_t, a_t, rng)

            # TD error; drop the bootstrap at termination (v_hat(terminal) = 0).
            delta = r + gamma * value_fn.value(s_next) * (1 - int(terminated)) - value_fn.value(s_t)

            # Semi-gradient weight update.
            value_fn.w += alpha * delta * value_fn.grad(s_t)

            s_t = s_next

    return value_fn