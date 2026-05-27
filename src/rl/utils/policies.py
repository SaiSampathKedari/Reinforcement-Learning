"""Action-selection helpers for tabular control algorithms.

Each helper takes an action-value row `Q(s, .)` (shape `(n_actions,)`) and
returns a single sampled action. They are *action selectors*, not full policy
callables -- to use as a `Policy` (matching `rl.utils.rollout.Policy`), wrap:

    policy = lambda s, rng: greedy_action(Q[s])
    policy = lambda s, rng: epsilon_greedy_action(Q[s], eps, rng)
"""

from __future__ import annotations
import numpy as np


def greedy_action(Q_row: np.ndarray) -> int:
    """Return the greedy action `argmax_a Q(s, a)`.

    Args:
        Q_row: action-value row of shape `(n_actions,)`.

    Returns:
        Action index in `[0, n_actions)`. Ties are broken by `np.argmax`
        (the lowest-index argmax wins).
    """
    return int(np.argmax(Q_row))


def epsilon_greedy_action(
    Q_row: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """epsilon-greedy action selection: explore vs exploit.

    With probability `epsilon`, return a uniformly random action.
    Otherwise, return the greedy action `argmax_a Q(s, a)`.

    Args:
        Q_row: action-value row of shape `(n_actions,)`.
        epsilon: exploration probability in `[0, 1]`.
        rng: numpy Generator for the explore/exploit coin flip and the
            random action when exploring.

    Returns:
        Action index in `[0, n_actions)`.
    """
    if rng.random() < epsilon:
        return int(rng.integers(Q_row.size))
    return int(np.argmax(Q_row))