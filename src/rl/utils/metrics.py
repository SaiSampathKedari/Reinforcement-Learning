"""Value-error metrics for prediction with function approximation (S&B 9.1)."""

from __future__ import annotations
import numpy as np


def rmsve(v_hat: np.ndarray, v_true: np.ndarray, mu: np.ndarray) -> float:
    """Root mean square value error: sqrt( sum_s mu(s) * (v_hat(s) - v_true(s))^2 ).

    The Chapter 9 objective (S&B eq. 9.1). It is keyed on *arrays*, not on the
    approximator, so it works unchanged for any model (linear, tile coding, a
    neural net) -- the caller just passes `vf.all_values(n_states)` as `v_hat`.

    `mu` is the state weighting: the on-policy distribution for the true VE, or a
    uniform weight for an unweighted RMS error (e.g. Fig 9.2 right, which averages
    RMS over all states to match the tabular Fig 7.2). `mu` need not be
    normalized; only the relative weights matter for comparisons at fixed `mu`.
    """
    return float(np.sqrt(mu @ (v_hat - v_true) ** 2))


def uniform_weights(states, n_states: int) -> np.ndarray:
    """Uniform distribution over `states`, as a full (n_states,) vector of zeros
    with mass 1/|states| on each listed state (terminals left at 0).

    `n_states` is the env's total state count (including terminals) so the result
    aligns with `vf.all_values(n_states)` and `v_true`. Convenience for the
    unweighted RMS error used in Fig 9.2 right.
    """
    states = list(states)
    mu = np.zeros(n_states)
    mu[states] = 1.0 / len(states)
    return mu
