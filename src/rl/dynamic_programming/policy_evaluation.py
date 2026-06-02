"""Exact model-based quantities under a fixed policy (direct linear solves).

Ground truth for scoring learned methods:
    mrp_from_policy        -> (P_pi, r_pi)
    policy_value           -> v_pi
    on_policy_distribution -> mu
Both solves restrict to the transient (non-terminal) states; terminals get 0,
which keeps the gamma = 1 case well-posed for a proper policy.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv


def mrp_from_policy(env: TabularEnv, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginalize the known model under `pi` into (P_pi, r_pi).

    P_pi[s, s'] = sum_a pi[s,a] p(s'|s,a) ,  r_pi[s] = sum_a pi[s,a] r(s,a).
    `pi` is (n_states, n_actions); terminal rows are all-zero and stay zero.
    """
    n = env.n_states
    P = np.zeros((n, n))
    r = np.zeros(n)
    for s in range(n):
        row = pi[s]
        if not row.any():                       # terminal / no action -> zero row
            continue
        for a in np.nonzero(row)[0]:            # only nonzero-probability actions
            pa = row[a]
            for p, s_next, reward, _terminated in env.transitions(s, int(a)):
                P[s, s_next] += pa * p
                r[s] += pa * p * reward
    return P, r


def policy_value(env: TabularEnv, pi: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Exact v_pi over all states (terminals = 0), via v = (I - gamma P_pi)^-1 r_pi.

    Solves only the transient block, which is well-posed at gamma = 1 for a proper
    policy. `gamma` defaults to `env.gamma`. Raises ValueError if the system is
    singular (improper policy at gamma = 1).
    """
    gamma = env.gamma if gamma is None else gamma
    P, r = mrp_from_policy(env, pi)

    n = env.n_states
    terminals = env.terminal_states()
    idx = np.array([s for s in range(n) if s not in terminals])  # transient states

    # Solve only the transient block; singularity (improper policy or a
    # continuing fully-stochastic chain at gamma = 1) surfaces as LinAlgError.
    A = np.eye(idx.size) - gamma * P[np.ix_(idx, idx)]
    try:
        v_transient = np.linalg.solve(A, r[idx])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"(I - gamma*P) is singular on the transient states at gamma={gamma}: "
            "no unique v_pi. At gamma = 1 this means the policy is improper -- some "
            "states never reach a terminal (a closed loop / continuing chain)."
        ) from exc

    v = np.zeros(n)
    v[idx] = v_transient
    return v


def on_policy_distribution(
    env: TabularEnv,
    pi: np.ndarray,
    start: int | np.ndarray | None = None,
) -> np.ndarray:
    """On-policy state distribution mu (S&B eq. 9.2) -- the VE weighting for Ch. 9.

    Expected visits eta solve  eta = h + P_pi^T eta  over the transient states,
    then mu = eta / eta.sum() (terminals = 0). `start` is the start-state
    distribution: an int (point mass), an array over states, or None (uses
    `env.START` if present, else uniform over transient states).
    """
    n = env.n_states
    terminals = env.terminal_states()
    idx = np.array([s for s in range(n) if s not in terminals])  # transient states

    # Start-state distribution h over all states.
    h = np.zeros(n)
    if start is None:
        s0 = getattr(env, "START", None)
        if s0 is not None:
            h[s0] = 1.0
        else:
            h[idx] = 1.0 / idx.size                  # uniform over transient states
    elif np.isscalar(start):
        h[int(start)] = 1.0
    else:
        h = np.asarray(start, dtype=float)
        h = h / h.sum()

    P, _ = mrp_from_policy(env, pi)
    A = np.eye(idx.size) - P[np.ix_(idx, idx)].T
    try:
        eta = np.linalg.solve(A, h[idx])             # expected visits per transient state
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "(I - P_pi^T) is singular on the transient states: the policy is "
            "improper (some states never reach a terminal)."
        ) from exc

    mu = np.zeros(n)
    mu[idx] = eta / eta.sum()
    return mu
