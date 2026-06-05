"""Plots for the 1000-state Random Walk (S&B Figures 9.1, 9.2)."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

TRUE_C = "#E63946"   # true value (red)
APPROX_C = "#1D3557" # approximate value (navy)
MU_C = "#9AA0A6"     # state distribution (grey)


def plot_figure_9_1(states, v_hat, v_true, mu, n_episodes=None):
    """Reproduce S&B Fig 9.1: approximate value v_hat vs true value v_pi, with mu.

    Left axis: the true value v_pi (smooth) and the state-aggregation estimate
    v_hat (a step function), over the chain `states`. Right axis (separate, small
    scale): the on-policy state distribution mu. Returns the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(states, v_true, color=TRUE_C, lw=2.0, label=r"true value $v_\pi$")
    ax.plot(states, v_hat, color=APPROX_C, lw=2.0,
            label=r"approximate MC value $\hat{v}$")
    ax.set_xlabel("State")
    ax.set_ylabel("Value")
    ax.set_ylim(-1.05, 1.05)
    ax.margins(x=0)
    ax.legend(loc="upper left", frameon=False)

    # State distribution on a separate, smaller right-hand scale (sits low, as in the book).
    ax2 = ax.twinx()
    ax2.fill_between(states, mu, color=MU_C, alpha=0.35)
    ax2.plot(states, mu, color=MU_C, lw=1.0, label=r"distribution $\mu$")
    ax2.set_ylabel(r"distribution $\mu$", color="0.45")
    ax2.tick_params(axis="y", colors="0.45")
    ax2.set_ylim(0, float(np.max(mu)) * 3.0)
    ax2.margins(x=0)

    title = "Figure 9.1 — gradient MC with state aggregation (1000-state random walk)"
    if n_episodes is not None:
        title += f"\n{n_episodes:,} episodes"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_figure_9_2(states, v_hat_td, v_true, alphas, n_values, errors, n_runs=None):
    """Reproduce S&B Fig 9.2 (two panels).

    Left: the near-asymptotic semi-gradient TD(0) value `v_hat_td` (a step
    function, 10 groups) against the true value `v_pi` -- worse than the MC
    approximation of Fig 9.1.

    Right: n-step semi-gradient TD with 20-group aggregation. `errors` is the
    average RMS error (over all states and the first few episodes), shape
    (len(n_values), len(alphas)); each row is one learning-rate curve for a given
    n, plotted against `alphas`. Returns the matplotlib figure.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: asymptotic TD value vs true value ---
    axL.plot(states, v_true, color=TRUE_C, lw=2.0, label=r"true value $v_\pi$")
    axL.plot(states, v_hat_td, color=APPROX_C, lw=2.0,
             label=r"approximate TD value $\hat{v}$")
    axL.set_xlabel("State")
    axL.set_ylabel("Value")
    axL.set_ylim(-1.05, 1.05)
    axL.margins(x=0)
    axL.legend(loc="upper left", frameon=False)
    axL.set_title("Left: asymptotic semi-gradient TD(0)\n(10 groups)", fontsize=11)

    # --- Right: RMS error vs alpha, one curve per n ---
    errors = np.asarray(errors)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(n_values)))
    for row, n, c in zip(errors, n_values, cmap):
        axR.plot(alphas, row, color=c, lw=1.6, label=f"n={n}")
    axR.set_xlabel(r"$\alpha$")
    axR.set_ylabel("Average RMS error\n(over 1000 states, first 10 episodes)")
    axR.set_xlim(0, 1)
    axR.set_ylim(0.25, 0.55)
    axR.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axR.set_title("Right: n-step semi-gradient TD\n(20 groups)", fontsize=11)

    sup = "Figure 9.2 — bootstrapping with state aggregation (1000-state random walk)"
    if n_runs is not None:
        sup += f"   ({n_runs} runs)"
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
