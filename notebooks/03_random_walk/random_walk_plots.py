"""Plot for the 1000-state Random Walk (S&B Figure 9.1)."""

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
