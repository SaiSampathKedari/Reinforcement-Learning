"""Generate the Windy Gridworld overview figure for the README (static PNG).

One panel that tells the whole story: the value function V(s) as a heatmap with
readable per-cell values, the per-column wind, and the optimal greedy path S -> G.

Run:  python make_overview.py   (from this directory)
Writes: windy_gridworld_overview.png
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from rl.envs.windy_gridworld import WindyGridworld
from rl.td.sarsa import sarsa
from windy_gridworld_plots import greedy_rollout

CMAP = "viridis"
PATH_C = "#E63946"   # red
START_C = "#2a9d8f"  # teal
GOAL_C = "#E63946"   # red


def _train_optimal(seed_pool=range(20), n_episodes=500):
    """Train Sarsa; return the first run whose greedy path is the 15-step optimum."""
    env = WindyGridworld()
    best = None
    for seed in seed_pool:
        Q, pi = sarsa(env, np.random.default_rng(seed), n_episodes=n_episodes,
                      alpha=0.5, epsilon=0.1)
        n = len(greedy_rollout(env, pi)) - 1
        if best is None or n < best[0]:
            best = (n, Q, pi)
        if n == 15:
            return env, Q, pi
    return env, best[1], best[2]   # fall back to the shortest found


def make_overview():
    env, Q, pi = _train_optimal()
    V = Q.max(axis=1).reshape(env.N_ROWS, env.N_COLS)
    visited = (Q != 0).any(axis=1).reshape(env.N_ROWS, env.N_COLS)  # unlearned cells stay 0
    Vm = np.where(visited, V, np.nan)
    path = greedy_rollout(env, pi)
    n_steps = len(path) - 1

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = mpl.colormaps[CMAP].with_extremes(bad="#ECECEC")  # grey for unvisited
    norm = mpl.colors.Normalize(np.nanmin(Vm), np.nanmax(Vm))
    im = ax.imshow(Vm, cmap=cmap, norm=norm, aspect="equal")

    # Per-cell value labels with contrast-aware color (white on dark, black on bright).
    for r in range(env.N_ROWS):
        for c in range(env.N_COLS):
            if not visited[r, c]:
                continue
            rgba = cmap(norm(V[r, c]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(c, r, f"{V[r, c]:.0f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="black" if lum > 0.6 else "white", zorder=3)

    # Cell borders.
    ax.set_xticks(np.arange(-0.5, env.N_COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, env.N_ROWS), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(env.N_COLS))
    ax.set_yticks(range(env.N_ROWS))
    ax.set_xlabel("column")
    ax.set_ylabel("row (0 = top)")

    # Optimal greedy path as directed red arrows + markers.
    xs = [c for _, c in path]
    ys = [r for r, _ in path]
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        ax.annotate("", xy=(c1, r1), xytext=(c0, r0),
                    arrowprops=dict(arrowstyle="-|>", color=PATH_C, lw=2.6,
                                    shrinkA=6, shrinkB=6), zorder=5)
    ax.plot(xs, ys, "o", color=PATH_C, ms=4, zorder=6)

    # Start / goal.
    for (rr, cc), col, lab in [(env.START, START_C, "S"), (env.GOAL, GOAL_C, "G")]:
        ax.scatter(cc, rr, s=620, color=col, edgecolors="white", linewidths=2, zorder=7)
        ax.text(cc, rr, lab, ha="center", va="center", color="white",
                fontsize=15, fontweight="bold", zorder=8)

    # Wind strength under each column.
    ax.set_ylim(env.N_ROWS - 0.5 + 1.0, -0.5)
    ax.text(-1.0, env.N_ROWS - 0.5 + 0.55, "wind ↑", ha="right", va="center",
            fontsize=12, color="0.3", fontweight="bold")
    for c, w in enumerate(env.WIND):
        ax.text(c, env.N_ROWS - 0.5 + 0.55, str(w), ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=("0.55" if w == 0 else "#1d4e89"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("V(s) = maxₐ Q(s, a)  ≈  −(steps to goal)", fontsize=11)

    ax.set_title(
        f"Windy Gridworld — value function & optimal path  ({n_steps}-step solution)\n"
        "Sarsa, ε=0.1, α=0.5, γ=1.  Reward −1/step; columns push the agent up by the wind value.",
        fontsize=13, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    return fig, n_steps


if __name__ == "__main__":
    fig, n = make_overview()
    out = "windy_gridworld_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (greedy path = {n} steps)")
