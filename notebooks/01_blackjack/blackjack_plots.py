"""Plotting helpers for Blackjack value functions.

Produces both 3D surface views and 2D heatmap views over the Blackjack state
grid `(player_sum x dealer_show)`, separately for usable-ace and no-usable-ace
cases. Reproduction layout matches S&B Fig 5.1. Uses a diverging colormap
(`coolwarm`) so the V = 0 level reads as neutral white, with red for wins
and blue for losses.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Decision-state grid: player_sum in [12, 21], dealer_show in [1, 10].
PLAYER_SUMS = np.arange(12, 22)
DEALER_SHOWS = np.arange(1, 11)

CMAP = "viridis"   # sequential: purple (loss) -> green -> yellow (win)
SUPTITLE = "Monte Carlo Prediction on Blackjack  --  S&B Fig 5.1"


def reshape_value(V: np.ndarray) -> np.ndarray:
    """Reshape V from `(201,)` to `(10, 10, 2)` indexed by
    `(player_sum - 12, dealer_show - 1, usable_ace)`.
    The terminal state (index 200) is dropped.
    """
    return V[:200].reshape(10, 10, 2)


def _surface_panel(V_grid: np.ndarray, usable_ace: int, ax, title: str) -> None:
    """3D surface plot for one usable-ace slice of V."""
    D, P = np.meshgrid(DEALER_SHOWS, PLAYER_SUMS)
    Z = V_grid[:, :, usable_ace]
    ax.plot_surface(
        D, P, Z,
        cmap=CMAP, vmin=-1, vmax=1,
        edgecolor="none", alpha=0.95,
    )
    ax.set_xlabel("Dealer showing", labelpad=6)
    ax.set_ylabel("Player sum", labelpad=6)
    ax.set_zlabel("V", labelpad=4)
    ax.set_xticks([1, 5, 10])
    ax.set_yticks([12, 15, 18, 21])
    ax.set_zticks([-1, 0, 1])
    ax.set_zlim(-1, 1)
    ax.set_title(title, fontsize=11, pad=12)
    ax.view_init(elev=25, azim=-50)


def _heatmap_panel(V_grid: np.ndarray, usable_ace: int, ax, title: str):
    """2D heatmap for one usable-ace slice of V."""
    Z = V_grid[:, :, usable_ace]
    im = ax.imshow(
        Z,
        origin="lower",
        cmap=CMAP,
        vmin=-1, vmax=1,
        extent=[0.5, 10.5, 11.5, 21.5],
        aspect="auto",
    )
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_xticks(np.arange(1, 11))
    ax.set_yticks(np.arange(12, 22))
    ax.set_title(title, fontsize=11)
    return im


def plot_fig_5_1_3d(
    V_short: np.ndarray,
    V_long: np.ndarray,
    n_short: int = 10_000,
    n_long: int = 500_000,
):
    """Reproduce S&B Fig 5.1 as 3D surface plots.

    Layout (2 rows x 2 cols):
        row 0 = usable ace,    cols = [n_short, n_long] episodes
        row 1 = no usable ace, cols = [n_short, n_long] episodes
    """
    fig = plt.figure(figsize=(14, 11))
    grids = [(n_short, reshape_value(V_short)), (n_long, reshape_value(V_long))]
    for col, (n, grid) in enumerate(grids):
        for row, ace in enumerate([1, 0]):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            label = "Usable ace" if ace else "No usable ace"
            _surface_panel(grid, ace, ax, f"{label}  |  {n:,} episodes")
    fig.suptitle(SUPTITLE, fontsize=14, fontweight="bold", y=0.98)
    # Manual margins -- constrained_layout doesn't play well with 3D + labels.
    plt.subplots_adjust(
        left=0.02, right=0.96, top=0.92, bottom=0.04,
        wspace=0.10, hspace=0.20,
    )
    return fig


def plot_fig_5_1_heatmap(
    V_short: np.ndarray,
    V_long: np.ndarray,
    n_short: int = 10_000,
    n_long: int = 500_000,
):
    """Reproduce S&B Fig 5.1 as 2D heatmaps (same layout as the 3D version).

    Single shared colorbar on the right; v-range pinned to [-1, 1] so colors
    are comparable across all four panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    grids = [(n_short, reshape_value(V_short)), (n_long, reshape_value(V_long))]
    im = None
    for col, (n, grid) in enumerate(grids):
        for row, ace in enumerate([1, 0]):
            label = "Usable ace" if ace else "No usable ace"
            im = _heatmap_panel(grid, ace, axes[row, col], f"{label}  |  {n:,} episodes")
    fig.colorbar(im, ax=axes, shrink=0.7, location="right", label="V")
    fig.suptitle(SUPTITLE, fontsize=14, fontweight="bold")
    return fig
