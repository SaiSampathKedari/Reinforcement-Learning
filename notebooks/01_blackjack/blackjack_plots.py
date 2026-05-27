"""Plotting helpers for Blackjack value functions and policies.

Produces 3D surface views, 2D value heatmaps, and binary policy heatmaps
over the Blackjack state grid `(player_sum x dealer_show)`, separately for
usable-ace and no-usable-ace cases.

Layouts:
    - Fig 5.1 (MC prediction): V^pi at two episode counts.
    - Fig 5.2 (MC control ES): pi* and V* = max_a Q*(s, a).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Decision-state grid.
PLAYER_SUMS  = np.arange(12, 22)                            # 12..21
DEALER_SHOWS = np.arange(1, 11)                              # 1..10
DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]  # book uses "A" for ace

# Colormaps.
CMAP = "viridis"
POLICY_CMAP = ListedColormap(["#B8D4E3", "#F4A261"])  # STICK=calm blue, HIT=warm orange


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def reshape_value(V: np.ndarray) -> np.ndarray:
    """Reshape V from `(201,)` to `(10, 10, 2)` indexed by
    `(player_sum - 12, dealer_show - 1, usable_ace)`.
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
    ax.set_xticklabels(["A", "5", "10"])
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
    ax.set_xticklabels(DEALER_LABELS)
    ax.set_yticks(np.arange(12, 22))
    ax.set_title(title, fontsize=11)
    return im


def _policy_panel(pi_slice: np.ndarray, ax, title: str) -> None:
    """2D heatmap of a binary tabular policy (0=STICK, 1=HIT).

    `pi_slice` has shape `(10, 10)`, indexed by `(player_sum-12, dealer_show-1)`.
    Includes cell grid lines and region labels matching S&B Fig 5.2.
    """
    ax.imshow(
        pi_slice,
        origin="lower",
        cmap=POLICY_CMAP,
        vmin=0, vmax=1,
        extent=[0.5, 10.5, 11.5, 21.5],
        aspect="auto",
    )
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_xticks(np.arange(1, 11))
    ax.set_xticklabels(DEALER_LABELS)
    ax.set_yticks(np.arange(12, 22))
    ax.set_title(title, fontsize=11)

    # Cell grid lines for readability.
    ax.set_xticks(np.arange(0.5, 11.5), minor=True)
    ax.set_yticks(np.arange(11.5, 22.5), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.tick_params(which="minor", length=0)

    # Region labels (placed at typical STICK/HIT centroids for Blackjack).
    ax.text(5.5, 20, "STICK", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2c3e50")
    ax.text(5.5, 14, "HIT", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#7f3b08")


# ---------------------------------------------------------------------------
# Figure-level functions (one per S&B figure)
# ---------------------------------------------------------------------------

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
    fig.suptitle("Monte Carlo Prediction  --  S&B Fig 5.1",
                 fontsize=14, fontweight="bold", y=0.98)
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
    """Reproduce S&B Fig 5.1 as 2D heatmaps (same layout as the 3D version)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    grids = [(n_short, reshape_value(V_short)), (n_long, reshape_value(V_long))]
    im = None
    for col, (n, grid) in enumerate(grids):
        for row, ace in enumerate([1, 0]):
            label = "Usable ace" if ace else "No usable ace"
            im = _heatmap_panel(grid, ace, axes[row, col], f"{label}  |  {n:,} episodes")
    fig.colorbar(im, ax=axes, shrink=0.7, location="right", label="V")
    fig.suptitle("Monte Carlo Prediction  --  S&B Fig 5.1",
                 fontsize=14, fontweight="bold")
    return fig


def plot_fig_5_2(
    Q: np.ndarray,
    pi: np.ndarray,
    n_episodes: int = 500_000,
):
    """Reproduce S&B Fig 5.2: pi* (policy heatmap) and V* (3D surface).

    Layout (2 rows x 2 cols):
        row 0 = usable ace,    cols = [pi*, V*]
        row 1 = no usable ace, cols = [pi*, V*]
    """
    V_star = Q.max(axis=1)
    pi_grid = pi[:200].reshape(10, 10, 2)
    V_grid = V_star[:200].reshape(10, 10, 2)

    fig = plt.figure(figsize=(13, 10))
    for row, ace in enumerate([1, 0]):
        label = "Usable ace" if ace else "No usable ace"
        # Left column: pi*
        ax_pi = fig.add_subplot(2, 2, row * 2 + 1)
        _policy_panel(pi_grid[:, :, ace], ax_pi, f"pi*  --  {label}")
        # Right column: V*
        ax_v = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
        _surface_panel(V_grid, ace, ax_v, f"V*  --  {label}")

    fig.suptitle(f"MC Control with Exploring Starts  --  S&B Fig 5.2  ({n_episodes:,} episodes)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(
        left=0.05, right=0.96, top=0.92, bottom=0.05,
        wspace=0.15, hspace=0.25,
    )
    return fig
