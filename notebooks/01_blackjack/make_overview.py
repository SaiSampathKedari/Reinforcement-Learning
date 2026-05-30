"""Generate the Blackjack overview figure for the README (static PNG).

Reproduces S&B Fig 5.2: optimal policy pi* (hit/stick maps) and value function
V* (3D surfaces), for usable-ace and no-usable-ace hands. The single best
representation of solved Blackjack.

Run:  python make_overview.py   (from this directory)
Writes: blackjack_overview.png
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")

from rl.envs.blackjack import Blackjack
from rl.monte_carlo.mc_control_es import mc_control_es
from blackjack_plots import plot_fig_5_2

N_EPISODES = 500_000


def main():
    env = Blackjack()
    Q, pi = mc_control_es(env, np.random.default_rng(0), n_episodes=N_EPISODES)
    fig = plot_fig_5_2(Q, pi, n_episodes=N_EPISODES)
    fig.savefig("blackjack_overview.png", dpi=150, bbox_inches="tight")
    print(f"wrote blackjack_overview.png  ({N_EPISODES:,} episodes)")


if __name__ == "__main__":
    main()
