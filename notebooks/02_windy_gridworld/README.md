# Windy Gridworld

7x10 gridworld with an upward, column-varying crosswind (S&B Example 6.5, §6.4).
State is a cell `(row, col)` encoded as an int in `[0, 70)`, with row 0 at the
top so the wind decreases the row index. Reward is `-1` per step until the goal;
undiscounted (`gamma = 1`). Environment code:
[`src/rl/envs/windy_gridworld.py`](../../src/rl/envs/windy_gridworld.py), with
optional King's-moves and stochastic-wind variants (Exercises 6.9 / 6.10).

## Notebooks

| File | Reproduces | Algorithm |
|---|---|---|
| [`01_sarsa_windy_gridworld.ipynb`](01_sarsa_windy_gridworld.ipynb) | S&B Example 6.5 | On-policy TD control (Sarsa) |

## Sutton & Barto cross-reference

| Figure / Exercise | Status |
|---|---|
| Example 6.5 (Sarsa learning curve) | [`01_sarsa_windy_gridworld.ipynb`](01_sarsa_windy_gridworld.ipynb) |
| Exercise 6.9 (King's moves) | planned |
| Exercise 6.10 (stochastic wind) | planned |

## Helpers

- [`windy_gridworld_plots.py`](windy_gridworld_plots.py): interactive Plotly
  figures (hover for exact values): gridworld layout, the episodes-vs-time-steps
  learning curve, steps-per-episode view, greedy-policy arrows, the `V(s)`
  heatmap + 3D surface, the greedy trajectory, and an animated policy-evolution
  figure (play / slider).
- [`make_overview.py`](make_overview.py): regenerates the gallery figure
  `windy_gridworld_overview.png`. Run `python make_overview.py`.

## Notes

- **Two telemetry planes.** Episode lengths (for the Example 6.5 curve) are an
  env-side signal, captured by the `EpisodeStats` wrapper in
  [`src/rl/envs/wrappers.py`](../../src/rl/envs/wrappers.py) (the tabular analogue
  of Gymnasium's `RecordEpisodeStatistics`). The evolving `Q`-table is an
  agent-side signal, captured by the agents' `on_episode_end` callback. Neither
  touches the algorithm.
- Training runs 500 episodes (vs the book's ~170) so the greedy policy fully
  converges to 15 steps. With constant epsilon the greedy policy can briefly hold
  a wind-canceling self-loop, the non-termination issue the book cites against
  Monte Carlo here; `TimeLimit` in `wrappers.py` bounds such runaway episodes.

![Windy Gridworld value function and optimal path](windy_gridworld_overview.png)
