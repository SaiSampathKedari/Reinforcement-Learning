# Windy Gridworld

7×10 gridworld with an upward, column-varying crosswind (S&B Example 6.5, §6.4).
State is a cell `(row, col)` encoded as an int in `[0, 70)`, with row 0 at the
top so the wind decreases the row index. Reward is `-1` per step until the goal;
undiscounted (`gamma = 1`). Env code lives in
[`src/rl/envs/windy_gridworld.py`](../../src/rl/envs/windy_gridworld.py), with
optional King's-moves and stochastic-wind variants for Exercises 6.9 / 6.10.

## Notebooks

| File | Reproduces | Algorithm |
|---|---|---|
| [01_sarsa_windy_gridworld.ipynb](01_sarsa_windy_gridworld.ipynb) | S&B Example 6.5 | On-policy TD control (Sarsa) |

## S&B figure cross-reference

| Figure / Exercise | Notebook |
|---|---|
| Example 6.5 (Sarsa learning curve) | [01_sarsa_windy_gridworld.ipynb](01_sarsa_windy_gridworld.ipynb) |
| Exercise 6.9 (King's moves) | _planned_ |
| Exercise 6.10 (Stochastic wind) | _planned_ |

## Helper files

- [windy_gridworld_plots.py](windy_gridworld_plots.py) — **interactive Plotly**
  figures (clean editorial style; hover for exact values): gridworld layout, the
  episodes-vs-time-steps learning curve, steps-per-episode view, greedy-policy
  arrow grid, `V(s)` heatmap, the greedy start-to-goal trajectory, and the
  learning-dynamics plots (`V(start)`, policy churn, and an **animated**
  policy-evolution figure with a play/slider).

## Notes

- **Two telemetry planes.** Episode lengths (for the Example 6.5 curve) are an
  env-side signal, captured with the `EpisodeStats` wrapper in
  [`src/rl/envs/wrappers.py`](../../src/rl/envs/wrappers.py) (tabular analogue of
  Gymnasium's `RecordEpisodeStatistics`). The evolving `Q`-table is an agent-side
  signal, captured with the agents' `on_episode_end` callback. Neither requires
  touching the algorithm.
- Training runs 500 episodes (vs the book's ~170) so the **greedy** policy fully
  converges to 15 steps: at ~8000 steps the greedy policy can still hold a
  wind-canceling self-loop, the very non-termination issue the book cites as the
  reason Monte Carlo is awkward here. (`TimeLimit` in `wrappers.py` bounds such
  runaway episodes.)
