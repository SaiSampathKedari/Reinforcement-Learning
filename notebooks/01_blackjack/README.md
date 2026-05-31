# Blackjack

Tabular Blackjack (S&B §5.1). State `(player_sum, dealer_show, usable_ace)`;
decisions are modelled only for `player_sum in [12, 21]`. Environment code:
[`src/rl/envs/blackjack.py`](../../src/rl/envs/blackjack.py).

## Notebooks

| File | Reproduces | Algorithm |
|---|---|---|
| [`01_mc_prediction_blackjack.ipynb`](01_mc_prediction_blackjack.ipynb) | S&B Fig 5.1 | First-visit MC prediction |
| [`02_mc_control_es_blackjack.ipynb`](02_mc_control_es_blackjack.ipynb) | S&B Fig 5.2 | MC control with exploring starts |

## Helpers

- [`blackjack_plots.py`](blackjack_plots.py): 3D surface and 2D heatmap views of
  the value function and policy over the `(player_sum, dealer_show)` grid, per
  usable-ace case.
- [`make_overview.py`](make_overview.py): regenerates the gallery figure
  `blackjack_overview.png` (S&B Fig 5.2). Run `python make_overview.py`.

![Blackjack optimal policy and value function](blackjack_overview.png)
