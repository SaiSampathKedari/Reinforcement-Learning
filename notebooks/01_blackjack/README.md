# Blackjack

Tabular Blackjack (S&B §5.1). State `(player_sum, dealer_show, usable_ace)`;
decisions only modelled for `player_sum in [12, 21]`. Env code lives in
[`src/rl/envs/blackjack.py`](../../src/rl/envs/blackjack.py).

## Notebooks

| File | Reproduces | Algorithm |
|---|---|---|
| [01_mc_prediction_blackjack.ipynb](01_mc_prediction_blackjack.ipynb) | S&B Fig 5.1 | First-visit MC prediction |
| [02_mc_control_es_blackjack.ipynb](02_mc_control_es_blackjack.ipynb) | S&B Fig 5.2 | MC Control with Exploring Starts |

## Helper files

- [blackjack_plots.py](blackjack_plots.py) -- 3D surface and 2D heatmap views
  over the Blackjack `(player_sum, dealer_show)` grid for each usable-ace case.
