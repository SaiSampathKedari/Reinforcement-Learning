# Notebooks

Experiments reproducing figures from Sutton & Barto. Each subdirectory is one
environment; notebooks are numbered in algorithm-progression order. Install once
from the repo root (`pip install -e .`), then run a notebook from its folder.
Figures are interactive where noted.

## Environments

| Folder | Environment | Methods | Source |
|---|---|---|---|
| [`01_blackjack/`](01_blackjack/) | Blackjack | Monte Carlo prediction & control | S&B §5.1 |
| [`02_windy_gridworld/`](02_windy_gridworld/) | Windy Gridworld | Sarsa (on-policy TD control) | S&B Example 6.5 |
| [`03_random_walk/`](03_random_walk/) | Random Walk _(planned)_ | TD / n-step / function-approximation prediction | S&B Examples 6.2 / 7.1 / 9.1 |
| [`04_mountain_car/`](04_mountain_car/) | Mountain Car | semi-gradient Sarsa control + tile coding | S&B Example 10.1 |

## Sutton & Barto cross-reference

| Figure / Example | Notebook |
|---|---|
| Fig 5.1 (MC prediction) | [`01_blackjack/01_mc_prediction_blackjack.ipynb`](01_blackjack/01_mc_prediction_blackjack.ipynb) |
| Fig 5.2 (MC control, exploring starts) | [`01_blackjack/02_mc_control_es_blackjack.ipynb`](01_blackjack/02_mc_control_es_blackjack.ipynb) |
| Example 6.5 (Sarsa learning curve) | [`02_windy_gridworld/01_sarsa_windy_gridworld.ipynb`](02_windy_gridworld/01_sarsa_windy_gridworld.ipynb) |
| Fig 9.1 (gradient MC, state aggregation) | [`03_random_walk/03_state_aggregation_mc.ipynb`](03_random_walk/03_state_aggregation_mc.ipynb) |
| Fig 9.2 (n-step bootstrapping) | [`03_random_walk/04_bootstrapping_nstep.ipynb`](03_random_walk/04_bootstrapping_nstep.ipynb) |
| Fig 10.1 (Mountain Car cost-to-go) | [`04_mountain_car/01_semi_gradient_sarsa.ipynb`](04_mountain_car/01_semi_gradient_sarsa.ipynb) |
