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

## Notes

- **Why only `player_sum` 12-21.** Below 12 the player can hit without any risk
  of going bust, so the choice is trivial -- the env auto-hits at reset and only
  the 200 non-trivial decision states are modelled.
- **Infinite deck.** Cards are drawn with replacement (`P(ace)=P(2..9)=1/13`,
  `P(10)=4/13`), so the state `(player_sum, dealer_show, usable_ace)` is Markov
  without tracking the cards already dealt. Card helpers are ported from
  Gymnasium so behaviour can be cross-checked.
- **Fixed dealer.** The dealer follows a fixed policy (hits until sum >= 17), so
  this is prediction / control against a stationary environment.
- **Why Monte Carlo fits.** Episodes are short and always terminate, and the
  model is unknown to the agent -- so full-return MC needs no bootstrapping.
  Fig 5.1 uses first-visit MC prediction; Fig 5.2 uses MC control with exploring
  starts to guarantee every `(state, action)` pair is sampled.
- **Noisier usable-ace surface.** Usable-ace states are visited far less often,
  so their value estimates are rougher -- the jaggedness in those plots is sample
  noise, matching the book.

![Blackjack optimal policy and value function](blackjack_overview.png)
