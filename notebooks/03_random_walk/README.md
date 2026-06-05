# Random Walk

Linear-chain random walk used to study **prediction** (S&B Examples 6.2, 7.1, 9.1).
A chain of `n_states` non-terminal states with a terminal at each end; episodes
start in the center. The same environment represents every walk: the jump range,
edge clipping, and any bias all live in the policy `pi(s' | s)`, so `±1`, `±10`,
and `±100` walks are different policies on one env. Environment code:
[`src/rl/envs/random_walk.py`](../../src/rl/envs/random_walk.py).

## Notebooks (planned)

| File | Reproduces | Method |
|---|---|---|
| `01_td_prediction_random_walk.ipynb` | S&B Example 6.2 (5-state) | TD(0) vs MC prediction |
| `02_n_step_td_random_walk.ipynb` | S&B Example 7.1 (19-state) | n-step TD prediction |
| `03_state_aggregation_mc.ipynb` | S&B Example 9.1 (1000-state) | gradient MC with state aggregation |
| `04_bootstrapping_nstep.ipynb` | S&B Example 9.2 / Figure 9.2 (1000-state) | semi-gradient TD(0) and n-step semi-gradient TD with state aggregation |

## Notes

- It is a Markov reward process encoded as an MDP: the action *is* the next
  state, transitions are deterministic, and the walk's randomness comes from the
  policy. So `pi(s' | s)` is the transition kernel, and true values follow from
  solving `v = (I - gamma P_pi)^{-1} r_pi`.
- Example 9.1 is the first function-approximation method (state aggregation); its
  agent will live in a new `src/rl/function_approximation/` module, not in `src/rl/td/`.
