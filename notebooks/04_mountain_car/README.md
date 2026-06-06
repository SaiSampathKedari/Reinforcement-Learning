# Mountain Car

Classic continuous-state control task (S&B Example 10.1). An underpowered car in
a valley must reach the goal on the right hill, but gravity beats the engine, so
it has to first reverse up the left slope to build momentum — things must get
*worse* (move away from the goal) before they can get better. Reward is `-1` every
step until the goal; state is continuous `(position, velocity)`; three throttle
actions `{-1, 0, +1}`. Environment code:
[`src/rl/envs/mountain_car.py`](../../src/rl/envs/mountain_car.py).

This is the first **control with approximation** task (Chapter 10): the value is
an *action*-value `q_hat(s, a) = w . x(s, a)`, the features are **tile coding**
over `(position, velocity)` with the action as a tile coordinate, and the policy
is learned by **episodic semi-gradient Sarsa**.

## Notebooks (planned)

| File | Reproduces | Method |
|---|---|---|
| `01_semi_gradient_sarsa.ipynb` | S&B Example 10.1 / Figure 10.1 | episodic semi-gradient one-step Sarsa + tile coding |
| `02_n_step_sarsa.ipynb` | S&B Figures 10.3 / 10.4 | n-step semi-gradient Sarsa |

## Policy animation

`01_semi_gradient_sarsa.ipynb` also animates the learned greedy policy driving the
car over the `sin(3x)` hill (`animate_policy` in
[`mountain_car_plots.py`](mountain_car_plots.py)) -- the car is tinted by its
throttle action (red reverse / grey coast / green forward) with a direction arrow
and a HUD. It shows the signature behavior: drive *away* from the goal first to
build momentum, then accelerate through to the flag. Saved as
[`mountain_car_policy.gif`](mountain_car_policy.gif).

![Mountain Car policy](mountain_car_policy.gif)

## Notes

- **Tile coding** uses Sutton's vendored `tiles3` (8 tilings, 8×8 over the bounded
  box, IHT size 4096). The action enters as an integer tile coordinate, so each
  action gets disjoint tiles in one shared weight vector `w`.
- **Exploration with `epsilon = 0`** comes from *optimistic initialization*:
  `w = 0` so `q_hat = 0` everywhere, but all true values are negative (every step
  costs `-1`), so visited states are pushed below the optimistic 0 and the agent
  is continually driven to explore new states until it finds the goal.
- **Cost-to-go** plotted in Fig 10.1 is `-max_a q_hat(s, a, w)` — roughly the
  number of steps to the goal from each state. It grows from a small bump (only
  the visited valley looks bad) to the converged ~120-step surface.
