# `rl` package map

Code is organized by **method**, not by book chapter. Tabular methods
(`monte_carlo`, `td`) stay separate from their function-approximation
counterparts (`*_approx`), since the two are mathematically distinct. The
representation machinery is factored out into two reusable axes shared across
Chapters 9-13:

- **`features/`** -- feature maps `x(s)`: the representation.
- **`approximators/`** -- parameterized models `v_hat(s) = f(s; w)`: hold the
  weights, predict, report gradients. They do **not** update themselves; the
  learner (algorithm) owns the update.

| Package | S&B | Contents |
|---|---|---|
| `envs/` | -- | Blackjack, Windy Gridworld, Random Walk (stateless `TabularEnv` contract) |
| `utils/` | -- | episode rollout, policy helpers |
| `dynamic_programming/` | Ch 4 | exact policy evaluation, on-policy distribution |
| `monte_carlo/` | Ch 5 | tabular MC prediction and control |
| `td/` | Ch 6-7, 12 | Sarsa, Q-learning, n-step, TD(lambda) (tabular) |
| `features/` | Ch 9 | feature maps `x(s)`: state aggregation (tile coding, Fourier later) |
| `approximators/` | Ch 9+ | value/policy models: linear (neural later) |
| `prediction_approx/` | Ch 9 | gradient MC, semi-gradient TD |
| `control_approx/` | Ch 10 | semi-gradient Sarsa |
| `offpolicy_approx/` | Ch 11 | gradient-TD (TDC), emphatic-TD |
| `traces_approx/` | Ch 12 | TD(lambda), true-online TD(lambda) with FA |
| `policy_gradient/` | Ch 13 | REINFORCE, actor-critic |
