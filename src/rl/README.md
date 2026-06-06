# `rl` package map

Code is organized by **method**, not by book chapter. Tabular methods
(`monte_carlo`, `td`) stay separate from their function-approximation
counterparts (`*_approx`), since the two are mathematically distinct. The
representation machinery is factored into two reusable axes shared across
Chapters 9-13, each split by what it operates on (prediction vs control):

- **`features/`** -- representations: state features `x(s)` (`FeatureMap`) for
  prediction, and state-action features `x(s, a)` (`StateActionFeatureMap`) for
  control.
- **`approximators/`** -- parameterized models holding the weights `w`:
  `v_hat(s)` (`ValueApproximator`) and `q_hat(s, a)` (`ActionValueApproximator`).
  They predict and report gradients but do **not** update themselves; the learner
  (algorithm) owns the update `w <- w + alpha * delta * grad`.

Environments mirror the same split: `Env` is the minimal `reset`/`step`
interface (any state type, e.g. continuous Mountain Car), and `TabularEnv` adds
the enumerable extras (`n_states`, `transitions`, `terminal_states`) that DP and
tabular methods need.

| Package | S&B | Contents |
|---|---|---|
| `envs/` | -- | tabular (Blackjack, Windy Gridworld, Random Walk) + continuous (Mountain Car); `Env` / `TabularEnv` contracts |
| `utils/` | -- | episode rollout, epsilon-greedy / greedy action selection, RMSVE metric |
| `dynamic_programming/` | Ch 4 | exact policy evaluation, policy value, on-policy distribution |
| `monte_carlo/` | Ch 5 | tabular MC prediction and control |
| `td/` | Ch 6-7, 12 | Sarsa, Q-learning, n-step Sarsa / n-step TD, TD(lambda), Sarsa(lambda) (tabular) |
| `features/` | Ch 9-10 | state aggregation, tile coding (vendored `tiles3`); Fourier / polynomial later |
| `approximators/` | Ch 9+ | linear state- and action-value models (neural later) |
| `prediction_approx/` | Ch 9 | gradient MC, semi-gradient TD, n-step semi-gradient TD |
| `control_approx/` | Ch 10 | semi-gradient Sarsa, n-step semi-gradient Sarsa |
| `offpolicy_approx/` | Ch 11 | gradient-TD (TDC), emphatic-TD _(planned)_ |
| `traces_approx/` | Ch 12 | TD(lambda), true-online TD(lambda) with FA _(planned)_ |
| `policy_gradient/` | Ch 13 | REINFORCE, actor-critic _(planned)_ |
