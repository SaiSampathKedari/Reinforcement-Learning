# Reports

Self-contained write-ups of the algorithms in this repo: derivations, update
rules, convergence arguments, and the link from theory to the implementation in
[`src/rl/`](../src/rl/). Compiled to PDF; numbered in reading order, which roughly
follows Sutton & Barto.

| # | Report | S&B | Topic |
|---|---|---|---|
| 01 | [Monte Carlo Control with Exploring Starts](01_Monte-Carlo-Control-with-Exploring-Starts.pdf) | §5.3 | MC control, exploring starts |
| 02 | [Monte Carlo Control without Exploring Starts](02_Monte-Carlo-Control-without-Exploring-Starts.pdf) | §5.4 | epsilon-soft on-policy MC control |
| 03 | [TD Prediction & Eligibility Traces](03_TD-Prediction-Eligibility-Traces.pdf) | §6.1, Ch 12 | TD(0), n-step, forward/backward TD(lambda) |
| 04 | [On-Policy TD Control: SARSA(0)](04_On-Policy-TD-Control-SARSA-0.pdf) | §6.4 | one-step Sarsa |
| 05 | [On-Policy TD Control: n-Step SARSA](05_On-Policy-TD-Control-n-Step-SARSA.pdf) | §7.2 | n-step Sarsa |
| 06 | [On-Policy TD Control: SARSA(λ)](06_On-Policy-TD-Control-SARSA-Lambda.pdf) | §12.7 | Sarsa(lambda), eligibility traces |
| 07 | [Off-Policy TD Control: Q-Learning](07_Off-Policy-TD-Control-Q-Learning.pdf) | §6.5 | Q-learning |
| 08 | [On-Policy Prediction: Value-Function Approximation](08_On-Policy-Prediction_Value-Function-Approximation.pdf) | Ch 9 | gradient MC, semi-gradient TD, the VE objective |
| 10 | [The Policy Gradient Theorem](10_Policy-Gradient-Theorem.pdf) | §13.2 | discounted objective, visitation measure, exact gradient |
| 11 | [Average-Reward Policy Gradient Theorem](11_Average-Reward-Policy-Gradient-Theorem.pdf) | §13.6 | continuing-task objective, stationary distribution |
| 12 | [Policy Gradient Theorem: Episodic Trajectory Route](12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf) | §13.2 | trajectory-likelihood derivation |
| 13 | [Policy Gradient Preliminaries](13_Policy-Gradient-Preliminaries.pdf) | -- | score function, log-derivative trick, expected score is zero |
| 14 | [REINFORCE](14_REINFORCE.pdf) | §13.3 | Monte Carlo policy gradient |
| 15 | [Actor-Critic](15_Actor-Critic.pdf) | §13.5 | GPI view; from the exact gradient to QAC, one move at a time |
| 16 | [Actor-Critic with a Baseline](16_Actor-Critic-with-a-Baseline.pdf) | §13.4-13.5 | baseline identity, advantage, TD error as one-sample advantage |
| 17 | [GAE Actor-Critic](17_GAE_Actor-Critic.pdf) | -- | n-step advantages and their λ-mixture (Schulman et al., 2016) |

### In progress

- **On-Policy Control with Approximation** (Ch 10) -- episodic semi-gradient
  Sarsa and n-step Sarsa; the Mountain Car study (reserved as report 09).
- **Trust-region line** -- natural policy gradient, TRPO, PPO.

> Theory companions to the code: prediction lives in
> [`prediction_approx/`](../src/rl/prediction_approx/), control in
> [`control_approx/`](../src/rl/control_approx/), and the shared representation
> machinery in [`features/`](../src/rl/features/) and
> [`approximators/`](../src/rl/approximators/).
