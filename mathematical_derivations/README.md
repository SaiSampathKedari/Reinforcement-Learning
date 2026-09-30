# Mathematical derivations

Self-contained PDFs covering the mathematical foundations, update rules, and
convergence arguments behind the algorithms in [`src/rl/`](../src/rl/).
Read 01–03 for the Sutton-notation MDP foundations, then continue through
tabular learning, function approximation, policy gradients, and actor-critic.

## Reading order and numbering

RL and [Deep RL](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning/tree/master/mathematical_derivations)
use one shared sequence. A derivation included in both repositories has the same
number and filename. RL covers 01–20; Deep RL shares 13–20 and continues from 21.
Reserved entries are explicitly marked and have no PDF yet. After this
reorganization, keep assigned numbers stable and use the indexes to guide readers
to future supplementary material.

The **Previous #** column records the former RL/Deep RL filename numbers.
Existing PDFs retain their original text, including historical report-number
references; use this column to locate reports from the former sequence. Existing
numbers shifted by three (for example, former report 15 is now 18). This mapping
does not apply to numbering in other repositories or to pre-existing inconsistent
citations within a PDF.

## MDP and dynamic-programming foundations

| # | Derivation | S&B | Topic |
|---|---|---|---|
| 01 | [Policy Evaluation in Sutton Notation](01_Policy-Evaluation-in-Sutton-Notation.pdf) | Ch 3–4 | infinite-horizon discounted returns, state/action values, Bellman expectation equations |
| 02 | [Bellman Operators](02_Bellman-Operators.pdf) | Ch 3–4 | expectation and optimality operators in state-value and action-value spaces; links to learning algorithms |
| 03 | [Policy Improvement Theorem](03_Policy-Improvement-Theorem.pdf) | §4.2 | componentwise and operator proofs; greedy policy corollary |

These are copies of reports
[09](https://github.com/SaiSampathKedari/Sequential-Decision-Making/blob/master/02_Infinite-Horizon-MDPs/09_Policy-Evaluation-in-Sutton-Notation.pdf),
[10](https://github.com/SaiSampathKedari/Sequential-Decision-Making/blob/master/02_Infinite-Horizon-MDPs/10_Bellman-Operators.pdf), and
[11](https://github.com/SaiSampathKedari/Sequential-Decision-Making/blob/master/02_Infinite-Horizon-MDPs/11_Policy-Improvement-Theorem.pdf)
from Sequential Decision Making. The originals keep their own numbering there.

## Tabular learning and function approximation

| # | Derivation | S&B | Topic | Previous # |
|---|---|---|---|---|
| 04 | [Monte Carlo Control with Exploring Starts](04_Monte-Carlo-Control-with-Exploring-Starts.pdf) | §5.3 | MC control, exploring starts | 01 |
| 05 | [Monte Carlo Control without Exploring Starts](05_Monte-Carlo-Control-without-Exploring-Starts.pdf) | §5.4 | epsilon-soft on-policy MC control | 02 |
| 06 | [TD Prediction & Eligibility Traces](06_TD-Prediction-Eligibility-Traces.pdf) | §6.1, Ch 12 | TD(0), n-step, forward/backward TD(lambda) | 03 |
| 07 | [On-Policy TD Control: SARSA(0)](07_On-Policy-TD-Control-SARSA-0.pdf) | §6.4 | one-step Sarsa | 04 |
| 08 | [On-Policy TD Control: n-Step SARSA](08_On-Policy-TD-Control-n-Step-SARSA.pdf) | §7.2 | n-step Sarsa | 05 |
| 09 | [On-Policy TD Control: SARSA(λ)](09_On-Policy-TD-Control-SARSA-Lambda.pdf) | §12.7 | Sarsa(lambda), eligibility traces | 06 |
| 10 | [Off-Policy TD Control: Q-Learning](10_Off-Policy-TD-Control-Q-Learning.pdf) | §6.5 | Q-learning | 07 |
| 11 | [On-Policy Prediction: Value-Function Approximation](11_On-Policy-Prediction_Value-Function-Approximation.pdf) | Ch 9 | gradient MC, semi-gradient TD, the VE objective | 08 |
| 12 | **On-Policy Control with Approximation — in progress** | Ch 10 | episodic semi-gradient Sarsa, n-step Sarsa, Mountain Car | 09 (reserved) |

## Policy gradients and actor-critic

| # | Derivation | S&B | Topic | Previous # |
|---|---|---|---|---|
| 13 | [The Policy Gradient Theorem](13_Policy-Gradient-Theorem.pdf) | §13.2 | discounted objective, visitation measure, exact gradient | 10 |
| 14 | [Average-Reward Policy Gradient Theorem](14_Average-Reward-Policy-Gradient-Theorem.pdf) | §13.6 | continuing-task objective, stationary distribution | 11 |
| 15 | [Policy Gradient Theorem: Episodic Trajectory Route](15_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf) | §13.2 | trajectory-likelihood derivation | 12 |
| 16 | [Policy Gradient Preliminaries](16_Policy-Gradient-Preliminaries.pdf) | — | objectives, stochastic gradient estimators, shared estimator decomposition | 13 |
| 17 | [REINFORCE](17_REINFORCE.pdf) | §13.3 | Monte Carlo policy gradient | 14 |
| 18 | [Actor-Critic](18_Actor-Critic.pdf) | §13.5 | GPI view; from the exact gradient to QAC, one move at a time | 15 |
| 19 | [Actor-Critic with a Baseline](19_Actor-Critic-with-a-Baseline.pdf) | §13.4–13.5 | baseline identity, advantage, TD error as one-sample advantage | 16 |
| 20 | [GAE Actor-Critic](20_GAE_Actor-Critic.pdf) | — | n-step advantages and their λ-mixture (Schulman et al., 2016) | 17 |

Report 16 builds on the theorem reports 13–15 and introduces the common estimator
notation for the algorithms that follow.

## Continue in Deep RL

The [Deep RL derivation index](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning/blob/master/mathematical_derivations/README.md)
continues with 21 Natural Policy Gradient, 22 TRPO, 23 PPO (reserved), 24 DQN,
25 Double DQN, 26 Deterministic Policy Gradient, and 27 DDPG.

> Theory companions to the code: prediction lives in
> [`prediction_approx/`](../src/rl/prediction_approx/), control in
> [`control_approx/`](../src/rl/control_approx/), and the shared representation
> machinery in [`features/`](../src/rl/features/) and
> [`approximators/`](../src/rl/approximators/).
