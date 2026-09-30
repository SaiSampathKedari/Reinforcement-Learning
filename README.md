# Reinforcement Learning

Reinforcement learning algorithms with mathematical derivations and reproductions of figures from Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020).

Broader MDP theory lives in [`Sequential-Decision-Making`](https://github.com/SaiSampathKedari/Sequential-Decision-Making); the Sutton-notation foundations are also included here as derivations 01–03. Deep-RL methods live in [`Deep-Reinforcement-Learning`](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning).

## Setup

```bash
git clone https://github.com/SaiSampathKedari/Reinforcement-Learning
cd Reinforcement-Learning
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or, with uv
uv sync
```

Requires Python ≥ 3.10.

## Layout

- `src/rl/`: algorithms and environments (see the [package map](src/rl/README.md)).
- `notebooks/`: experiments, one folder per environment (see the [notebook index](notebooks/README.md)).
- `mathematical_derivations/`: derivations, proofs, and convergence analysis (see the [derivation index](mathematical_derivations/README.md)).

## Algorithms

| Category | Algorithms | Status |
|---|---|:--:|
| Model-free prediction | MC prediction, TD(0), n-step TD, TD(λ) | ● |
| Model-free control | Sarsa, Q-learning, n-step Sarsa, Sarsa(λ) | ● |
| Dynamic programming | policy evaluation; policy & value iteration | ◐ |
| Model-based & planning | Dyna-Q, prioritized sweeping | ○ |
| Value-function approximation | gradient MC, state aggregation; semi-gradient TD / Sarsa, n-step semi-gradient TD / Sarsa, tile coding | ◐ |
| Policy gradient | REINFORCE, REINFORCE with baseline | ○ |
| Actor-critic | one-step actor-critic, advantage actor-critic (A2C) | ○ |
| Trust-region & proximal methods | natural policy gradient, TRPO, PPO | ○ |

<sub>● implemented &nbsp;·&nbsp; ◐ partial &nbsp;·&nbsp; ○ planned</sub>

## Mathematical derivations

Self-contained PDF write-ups in a shared reading sequence with Deep RL (see the [derivation index](mathematical_derivations/README.md)). This repository covers 01–20, with 12 reserved for on-policy control with approximation; Deep RL shares 13–20 and continues from 21.

- **MDP & dynamic-programming foundations (01–03)** —
  [policy evaluation in Sutton notation](mathematical_derivations/01_Policy-Evaluation-in-Sutton-Notation.pdf),
  [Bellman operators](mathematical_derivations/02_Bellman-Operators.pdf),
  [policy improvement theorem](mathematical_derivations/03_Policy-Improvement-Theorem.pdf).
- **Tabular & approximation** —
  [MC control (ES)](mathematical_derivations/04_Monte-Carlo-Control-with-Exploring-Starts.pdf),
  [ε-soft MC control](mathematical_derivations/05_Monte-Carlo-Control-without-Exploring-Starts.pdf),
  [TD & eligibility traces](mathematical_derivations/06_TD-Prediction-Eligibility-Traces.pdf),
  [Sarsa(0)](mathematical_derivations/07_On-Policy-TD-Control-SARSA-0.pdf),
  [n-step Sarsa](mathematical_derivations/08_On-Policy-TD-Control-n-Step-SARSA.pdf),
  [Sarsa(λ)](mathematical_derivations/09_On-Policy-TD-Control-SARSA-Lambda.pdf),
  [Q-learning](mathematical_derivations/10_Off-Policy-TD-Control-Q-Learning.pdf),
  [value-function approximation](mathematical_derivations/11_On-Policy-Prediction_Value-Function-Approximation.pdf).
- **Policy gradient & actor-critic** —
  [policy gradient theorem](mathematical_derivations/13_Policy-Gradient-Theorem.pdf),
  [average-reward PG](mathematical_derivations/14_Average-Reward-Policy-Gradient-Theorem.pdf),
  [trajectory route](mathematical_derivations/15_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf),
  [PG preliminaries](mathematical_derivations/16_Policy-Gradient-Preliminaries.pdf),
  [REINFORCE](mathematical_derivations/17_REINFORCE.pdf),
  [actor-critic](mathematical_derivations/18_Actor-Critic.pdf),
  [baseline & advantage](mathematical_derivations/19_Actor-Critic-with-a-Baseline.pdf),
  [GAE](mathematical_derivations/20_GAE_Actor-Critic.pdf).

## Environments

| Environment | Source | Status | Notebook |
|---|---|:--:|---|
| Blackjack | S&B §5.1 | ● | [`01_blackjack`](notebooks/01_blackjack/) |
| Windy Gridworld | S&B Example 6.5 | ● | [`02_windy_gridworld`](notebooks/02_windy_gridworld/) |
| Random Walk | S&B Examples 6.2 / 7.1 / 9.1 | ● | [`03_random_walk`](notebooks/03_random_walk/) |
| Cliff Walking | S&B Example 6.6 | ○ | |
| FrozenLake | Gymnasium | ○ | |
| Dyna Maze | S&B Example 8.1 | ○ | |
| Mountain Car | S&B Example 10.1 | ● | [`04_mountain_car`](notebooks/04_mountain_car/) |
| Acrobot | Classic control | ○ | |
| CartPole | Classic control | ○ | |
| Short Corridor | S&B Example 13.1 | ○ | |

<sub>● implemented &nbsp;·&nbsp; ○ planned</sub>

## Related repositories

A sequence from mathematical foundations to deep RL:

- **Foundations** — [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) · [Probability & Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) · [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **RL theory** — [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **This repo** — Reinforcement Learning: S&B algorithms and figure reproductions
- **Next** — [Deep Reinforcement Learning](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning)

## Contact

- Email: sampath@umich.edu
- LinkedIn: [sai-sampath-kedari](https://www.linkedin.com/in/sai-sampath-kedari)

## Gallery

<table>
<tr>
<td align="center" width="50%">
<b>Blackjack</b>: optimal policy π* and value V* (S&amp;B Fig 5.2)<br><br>
<img src="notebooks/01_blackjack/blackjack_overview.png" alt="Blackjack optimal policy and value function" height="300">
</td>
<td align="center" width="50%">
<b>Windy Gridworld</b>: value V(s) and 15-step optimal path (S&amp;B Ex. 6.5)<br><br>
<img src="notebooks/02_windy_gridworld/windy_gridworld_overview.png" alt="Windy Gridworld value function and optimal path" height="300">
</td>
</tr>
<tr>
<td align="center" width="50%">
<b>Random Walk</b>: gradient MC with state aggregation vs true v&#960; (S&amp;B Fig 9.1)<br><br>
<img src="notebooks/03_random_walk/figure_9_1.png" alt="Gradient Monte Carlo with state aggregation on the 1000-state random walk" height="290">
</td>
<td align="center" width="50%">
<b>Mountain Car</b>: cost-to-go &minus;max<sub>a</sub>&nbsp;q&#770;(s,a,<b>w</b>) learned by semi-gradient Sarsa (S&amp;B Fig 10.1)<br><br>
<img src="notebooks/04_mountain_car/figure_10_1.png" alt="Mountain Car cost-to-go surfaces learned by semi-gradient Sarsa" height="290">
</td>
</tr>
<tr>
<td align="center" colspan="2">
<b>Mountain Car</b>: the learned greedy policy in action &mdash; reverse up the left slope to build momentum, then accelerate through to the goal<br><br>
<img src="notebooks/04_mountain_car/mountain_car_policy.gif" alt="Mountain Car learned greedy policy animation" width="86%">
</td>
</tr>
</table>
