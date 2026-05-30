# Reinforcement Learning

Reinforcement learning algorithms with mathematical derivations and reproductions of figures from Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020).

Theoretical foundations live in [`Sequential-Decision-Making`](https://github.com/SaiSampathKedari/Sequential-Decision-Making). Deep-RL methods will live in [`Deep-Reinforcement-Learning`](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning).

## Setup

```bash
git clone https://github.com/SaiSampathKedari/Reinforcement-Learning
cd Reinforcement-Learning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.10.

## Layout

- `src/rl/`: algorithms and environments.
- `notebooks/`: experiments, one folder per environment.
- `reports/`: derivations, proofs, and convergence analysis.

## Algorithms

| Topic (Sutton & Barto) | Status |
|---|:--:|
| Monte Carlo methods (Ch. 5) | ● |
| Temporal-difference learning (Ch. 6) | ● |
| n-step bootstrapping (Ch. 7) | ● |
| Eligibility traces (Ch. 12) | ● |
| Planning & model-based learning (Ch. 8) | ○ |
| Value-function approximation (Ch. 9-10) | ○ |
| Policy-gradient & actor-critic (Ch. 13) | ○ |

<sub>● implemented &nbsp;·&nbsp; ○ planned</sub>

## Environments

| Environment | Source | Status | Notebook |
|---|---|:--:|---|
| Blackjack | S&B §5.1 | ● | [`01_blackjack`](notebooks/01_blackjack/) |
| Windy Gridworld | S&B Example 6.5 | ● | [`02_windy_gridworld`](notebooks/02_windy_gridworld/) |
| Cliff Walking | S&B Example 6.6 | ○ | |
| Dyna Maze | S&B Example 8.1 | ○ | |
| Mountain Car | S&B Example 10.1 | ○ | |
| Short Corridor | S&B Example 13.1 | ○ | |
| CartPole | Classic control | ○ | |

<sub>● implemented &nbsp;·&nbsp; ○ planned</sub>

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
</table>
