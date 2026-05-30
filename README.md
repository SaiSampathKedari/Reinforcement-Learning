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

- `src/rl/` — algorithms and environments.
- `notebooks/` — experiments, one folder per environment.
- `reports/` — LaTeX derivations (PDFs).

## Algorithms

Monte Carlo (prediction, exploring-starts and ε-soft control) · TD prediction · n-step TD · TD(λ) · Sarsa(0) · n-step Sarsa · Sarsa(λ) · Q-learning. Mathematical derivations live in [`reports/`](reports/) (LaTeX PDFs).

## Environments

### Blackjack — [`notebooks/01_blackjack/`](notebooks/01_blackjack/)

Monte Carlo prediction and control (S&B §5). Reproduces Figs 5.1–5.2.

![Blackjack — optimal policy and value function](notebooks/01_blackjack/blackjack_overview.png)

### Windy Gridworld — [`notebooks/02_windy_gridworld/`](notebooks/02_windy_gridworld/)

Sarsa on-policy TD control (S&B Example 6.5), with King's-moves and stochastic-wind variants. Interactive Plotly notebook (hover for values, animated policy formation).

![Windy Gridworld — value function and optimal path](notebooks/02_windy_gridworld/windy_gridworld_overview.png)

## Contact

- Email: sampath@umich.edu
- LinkedIn: [sai-sampath-kedari](https://www.linkedin.com/in/sai-sampath-kedari)
