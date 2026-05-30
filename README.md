# Reinforcement Learning

Reinforcement learning algorithms with mathematical derivations and reproductions of figures from Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020).

Theoretical foundations live in [`Sequential-Decision-Making`](https://github.com/SaiSampathKedari/Sequential-Decision-Making). Deep-RL methods will live in [`Deep-Reinforcement-Learning`](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning).

![Windy Gridworld — value function and optimal path](notebooks/02_windy_gridworld/windy_gridworld_overview.png)

> **Windy Gridworld** (S&B Example 6.5): the state-value function `V(s)` learned by Sarsa, and the resulting 15-step optimal path from **S** to **G**. The upward wind (per-column strength shown below) bends the path. Interactive notebook: [`notebooks/02_windy_gridworld/`](notebooks/02_windy_gridworld/).

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

## Status

- **Blackjack** environment with Monte Carlo prediction and control (reproduces S&B Figs 5.1–5.2).
- **Windy Gridworld** environment with Sarsa on-policy TD control (reproduces S&B Example 6.5), plus King's-moves / stochastic-wind variants.
- TD control algorithms: Sarsa(0), n-step Sarsa, Sarsa(λ), Q-learning.
- Reports for Monte Carlo control, eligibility traces, and TD control.

## Contact

- Email: sampath@umich.edu
- LinkedIn: [sai-sampath-kedari](https://www.linkedin.com/in/sai-sampath-kedari)
