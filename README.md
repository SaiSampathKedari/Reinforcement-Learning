# Reinforcement Learning — Algorithms and Implementations

This repository contains my **theoretical derivations** and **from-scratch implementations** of classical reinforcement learning algorithms, following *Sutton & Barto: Reinforcement Learning: An Introduction* (2nd ed.) and *Mathematical Foundations of Reinforcement Learning* (Shiyu Zhao). The scope is **tabular methods, linear function approximation, and policy gradient methods** — implemented in **pure NumPy** for clarity and pedagogical transparency. Deep-RL methods (neural networks, replay buffers, target networks) live in the sister repository [`Deep-Reinforcement-Learning`](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning).

## Key Topics Covered

1. **Monte Carlo Methods**: On-policy and off-policy Monte Carlo prediction and control, exploring starts, ε-soft policies, GLIE Monte Carlo with convergence guarantees, importance sampling for off-policy learning.
2. **Temporal-Difference Learning**: TD(0), SARSA, Q-Learning, Expected SARSA, Double Q-Learning; on-policy vs off-policy bootstrapping.
3. **n-Step Bootstrapping**: n-step SARSA (on/off-policy), n-step tree backup, Q(σ) unifying framework.
4. **Planning and Model-Based Methods**: Dyna-Q, prioritized sweeping, trajectory sampling.
5. **Function Approximation**: Linear value function approximation with tile coding, semi-gradient SARSA and Q-learning, differential and episodic semi-gradient methods.
6. **Eligibility Traces**: TD(λ), forward/backward view equivalence, true online TD(λ), Sarsa(λ) with linear function approximation.
7. **Policy Gradient Methods**: REINFORCE (with and without baseline), Actor-Critic (one-step, eligibility-trace, continuing), policy gradient theorem.

## Repository Structure
- **`reports/`** — Typeset LaTeX derivations for each algorithm, including convergence proofs and the mathematical motivation behind every implementation choice.
- **`notebooks/`** — Jupyter notebooks organized by chapter (`ch05_monte_carlo/`, `ch06_temporal_difference/`, etc.), pairing theory cells with implementations and reproducing key figures from Sutton & Barto.
- **`src/`** — Modular Python package containing agent classes, environments, feature extractors (tile coding), and diagnostic utilities.
- **`images/`** — Generated plots and animations: learning curves, value-function heatmaps, policy visualizations.
- **`external/sequential-decision-making/`** — Git submodule pointing to the upstream theory repository [`Sequential-Decision-Making`](https://github.com/SaiSampathKedari/Sequential-Decision-Making), which contains the MDP and dynamic programming foundations.

## Purpose

This repository sits **between theory and deep learning** in my study of reinforcement learning. Its role mirrors how [`MonteCarlo-Statistical-Methods`](https://github.com/SaiSampathKedari/MonteCarlo-Statistical-Methods) bridges [`Probability-and-Distribution-Theory`](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) and [`Bayesian-Filtering-and-Smoothing`](https://github.com/SaiSampathKedari/Bayesian-Filtering-and-Smoothing): it implements the algorithms whose theoretical foundations are derived upstream in `Sequential-Decision-Making`, and is itself a prerequisite for the deep RL methods downstream in `Deep-Reinforcement-Learning`.

Every algorithm is implemented from scratch in NumPy and validated by reproducing the corresponding figure(s) from Sutton & Barto. Experiments are seeded and reported with variance across runs — RL is notoriously seed-sensitive, and learning curves without confidence bands are unfalsifiable.

All derivations are typeset in **LaTeX** for clarity.

## Getting Started

```bash
# Clone with submodules
git clone --recursive git@github.com:SaiSampathKedari/Reinforcement-Learning.git
cd Reinforcement-Learning

# Or initialize submodules after cloning
git submodule update --init --recursive

# Install in editable mode
pip install -e .
```

## References
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- S. Zhao, *Mathematical Foundations of Reinforcement Learning*
- C. Szepesvári, *Algorithms for Reinforcement Learning*
- A. Agarwal, N. Jiang, S. M. Kakade, W. Sun, *Reinforcement Learning: Theory and Algorithms*

---

### **About Me**
I am focused on building strong **mathematical and statistical foundations** to support my work in **Robotics, AI, and State Estimation**.

---

### **Contact**
Feel free to connect with me:
- **Email**: sampath@umich.edu
- **LinkedIn**: www.linkedin.com/in/sai-sampath-kedari
