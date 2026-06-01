"""Random Walk (S&B Examples 6.2, 7.1, 9.1).

A linear chain of `n_states` non-terminal states with a terminal at each end;
episodes start in the center. The task is prediction: estimate the value
function v_pi of a fixed policy and compare it to the true values.

The environment is a pure deterministic chain. An action names the next state to
move to (action `a` means "go to state `a`"); the move is deterministic and a
reward is given only on entering a terminal. Everything that makes this a
"random walk" (the jump range, the clipping at the edges, any bias) lives in the
policy, a distribution over next states pi(s' | s). So one environment
represents every walk, selected purely by the policy:

    RandomWalk(n_states=5,    left_reward=0.0)  with a +/-1 uniform policy   -> Example 6.2
    RandomWalk(n_states=19)                      with a +/-1 uniform policy   -> Example 7.1
    RandomWalk(n_states=1000)                    with a +/-100 uniform policy -> Example 9.1

Because the action is the next state, pi(s' | s) is exactly the transition
kernel, so the true values follow from solving v = (I - gamma P_pi)^{-1} r_pi.

State layout (integer indices):
    0              left terminal
    1 .. n_states  the chain (start = center)
    n_states + 1   right terminal
So `n_states` (total, including terminals) and `n_actions` both equal the
constructor's `n_states` + 2 (an action is a target-state index).
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv


class RandomWalk(TabularEnv):
    """Deterministic linear-chain MDP; the walk is defined entirely by the policy.

    See the module docstring for the state layout and the policy-as-walk design.
    """

    def __init__(
        self,
        n_states: int = 1000,
        left_reward: float = -1.0,
        right_reward: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Args:
            n_states: number of non-terminal chain states (e.g. 5, 19, 1000).
            left_reward: reward on terminating at the left end (0.0 for the
                5-state walk; -1.0 for the 19- and 1000-state walks).
            right_reward: reward on terminating at the right end (+1.0).
            gamma: discount factor (1.0; these walks are undiscounted episodic).
        """
        self._chain = n_states                 # number of non-terminal chain states
        self.n_states = n_states + 2           # + left terminal (0) and right terminal (n+1)
        self.n_actions = self.n_states         # action a = "go to state a"

        self.left_reward = left_reward
        self.right_reward = right_reward
        self.gamma = gamma

        self.LEFT_TERMINAL = 0
        self.RIGHT_TERMINAL = self.n_states - 1     # == n_states + 1
        self.START = (n_states + 1) // 2            # center of the chain (500 for n=1000)

    # --- TabularEnv interface ---

    def reset(self, rng: np.random.Generator) -> int:
        """Return the fixed center start state. (rng unused.)"""
        return self.START

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        """Move to the chosen next state `a`. Returns (next_state, reward, terminated).

        The move is deterministic: the next state is exactly `a`. A reward is
        given only on entering a terminal (`left_reward` / `right_reward`), and 0
        otherwise. `s` and `rng` are unused: the next state is the action, and the
        walk's randomness comes from the policy that samples `a`.
        """
        if a == self.LEFT_TERMINAL:
            return self.LEFT_TERMINAL, self.left_reward, True
        elif a == self.RIGHT_TERMINAL:
            return self.RIGHT_TERMINAL, self.right_reward, True
        else:
            return a, 0.0, False

    def transitions(
        self,
        s: int,
        a: int,
    ) -> list[tuple[float, int, float, bool]]:
        """p(s', r | s, a): a single deterministic outcome with probability 1.0.

        Mirrors `step`. Since the action is the next state, a policy's pi(s' | s)
        is itself the transition kernel P_pi, so the true values follow from
        solving v = (I - gamma P_pi)^{-1} r_pi over the chain.
        """
        if a == self.LEFT_TERMINAL:
            return [(1.0, self.LEFT_TERMINAL, self.left_reward, True)]
        elif a == self.RIGHT_TERMINAL:
            return [(1.0, self.RIGHT_TERMINAL, self.right_reward, True)]
        else:
            return [(1.0, a, 0.0, False)]

    def terminal_states(self) -> set[int]:
        """The two end terminals."""
        return {self.LEFT_TERMINAL, self.RIGHT_TERMINAL}
