"""Base class for tabular environments. Every tabular env in this repo
subclasses `TabularEnv` and implements `reset` and `step`."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class TabularEnv(ABC):
    """Finite MDP with integer states and actions.

    Subclasses set `n_states`, `n_actions`, `gamma` in their own `__init__`.
    Envs are stateless: the agent loop owns the current state, the env is
    a pure mapping `(s, a) -> (s', r, terminated)`.
    """

    n_states: int   # total states, including terminal
    n_actions: int  # total actions
    gamma: float    # discount factor

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> int:
        """Return an initial state."""

    @abstractmethod
    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        """Take one step. Returns `(next_state, reward, terminated)`."""

    def transitions(
        self,
        s: int,
        a: int,
    ) -> list[tuple[float, int, float, bool]]:
        """Return `p(s', r | s, a)` as a list of `(prob, s', r, terminated)`.

        Override only in envs with a known model (used by DP).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no transition model; "
            "override `transitions` to use DP on this env."
        )

    def terminal_states(self) -> set[int]:
        """Return the set of terminal state indices. Empty by default."""
        return set()
