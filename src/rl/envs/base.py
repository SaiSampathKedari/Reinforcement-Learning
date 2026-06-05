"""Environment base classes.

`Env` is the minimal interaction interface (`reset` + `step`), agnostic to the
state type -- subclassed by continuous envs (e.g. Mountain Car) whose state is a
real-valued vector. `TabularEnv` adds the enumerable / known-model extras
(`n_states`, `transitions`, `terminal_states`) that DP and tabular methods need;
every tabular env in this repo subclasses it.

Envs are stateless: the agent loop owns the current state, the env is a pure
mapping `(s, a) -> (s', r, terminated)`. Termination is universal -- it is the
`terminated` flag returned by `step`, not the (tabular-only) `terminal_states`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

State = Any  # int for a tabular env, np.ndarray([...]) for a continuous one


class Env(ABC):
    """Minimal agent-interaction interface, agnostic to the state type."""

    n_actions: int  # number of (discrete) actions
    gamma: float    # discount factor

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> State:
        """Return an initial state."""

    @abstractmethod
    def step(
        self,
        s: State,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[State, float, bool]:
        """Take one step. Returns `(next_state, reward, terminated)`."""


class TabularEnv(Env):
    """Finite MDP with integer states and actions.

    Adds the enumerable extras on top of `Env`: a state count, an optional known
    model (`transitions`, for DP), and the set of terminal states. Subclasses set
    `n_states`, `n_actions`, `gamma` in their own `__init__`.
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
