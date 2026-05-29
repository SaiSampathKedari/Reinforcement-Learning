"""Windy Gridworld (S&B 2020, Example 6.5, p.130).

A 7x10 gridworld with an upward crosswind whose strength varies by column.
Undiscounted episodic task: reward -1 per step until the goal is reached.

Variants (constructor flags):
    king_moves      -- 8 diagonal-inclusive actions (Exercise 6.9).
    no_op           -- add a 9th "stay" action (Exercise 6.9, king moves + stop).
    stochastic_wind -- in windy columns, the shift is {w-1, w, w+1} with equal
                       prob; calm columns (w=0) stay deterministic (Exercise 6.10).

Coordinate convention: state is a cell (row, col) with row 0 at the TOP, so the
wind ("upward") DECREASES the row index. Cells are clipped to the grid edges.
"""

from __future__ import annotations
import numpy as np
from rl.envs.base import TabularEnv


class WindyGridworld(TabularEnv):
    """Tabular Windy Gridworld. State = grid cell encoded as an int in [0, 70).

    State space:
        row in [0, 6]  (7 values)
        col in [0, 9]  (10 values)
        -> 70 states. The goal cell is terminal (no separate absorbing index).

    Reward:
        -1 on every transition until the goal cell is entered.
    """

    # --- grid geometry (facts from Example 6.5) ---
    N_ROWS = 7
    N_COLS = 10
    n_states = N_ROWS * N_COLS  # 70 states total

    # Wind strength per column, in cells pushed upward.
    WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)

    # Start and goal cells as (row, col).
    START = (3, 0)
    GOAL = (3, 7)

    # --- action encodings as (drow, dcol) deltas; row 0 is the top ---
    # Standard 4 moves.
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    _MOVE_DELTAS = (
        (-1, 0),  # UP
        (+1, 0),  # DOWN
        (0, +1),  # RIGHT
        (0, -1),  # LEFT
    )
    # Four diagonals appended for king moves (Exercise 6.9).
    _DIAG_DELTAS = (
        (-1, +1),  # UP-RIGHT
        (+1, +1),  # DOWN-RIGHT
        (+1, -1),  # DOWN-LEFT
        (-1, -1),  # UP-LEFT
    )
    _NO_OP_DELTA = (0, 0)  # optional "stay" action (the 9th, with king moves)

    # set per-instance in __init__ (depends on the chosen action set)
    n_actions: int

    def __init__(
        self,
        king_moves: bool = False,
        no_op: bool = False,
        stochastic_wind: bool = False,
        gamma: float = 1.0,
    ):
        # Assemble the action set: 4 moves, + diagonals, + no-op.
        deltas = list(self._MOVE_DELTAS)
        if king_moves:
            deltas += list(self._DIAG_DELTAS)
        if no_op:
            deltas.append(self._NO_OP_DELTA)
        self._deltas = tuple(deltas)
        self.n_actions = len(self._deltas)

        self.stochastic_wind = stochastic_wind
        self.gamma = gamma
        self._goal = self._encode(*self.GOAL)

    # --- TabularEnv interface ---

    def reset(self, rng: np.random.Generator) -> int:
        """Return the start state. (rng unused: the start is fixed.)"""
        return self._encode(*self.START)

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        """Take one step. Returns (next_state, reward, terminated).

        Reward is -1 on every step, including the one that reaches the goal,
        so the return equals the (negated) number of steps taken.
        """
        row, col = self._decode(s)
        wind = self._sample_wind(col, rng)  # wind comes from the CURRENT column (S&B convention)
        s_next = self._step_from_cell(row, col, a, wind)
        return s_next, -1.0, s_next == self._goal

    def transitions(
        self,
        s: int,
        a: int,
    ) -> list[tuple[float, int, float, bool]]:
        """Enumerate p(s', r | s, a) as (prob, s', r, terminated).

        Deterministic wind (or a calm column) -> one outcome with prob 1.0.
        stochastic_wind in a windy column      -> three outcomes (wind w-1, w,
                             w+1), each prob 1/3, with duplicate next states merged.
        """
        row, col = self._decode(s)
        base = self.WIND[col]
        if self.stochastic_wind and base > 0:
            winds = (base - 1, base, base + 1)
        else:
            winds = (base,)
        prob = 1.0 / len(winds)

        merged: dict[int, float] = {}
        for wind in winds:
            s_next = self._step_from_cell(row, col, a, wind)
            merged[s_next] = merged.get(s_next, 0.0) + prob

        return [
            (p, s_next, -1.0, s_next == self._goal)
            for s_next, p in merged.items()
        ]

    def terminal_states(self) -> set[int]:
        """The goal cell is the only terminal state."""
        return {self._goal}

    # --- dynamics shared by step() and transitions() ---

    def _sample_wind(self, col: int, rng: np.random.Generator) -> int:
        """Wind shift for `col`. Adds +/-1 noise when stochastic_wind is on."""
        wind = self.WIND[col]
        if self.stochastic_wind and wind > 0:
            wind += int(rng.integers(-1, 2))  # -1, 0, or +1
        return wind

    def _step_from_cell(
        self,
        row: int,
        col: int,
        a: int,
        wind: int,
    ) -> int:
        """Apply action `a` then a fixed `wind` shift; clip; return encoded s'.

        Wind pushes upward, which decreases the row index (row 0 is the top).
        The action delta and the wind are summed, then clipped once at the edge.
        """
        drow, dcol = self._deltas[a]
        new_row = self._clip(row + drow - wind, 0, self.N_ROWS - 1)
        new_col = self._clip(col + dcol, 0, self.N_COLS - 1)
        return self._encode(new_row, new_col)

    # --- index <-> cell helpers ---

    def _encode(self, row: int, col: int) -> int:
        """Pack (row, col) into a unique index: row * N_COLS + col."""
        return row * self.N_COLS + col

    def _decode(self, s: int) -> tuple[int, int]:
        """Inverse of `_encode`. Returns (row, col)."""
        row = s // self.N_COLS
        col = s % self.N_COLS
        return row, col

    @staticmethod
    def _clip(value: int, lo: int, hi: int) -> int:
        """Clamp `value` into the inclusive range [lo, hi]."""
        return max(lo, min(value, hi))
