"""Environment wrappers for `TabularEnv` (cf. `gymnasium.wrappers`).

A wrapper *is-a* `TabularEnv`: it satisfies the same interface and slots between
the agent and the underlying env, adding behaviour without touching the agent or
the concrete environment. Wrappers compose -- stack them like layers:

    env = EpisodeStats(TimeLimit(WindyGridworld(), max_steps=2000))

`TabularWrapper` provides the delegation boilerplate; subclasses override only
`reset`/`step`. Unknown attributes forward to the wrapped env, so the layout
attributes of a concrete env (`WIND`, `START`, `_encode`, ...) remain reachable
through the wrapper.
"""

from __future__ import annotations
import numpy as np

from rl.envs.base import TabularEnv


class TabularWrapper(TabularEnv):
    """Base class for `TabularEnv` wrappers (cf. `gymnasium.Wrapper`).

    Forwards the full env interface to the wrapped `env`. Subclasses override
    `reset` / `step` to add behaviour and call through to `self.env`.
    """

    def __init__(self, env: TabularEnv):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma

    def reset(self, rng: np.random.Generator) -> int:
        return self.env.reset(rng)

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        return self.env.step(s, a, rng)

    def transitions(self, s: int, a: int):
        return self.env.transitions(s, a)

    def terminal_states(self) -> set[int]:
        return self.env.terminal_states()

    def __getattr__(self, name: str):
        # Called only when normal lookup fails: forward to the wrapped env.
        # Guard `env` to avoid infinite recursion before __init__ sets it.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


class EpisodeStats(TabularWrapper):
    """Record per-episode return and length (cf. `RecordEpisodeStatistics`).

    Counts `step` calls and accumulates reward between terminations, leaving the
    agent untouched. Results are exposed as growing lists:

        env = EpisodeStats(WindyGridworld())
        Q, pi = sarsa(env, rng, n_episodes=500, alpha=0.5)
        env.episode_lengths   # [steps in ep 1, ep 2, ...]
        env.episode_returns   # [return of ep 1, ep 2, ...]
    """

    def __init__(self, env: TabularEnv):
        super().__init__(env)
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self._len = 0
        self._ret = 0.0

    def reset(self, rng: np.random.Generator) -> int:
        self._len = 0
        self._ret = 0.0
        return self.env.reset(rng)

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        s_next, r, terminated = self.env.step(s, a, rng)
        self._len += 1
        self._ret += r
        if terminated:
            self.episode_lengths.append(self._len)
            self.episode_returns.append(self._ret)
        return s_next, r, terminated


class TimeLimit(TabularWrapper):
    """Force termination after `max_steps` per episode (cf. `gymnasium.TimeLimit`).

    Useful to bound runaway episodes (e.g. a greedy policy with a self-loop).

    Caveat: `TabularEnv.step` has no `truncated` flag, so a time-out is reported
    as `terminated=True`. That conflates truncation with genuine termination, so
    do not use this where the terminated/truncated distinction matters for
    bootstrapping (a value-based agent will treat the cut-off state as terminal).
    """

    def __init__(self, env: TabularEnv, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._t = 0

    def reset(self, rng: np.random.Generator) -> int:
        self._t = 0
        return self.env.reset(rng)

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        s_next, r, terminated = self.env.step(s, a, rng)
        self._t += 1
        if self._t >= self.max_steps:
            terminated = True
        return s_next, r, terminated
