"""Blackjack (S&B 2020, §5.1, p.93). Infinite deck.

Card-handling helpers are ported almost verbatim from
gymnasium/envs/toy_text/blackjack.py so behaviour can be cross-checked.
"""

from __future__ import annotations
import numpy as np
from rl.envs.base import TabularEnv


class Blackjack(TabularEnv):
    """Tabular Blackjack. State = (player_sum, dealer_show, usable_ace).

    State space:
        player_sum  in [12, 21]    (10 values)
        dealer_show in  [1, 10]    (10 values)
        usable_ace  in  {0, 1}     ( 2 values)
        -> 200 decision states + 1 terminal state = 201 total.

    Rules:
        1. Sums below 12 are not modelled: the player auto-hits at reset.
        2. Cards drawn with replacement (infinite deck):
           P(ace) = P(2..9) = 1/13,  P(10/J/Q/K) = 4/13.
        3. Dealer hits until sum >= 17.
        4. Reward: +1 win, -1 loss, 0 draw. No natural-blackjack bonus.
    """

    # Action labels
    STICK = 0
    HIT = 1

    # Sizes
    n_actions = 2
    n_states = 201
    TERMINAL = 200

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def reset(self, rng: np.random.Generator) -> int:
        hand = [self._draw_card(rng), self._draw_card(rng)]
        while self._sum_hand(hand) < 12:
            hand.append(self._draw_card(rng))
        dealer_show = self._draw_card(rng)
        return self._encode(self._sum_hand(hand), dealer_show, self._usable_ace(hand))

    def step(
        self,
        s: int,
        a: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool]:
        player_sum, dealer_show, usable_ace = self._decode(s)

        if a == self.HIT:
            hand = [1, player_sum - 11] if usable_ace else [player_sum]
            hand.append(self._draw_card(rng))
            if self._is_bust(hand):
                return self.TERMINAL, -1.0, True
            return self._encode(
                self._sum_hand(hand),
                dealer_show,
                self._usable_ace(hand),
            ), 0.0, False

        # STICK: dealer plays out, compare scores.
        dealer_score = self._play_dealer(dealer_show, rng)
        return self.TERMINAL, self._cmp(player_sum, dealer_score), True

    def terminal_states(self) -> set[int]:
        return {self.TERMINAL}

    # --- card mechanics (ported from gymnasium/envs/toy_text/blackjack.py) ---

    @staticmethod
    def _draw_card(rng: np.random.Generator) -> int:
        """Draw one card. 1 = ace; 10 covers 10/J/Q/K."""
        return int(min(rng.integers(1, 14), 10))

    @staticmethod
    def _usable_ace(hand: list[int]) -> bool:
        """True if the hand has an ace that can count as 11 without busting."""
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def _sum_hand(hand: list[int]) -> int:
        """Hand total. Ace counts as 11 when usable, else 1."""
        return sum(hand) + 10 if Blackjack._usable_ace(hand) else sum(hand)

    @staticmethod
    def _is_bust(hand: list[int]) -> bool:
        return Blackjack._sum_hand(hand) > 21

    @staticmethod
    def _score(hand: list[int]) -> int:
        """0 if bust, else `_sum_hand`. Used for player-vs-dealer comparison."""
        return 0 if Blackjack._is_bust(hand) else Blackjack._sum_hand(hand)

    @staticmethod
    def _cmp(a: int, b: int) -> float:
        """+1 if a>b, -1 if a<b, 0 if a==b."""
        return float(a > b) - float(a < b)

    @staticmethod
    def _play_dealer(show: int, rng: np.random.Generator) -> int:
        """Play the dealer to completion. Dealer hits until sum >= 17."""
        hand = [show, Blackjack._draw_card(rng)]
        while Blackjack._sum_hand(hand) < 17:
            hand.append(Blackjack._draw_card(rng))
        return Blackjack._score(hand)

    @staticmethod
    def _encode(player_sum: int, dealer_show: int, usable_ace: bool) -> int:
        """Pack (player_sum, dealer_show, usable_ace) into a unique index in [0, 200).

        Mixed-base positional encoding with strides 20 / 2 / 1:
            (player_sum - 12) * 20  +  (dealer_show - 1) * 2  +  int(usable_ace)
        """
        return (player_sum - 12) * 20 + (dealer_show - 1) * 2 + int(usable_ace)

    @staticmethod
    def _decode(s: int) -> tuple[int, int, bool]:
        """Inverse of `_encode`. Returns (player_sum, dealer_show, usable_ace)."""
        player_sum = 12 + s // 20
        rest = s % 20
        dealer_show = 1 + rest // 2
        usable_ace = bool(rest % 2)
        return player_sum, dealer_show, usable_ace
