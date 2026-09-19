"""Reinforcement-learning environment for Game of the Amazons.

This module is deliberately independent of PyTorch.  It adapts the supplied
``game_backend.Game`` into a three-stage Markov decision process:

    SELECT_PIECE -> SELECT_MOVE -> SELECT_ARROW -> opponent's turn

The split keeps the policy space at N*N actions for every decision.  A full
10x10 move would otherwise require a very large (origin, destination, arrow)
action head.
"""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from rule.game_backend import Game


class Phase(IntEnum):
    SELECT_PIECE = 0
    SELECT_MOVE = 1
    SELECT_ARROW = 2


class AmazonsEnv:
    """Three-stage environment backed by the user's original rule engine."""

    INPUT_CHANNELS = 8

    def __init__(self, board_size: int = 4):
        if board_size not in (3, 4, 5, 10):
            raise ValueError(
                "The supplied backend defines starting positions only for "
                "board sizes 3, 4, 5 and 10."
            )
        self.game = Game(testgame=True, board_width=board_size)
        self.board_size = board_size
        self.phase = Phase.SELECT_PIECE
        self.selected_piece: Optional[int] = None
        self.selected_move: Optional[int] = None
        self._plays_cache: Optional[Dict] = None

    @property
    def action_size(self) -> int:
        return self.board_size**2

    @property
    def current_player(self) -> int:
        return int(self.game.turn)

    @property
    def completed_turns(self) -> int:
        return self.game.turncount - 1

    @property
    def winner(self) -> Optional[int]:
        """Winner at a terminal state, otherwise ``None``."""
        return 3 - self.current_player if self.is_terminal() else None

    def clone(self) -> "AmazonsEnv":
        return copy.deepcopy(self)

    def _available_plays(self) -> Dict:
        # Enumerating every complete play is expensive on 10x10.  The board
        # does not change during piece and destination selection, so cache it
        # until the arrow completes the turn.
        if self._plays_cache is None:
            self._plays_cache = self.game.find_available_plays()
        return self._plays_cache

    def initial_complete_move_count(self) -> int:
        """Number of legal (piece, move, arrow) triples in this position."""
        return sum(
            len(arrows)
            for moves in self._available_plays().values()
            for arrows in moves.values()
        )

    def legal_actions(self) -> List[int]:
        """Legal board-cell actions for the current internal phase."""
        plays = self._available_plays()

        if self.phase == Phase.SELECT_PIECE:
            return list(plays)

        if self.phase == Phase.SELECT_MOVE:
            moves = plays.get(self.selected_piece, {})
            return list(moves)

        if self.phase == Phase.SELECT_ARROW:
            moves = plays.get(self.selected_piece, {})
            return list(moves.get(self.selected_move, []))

        raise RuntimeError(f"Unknown phase {self.phase!r}")

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def is_terminal(self) -> bool:
        # An intermediate state is always reached through a legal choice and
        # therefore has at least one continuation.  Only SELECT_PIECE can be a
        # terminal state in the staged representation.
        return self.phase == Phase.SELECT_PIECE and not self.legal_actions()

    def child_player_after(self, action: int) -> int:
        """Player-to-move after ``action`` without mutating the environment."""
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}")
        return 3 - self.current_player if self.phase == Phase.SELECT_ARROW else self.current_player

    def step(self, action: int) -> bool:
        """Apply one staged action.

        Returns ``True`` only when an arrow was placed and the full turn ended.
        """
        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action {action}; player={self.current_player}, "
                f"phase={self.phase.name}"
            )

        if self.phase == Phase.SELECT_PIECE:
            self.selected_piece = action
            self.phase = Phase.SELECT_MOVE
            return False

        if self.phase == Phase.SELECT_MOVE:
            self.selected_move = action
            self.phase = Phase.SELECT_ARROW
            return False

        assert self.selected_piece is not None
        assert self.selected_move is not None
        self.game.make_play(self.selected_piece, self.selected_move, action)
        self.phase = Phase.SELECT_PIECE
        self.selected_piece = None
        self.selected_move = None
        self._plays_cache = None
        return True

    def encode(self) -> np.ndarray:
        """Encode state as eight N x N float32 planes.

        Planes 0-2: current player, opponent, arrows.
        Planes 3-4: selected piece and selected destination.
        Planes 5-7: one-hot phase planes.
        """
        n = self.board_size
        encoded = np.zeros((self.INPUT_CHANNELS, n, n), dtype=np.float32)
        own_symbol = str(self.current_player)
        opponent_symbol = "2" if own_symbol == "1" else "1"

        for index, tile in enumerate(self.game.board.game_tiles):
            row, column = divmod(index, n)
            symbol = tile.to_string()
            if symbol == own_symbol:
                encoded[0, row, column] = 1.0
            elif symbol == opponent_symbol:
                encoded[1, row, column] = 1.0
            elif symbol == "0":
                encoded[2, row, column] = 1.0

        if self.selected_piece is not None:
            row, column = divmod(self.selected_piece, n)
            encoded[3, row, column] = 1.0
        if self.selected_move is not None:
            row, column = divmod(self.selected_move, n)
            encoded[4, row, column] = 1.0

        encoded[5 + int(self.phase), :, :] = 1.0
        return encoded


def run_environment_smoke_tests() -> None:
    """Rule/adapter checks that do not require PyTorch."""
    expected_opening_moves = {3: 29, 4: 104, 5: 249, 10: 2176}

    for board_size, expected in expected_opening_moves.items():
        env = AmazonsEnv(board_size)
        assert env.encode().shape == (8, board_size, board_size)
        assert env.initial_complete_move_count() == expected

        player = env.current_player
        assert not env.step(env.legal_actions()[0])
        assert env.phase == Phase.SELECT_MOVE and env.current_player == player
        assert not env.step(env.legal_actions()[0])
        assert env.phase == Phase.SELECT_ARROW and env.current_player == player
        assert env.step(env.legal_actions()[0])
        assert env.phase == Phase.SELECT_PIECE and env.current_player != player

        print(
            f"{board_size}x{board_size}: passed; "
            f"opening complete moves={expected}"
        )


if __name__ == "__main__":
    run_environment_smoke_tests()
