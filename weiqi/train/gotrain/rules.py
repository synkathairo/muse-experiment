"""Minimal 9x9 Go rules for SGF replay (dataset building only).

Tromp-Taylor-style legality: a move is legal if the point is empty, not ko-banned,
not suicide, and does not repeat any earlier board (positional superko).
The one-move ko point is kept as a fast path; passes are always legal.
This is intentionally lenient: for behavioral cloning it only gates which
(position -> move) pairs we keep, and Japanese-rules (KGS) vs Chinese-rules
legality differences are negligible here.
"""

EMPTY, BLACK, WHITE = 0, 1, 2


def opponent(color):
    return WHITE if color == BLACK else BLACK


class Board:
    def __init__(self, size=9):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.to_play = BLACK
        self.ko = None          # (r, c) or None: simple-ko-banned point
        self.last_move = None   # (r, c) or None (None also means pass / no move yet)
        self.history = {self._tuple()}  # board tuples seen (positional superko)

    def _tuple(self):
        return tuple(v for row in self.grid for v in row)

    # -- group / liberty helpers -------------------------------------------------
    def _neighbors(self, r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def _group(self, r, c):
        """Flood fill from (r,c): returns (stones set, liberties set)."""
        color = self.grid[r][c]
        stones, liberties, stack = set(), set(), [(r, c)]
        seen = {(r, c)}
        while stack:
            sr, sc = stack.pop()
            stones.add((sr, sc))
            for nr, nc in self._neighbors(sr, sc):
                v = self.grid[nr][nc]
                if v == EMPTY:
                    liberties.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return stones, liberties

    # -- legality / play ----------------------------------------------------------
    def _would_capture(self, r, c, color):
        opp = opponent(color)
        captured = []
        for nr, nc in self._neighbors(r, c):
            if self.grid[nr][nc] == opp:
                stones, libs = self._group(nr, nc)
                if libs == {(r, c)}:
                    captured.extend(stones)
        return captured

    def is_legal(self, r, c, color):
        if self.grid[r][c] != EMPTY:
            return False
        if self.ko is not None and (r, c) == self.ko:
            return False
        captured = self._would_capture(r, c, color)
        # Tentatively apply, so suicide and superko are tested on the result.
        for cr, cc in captured:
            self.grid[cr][cc] = EMPTY
        self.grid[r][c] = color
        if captured:
            legal = True
        else:
            _, libs = self._group(r, c)
            legal = bool(libs)
        board_t = self._tuple()
        # Revert the tentative move.
        self.grid[r][c] = EMPTY
        for cr, cc in captured:
            self.grid[cr][cc] = opponent(color)
        if not legal:
            return False
        # Positional superko: the resulting board must not repeat any earlier
        # one. (Passes are exempt — they create no new board — and stay legal.)
        return board_t not in self.history

    def play(self, move, color):
        """Apply a move. move = (r, c) or None for pass. Returns True if legal."""
        if move is None:
            self.history.add(self._tuple())  # board unchanged; harmless dup
            self.last_move = None
            self.ko = None
            self.to_play = opponent(color)
            return True
        r, c = move
        if not self.is_legal(r, c, color):
            return False
        captured = self._would_capture(r, c, color)
        for cr, cc in captured:
            self.grid[cr][cc] = EMPTY
        self.grid[r][c] = color
        self.history.add(self._tuple())
        # simple ko: exactly one stone captured, and the played stone is now a
        # lone single stone with exactly one liberty (the vacated point).
        stones, libs = self._group(r, c)
        if len(captured) == 1 and len(stones) == 1 and len(libs) == 1:
            self.ko = captured[0]
        else:
            self.ko = None
        self.last_move = (r, c)
        self.to_play = opponent(color)
        return True

    def stones(self, color):
        """(9,9) bool array of `color`'s stones. Row 0 = top, col 0 = left."""
        import numpy as np

        arr = np.zeros((self.size, self.size), dtype=bool)
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == color:
                    arr[r, c] = True
        return arr
