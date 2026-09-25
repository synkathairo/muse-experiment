"""Minimal 9x9 Go rules for SGF replay (dataset building only).

Tromp-Taylor-style legality: a move is legal if the point is empty, not ko-banned,
not suicide, and does not repeat any earlier board (positional superko).
The one-move ko point is kept as a fast path; passes are always legal.
This is intentionally lenient: for behavioral cloning it only gates which
(position -> move) pairs we keep, and Japanese-rules (KGS) vs Chinese-rules
legality differences are negligible here.

Performance note: this module is the hot loop of self-play training
(~87% of rollout wall-clock in profiling: legal_mask calls is_legal per
empty point, each doing flood fills). The internals below are optimized
accordingly — int-encoded points, precomputed neighbor tables, bytearray
`seen` sets, bytes-based board hashes — while the public API and exact
legality semantics are unchanged (pinned by tests/test_rules_fuzz.py's
differential fuzz against the naive reference).
"""

import itertools

EMPTY, BLACK, WHITE = 0, 1, 2


def opponent(color):
    return WHITE if color == BLACK else BLACK


# -- precomputed per-size tables (module-level cache) --------------------------
# _NB[size][p] = tuple of int-encoded neighbor points of p = r*size+c.
# _RC[size][p] = (r, c) for int-encoded p.
_NB = {}
_RC = {}


def _tables(size):
    if size not in _NB:
        nb, rc = [], []
        for r in range(size):
            for c in range(size):
                rc.append((r, c))
                nbs = []
                if r > 0:
                    nbs.append((r - 1) * size + c)
                if r < size - 1:
                    nbs.append((r + 1) * size + c)
                if c > 0:
                    nbs.append(r * size + c - 1)
                if c < size - 1:
                    nbs.append(r * size + c + 1)
                nb.append(tuple(nbs))
        _NB[size] = tuple(nb)
        _RC[size] = tuple(rc)
    return _NB[size], _RC[size]


class Board:
    def __init__(self, size=9):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.to_play = BLACK
        self.ko = None          # (r, c) or None: simple-ko-banned point
        self.last_move = None   # (r, c) or None (None also means pass / no move yet)
        self._nb, self._rc = _tables(size)
        self.history = {self._tuple()}  # board hashes seen (positional superko)

    def _tuple(self):
        # bytes, not tuple-of-ints: built at C speed by itertools.chain, and
        # hashing 81 bytes is far cheaper than hashing 81 Python ints.
        return bytes(itertools.chain.from_iterable(self.grid))

    # -- group / liberty helpers -------------------------------------------------
    def _group_int(self, p0):
        """Flood fill from int-encoded point p0.

        Returns (stones, liberties): stones as a list of int points, liberties
        as a set of int points. Internal fast path; callers convert at the
        API boundary only where needed.
        """
        grid = self.grid
        rc = self._rc
        nb = self._nb
        r0, c0 = rc[p0]
        color = grid[r0][c0]
        n = self.size * self.size
        seen = bytearray(n)
        seen[p0] = 1
        stones = [p0]
        libs = set()
        stack = [p0]
        while stack:
            p = stack.pop()
            for q in nb[p]:
                qr, qc = rc[q]
                v = grid[qr][qc]
                if v == EMPTY:
                    libs.add(q)
                elif v == color and not seen[q]:
                    seen[q] = 1
                    stones.append(q)
                    stack.append(q)
        return stones, libs

    # -- legality / play ----------------------------------------------------------
    def _would_capture(self, p, color):
        """Int-encoded points captured by playing p for color.

        Returns a list of int points (deduplicated: each captured group is
        flood-filled once). Idempotent application makes dedup behavior-safe:
        the only length-sensitive use is the `len(captured) == 1` ko check,
        and a 1-stone capture touches exactly one neighbor point, so it can
        never have appeared twice.
        """
        opp = opponent(color)
        grid = self.grid
        rc = self._rc
        captured = []
        seen_groups = set()
        for q in self._nb[p]:
            qr, qc = rc[q]
            if grid[qr][qc] == opp and q not in seen_groups:
                stones, libs = self._group_int(q)
                for s in stones:
                    seen_groups.add(s)
                if len(libs) == 1 and p in libs:
                    captured.extend(stones)
        return captured

    def is_legal(self, r, c, color):
        if self.grid[r][c] != EMPTY:
            return False
        if self.ko is not None and (r, c) == self.ko:
            return False
        p = r * self.size + c
        captured = self._would_capture(p, color)
        # Tentatively apply, so suicide and superko are tested on the result.
        grid = self.grid
        rc = self._rc
        for s in captured:
            sr, sc = rc[s]
            grid[sr][sc] = EMPTY
        grid[r][c] = color
        if captured:
            legal = True
        else:
            _, libs = self._group_int(p)
            legal = bool(libs)
        if not legal:
            # revert the tentative move before returning
            grid[r][c] = EMPTY
            opp = opponent(color)
            for s in captured:
                sr, sc = rc[s]
                grid[sr][sc] = opp
            return False
        board_t = self._tuple()
        # Revert the tentative move.
        grid[r][c] = EMPTY
        opp = opponent(color)
        for s in captured:
            sr, sc = rc[s]
            grid[sr][sc] = opp
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
        p = r * self.size + c
        captured = self._would_capture(p, color)
        grid = self.grid
        rc = self._rc
        for s in captured:
            sr, sc = rc[s]
            grid[sr][sc] = EMPTY
        grid[r][c] = color
        self.history.add(self._tuple())
        # simple ko: exactly one stone captured, and the played stone is now a
        # lone single stone with exactly one liberty (the vacated point).
        stones, libs = self._group_int(p)
        if len(captured) == 1 and len(stones) == 1 and len(libs) == 1:
            self.ko = rc[captured[0]]
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
