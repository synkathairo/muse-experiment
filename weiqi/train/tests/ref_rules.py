"""Deliberately naive, clarity-first 9x9 Go rules engine.

Purpose: an INDEPENDENT oracle for differential fuzzing of gotrain.rules
(Python training env) and weiqi/engine (Rust). Every algorithm here is chosen
to differ from both:

- board: flat list of 81 ints (vs list-of-lists in rules.py, [Option<Color>; 81]
  in Rust),
- legality: copy-the-board-and-simulate (vs precomputed _would_capture in
  rules.py, tentative-place-with-rollback in Rust),
- ko: POSITIONAL SUPERKO — a move is banned iff the resulting board equals
  ANY previous board position in this game (vs the "lone stone with one
  liberty" heuristic in both other engines). The ko *point* is found by brute
  force: the empty point whose play would recreate the previous position. If
  the heuristic and the superko test ever disagree on a single ko, that's a
  finding.
- scoring: flood fill with explicit adjacency sets (structurally different loop
  from selfplay.score).

This file is intentionally slow and obvious. Correctness over speed.
"""
from __future__ import annotations

from collections.abc import Iterator

N: int = 9
EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2
KOMI: float = 7.5


def _neighbors(i: int) -> Iterator[int]:
    r, c = divmod(i, N)
    if r > 0:
        yield i - N
    if r < N - 1:
        yield i + N
    if c > 0:
        yield i - 1
    if c < N - 1:
        yield i + 1


class RefBoard:
    """Naive 9x9 board. move: int 0..80 or None (pass). color: BLACK/WHITE."""

    def __init__(self) -> None:
        self.b: list[int] = [EMPTY] * (N * N)
        self.ko: int | None = None   # int point or None, for the side to move
        self.prev: list[tuple[int, ...]] = []  # previous boards (positional ko)
        self.caps: dict[int, int] = {BLACK: 0, WHITE: 0}
        self.passes: int = 0

    # -- core simulation -------------------------------------------------
    def _group(self, b: list[int], i: int) -> tuple[set[int], set[int]]:
        color = b[i]
        stones: set[int] = set()
        libs: set[int] = set()
        stack: list[int] = [i]
        seen: set[int] = {i}
        while stack:
            s = stack.pop()
            stones.add(s)
            for nb in _neighbors(s):
                if b[nb] == EMPTY:
                    libs.add(nb)
                elif b[nb] == color and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return stones, libs

    def _simulate(self, i: int, color: int) -> tuple[list[int], int] | None:
        """Copy board, play, remove dead foe groups. Returns (board, n_captured)
        or None for suicide."""
        b = self.b[:]
        b[i] = color
        foe = WHITE if color == BLACK else BLACK
        captured: int = 0
        seen: set[int] = set()
        for nb in _neighbors(i):
            if b[nb] == foe and nb not in seen:
                stones, libs = self._group(b, nb)
                seen |= stones
                if not libs:
                    captured += len(stones)
                    for s in stones:
                        b[s] = EMPTY
        if not self._group(b, i)[1]:
            return None  # suicide: own group has no liberties, captured nothing
        return b, captured

    # -- public API ------------------------------------------------------
    def legal_moves(self, color: int) -> list[bool]:
        """[bool]*82: legal moves for color. Index 81 = pass (always legal)."""
        mask = [False] * 82
        mask[81] = True
        for i in range(N * N):
            if self.b[i] != EMPTY or i == self.ko:
                continue
            sim = self._simulate(i, color)
            if sim is None:
                continue
            if tuple(sim[0]) in self.prev:
                continue  # positional superko: would recreate an old position
            mask[i] = True
        return mask

    def play(self, move: int | None, color: int) -> int:
        """Apply move (int or None). Returns stones captured. Raises on illegal."""
        if move is None:
            self.prev.append(tuple(self.b))
            self.ko = None
            self.passes += 1
            return 0
        i = move
        if not (0 <= i < N * N):
            raise ValueError("out of bounds")
        if self.b[i] != EMPTY:
            raise ValueError("occupied")
        if i == self.ko:
            raise ValueError("ko")
        sim = self._simulate(i, color)
        if sim is None:
            raise ValueError("suicide")
        nb, captured = sim
        if tuple(nb) in self.prev:
            raise ValueError("ko (positional superko)")
        self.prev.append(tuple(self.b))
        before = self.prev[-1]
        self.b = nb
        self.caps[color] += captured
        self.passes = 0
        # ko point for the NEXT player, by brute-force positional check: the
        # empty point whose play would recreate the position before this move.
        self.ko = None
        foe = WHITE if color == BLACK else BLACK
        for q in range(N * N):
            if self.b[q] != EMPTY:
                continue
            s2 = self._simulate(q, foe)
            if s2 is not None and tuple(s2[0]) == before:
                self.ko = q
                break
        return captured

    def score(self) -> tuple[float, float]:
        """Tromp-Taylor area score: (black, white + komi)."""
        black = sum(1 for v in self.b if v == BLACK)
        white = sum(1 for v in self.b if v == WHITE)
        seen = [False] * (N * N)
        bt = wt = 0
        for i in range(N * N):
            if self.b[i] != EMPTY or seen[i]:
                continue
            stack, seen[i], region, adj = [i], True, 0, set()
            while stack:
                s = stack.pop()
                region += 1
                for nb in _neighbors(s):
                    if self.b[nb] == EMPTY and not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
                    elif self.b[nb] != EMPTY:
                        adj.add(self.b[nb])
            if adj == {BLACK}:
                bt += region
            elif adj == {WHITE}:
                wt += region
        return black + bt, white + wt + KOMI
