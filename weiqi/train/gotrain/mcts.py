"""PUCT search over gotrain.rules positions with batched torch inference.

A faithful port of weiqi/engine/src/mcts.rs: same PUCT (c_puct=1.5),
Dirichlet root noise, negamax backups, and sign-of-score-margin terminal
values. The one deliberate difference is evaluation batching: `--batch N`
evaluates up to N leaves per forward pass (virtual loss keeps the batched
selections diverse); `--batch 1` is exactly the sequential algorithm the
Rust binary and the demo toggle run.

Values are always from the side-to-move's perspective at the node they sit
on; backups negate at every ply (negamax).
"""
from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch

from . import rules, selfplay

PASS: int = 81


class SearchConfig:
    def __init__(self, simulations: int = 100, batch: int = 16,
                 c_puct: float = 1.5,
                 dirichlet_eps: float = 0.15, dirichlet_alpha: float = 0.15,
                 temperature: float = 0.0,
                 seed: int = 0x9E3779B97F4A7C15) -> None:
        self.simulations: int = simulations
        self.batch: int = max(1, batch)
        self.c_puct: float = c_puct
        self.dirichlet_eps: float = dirichlet_eps
        self.dirichlet_alpha: float = dirichlet_alpha
        self.temperature: float = temperature
        self.seed: int = seed


def _clone(board: rules.Board) -> rules.Board:
    nb = rules.Board.__new__(rules.Board)
    nb.size = board.size
    nb.grid = [row[:] for row in board.grid]
    nb.to_play = board.to_play
    nb.ko = board.ko
    nb.last_move = board.last_move
    nb.history = set(board.history)
    nb._nb, nb._rc = board._nb, board._rc
    return nb


def _idx_to_move(m: int) -> tuple[int, int] | None:
    return None if m == PASS else divmod(m, 9)


class _Node:
    __slots__ = ("board", "passes", "to_move", "legal", "P",
                 "N", "W", "V", "children", "expanded")

    def __init__(self) -> None:
        # board/legal/P/N/W/V start unset; the search fills them in
        # (board at creation, the arrays at expansion) before any read.
        self.board: rules.Board | None = None
        self.passes: int = 0
        self.to_move: int = rules.BLACK
        self.legal: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.V: np.ndarray | None = None   # virtual loss (in-flight batch selections)
        self.children: dict[int, int] = {}
        self.expanded: bool = False


# Tagged union: the literal first element lets the caller narrow with `kind ==`.
_SelectOutcome = (
    tuple[Literal["leaf"], _Node, list[tuple[int, int]]]
    | tuple[Literal["terminal"], float, list[tuple[int, int]]]
    | tuple[Literal["dup"]]
)


class Searcher:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 cfg: SearchConfig) -> None:
        self.model: torch.nn.Module = model
        self.model.eval()
        self.device: torch.device = device
        self.cfg: SearchConfig = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # -- search -----------------------------------------------------------
    def search(self, board: rules.Board, passes: int) -> int:
        """Return the chosen move index (0..80 point, 81 pass)."""
        if passes >= 2:
            return PASS
        root = _Node()
        root.board = _clone(board)
        root.passes = passes
        root.to_move = board.to_play
        root.legal = selfplay.bot_mask(root.board, root.to_move)
        if int(root.legal.sum()) <= 1:  # only pass is legal
            return PASS
        nodes = [root]
        self._evaluate([root])          # root priors
        self._add_root_noise(root)

        sims = 0
        while sims < self.cfg.simulations:
            leaves: list[tuple[_Node, list[tuple[int, int]]]] = []  # (node, path)
            pending: set[int] = set()
            while len(leaves) < self.cfg.batch and sims < self.cfg.simulations:
                outcome = self._select_leaf(nodes, pending)
                kind = outcome[0]
                if kind == "leaf":
                    # ty does not narrow the _SelectOutcome union through the
                    # `kind` alias; the literal tag makes the cast exact.
                    _, node, path = cast("tuple[Literal['leaf'], _Node, list[tuple[int, int]]]",
                                         outcome)
                    leaves.append((node, path))
                    pending.add(id(node))
                elif kind == "terminal":
                    _, value, path = cast("tuple[Literal['terminal'], float, list[tuple[int, int]]]",
                                          outcome)
                    self._backup(nodes, path, value)
                    sims += 1
                else:  # "dup": batch is full enough
                    break
            if leaves:
                values = self._evaluate([n for n, _ in leaves])
                for (node, path), v in zip(leaves, values, strict=True):
                    self._backup(nodes, path, v)
                    sims += 1
        return self._choose_move(root)

    def _select_leaf(self, nodes: list[_Node],
                     pending: set[int]) -> _SelectOutcome:
        """Descend from the root with PUCT + virtual loss.

        Returns ("leaf", node, path), ("terminal", value, path), or ("dup",).
        """
        node_idx = 0
        path: list[tuple[int, int]] = []
        while True:
            node = nodes[node_idx]
            if node.passes >= 2:
                return ("terminal", self._terminal_value(node), path)
            if not node.expanded:
                if id(node) in pending:
                    return ("dup",)
                return ("leaf", node, path)
            m = self._select(node)
            path.append((node_idx, m))
            cast(np.ndarray, node.V)[m] += 1
            nxt = node.children.get(m)
            if nxt is None:
                nxt = len(nodes)
                child = _Node()
                child.board = _clone(cast(rules.Board, node.board))
                ok = child.board.play(_idx_to_move(m), node.to_move)
                assert ok, f"search played illegal move {m}"
                child.passes = node.passes + 1 if m == PASS else 0
                child.to_move = rules.opponent(node.to_move)
                node.children[m] = nxt
                nodes.append(child)
            node_idx = nxt

    def _select(self, node: _Node) -> int:
        # Only called on expanded nodes: the stat arrays are all set.
        N = cast(np.ndarray, node.N)
        V = cast(np.ndarray, node.V)
        W = cast(np.ndarray, node.W)
        P = cast(np.ndarray, node.P)
        legal = cast(np.ndarray, node.legal)
        visits = N + V
        total = float(visits.sum())
        q = np.zeros(82, dtype=np.float64)
        nz = visits > 0
        q[nz] = W[nz] / visits[nz]
        # child's value is from the opponent's perspective -> negate
        u = (self.cfg.c_puct * P * np.sqrt(total) / (1.0 + visits))
        score = np.where(legal, -q + u, -np.inf)
        return int(np.argmax(score))

    def _evaluate(self, leaves: list[_Node]) -> list[float]:
        """Run one batched forward pass; expand the leaves. Returns values."""
        planes = np.stack([selfplay.observe(cast(rules.Board, n.board), n.to_move)
                           for n in leaves])
        with torch.no_grad():
            logits, values = self.model(torch.from_numpy(planes).to(self.device))
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy().ravel()
        out: list[float] = []
        for node, lg, v in zip(leaves, logits, values, strict=True):
            node.legal = selfplay.bot_mask(cast(rules.Board, node.board),
                                           node.to_move)
            legal = node.legal
            l = lg[legal]
            l = l - l.max()
            e = np.exp(l)
            p = np.zeros(82, dtype=np.float32)
            p[legal] = (e / e.sum()).astype(np.float32)
            node.P = p
            node.N = np.zeros(82, dtype=np.int32)
            node.W = np.zeros(82, dtype=np.float64)
            node.V = np.zeros(82, dtype=np.int32)
            node.children = {}
            node.expanded = True
            out.append(float(np.clip(v, -1.0, 1.0)))
        return out

    def _backup(self, nodes: list[_Node], path: list[tuple[int, int]],
                v: float) -> None:
        for node_idx, m in reversed(path):
            node = nodes[node_idx]
            V = cast(np.ndarray, node.V)
            N = cast(np.ndarray, node.N)
            W = cast(np.ndarray, node.W)
            V[m] -= 1
            N[m] += 1
            W[m] += v
            v = -v

    def _terminal_value(self, node: _Node) -> float:
        b, w = selfplay.score(cast(rules.Board, node.board))
        m = b - w
        v = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
        return v if node.to_move == rules.BLACK else -v

    def _add_root_noise(self, root: _Node) -> None:
        eps = self.cfg.dirichlet_eps
        if eps <= 0:
            return
        legal = cast(np.ndarray, root.legal)
        P = cast(np.ndarray, root.P)
        idx = np.flatnonzero(legal)
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(idx))
        p = P.copy()
        p[idx] = (1.0 - eps) * p[idx] + eps * noise
        root.P = p.astype(np.float32)

    def _choose_move(self, root: _Node) -> int:
        n = cast(np.ndarray, root.N).astype(np.float64)
        if self.cfg.temperature <= 0 or n.sum() == 0:
            return int(np.argmax(n))
        p = n ** (1.0 / self.cfg.temperature)
        p /= p.sum()
        return int(self.rng.choice(82, p=p))
