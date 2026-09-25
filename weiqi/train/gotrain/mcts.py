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

import numpy as np
import torch

from . import rules, selfplay

PASS = 81


class SearchConfig:
    def __init__(self, simulations=100, batch=16, c_puct=1.5,
                 dirichlet_eps=0.15, dirichlet_alpha=0.15,
                 temperature=0.0, seed=0x9E3779B97F4A7C15):
        self.simulations = simulations
        self.batch = max(1, batch)
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        self.seed = seed


def _clone(board):
    nb = rules.Board.__new__(rules.Board)
    nb.size = board.size
    nb.grid = [row[:] for row in board.grid]
    nb.to_play = board.to_play
    nb.ko = board.ko
    nb.last_move = board.last_move
    nb.history = set(board.history)
    return nb


def _idx_to_move(m):
    return None if m == PASS else divmod(m, 9)


class _Node:
    __slots__ = ("board", "passes", "to_move", "legal", "P",
                 "N", "W", "V", "children", "expanded")

    def __init__(self):
        self.board = None
        self.passes = 0
        self.to_move = rules.BLACK
        self.legal = None
        self.P = None
        self.N = None
        self.W = None
        self.V = None          # virtual loss (in-flight batch selections)
        self.children = {}
        self.expanded = False


class Searcher:
    def __init__(self, model, device, cfg):
        self.model = model
        self.model.eval()
        self.device = device
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # -- search -----------------------------------------------------------
    def search(self, board, passes):
        """Return the chosen move index (0..80 point, 81 pass)."""
        if passes >= 2:
            return PASS
        root = _Node()
        root.board = _clone(board)
        root.passes = passes
        root.to_move = board.to_play
        root.legal = selfplay.legal_mask(root.board, root.to_move)
        if int(root.legal.sum()) <= 1:  # only pass is legal
            return PASS
        nodes = [root]
        self._evaluate([root])          # root priors
        self._add_root_noise(root)

        sims = 0
        while sims < self.cfg.simulations:
            leaves = []                # (node, path)
            pending = set()
            while len(leaves) < self.cfg.batch and sims < self.cfg.simulations:
                outcome = self._select_leaf(nodes, pending)
                kind = outcome[0]
                if kind == "leaf":
                    _, node, path = outcome
                    leaves.append((node, path))
                    pending.add(id(node))
                elif kind == "terminal":
                    _, value, path = outcome
                    self._backup(nodes, path, value)
                    sims += 1
                else:  # "dup": batch is full enough
                    break
            if leaves:
                values = self._evaluate([n for n, _ in leaves])
                for (node, path), v in zip(leaves, values):
                    self._backup(nodes, path, v)
                    sims += 1
        return self._choose_move(root)

    def _select_leaf(self, nodes, pending):
        """Descend from the root with PUCT + virtual loss.

        Returns ("leaf", node, path), ("terminal", value, path), or ("dup",).
        """
        node_idx, path = 0, []
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
            node.V[m] += 1
            nxt = node.children.get(m)
            if nxt is None:
                nxt = len(nodes)
                child = _Node()
                child.board = _clone(node.board)
                ok = child.board.play(_idx_to_move(m), node.to_move)
                assert ok, f"search played illegal move {m}"
                child.passes = node.passes + 1 if m == PASS else 0
                child.to_move = rules.opponent(node.to_move)
                node.children[m] = nxt
                nodes.append(child)
            node_idx = nxt

    def _select(self, node):
        visits = node.N + node.V
        total = float(visits.sum())
        q = np.zeros(82, dtype=np.float64)
        nz = visits > 0
        q[nz] = node.W[nz] / visits[nz]
        # child's value is from the opponent's perspective -> negate
        u = (self.cfg.c_puct * node.P * np.sqrt(total) / (1.0 + visits))
        score = np.where(node.legal, -q + u, -np.inf)
        return int(np.argmax(score))

    def _evaluate(self, leaves):
        """Run one batched forward pass; expand the leaves. Returns values."""
        planes = np.stack([selfplay.observe(n.board, n.to_move) for n in leaves])
        with torch.no_grad():
            logits, values = self.model(torch.from_numpy(planes).to(self.device))
        logits = logits.float().cpu().numpy()
        values = np.asarray(values, dtype=np.float64).ravel()
        out = []
        for node, lg, v in zip(leaves, logits, values):
            node.legal = selfplay.legal_mask(node.board, node.to_move)
            l = lg[node.legal]
            l = l - l.max()
            e = np.exp(l)
            p = np.zeros(82, dtype=np.float32)
            p[node.legal] = (e / e.sum()).astype(np.float32)
            node.P = p
            node.N = np.zeros(82, dtype=np.int32)
            node.W = np.zeros(82, dtype=np.float64)
            node.V = np.zeros(82, dtype=np.int32)
            node.children = {}
            node.expanded = True
            out.append(float(np.clip(v, -1.0, 1.0)))
        return out

    def _backup(self, nodes, path, v):
        for node_idx, m in reversed(path):
            node = nodes[node_idx]
            node.V[m] -= 1
            node.N[m] += 1
            node.W[m] += v
            v = -v

    def _terminal_value(self, node):
        b, w = selfplay.score(node.board)
        m = b - w
        v = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
        return v if node.to_move == rules.BLACK else -v

    def _add_root_noise(self, root):
        eps = self.cfg.dirichlet_eps
        if eps <= 0:
            return
        idx = np.flatnonzero(root.legal)
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(idx))
        p = root.P.copy()
        p[idx] = (1.0 - eps) * p[idx] + eps * noise
        root.P = p.astype(np.float32)

    def _choose_move(self, root):
        n = root.N.astype(np.float64)
        if self.cfg.temperature <= 0 or n.sum() == 0:
            return int(np.argmax(n))
        p = n ** (1.0 / self.cfg.temperature)
        p /= p.sum()
        return int(self.rng.choice(82, p=p))
