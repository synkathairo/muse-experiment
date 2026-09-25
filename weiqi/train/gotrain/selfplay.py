"""Self-play Go environment for the Autodidact leg (PLAN.md §5).

No PufferLib anywhere: a small numpy/rules.py-based vectorized env with the exact
§3 input planes and §1 rules, driven by a hand-rolled PPO loop (gotrain.ppo).

Rules (PLAN.md §1, provisional but locked for this leg):
  - 9x9, Tromp-Taylor-style legality (via gotrain.rules: no suicide, positional superko),
  - two-pass termination, Tromp-Taylor area scoring, komi 7.5,
  - reward 0 on non-terminal steps, +/-1 at game end from the LEARNER's perspective.

Opponent scheme (why this and not raw self-play):
  - The learner alternates color every episode, so it must attack and defend with
    both colors; this stops color-specific exploits (e.g. learning only to win as
    Black against a weak-White snapshot) and gives the value head both perspectives.
  - The opponent is a FROZEN snapshot copy of the learner's policy, refreshed every
    K PPO iterations. A frozen opponent is stationary inside each PPO update window,
    which is what PPO's clipped trust region assumes; chasing a live copy of
    yourself every step is non-stationary and prone to strategy cycles
    (rock-paper-scissors dynamics). This is the poor-man's version of AlphaZero's
    "promote the best player" gating: the bar ratchets up in discrete steps.
  - Refreshing every K>>1 iterations (not every iteration) keeps consecutive
    iterations' data distributions close, which is what makes the value targets
    and advantages meaningful across the refresh boundary.

Termination subtlety: two consecutive passes = true terminal (bootstrap value 0).
Hitting max_plies is scored for a +/-1 reward and then treated as an EPISODIC
TERMINAL for GAE (bootstrap value 0 as well). The env still reports the
Gymnasium terminated/truncated split (see step()), but the trainer must not
bootstrap truncations: only the final rollout observation is preserved, so a
within-rollout truncation would otherwise bootstrap from the NEXT episode's
value. This matters because random early play almost never passes twice.

All observations are encoded from the side-to-move's perspective (plane 0 = mover's
own stones), exactly as gotrain.features.encode specifies, so one policy net serves
both colors and both roles.
"""

import numpy as np

from . import features
from .rules import BLACK, WHITE, EMPTY, Board, opponent

N = 9
N_MOVES = 82
PASS = features.move_to_index(None)  # 81
KOMI = 7.5
MAX_PLIES = 3 * N * N  # 243; safety cap guaranteeing termination


# ---------------------------------------------------------------------------
# Tromp-Taylor area scoring
# ---------------------------------------------------------------------------
def score(board):
    """Return (black_score, white_score) with komi 7.5 added to White.

    Tromp-Taylor: score = stones on board + empty points whose bordering stones
    are all one color. Empty regions touching both colors are neutral.
    """
    n = board.size
    black_stones = white_stones = 0
    for r in range(n):
        for c in range(n):
            v = board.grid[r][c]
            if v == BLACK:
                black_stones += 1
            elif v == WHITE:
                white_stones += 1

    black_terr = white_terr = 0
    seen = [[False] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if board.grid[r][c] != EMPTY or seen[r][c]:
                continue
            # flood-fill one empty region, tracking bordering colors
            region = 0
            borders = set()
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                sr, sc = stack.pop()
                region += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        v = board.grid[nr][nc]
                        if v == EMPTY and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                        elif v == BLACK:
                            borders.add(BLACK)
                        elif v == WHITE:
                            borders.add(WHITE)
            if borders == {BLACK}:
                black_terr += region
            elif borders == {WHITE}:
                white_terr += region
            # else: neutral (seki-like / dame) — counts for nobody
    return (black_stones + black_terr, white_stones + white_terr + KOMI)


def winner(board):
    """BLACK, WHITE, or EMPTY (draw — impossible with 7.5 komi, kept for safety)."""
    b, w = score(board)
    if b > w:
        return BLACK
    if w > b:
        return WHITE
    return EMPTY


# ---------------------------------------------------------------------------
# Observation / legality helpers
# ---------------------------------------------------------------------------
def legal_mask(board, color):
    """bool[82]: legal moves for `color`. Pass (81) is always legal.

    Masking + renormalizing the policy over this mask guarantees no illegal move
    can ever be sampled, regardless of what the net outputs.
    """
    mask = np.zeros(N_MOVES, dtype=bool)
    n = board.size
    for r in range(n):
        for c in range(n):
            if board.grid[r][c] == EMPTY and board.is_legal(r, c, color):
                mask[r * n + c] = True
    mask[PASS] = True
    return mask


def observe(board, color):
    """(6,9,9) float32 planes from `color`-to-move's perspective (PLAN.md §3)."""
    return features.encode(
        board.stones(color),
        board.stones(opponent(color)),
        last_move=board.last_move,
        black_to_move=(color == BLACK),
        ko_point=board.ko,
    )


# ---------------------------------------------------------------------------
# Vectorized self-play environment
# ---------------------------------------------------------------------------
class SelfPlayGo:
    """num_envs parallel 9x9 games: learner (alternating color) vs frozen snapshot.

    opponent_fn: callable (obs (B,6,9,9) float32, masks (B,82) bool)
                 -> int64 (B,) move indices. The trainer wires this to the frozen
                 snapshot net (sampled, masked); tests may pass a random player.

    step() applies one learner move and then the opponent's reply in every live
    env, so each call advances the learner by exactly one PPO sample. Returns
    (obs, rewards, dones, terms, ep_lens):
      obs      (num_envs,6,9,9) float32, always the learner's turn to move
      rewards  float32, 0 except +/-1 (learner's perspective) on game end
      dones    bool, game over (terminal or truncated)
      terms    bool, TRUE terminal (two passes); False => truncation (max plies)
      ep_lens  int, plies played for finished games, else 0
    Finished envs are NOT auto-reset: call reset(idxs) for dones before stepping.
    """

    def __init__(self, num_envs=32, seed=0, max_plies=MAX_PLIES, opponent_fn=None):
        self.num_envs = num_envs
        self.max_plies = max_plies
        self.opponent_fn = opponent_fn
        self.rng = np.random.default_rng(seed)
        self.boards = [Board(N) for _ in range(num_envs)]
        self.learner_color = np.full(num_envs, BLACK, dtype=np.int64)
        self.color_counter = np.zeros(num_envs, dtype=np.int64)  # alternates colors
        self.plies = np.zeros(num_envs, dtype=np.int64)
        self.consec_passes = np.zeros(num_envs, dtype=np.int64)
        self.done = np.zeros(num_envs, dtype=bool)

    # -- reset ---------------------------------------------------------------
    def _new_game(self, i):
        self.boards[i] = Board(N)
        # alternate the learner's color every episode: even counter -> Black
        self.learner_color[i] = BLACK if self.color_counter[i] % 2 == 0 else WHITE
        self.color_counter[i] += 1
        self.plies[i] = 0
        self.consec_passes[i] = 0
        self.done[i] = False
        # if the learner is White, the opponent (Black) moves first so that the
        # returned observation is always the learner's turn
        if self.learner_color[i] == WHITE:
            self._opponent_move(np.array([i]))

    def reset(self, idxs=None):
        """Reset games; return learner-perspective obs for `idxs` (default: all)."""
        if idxs is None:
            idxs = np.arange(self.num_envs)
        idxs = np.asarray(idxs)
        for i in idxs:
            self._new_game(int(i))
        return self._observe_learner(idxs)

    # -- internal move plumbing ----------------------------------------------
    def _observe_learner(self, idxs):
        obs = np.zeros((len(idxs), 6, N, N), dtype=np.float32)
        for k, i in enumerate(idxs):
            obs[k] = observe(self.boards[int(i)], self.learner_color[int(i)])
        return obs

    def _apply(self, i, move_idx, color):
        """Apply move_idx for color in env i. Returns True if the game ended."""
        board = self.boards[i]
        move = features.index_to_move(int(move_idx))
        ok = board.play(move, int(color))
        assert ok, f"illegal move emitted: idx={move_idx} color={color}"  # mask guarantees this
        self.plies[i] += 1
        self.consec_passes[i] = self.consec_passes[i] + 1 if move is None else 0
        if self.consec_passes[i] >= 2 or self.plies[i] >= self.max_plies:
            self.done[i] = True
            return True
        return False

    def _reward(self, i):
        """+/-1 terminal reward from the learner's perspective (0 on draw)."""
        w = winner(self.boards[i])
        if w == EMPTY:
            return 0.0
        return 1.0 if w == self.learner_color[i] else -1.0

    def _opponent_move(self, idxs):
        """Play the frozen-snapshot opponent's reply in envs `idxs` (its turn)."""
        idxs = np.asarray(idxs, dtype=np.int64)
        if len(idxs) == 0 or self.opponent_fn is None:
            return
        obs = np.zeros((len(idxs), 6, N, N), dtype=np.float32)
        masks = np.zeros((len(idxs), N_MOVES), dtype=bool)
        for k, i in enumerate(idxs):
            i = int(i)
            color = opponent(self.learner_color[i])
            obs[k] = observe(self.boards[i], color)
            masks[k] = legal_mask(self.boards[i], color)
        opp_actions = np.asarray(
            self.opponent_fn(obs, masks, self.plies[idxs]), dtype=np.int64)
        assert opp_actions.shape == (len(idxs),)
        for k, i in enumerate(idxs):
            i = int(i)
            if not self.done[i]:
                self._apply(i, opp_actions[k], opponent(self.learner_color[i]))

    # -- main step -------------------------------------------------------------
    def step(self, actions):
        """One learner move (+ opponent reply) per live env. See class docstring."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.num_envs,)
        live = np.where(~self.done)[0]
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terms = np.zeros(self.num_envs, dtype=bool)   # true terminal (two passes)
        ep_lens = np.zeros(self.num_envs, dtype=np.int64)
        ended_on_learner = np.zeros(self.num_envs, dtype=bool)

        for i in live:
            i = int(i)
            color = int(self.learner_color[i])
            # defense in depth: the trainer masks, but never trust the caller
            assert legal_mask(self.boards[i], color)[actions[i]], \
                f"learner played illegal move {actions[i]}"
            if self._apply(i, actions[i], color):
                ended_on_learner[i] = True
                terms[i] = self.consec_passes[i] >= 2
                rewards[i] = self._reward(i)
                ep_lens[i] = self.plies[i]

        # opponent replies everywhere still live and it is now the opponent's turn
        need_opp = np.array([i for i in live if not self.done[int(i)]], dtype=np.int64)
        self._opponent_move(need_opp)
        for i in need_opp:
            i = int(i)
            if self.done[i] and not ended_on_learner[i]:
                # game ended on the opponent's move
                terms[i] = self.consec_passes[i] >= 2
                rewards[i] = self._reward(i)
                ep_lens[i] = self.plies[i]

        obs = self._observe_learner(np.arange(self.num_envs))
        return obs, rewards, self.done.copy(), terms, ep_lens

    def legal_masks_learner(self):
        """(num_envs,82) bool legal masks for the learner's current turn."""
        masks = np.zeros((self.num_envs, N_MOVES), dtype=bool)
        for i in range(self.num_envs):
            if not self.done[i]:
                masks[i] = legal_mask(self.boards[i], int(self.learner_color[i]))
        return masks
