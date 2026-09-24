"""Input feature planes — LOCKED spec, see PLAN.md §3.

6 planes x 9x9, float32, row-major. Row 0 = top of the board (SGF row 'a'),
col 0 = left (SGF col 'a').

Plane 0: own stones (side to move)
Plane 1: opponent stones
Plane 2: empty points
Plane 3: last move (1.0 at last played point; zeros if none or pass)
Plane 4: color to play (all 1.0 if Black to move, else 0.0)
Plane 5: ko-banned point (1.0, else zeros)

The Rust engine (weiqi/engine) must encode these bit-identically; golden
vectors in weiqi/golden/ pin the agreement.
"""

import numpy as np

N = 9
N_PLANES = 6


def encode(own, opp, last_move=None, black_to_move=True, ko_point=None):
    """Build the 6-plane input.

    own, opp: (9,9) bool arrays. last_move / ko_point: (row, col) or None.
    """
    planes = np.zeros((N_PLANES, N, N), dtype=np.float32)
    own = np.asarray(own, dtype=bool)
    opp = np.asarray(opp, dtype=bool)
    planes[0] = own
    planes[1] = opp
    planes[2] = ~(own | opp)
    if last_move is not None:
        planes[3, last_move[0], last_move[1]] = 1.0
    if black_to_move:
        planes[4, :, :] = 1.0
    if ko_point is not None:
        planes[5, ko_point[0], ko_point[1]] = 1.0
    return planes


def move_to_index(move):
    """move: (row, col) or None (pass) -> 0..81."""
    if move is None:
        return 81
    return move[0] * N + move[1]


def index_to_move(idx):
    """0..81 -> (row, col) or None (pass)."""
    if idx == 81:
        return None
    return (idx // N, idx % N)
