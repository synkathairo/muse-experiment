"""
Round-robin tournament engine for the iterated prisoner's dilemma.

Every strategy plays every other strategy (and itself) for `rounds`
rounds per match. Total accumulated points decide the standings.
"""

from strategies import PAYOFFS


def play_match(strat_a, strat_b, rounds=200):
    """Play one match. Returns (score_a, score_b)."""
    hist_a, hist_b = [], []
    score_a, score_b = 0, 0
    for _ in range(rounds):
        move_a = strat_a(hist_a, hist_b)
        move_b = strat_b(hist_b, hist_a)
        pa, pb = PAYOFFS[(move_a, move_b)]
        score_a += pa
        score_b += pb
        hist_a.append(move_a)
        hist_b.append(move_b)
    return score_a, score_b


def run_tournament(strategies, rounds=200):
    """strategies: dict name -> strategy fn.

    Returns (standings, matrix) where standings is a list of
    (name, total_score) sorted descending, and matrix[name_a][name_b]
    is name_a's score against name_b.
    """
    names = list(strategies)
    matrix = {a: {} for a in names}
    totals = {a: 0 for a in names}
    for i, a in enumerate(names):
        for b in names[i:]:
            sa, sb = play_match(strategies[a], strategies[b], rounds)
            matrix[a][b] = sa
            matrix[b][a] = sb
            totals[a] += sa
            totals[b] += sb
    standings = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return standings, matrix
