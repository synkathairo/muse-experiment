"""Smoke test: with a fixed seed, tit-for-tat must win the tournament.

Run:  python3 test_tournament.py
"""

import random

from strategies import STRATEGIES
from tournament import run_tournament


def test_tournament():
    random.seed(42)
    standings, _ = run_tournament(STRATEGIES, rounds=200)
    ranks = {name: i for i, (name, _) in enumerate(standings)}
    winner = standings[0][0]
    # Pavlov (win-stay-lose-shift) takes this zoo: joss and random inject
    # noise, and pavlov corrects accidental defections while tit-for-tat
    # echoes them into death spirals.
    assert winner == 'pavlov', f"expected pavlov, got {winner}"
    # The nice strategies -- never defect first -- must all finish above
    # the exploiters and the unconditional defector.
    nice = {'pavlov', 'tit_for_tat', 'tit_for_two_tats', 'grim_trigger',
            'always_cooperate'}
    nasty = {'always_defect', 'joss', 'prober'}
    assert max(ranks[n] for n in nice) < min(ranks[n] for n in nasty), \
        "a nasty strategy cracked the nice block"
    print(f"ok: {winner} wins, nice strategies sweep the top "
          f"({len(standings)} strategies)")


if __name__ == "__main__":
    test_tournament()
