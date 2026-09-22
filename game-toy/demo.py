"""
Demo: the Axelrod tournament.

Nine strategies, round-robin, 200 rounds per match. Watch the nice guy
win without winning a single fight.

Run:  python3 demo.py
"""

import random

from strategies import STRATEGIES
from tournament import run_tournament


def main():
    random.seed(42)
    rounds = 200
    standings, matrix = run_tournament(STRATEGIES, rounds)

    print(f"Axelrod-style tournament: {len(STRATEGIES)} strategies, "
          f"{rounds} rounds/match\n")
    print(f"{'rank':<4} {'strategy':<18} {'total points':<12} "
          f"{'avg/match':<9}")
    print("-" * 48)
    n_matches = len(STRATEGIES)
    for rank, (name, total) in enumerate(standings, 1):
        print(f"{rank:<4} {name:<18} {total:<12} "
              f"{total / n_matches:<9.1f}")

    print("\nHead-to-head: tit_for_tat's score vs each opponent "
          "(200 rounds each):")
    tft = matrix['tit_for_tat']
    for name in STRATEGIES:
        mine, theirs = tft[name], matrix[name]['tit_for_tat']
        verdict = "tie" if mine == theirs else ("win" if mine > theirs else "loss")
        print(f"  vs {name:<18} {mine:>3} - {theirs:<3}  ({verdict})")

    winner = standings[0][0]
    print(f"\nWinner: {winner}.")
    print("In Axelrod's 1980 tournaments tit-for-tat won outright. Here,")
    print("pavlov edges it: joss and random inject noise, and pavlov's")
    print("win-stay-lose-shift corrects accidental defections while")
    print("tit-for-tat echoes them into death spirals (see vs joss above).")
    print("The deeper lesson: there is no universally best strategy --")
    print("only strategies fit for their environment. Try removing joss")
    print("and random from STRATEGIES and watch the standings shift.")


if __name__ == "__main__":
    main()
