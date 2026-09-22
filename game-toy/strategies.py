"""
Strategies for the iterated prisoner's dilemma tournament.

A strategy is a function `move(my_history, opp_history) -> 'C' | 'D'`,
where the histories are lists of past moves ('C' = cooperate, 'D' = defect),
most recent last. Stateless one-liners and stateful grudge-holders alike.

The classic Axelrod (1980) result this recreates: tit-for-tat -- the
simplest strategy here -- wins the round-robin despite never outscoring
any single opponent head-to-head.
"""

import random

C, D = 'C', 'D'

# Standard Axelrod payoffs: (my_move, opp_move) -> (my_points, opp_points)
# T=5 (temptation), R=3 (reward), P=1 (punishment), S=0 (sucker)
PAYOFFS = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}


def tit_for_tat(my, opp):
    """Cooperate first, then copy the opponent's last move.

    Nice, retaliatory, forgiving, clear. Winner of both Axelrod tournaments.
    """
    return C if not opp else opp[-1]


def grim_trigger(my, opp):
    """Cooperate until the opponent defects once, then defect forever."""
    return D if D in opp else C


def always_cooperate(my, opp):
    return C


def always_defect(my, opp):
    return D


def random_player(my, opp, p=0.5):
    """Cooperate with probability p, else defect."""
    return C if random.random() < p else D


def tit_for_two_tats(my, opp):
    """Like tit-for-tat but only retaliates after two consecutive defections."""
    return D if opp[-2:] == [D, D] else C


def pavlov(my, opp):
    """Win-stay, lose-shift: repeat last move after a good round (3 or 5
    points), switch after a bad one (0 or 1). Starts by cooperating."""
    if not my:
        return C
    last_points = PAYOFFS[(my[-1], opp[-1])][0]
    return my[-1] if last_points >= 3 else (D if my[-1] == C else C)


def joss(my, opp):
    """Tit-for-tat with a sneaky 10% random defection -- tries to exploit
    the forgiving strategies. Axelrod's second tournament was full of these;
    they mostly just hurt themselves."""
    if opp and random.random() < 0.10:
        return D
    return C if not opp else opp[-1]


def prober(my, opp):
    """Defect on moves 2-4 to probe, then: if the opponent never retaliated,
    keep defecting; otherwise play tit-for-tat. An exploiter with a conscience
    of convenience."""
    n = len(my)
    if n < 3:
        return D if n > 0 else C
    if D not in opp[1:4]:
        return D
    return opp[-1]


STRATEGIES = {
    'tit_for_tat': tit_for_tat,
    'grim_trigger': grim_trigger,
    'always_cooperate': always_cooperate,
    'always_defect': always_defect,
    'random': random_player,
    'tit_for_two_tats': tit_for_two_tats,
    'pavlov': pavlov,
    'joss': joss,
    'prober': prober,
}
