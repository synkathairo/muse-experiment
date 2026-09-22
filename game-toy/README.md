# game-toy

An Axelrod-style iterated prisoner's dilemma tournament. Nine strategies,
round-robin, 200 rounds per match — run it and watch cooperation win.

```
python3 demo.py          # the tournament
python3 test_tournament.py  # smoke test (fixed seed)
```

Stdlib only.

## What's here

- `strategies.py` — the strategy zoo. A strategy is a function
  `(my_history, opp_history) -> 'C' | 'D'`: tit-for-tat, grim trigger,
  always-cooperate, always-defect, random, tit-for-two-tats, pavlov
  (win-stay-lose-shift), joss (tit-for-tat with sneaky defections), prober.
- `tournament.py` — round-robin engine: every pair (including self-play),
  total points decide the standings.
- `demo.py` — prints the standings plus tit-for-tat's head-to-head record.

## The idea

One round pays: mutual cooperation 3–3, mutual defection 1–1, and 5 for
the defector against a lone cooperator (who gets 0). Defection tempts
every round — and yet, in Axelrod's 1980 tournaments, the winning
strategy was tit-for-tat: cooperate first, then copy the opponent.
Nice, retaliatory, forgiving, clear.

The twist this demo surfaced honestly: tit-for-tat does *not* win this
particular zoo — pavlov does. The strategy pool includes noisy neighbors
(joss, random), and under noise, pavlov's win-stay-lose-shift corrects
accidental defections while tit-for-tat echoes them into defection
spirals (TFT scores 239 vs joss; pavlov scores 397). That's a real,
documented result, not a bug — and it's the deeper lesson: there is no
universally best strategy, only strategies fit for their environment.
Remove the noisy strategies and the standings shift; try it.

What *does* hold across every pool: the nice strategies — the ones that
never defect first — sweep the top of the table, and the exploiters
finish at the bottom. Defection wins rounds; cooperation wins tournaments.

## Try

- Add your own strategy to `STRATEGIES` and enter the tournament.
- Change the payoffs or the round count; watch grim trigger's fortunes.
- Seed the population and iterate: replace the bottom strategies with
  copies of the winners — a poor man's evolutionary tournament.
