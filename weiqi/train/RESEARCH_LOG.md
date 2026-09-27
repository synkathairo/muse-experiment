# Weiqi research log

Chronological record of training experiments on the 9x9 Go demo net
(130K-param CNN, search-free PPO self-play unless noted).
Each entry: hypothesis, setup, result, verdict.

## Closed levers

### Training komi 6.5 (2026-09-25) — small win
- Hypothesis: 7.5 komi on 9x9 overcompensates White; training at lower komi teaches Black to fight.
- Setup: 6M → 15.5M continuation at komi 6.5 (Dirichlet openings 12/0.05/0.25, ent-coef 0.03).
- Result: 7–25 vs GNU Go ladder; first Black wins ever (2–14 as Black).
- Verdict: weak positive, closed. Ladder stays at 7.5 for comparability.

### Ownership auxiliary heads (2026-09-25/26) — null
- Hypothesis: KataGo-style ownership + score-margin heads improve sample efficiency / trunk features.
- Setup: matched 3M from-scratch screens, seed 7 — plain 5–27 vs ownership 3–29.
  20M ownership continuation vs 15.5M baseline — 5–27 vs 7–25.
- Result: no improvement. Trunk-drift diagnostic (61% argmax agreement, KL 0.31)
  showed the aux signal reaches the trunk without helping strength.
- Verdict: closed, do not pursue.

### Score-graded terminal rewards (2026-09-26) — null
- Hypothesis: tanh(margin/15) terminal rewards teach margin-awareness, KataGo-style.
- Setup: matched 3M screen, seed 7, `--reward-mode score`.
- Result: 5–27, per-level identical to plain PPO (0–8, 0–8, 1–7, 4–4).
- Verdict: perfect null, closed. Commit `554f300` local-only, not pushed.

### MCTS behavior policy for PPO, "option A" (2026-09-26) — dead on arrival
- Hypothesis: MCTS-chosen rollout actions give PPO stronger trajectories.
- Pre-check: MCTS-100 + 15.5M net vs greedy ladder → 5–27 vs 7–25.
  The search adds no playing strength (weak priors/values: blind leading the blind).
- Verdict: premise fails; the ~24h build is not justified. Closed.

## Open directions

### Knowledge distillation from KataGo 9x9 (2026-09-26) — clean negative, direction closed
- Idea: distill a released KataGo 9x9 net's policy into our 130K trunk via KL
  on soft targets. No published KataGo→tiny-net result found (open research).
  Built KataGo v1.18.2 CPU (Eigen) + 10.5M-param transformer teacher;
  raw 82-way priors via analysis engine (maxVisits=1 + includePolicy).
- From-scratch spike FAILED both gates: 2,025 positions, 60 epochs policy-only —
  held-out top-1 29.1% (gate >40%), CE 3.24 nats; reduced ladder 2–14, identical
  to a matched plain-PPO baseline; student argmax legal only 92.6%.
- Fine-tune follow-up (the plausible fix): 15.5M checkpoint + same 2,025 teacher
  positions, policy head only, LR 1e-4, 25 epochs. Imitation worked this time —
  held-out CE 9.58 → 3.13, top-1 6.2% → 29.9%, top-5 27.2% → 66.2%.
  Side finding: the base 15.5M net's raw argmax is legal only 18–24% of the time
  (the GTP harness masks illegal moves, hiding this; PPO never penalizes illegal
  raw logits); distillation raised it to 86%.
- Full 32-game ladder: 7–25, identical to baseline (gate was >7 wins).
  Black 0–8 vs 2–14, White 7–9 vs 5–11 — noise-sized swing. No forgetting, no gain.
- Verdict: strongest form of the negative — the teacher's policy shape was
  absorbed and still didn't convert to wins. Distillation closed as a strength
  method for the 130K net; the capacity verdict hardens.

### Supervised warm-start from OGS games (parked)
- Needs explicit user word: gentle rate-limited retry or forum ask.
  OGS global /api/v1/games/ 404s; only per-player game lists remain.

### League play / opponent pool (proposed, not started)
- Replace the single snapshot opponent with past checkpoints (AlphaStar-style)
  to stop the net overfitting to its own current style.

### Small-board curriculum 5x5 → 7x7 → 9x9 (proposed, not started)
- Trunk is fully convolutional; needs a value-head transfer check first.

### Bigger net + distill to demo size (proposed, expensive)
- Train large, distill to the 130K demo trunk. Needs real compute (user's Mac).

### Codex coworker consult (2026-09-26, gpt-reserve) — unconsidered ideas
Asked what we haven't tried under the ~130K WASM constraint. Ranked list;
converged independently with our masked-loss, komi-conditioning, league-play,
and curriculum ideas. Genuinely new:
- Tactical input planes (liberties, atari markers, ko point, last moves):
  stop asking the tiny net to infer liberties through convolutions; hand it
  the answers. Challenges the capacity verdict — the wall may be an input
  representation wall.
- Factorized action head: spatial logits per intersection + scalar pass logit,
  replacing the flat 82-way softmax; lets conv structure work, less memorization.
- Deeper trunk at fixed budget: reallocate flat-head params to conv depth /
  dilated layers; fully conv policy head, global pooling only for value.
- Shallow tactical solver (1-2 ply capture/atari) generating training targets;
  deployment stays search-free. Distinct from the killed MCTS-behavior idea.
- AWR (advantage-weighted replay imitation) instead of pure PPO on the same
  trajectories; less variance-chasing.
- Key insight: distillation's failure to convert despite better top-1/top-5
  imitation is consistent with the true failure mode being specific tactical
  blunders (ladder defense, legal-action errors), not average policy shape.
