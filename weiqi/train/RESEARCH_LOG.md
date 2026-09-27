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

### Masked PPO (2026-09-26) — premise falsified, no experiment needed
- Hypothesis (from the distillation legality finding): PPO optimizes the full
  82-way distribution while inference masks illegal moves — a behavior/
  optimization mismatch wasting capacity. Codex (gpt-5.6-sol) picked this as
  the top direction with a head-to-head kill test.
- Investigation killed it before any compute: PPO was ALREADY fully masked.
  train_selfplay.py `masked_dist` (-inf fill before Categorical), ppo.py
  `_masked_logps_entropy` (-inf before log-softmax; entropy over legal moves
  only, NaN-guarded). Present since the original PPO commit 9689ec2; the
  premise was never true of this codebase. A --mask-illegal flag would have
  been a literal no-op.
- The 18-24% raw-argmax legality is therefore NOT a training bug: illegal
  policy-head rows get exactly zero gradient (proven by test), stay at init,
  and drift only via trunk features — random-init projections, masked to -inf
  at inference. Harmless. The earlier claim that "the entropy bonus rewards
  illegal spread" was wrong; retracted.
- Tests committed locally as 258ac9f (9 tests, suite 65/65): illegal probs
  exactly 0, legal sum to 1, zero gradient on illegal logits, 2000/2000
  rollout samples legal, greedy argmax legal even when raw argmax is forced
  illegal. These lock in the masking behavior.
- Remaining micro-candidate: entropy is not normalized by legal-move count
  (early-game positions get larger-magnitude entropy bonus). Assessed as
  second-order; no standard implementation does it. Thread otherwise dropped —
  the real uncertainty is representation (tactical planes), not optimization.

### Tactical input planes (2026-09-27) — IN FLIGHT
- Hypothesis: the 130K net wastes capacity computing group liberties through
  convolutions; handing it tactical state directly tests whether the wall is
  representation rather than capacity. This is the decisive branch of the
  distillation post-mortem (blunders, not average policy shape).
- Encoding: 13 planes (planes 0-5 identical to the locked 6-plane spec; new
  planes 6/7/8 = own stones with group liberties 1/2/>=3, 9/10/11 = opponent
  stones with liberties 1/2/>=3, 12 = ko-prohibition point, same semantics as
  plane 5). Liberties from exact connected-group flood fill
  (Board.liberty_map). Phase 1 is Python training only — web/Rust/WASM untouched.
- Network: GoNetTactical (identical trunk/heads, conv1 6->13 channels);
  134,554 params (+4,032 = 64*7*3*3, asserted by test).
- Branching: 15.5M checkpoint -> 13-plane net, 6 old channels copied, 7 new
  channels zero-initialized => branched policy bit-identical to baseline
  (test asserts policy/value match to 1e-5 on real positions).
- Kill test: 4 x 500k-step continuations from 15.5M, matched (komi 6.5,
  dirichlet 12/0.05/0.25, ent 0.03, lr 1e-4 constant via --no-anneal-lr since
  the annealed schedule sits at ~0 after 15.5M): planes_feat_s7/s8 (tactical)
  vs planes_ctrl_s7/s8 (plain GoNetAux continuation).
- Eval per final checkpoint: full 32-game GNU Go ladder (baseline 15.5M: 7-25)
  + tactical blunder rate on a mined set (1-ply capture/escape solver over
  frozen-baseline self-play games and vs-GNUGo losses).
- Kill criterion: tactical branch must beat control on tactics and/or match
  results; if it improves neither, shelve the direction. (One tied ladder
  alone does not kill it if tactical errors clearly improve.)

### Tactical input planes (2026-09-27) — RESULTS (mixed; seed bug found)

**Training completed** — all 4x500k branches 15.5M -> 16,007,168 steps:
- planes_feat_s7 (tactical): done 23:43
- planes_ctrl_s7 (plain GoNet): done 01:35
- planes_feat_s8 (tactical): done 03:22
- planes_ctrl_s8 (plain GoNet): done 12:38 (survived 4 VM reboots via flock watchdog; /usr/games/gnugo was wiped by a reboot, ladders used ~/workspace/katago/distill/gnugo instead)

**SEED BUG (critical, found during eval)**: planes_feat_s7 and planes_feat_s8 are
BIT-IDENTICAL (0/18 state-dict tensors differ; bit-identical logits on all 612
blunder positions). Root causes, all in the resume path:
1. Both feat runs were launched with `--resume runs/planes_feat_s7/branch_init.pt`
   — the s8 command pointed at s7's branch file, not its own (which existed).
2. `load_ckpt` (train_selfplay.py:405-406) restores torch/numpy RNG state AFTER
   `torch.manual_seed(args.seed)` / `np.random.seed(args.seed)` (lines 555-556),
   so `--seed` is a complete no-op on resume.
3. `SelfPlayGo.self.rng` (selfplay.py:246) is created but never used — the env
   seed is dead code.
Identical resume file + identical restored RNG => identical training. The ctrl
branches (26/26 tensors differ) empirically diverged and count as two valid
seeds. **Effective design: feat n=1, ctrl n=2.** Fix `--seed`-on-resume (seed
must take precedence over restored RNG) and the dead env RNG regardless.

**Blunder rate** — 612 positions mined from frozen-baseline losses (baseline
15.5M itself: 63.9%):
- planes_ctrl_s7: 64.5% (395/612; capture 58.2%, escape 65.4%)
- planes_ctrl_s8: 68.8% (421/612; capture 62.8%, escape 69.5%)
- planes_feat (s7=s8, one model): 68.6% (420/612; capture 63.0%, escape 66.8%)
Verdict: NO IMPROVEMENT on tactical decisions. Feat sits at the worse end of
the control range and below baseline. (Feat vs ctrl_s7: 4.1pp, ~1.5 SE, n.s.)

**GNU Go ladder** — 32 games, levels 1/3/5/10 x 8 games alternating colors,
greedy; baseline 15.5M: 7-25:
- planes_feat_s7: 10-22 [L1:1-7, L3:3-5, L5:4-4, L10:2-6]
- planes_feat_s8: 13-19 [L1:3-5, L3:3-5, L5:5-3, L10:2-6] (same weights as s7;
  ladder noise on identical weights ~+/-1.5 wins)
- planes_ctrl_s7: 5-27 [L1:3-5, L3:0-8, L5:2-6, L10:0-8]
- planes_ctrl_s8: 6-26 [L1:0-8, L3:1-7, L5:4-4, L10:1-7]
Verdict: feat beats matched controls 23/64 (36%) vs 11/64 (17%), two-proportion
z~2.4, p~0.02. vs baseline 7/32 (22%): 36% vs 22%, z~1.4, p~0.16 (n.s.).
Both ctrls underperform baseline; both feat ladders outperform both ctrls.

**Interpretation**: MIXED. The direct hypothesis (tactical planes -> better
tactical decisions) is NOT supported — blunder rate flat/worse. But overall
GNU Go strength improved significantly vs plain-continuation controls, via an
unclear mechanism (not fewer tactical blunders on the mined set; candidates:
positional value of liberty features, regularization/optimization effects of
the wider input layer, or a lucky run — feat is n=1). The kill criterion
(shelve if NEITHER improves) is not met: ladders improved. NOT shelved.
Next: peer-review the interpretation (Sol), then likely a replication leg —
proper second feat seed with the RNG bug fixed — before any demo/engine work.

### Tactical input planes (2026-09-27) — REPLICATION (in flight)

**Bugs fixed before replication:**
1. `branch_checkpoint` now migrates Adam moments (was: fresh optimizer). The 6 old conv1 channels keep their moments; 7 new channels start at zero. Verified: 18/18 params have moments, old channels bit-identical to source.
2. `train_selfplay.py` now re-seeds torch/numpy AFTER `load_ckpt`, so `--seed` takes precedence over restored RNG. (Previously: seed nullified on resume → identical trajectories.)

**Replication design (Sol's recommendation):**
- Fresh matched pair, both with seed 21, both with migrated/warm Adam, both 500K steps from 15.5M.
- `repl_tact_s21`: GoNetTactical (13 planes), branch_init.pt with migrated optimizer.
- `repl_ctrl_s21`: Plain GoNetAux (6 planes), resumed from 15.5M via load_ckpt (migrates trunk Adam).
- Only difference: tactical input planes. RNG and optimizer handling identical.

**Status:** Launched 2026-09-27 13:05 UTC. Watchdog: `~/workspace/repl_watchdog.sh`. Estimated 5h + reboot overhead.

**Replication results (tactical benchmark, 612 positions):**
- `repl_tact_s21`: 396/612 = 64.7% (capture 60.0%, escape 63.4%)
- `repl_ctrl_s21`: 422/612 = 69.0% (capture 64.9%, escape 64.4%)
- Baseline 15.5M: 63.9%

The replication tactical beats the replication control by 4.3pp on blunders, and matches baseline (64.7% vs 63.9%). Unlike the original kill test where tactical was WORSE than baseline, the fixed-methodology tactical now performs at baseline level on the tactical set.

**Replication results (GNU Go ladders, 32 games each):**
- `repl_tact_s21`: 9-23 (L1 3-5, L3 2-6, L5 1-7, L10 3-5)
- `repl_ctrl_s21`: 11-21 (L1 4-4, L3 1-7, L5 4-4, L10 2-6)
- Baseline 15.5M: 7-25

**VERDICT: Shelve tactical planes.**

The original ladder edge (tactical 23/64 vs controls 11/64) did NOT replicate under clean methodology. With matched seeds, matched warm Adam, and fixed RNG:
- Ladders: control beats tactical 11-21 vs 9-23 (n.s.)
- Blunders: tactical 64.7% vs control 69.0% (tactical better, but neither beats baseline 63.9%)

The original gap was likely the optimizer confound (fresh vs warm Adam), not the planes. Tactical features do not improve strength over baseline within a 500K-step budget. The blunder-rate mitigation (vs control) is interesting but doesn't translate to wins and doesn't exceed baseline.

**Lessons:**
1. Always verify seed actually controls RNG after resume (test: different seeds → different trajectories).
2. Match optimizer state handling between arms (fresh vs migrated is a confound).
3. A single training replicate (n=1) is not evidence; the seed bug made this worse.

### Future directions: candidate strategies (2026-09-27) — from Sol + Astra consultations

No OGS/supervised data allowed. All ideas target the 130K-param PPO self-play setup.

**1. Archived-position continuation self-play (Astra's top bet, Sol's #1)**
*Reasoning:* The agent rarely trains on meaningful midgame/endgame states because every episode starts from move zero. Most learning signal comes from opening trajectories; the critic never sees enough diverse late-game positions to calibrate. By archiving self-play positions and restarting 50% of episodes from midgame states, the policy and value heads get dense training on the positions where games are actually decided.
*Variants:* (a) Simple: uniform sample from rolling archive at 25/50/75/100% game progress. (b) Adaptive (Astra): prioritize positions where critic is uncertain, policy entropy is high, or consecutive checkpoints disagree: $w(s) = |V_{current}(s) - V_{older}(s)|$. Keep fraction of normal games to avoid distribution collapse. Compatible with on-policy PPO if each restart is a fresh episode.
*Evidence:* Go-Exploit 9x9 study (arXiv:2302.12359) supports the simple version; Cheng et al. report 58.5% sample efficiency gain from uncertainty-guided branching (but in MCTS/AlphaZero, not PPO — transfer uncertain).
*Kill criterion (Astra):* No improvement in 200-game GNU Go win rate AND critic calibration on archived positions by 1-2M steps.

**2. Bigger network (my #1, Sol's #2, Astra's #3)**
*Reasoning:* The 130K net may be at capacity. Distillation failed (student couldn't absorb 10.5M teacher), PPO has plateaued, auxiliary heads and input planes both null. A 500K-1M param net raises the ceiling. But: retrain from scratch, update Rust/WASM inference, re-export — multi-day project.
*Sol's caution:* Run a controlled ~500K comparison before committing to 1M. Benchmark CPU inference throughput first — if it's too slow, the strength gain isn't usable.
*Astra's caution:* Don't overclaim capacity from distillation; 29.9% top-1 doesn't measure consequential choices. The value target may be the bottleneck, not the trunk.

**3. Short-horizon value targets (Sol's #3, Astra's #2)**
*Reasoning:* The critic trains against a noisy binary terminal outcome propagated over 100+ moves. KataGo trains auxiliary value heads at multiple horizons (~6, ~16, ~50 moves on 19x19, shorter on 9x9), giving lower-variance feedback. This is a bias-variance tradeoff: short-horizon targets are biased (bootstrap from current value) but much less noisy.
*Caveat:* KataGo's targets average future MCTS values; ours would bootstrap from PPO value predictions — materially different. Cheap to test (auxiliary heads, no architecture lock-in).

**4. Global pooling path (Sol's missing idea)**
*Reasoning:* Four 3x3 conv layers have a 9-point nominal receptive field, but thin CNNs struggle to combine board-wide information into the policy/value heads. A small global-pooling branch (e.g., mean/max pool over spatial dims → FC → concat with head input) gives the heads direct access to whole-board features. KataGo's ablation showed clear learning-efficiency benefit. Sharper capacity test than input planes, at roughly fixed parameter count.

**5. Symmetry augmentation (Astra's free lunch — CURRENTLY MISSING)**
*Reasoning:* Go is invariant under all 8 dihedral symmetries (rotations + reflections). Training on all 8 versions of each position is ~8x effective data for free. Astra: "nearly free and should beat almost any hand-designed input plane." Verified 2026-09-27: NOT currently implemented in gotrain/. This should be added immediately regardless of other directions.

**6. Population Based Training (my #2, Sol's #4)**
*Reasoning:* Instead of fixed hyperparameters, maintain a population that copies the best and mutates hyperparameters (LR, ent-coef, etc.) mid-training. Wu et al. 2020 showed gains on 9x9 Go specifically.
*Sol's caution:* Their result used 16 agents sharing AlphaZero self-play data; low self-play overhead doesn't transfer to our CPU PPO trainer. Try LR decay on continuation first — cheaper.

**7. MCTS-generated policy targets / AlphaZero-style (my #4, Sol's #5, Astra's "serious alternative")**
*Reasoning:* Use search during training to produce improved move targets (visit distributions), train policy to imitate them. This is approximate policy iteration, not policy gradient — a different learning paradigm. Our MCTS-100-at-play-time null does NOT test this; search at evaluation exposes a weak policy, search during training improves the targets.
*Astra's practical suggestion:* "Search-lite" — search only a subset of moves (8-16 early/midgame decisions) to avoid making CPU throughput the confound. Full AlphaZero changes the entire training loop.

**Evaluation methodology (Sol + Astra agree):**
- 32-game ladder is a smoke test, not a decision tool. 7-25 has 95% CI of 11%-39%.
- For decisions: 200+ games per comparison, fixed openings, alternating colors, balanced by level.
- Add direct head-to-head vs incumbent checkpoint (not just GNU Go).
- Keep a final opponent set out of checkpoint selection to avoid overfitting to the ladder.
- Our komi 6.5 conclusion (from 32 games) should be treated as provisional.

**Consensus next steps:**
1. Add symmetry augmentation immediately (free).
2. Test archived-position continuation (Astra's bet).
3. Benchmark 500K net CPU throughput (informs bigger-net decision).
4. Consider global pooling as a cheap architectural test.

### Symmetry augmentation (2026-09-27) — IMPLEMENTED

Astra flagged that all 8 dihedral symmetries were not used during PPO. Verified: no augmentation existed. Implemented in `gotrain/train_selfplay.py`:
- Per-env random symmetry (rot90 k=0..3 x flip/no-flip) applied to obs and masks before `sample_actions`.
- Actions sampled in the augmented frame; mapped back via inverse for `env.step`.
- Stored (obs, masks, actions, logps) are all in the augmented frame, so PPO's importance ratio stays valid.
- Pass (81) unaffected. Value targets unchanged (win/loss is symmetric).

Bugs caught during implementation:
1. First attempted augmentation in `ppo.py::ppo_update` — wrong, because stored logps were computed pre-transform. Moved to rollout time.
2. Inserted helpers stole `@torch.no_grad()` from `sample_actions` (decorator ended up on wrong function). Restored.

Tests: 75 passed. Smoke training run (256 steps) completes. This is ~8x free data; should have been there from the start.

### Position archive for restarted self-play (2026-09-27) — IMPLEMENTED

Astra's top bet, Sol's #1. The agent rarely trains on midgame/endgame because every episode starts from move zero.

Implementation (`gotrain/selfplay.py`, `gotrain/train_selfplay.py`):
- `PositionArchive`: FIFO rolling buffer (default 10K) of deep-copied Boards + env metadata (learner_color, plies, consec_passes).
- During rollout: each live env archives its position with prob `--archive-prob` (default 0.02) per step. Board is at learner's turn (post-step invariant), so restoration is clean.
- On env done: with prob `--archive-restart-prob` (default 0.0 = disabled), restarts from a random archived position via `SelfPlayGo.reset_from_archive()` instead of a fresh game. Falls back to fresh if archive empty.
- `color_counter` NOT incremented on archive restore (the archived game had a fixed learner color; alternation resumes on next full reset).
- Superko history preserved via deepcopy.

CLI: `--archive-restart-prob 0.5 --archive-size 10000 --archive-prob 0.02`

Tests: 3 new (roundtrip, empty fallback, FIFO), 78 total pass. Smoke training with archive enabled completes at full throughput.

Kill test (per Astra): 1-2M steps, 200-game eval. Kill if no improvement in win rate AND critic calibration vs baseline.
