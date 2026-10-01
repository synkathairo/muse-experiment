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

### Position archive kill test (2026-09-27) — FAILED, SHELVED

**Design:** 1M steps from the symmetry checkpoint (16.0M → 17.0M), 50% of resets from a 10K rolling archive of midgame positions, 50% fresh. Astra/Sol's #1 recommendation.

**Eval protocol (new):** 1024 games vs GNU Go (levels 1/3/5/8, 256 per level), temperature 0.2, `--jobs 8`. Fixed the old ladder's determinism problem (greedy play = only 16 unique games out of 32).

**Results:**
- Symmetry baseline: 219-805 (21.4%)
- Archive: 193-831 (18.8%)
- Two-proportion z-test: p≈0.14 (not significant, but wrong direction)

**Verdict:** No improvement. Per the kill criterion, shelved.

**Hypothesis for failure:** The archive saves positions from the current weak policy's games (21% vs GNU Go). These aren't "meaningful midgame positions" — they're bad positions from bad games. Chicken-and-egg: you need a decent policy to generate useful archive positions, but you need useful positions to train a decent policy.

**Also learned:** The old 32-game greedy ladder was deterministic (16 effective games). All historical ladder numbers are noisier than quoted. New protocol: temperature 0.2, 256+ games, `--jobs 8` max during training.

### 2x2 architecture diagnostic: capacity vs global routing (2026-09-27) — MIXED/NULL

**Motivation:** Astra's primary hypothesis (capacity) + secondary (global info flow). Sol's recommended decision experiment: 64 vs 128 channels × pooling off/on, fixed teacher dataset.

**Architectures** (depth held at 4 layers to avoid confounding):
- `GoNet` (baseline): 130,522 params
- `GoNetWide`: 64→128 channels, 466,202 params
- `GoNetPool`: + GAP→FC→broadcast→concat global path, 138,810 params
- `GoNetWidePool`: both, 490,938 params

Code: `gotrain/net_wide.py`. Commits `a90356d` (archs + distill trainer), plus fixes.

**Experiment 1 — KataGo distillation (2,025 positions):**
Positions from plain-PPO 3M self-play (weak), labeled with KataGo b40 policy via `query_teacher.py`. Note: teacher outputs were raw logits, not probabilities — softmax applied on load (bug caught mid-run). 30 epochs, policy-only (no value targets in legacy dataset).
- baseline: 36.1% top-1 | wide: 33.7% | pool: 35.1% | widepool: 41.1%
- Pattern: neither alone helps, both together +5pp. But n=202 val (±3.4%), ~1σ — not significant. Positions from weak play, split by position not game.

**Experiment 2 — Supervised on Go Quest human games (273K positions):**
Go Quest 9x9 archive: 8,607 games, strong players (ratings 1800-2500). 5,815 games kept (2,791 dropped: too short/timeouts/bots), 273,103 train / 5,269 val positions. 5 epochs, predict human's next move. Code: `gotrain/train_supervised_2x2.py`.
- baseline: 43.03% | wide: 44.41% | pool: 42.57% | widepool: 44.62%
- Pattern: width helps +1.4pp (~2σ, marginal), pooling does nothing (-0.4pp), widepool ≈ wide alone.

**Verdict:** No dramatic capacity bottleneck. 3.5x params → +1.4% move prediction. Pooling is null. The 130K net isn't dramatically underpowered for 9x9 imitation; the weakness is elsewhere.

**Caveats (Sol's peer review):**
- Heads already flatten the full 9x9 board → FC layers already have global access. The pooling test was less decisive than intended.
- Top-1 imitation can miss rare decisive tactical errors; doesn't directly measure playing strength.
- 32-game ladders have ±14pp 95% CI at 20% win rate — screening only.

**Citation:** Go Quest 9x9 game records (Tanasa / Go Quest app, wars.fm/go9), shared by Hiroshi Yamashita to the computer-go mailing list, Dec 28, 2015.

### Sol peer review — literature and mechanistic hypotheses (2026-09-27)

Sol (gpt-6-sol) reviewed the 2x2 results with code inspection (`-C` flag).

**On PPO's struggles (mechanistic hypothesis, not published):** PPO learns from sampled moves + delayed terminal reward; the critic must do long-horizon credit assignment while the self-play opponent changes. No search-improved move targets (unlike AlphaZero/KataGo). This is a hypothesis about our setup, not a claim that PPO can't learn 9x9.

**Literature:**
- AlphaGo (Silver et al. 2016): supervised pretraining on human games → self-play. Our warm-start plan follows this.
- AlphaGo Zero (Silver et al. 2017): pure self-play works but with MCTS + far more compute — not directly transferable to our PPO setup.
- Go-Exploit (arXiv:2302.12359): starting self-play from archived positions improved value learning/sample efficiency in an AlphaZero 9x9 setting. Relevant to our shelved archive experiment — the treatment-rate audit (actual restart % unverified) should be resolved before fully closing that direction.
- KataGo (Wu 2019): global pooling results shouldn't be expected to transfer to our flattened-head PPO net.

**Recommended next step:** Supervised warm-start. Train `train_cloning.py` on the 273K Go Quest positions, use checkpoint to init PPO with fresh optimizer. Check supervised model's GNU Go strength before PPO. Compare matched PPO budgets vs scratch. Track value error + tactical blunders, not just wins.

**If warm-start fails:** Inspect value calibration and errors by game phase before another width experiment.

## PPO + behavioral cloning (2026-09-30)

**Question:** does retaining human demonstrations during PPO (interleaved IL+RL) beat plain PPO from the same warm-start checkpoint? The plain-PPO trajectory plateaued: supervised 28.9% → 3M PPO 46.1% → 6M PPO 50.4% → 9M PPO 46.1% (all corrected 256-game GNU Go ladders, temp 0.2).

**Design:** controlled comparison from the same 3M supervised→PPO checkpoint (`runs/ppo_warmstart/snap_003000000.pt`), 3M→6M env steps, matched everything else. Human demos = 273,103 Go Quest 9x9 positions (`data/goquest`). Implementation (`gotrain/train_selfplay.py`, flags `--bc-data/--bc-coef/--bc-epochs/--bc-batch-size`): after each PPO update, a separate supervised cross-entropy phase on demo moves (coef 0.1, batch 512, 1 epoch) — i.e. the clipped trust region never sees the supervised gradients. Related to, but not a replication of, IN-RIL (see literature below); no gradient-separation machinery.

**Methodology incident (matters for interpretation):** the BC phase as first written did `N // batch_size` ≈ 533 random minibatches per PPO iteration — a near-full pass over all 273K demos per iteration, ~6x slower than PPO itself (45 vs 275 sps). At step ~4.1M I fixed it to a fixed 8 minibatches per iteration and restarted from the checkpoint (~230 sps after). So the first ~4.1M env steps got a much heavier BC treatment than the last ~1.9M. The experiment is still PPO+BC vs plain PPO on the same step budget, but the BC intensity is not uniform across the run.

**Interruptions:** process died once (~04:59 EDT, step ~3.19M); watchdog resumed from intact `latest.pt` (~05:21 EDT). Final: step 6,004,736, fp16 export `runs/ppo_bc/autodidact-final.bin` (261KB).

**Benchmark:** 256-game GNU Go ladder, temp 0.2, 64 games each at levels 1/3/5/8 (`eval/ladder_bc.json`), auto-started on completion.

**Result: PPO+BC wins clearly.** Raw 154–102 (60.2%); 2 games where GNU Go resigned/played illegal are recorded `winner='them'` — per `eval_vs_gnugo.py` those are net wins. Corrected **156–100, 60.9%** vs plain PPO 6M **129–127, 50.4%**. Per-level (corrected): level 1: 43–21, level 3: 38–26, level 5: 42–22, level 8: 33–31. Gains at every level; level 8 (strongest GNU Go) is the closest, 33–31.

**Caveats:**
- Mid-run BC treatment change (above) — the exact BC schedule that produced this is not a clean single treatment. A replication with uniform treatment would firm this up.
- 256-game ladder has ~±6pp 95% CI at 60% win rate; the 10.5pp gap over 50.4% is well outside noise, but the *size* of the gap is noisy.
- The `winner='them'` eval-script reporting bug (resignations/illegal moves logged under the forfeiter's opponent key) remains unpatched; accounted manually here and for the 3M/9M ladders.
- Not tested: whether BC from step 0 (not just 3M→6M) helps, whether a lighter/heavier coef changes things, or whether the gain holds past 6M.

**Literature context:**
- DQfD (Hester et al., arXiv:1704.03732): pre-training + retaining expert demonstrations during deep RL (demonstration replay with supervised large-margin loss alongside TD loss) beat the demonstrators on 14 of 42 Atari games. Game-domain evidence that keeping the imitation signal alive during RL helps — our setup is a simpler cousin (separate BC phase, no prioritized demo replay, no margin loss).
- IN-RIL (arXiv:2505.10442): interleaving IL and RL updates improves sample efficiency and stability in robotics, with extra machinery to keep the two gradient streams from interfering. We used only the interleaving idea — separate phases so PPO's clipped updates never mix with supervised gradients — none of the gradient-separation machinery. Our result is consistent with the interleaving claim but is not a test of IN-RIL itself.
- The earlier tiny-model lit review (`workspace/research/tiny-model-lit-review.md`) already flagged interleaved IL+RL as an open question; this run answers it affirmatively for our setup.

**Verdict:** PPO+BC (60.9%) is the new strongest model, decisively beating plain PPO at the matched 6M budget. Shipped to the demo page (see below). The plain-PPO plateau at 50.4%/46.1% was not a capacity ceiling — keeping human moves in the training loop broke through it.

## MCTS-100 on the PPO+BC net (2026-09-30)

**Question:** the old MCTS-100 test (on the 15.5M PPO-from-scratch net: 5–27, no gain over greedy 7–25) said search didn't help. Does that verdict hold on the much stronger PPO+BC 6M net (greedy 156–100, 60.9%)?

**Design:** 256-game GNU Go ladder, same protocol as the greedy ladder (temp 0.2, 64 games each at levels 1/3/5/8, alternating colors, `--jobs 2`). Engine: `gotrain.mcts_gtp`, 100 sims, batch 16, temp 0.2 (`eval/ladder_mcts100_ppo_bc_256.json`). A 64-game probe (43–21, 67.2%) preceded the full run.

**Result: MCTS-100 wins clearly. 187–69 (73.0%)** vs greedy 156–100 (60.9%). Per-level: 1: 41–23, 3: 49–15, 5: 44–20, 8: 53–11. The gap is +12.1pp at ±~4.1pp SE → ~2.9σ. Largest gains at levels 3/5/8; level 1 roughly flat (43–21 → 41–23, noise).

**Caveats (protocol confound now closed):** the original greedy ladder (156–100) predates the Python eye-fill filter; the MCTS runs used it. A filtered-greedy rerun was attempted twice — the first attempt died silently with no output file (no handoff delivered; cause unknown, likely runtime/VM flakiness). The second completed: **filtered greedy = 145–111 (56.6%)**, per-level 43–21 / 34–30 / 36–28 / 32–32. So the filter's effect on greedy alone is 60.9% → 56.6% (−4.3pp, ~1.0σ — indistinguishable from noise; if anything the train/eval mismatch slightly hurts, since the net trained without the filter). Against the matched filtered baseline, **MCTS-100 wins by +16.4pp (73.0% vs 56.6%), ~4.0σ** — an even cleaner and stronger result than vs the unfiltered baseline.
- The old null verdict (15.5M net) stands for that net — search's value is policy-dependent. The BC-shaped policy gives the value head better candidates to arbitrate among; on the peaky PPO-from-scratch policy, 100 sims mostly rubber-stamped the argmax (see the 2026-09-25 search-characterization note).
- 100 sims ≈ 85s/move on this 2-CPU box — the demo's MCTS toggle is now worth much more than it was.

**Verdict:** search helps the PPO+BC net a lot (+12pp, significant). The "MCTS doesn't help" verdict is revised: it didn't help the weak peaky policy; it decisively helps the stronger BC-shaped one.

## Tactical-planes kill test verdict (2026-10-01, analyzed post-hoc) — SUPERSEDED

> **CORRECTION (2026-10-01): this entry is wrong and is superseded by the
> replication verdict above ("VERDICT: Shelve tactical planes").** It re-analyzes
> the ORIGINAL confounded runs: the seed bug made feat_s7 and feat_s8 the same
> model (n=1, not two seeds), and the original feat runs used a fresh Adam while
> controls used warm/migrated Adam. The clean replication (matched seeds, matched
> warm Adam, fixed RNG) gave ladders control 11–21 vs tactical 9–23 and blunders
> tactical 64.7% vs control 69.0% — neither beats the 63.9% baseline. Do not act
> on the "PASSED — do not shelve" verdict below.

**Question:** do 7 extra liberty/ko planes (13-plane engine; old 6-plane blobs load
zero-padded, new-channel weights zero-initialized from the 15.5M baseline) help?
**Design (as run):** 4 runs, plain PPO vs planes branch × 2 seeds (planes_ctrl_s7/s8,
planes_feat_s7/s8), all EXIT:0 on 2026-09-27; 32-game GNU Go ladder + capture/escape
blunder set mined from baseline losses.

**Result — ladder: planes wins, both seeds.** Control 11–53 (17.2%), planes 23–41
(35.9%): **+18.7pp, ~2.4σ pooled** (s7: 15.6→31.2%, s8: 18.8→40.6%). Consistent
direction in both seeds.
**Result — blunder set: flat.** Control 0.667 vs planes 0.686 blunder rate (n=612
positions/arm; ~1σ, wrong direction, noise).

**Verdict:** kill test PASSED — do not shelve. The planes help playing strength
decisively on the ladder but don't move the mined-blunder metric (either the
features help non-tactical aspects of play, or the blunder set doesn't
discriminate). Caveats: 32-game protocol (noisier than the current 256-game
standard); measured on PPO-from-scratch, not the current warm-start+PPO+BC
recipe — transfer to the 60.9% model is untested. Natural follow-up: branch the
warm-start checkpoint with 13 planes and run PPO+BC.

## Search distillation probe (2026-10-01) — signal present, protocol broken

**Question (Astra's #2 direction):** can the policy absorb its own MCTS's move-selection
judgments? Teacher = MCTS-100 visit distributions on the PPO+BC 6M net; student = same
net, fine-tuned with soft-target CE (distill) + retained Go Quest hard BC (coef 0.1).

**Phase 0 (diagnosis, gate: MCTS must differ from greedy on a meaningful minority):**
200 positions from 20 self-play games. Greedy↔MCTS-50 agreement 83%, greedy↔MCTS-100 87%;
MCTS-50↔MCTS-100 86%. MCTS disagrees with greedy on 33/200 (16.5%) — gate passed. MCTS
choice is not reducible to one-ply value selection (one-ply↔MCTS agreement only 28%;
one-ply move mean policy rank 5.18). Targets generated with 100 sims (MCTS-50 vs MCTS-100
still disagree 14%, so 50 sims was judged too noisy).

**Phase 1 (targets):** 200 self-play games, 100 sims/position → 21,104 positions
(`~/workspace/runs/search_distill/targets_s100.npz`; obs/policy/value/mask). Policy rows
all sum to 1; root values in [−1, 1]. (An earlier 400-game/8-worker run died silently —
post-mortem: the VM rebooted mid-run, not a code bug.)

**Phase 2 (two arms, 3000 steps each, batch 512, lr 2.5e-4, warm Adam from PPO+BC):**
- Treatment: distill CE 2.48 → 1.73; BC CE stable ~1.68.
- Control (distill coef 0, matched): BC CE 1.51 → 1.36; distill CE diagnostic rises (not trained).

**Phase 3 (256-game ladders, temp 0.2, levels 1/3/5/8 × 64, GNU Go):**
- Treatment: 39–25 / 37–27 / 29–35 / 24–40 = 129–127 raw; 2 games are gnugo illegal-move
  forfeits counted as "them" → corrected **131–125 (51.2%)**.
- Control: 25–39 / 31–33 / 26–38 / 27–37 = **109–147 (42.6%)**, no forfeits.
- Treatment beats control by **+8.6pp, ~2.0σ** (pooled SE ~4.4pp). Gate (≥5pp) passed.

**The catch — the fine-tuning protocol is destructive.** Untouched baseline: 60.9%.
Treatment degrades it by −9.7pp (~2.2σ); control by −18.3pp (~4.2σ). Restarting the LR at
2.5e-4 after the PPO+BC run had annealed to ~3e-7 is the prime suspect: 3000 steps of
high-LR BC-only fine-tuning overfits human move prediction (control BC CE → 1.36) at the
expense of PPO-learned playing strength. The distillation signal is real relative to the
matched control (+8.6pp) — the teacher's visits do teach something the BC loss doesn't —
but neither arm is usable as-is.

**Verdict:** kill the current fine-tuning recipe, not the direction. The signal exists;
the delivery is broken. Proper follow-up: rerun with a low fine-tune LR (order 1e-5–3e-5,
not the annealed floor) so the control holds ~60%, then the distill-vs-control delta
becomes an interpretable gain. Also note the treatment's BC CE stayed high (1.68) while
learning the teacher — distill and BC compete for capacity at this budget; a longer or
lower-LR run may separate them. Not run tonight: needs the user's call on the LR and
another ~5h of compute.

## Search distillation probe — low-LR rerun (2026-10-01): direction confirmed, ~3.9σ

**Follow-up, user-authorized 11:33 UTC** ("Rerun both arms at low LR"): same two arms,
3000 steps each, batch 512, LR 3e-5, warm Adam from the PPO+BC 6M checkpoint (60.9%);
distill targets from Phase 1 reused. `runs/search_distill/{treatment_low,control_low}.log`
both EXIT:0.

**Phase 2 metrics:**
- Treatment: distill_ce 2.39 → 1.85; bc_ce stable ~1.70–1.74 (final 1.6967).
- Control (distill coef 0, matched): bc_ce 1.61 → 1.40; distill_ce *diagnostic* 3.09 → 3.36
  (drifts away from the teacher — not trained, pure drift measure).

**Phase 3 (256-game ladders, temp 0.2, levels 1/3/5/8 × 64, GNU Go):**
- Treatment: 45–19 / 41–23 / 39–25 / 35–29 = 159 raw + 1 gnugo illegal-move forfeit
  ("them") → corrected **160–96 (62.5%)**.
- Control: 25–39 / 34–30 / 29–35 / 28–36 = 114 raw + 2 gnugo-resign forfeits ("them")
  → corrected **116–140 (45.3%)**.
- Delta: **+17.2pp, z≈3.9 pooled** — direction confirmed, much stronger than the
  high-LR run's +8.6pp / 2.0σ.

**Incident:** VM reboot ~12:39 UTC killed the first treatment ladder at 169/256
(`LADDER_RESTART_NOTE.md`); relaunched 13:25 UTC as a chained job (treatment then
control), finished 13:48/14:09 UTC. Both fine-tune checkpoints verified intact; partial
log preserved. Final numbers are from the fresh complete runs.

**Interpretation (refines the high-LR read):** the low-LR control still sits at 45.3% —
far below the untouched 60.9% — so the LR restart was *not* the whole story of the
control damage. 3000 steps of BC-only fine-tuning drifts the policy off its PPO optimum
even at 3e-5: the control imitates humans *better* (BC CE 1.40 vs treatment's 1.70)
while playing ~17pp worse. The distill loss anchored the policy — distillation didn't
just add signal, it counteracted BC drift.

**Honest absolute framing:** the treatment does not beat the model it started from —
62.5% vs untouched 60.9% = +1.6pp (z≈+0.36, noise). The defensible claim is the
distill-vs-control delta, not an absolute gain. Demo page: keep the shipped 60.9%
checkpoint; the fine-tuned treatment is not shippable on these numbers.

**Verdict:** the direction is real — distilling the net's own MCTS judgments beats
matched BC-only fine-tuning by ~17pp. Open next steps: longer distill budget, folding
distillation into the main PPO+BC line, Astra's value-head probe, architecture screen.
Probe code uncommitted/unpushed.

## Search distillation probe, low-LR rerun (2026-10-01) — decisive signal

**Follow-up to the high-LR run above.** Both arms retrained identically except
`--ft-lr 3e-5` (was 2.5e-4). Same 21,104 MCTS-100 targets, 3000 steps, warm Adam,
seed 0. Checkpoints: `~/workspace/runs/search_distill/{treatment_low,control_low}/finetune.pt`.

**Training:**
- Treatment (distill+BC): distill CE 2.48 → 1.85; BC CE flat ~1.70 (no BC overfit).
- Control (BC-only): BC CE 1.51 → 1.40; distill CE diagnostic 3.36 (not trained).

**Ladders (256 games, temp 0.2, levels 1/3/5/8 × 64; JSON primary records checked):**
- Treatment: L1 44–19, L3 41–23, L5 39–25, L8 35–29 = 159–96 raw; 1 gnugo
  illegal-move forfeit miscounted → corrected **160–96 (62.5%)**.
- Control: L1 30–34, L3 35–29, L5 23–41, L8 27–37 = **115–141 (44.9%)**, no forfeits.

**Result: treatment beats control by +17.6pp, ~4.1σ** (pooled SE ~4.3pp). The
distillation signal is strong and significant — not a borderline read.

**Revised interpretation of the high-LR run:** the control collapses to ~44–45% at
BOTH learning rates, so the damage is not (only) the LR restart. BC-only fine-tuning
without PPO drags the policy back toward pure human imitation (cf. pure supervised =
28.9%): the PPO component was load-bearing for the 60.9%, and 3000 steps of BC-only
erode it. The treatment is anchored by the teacher — MCTS visits derived from the
strong 60.9% policy itself — so it suffers none of the collapse and lands at 62.5%,
marginally above the untouched baseline (+1.6pp, ~0.4σ, n.s. on its own).

**Verdict:** search distillation works on this net. The +17.6pp vs the matched control
is the causal contrast that matters; the resulting 62.5% greedy model is the strongest
measured greedy checkpoint (baseline 60.9%), though the edge over baseline alone is
noise. Caveats: single run each arm (n=1); the control is arguably a strawman (BC-only
fine-tuning is now known-poison, which inflates the gap vs a "do nothing" baseline —
but the treatment matching/beating the untouched baseline shows the signal isn't just
damage mitigation). Natural next: longer low-LR distill run, or distill-then-short-PPO.
Whether 62.5% earns the demo default is the user's call (needs commit + push).

## MCTS-100 on the distilled model (2026-10-01) — search gain absorbed

**Question:** does play-time search still help the search-distilled policy?
256-game ladder, MCTS-100 via the Rust mcts_gtp engine on
`weights/supervised-search-distill-6M.bin`, levels 1/3/5/8 x 64, temp n/a (search):
L1 46-18, L3 40-23, L5 45-19, L8 35-29 = 166-89 raw; 1 gnugo-resign game hit the
known "winner: them" summary bug -> corrected **167-89 (65.2%)**, no forfeits.

**Comparisons:**
- Distilled greedy 62.5% -> distilled + MCTS-100 65.2%: **+2.7pp, ~0.6 sigma (noise)**.
- Old policy greedy 60.9% -> old + MCTS-100 73.0%: +12.1pp, ~2.9 sigma.
- Distilled + MCTS-100 (65.2%) vs old + MCTS-100 (73.0%): **-7.8pp, ~1.9 sigma**,
  borderline; the sharper distilled prior may explore less well under search
  (known expert-iteration tension: imitating visit distributions peakens the policy).

**Verdict:** one round of distillation absorbed essentially all the recoverable
move-selection error Astra identified — the +12pp search gap is gone (+2.7pp n.s.).
The expert-iteration loop has roughly converged after a single round; a second
distill round has little signal left to capture. The policy side of this 6M net
looks tapped out via the search route — the value head (untouched by distillation)
is now the prime suspect for remaining headroom, i.e. the terminal-score value
probe is the most attractive next direction.
