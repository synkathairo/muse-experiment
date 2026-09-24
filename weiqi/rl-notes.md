# RL recon notes — PufferLib Ocean Go env (2026-09-24)

**Verdict up front: do NOT use PufferLib's Ocean Go env for the Autodidact.**
It cannot do self-play (hardcoded heuristic opponent), its rules deviate from our
demo ruleset, and its observation doesn't match our 6 planes. See below.
Recommended instead: small custom Python self-play env (design sketched at bottom),
trained on better hardware (user machine / Colab GPU).

## What was examined

PufferLib 3.0.0 sdist (from PyPI), files:
- `pufferlib/ocean/go/go.h` — C game logic (~700 lines)
- `pufferlib/ocean/go/go.py` — Python PufferEnv wrapper
- `pufferlib/ocean/go/binding.c` — C binding glue
- `pufferlib/pufferl.py`, `pufferlib/models.py` — PPO trainer + policy API
- `pufferlib/config/ocean/go.ini` — reference PPO config (targets 100M steps)

## Env facts

- **Board size**: `grid_size` param, **default 7** (go.py:17). Settable to 9; verified
  working at 9×9 (obs shape (164,), 82 actions). Render code assumes 9 for layout
  but that's display-only.
- **Agent structure**: SINGLE-AGENT. Learner is always Black (player 1). After every
  learner move, `enemy_greedy_hard()` (go.h:555, called at go.h:675 and :689) plays
  White. There is no way to control White — **self-play is impossible** with this env.
  `enemy_greedy_easy` (go.h:591) exists but is never called.
- **Observation** (go.h:182 `compute_observations`): flat float32,
  `2*grid_size² + 2` = 164 floats at 9×9. Plane 0 = own stones, plane 1 = opponent
  stones (always from Black's perspective), then 2 scalars = capture counts.
  **Not our 6 planes** (no last-move, no ko point, no color plane).
- **Actions**: `Discrete(grid_size² + 1)` = 82 at 9×9 (go.py:37). Index 0 = pass,
  1..81 = points (point = action-1). Matches our 82-way policy by luck.
- **Scoring**: Tromp-Taylor-ish area scoring (go.h:232 `compute_score_tromp_taylor`:
  stones + surrounded empty points), komi 7.5 default, score from Black's perspective.
- **Ko** (go.h:410 `is_ko`): compares post-move board to pre-move board, but only
  checked when a capture occurred AND `previous_board_state` is only refreshed on
  the *learner's* move — the enemy's ko recaptures are checked against a stale board.
  Buggy for White; approximately right for Black.
- **Suicide**: illegal (go.h `make_move` returns 0 on zero liberties).
- **Game end**: NO two-pass termination. Ends at `tick > 3*grid_size²` (243 plies at
  9×9) or when the enemy has no legal move (go.h:660 `c_step`).
- **Pass**: action 0 = skip turn, does NOT end game, penalized −0.25 by default.
  A net trained here would learn passing is bad — bad habit for real Go.
- **Rewards**: dense shaping, NOT pure win/loss (defaults go.py:24-29):
  valid move +0.1, invalid −0.1, pass −0.25, capture +0.25/opponent capture −0.25.
  Terminal step reward is overwritten to ±1/0 (go.h:645 `end_game`).
  Reference sweep (go.ini) uses different shaped values (e.g. pass −0.60).
- **Custom policy API**: `pufferl.train(env_name, vecenv=..., policy=...)`; any
  `nn.Module` with `forward_eval(obs, state) -> (logits, values)`. No base class.
  (models.py docstring: "PufferLib is not a framework.")

## Benchmarks (this box: 2 CPUs, 7GB RAM; contended — parallel training agent
## running, so treat as conservative lower bounds)

| what | measured |
|---|---|
| Our net (130,522 params ✓) fwd, batch 512 | ~1,700–3,900 samples/sec |
| Our net fwd+bwd, batch 512 | ~1,165 samples/sec |
| Our net fwd, batch 64 (rollout batch size) | ~1,183 samples/sec |
| Our net fwd, batch 1 (in-env opponent inference) | ~238 samples/sec |
| Ocean Go env, 9×9, random acts, 1 env | 62,261 env-steps/sec |
| Ocean Go env, 9×9, random acts, 64 envs | 260,985 env-steps/sec |

Each env "step" = learner move + heuristic opponent move (both sides simulated).

## Throughput model (PPO, 64 envs × 256 steps = 16,384 samples/iter, 4 epochs)

- Rollout inference: 16,384 / 1,183 ≈ 14 s
- Env stepping (Ocean C env): 16,384 / 261,000 ≈ 0.1 s — negligible
- Learning: 4 × 16,384 / 1,165 ≈ 56 s
- **≈ 70 s / 16,384 steps ≈ 234 env-steps/sec ≈ 20M steps/day ≈ 60M per 3 days**

The bottleneck is the net on CPU, not the env. A pure-Python custom env
(est. 5–25k steps/sec — still ≪ net cost) would not change this materially.

PufferLib's own go.ini targets 100M steps → 4.3 days here. Over the 3-day rule
even before questioning sample efficiency.

## Recommended self-play design (custom env, "S1")

Single-agent PufferEnv in Python (~250 lines), NOT the Ocean env:
- Exact §3 planes (6×9×9, color-relative), exact §6 rules (Tromp-Taylor, simple ko,
  two-pass end, 7.5 komi, suicide illegal).
- Agent assigned a color per episode (alternate); opponent = frozen snapshot of the
  policy, refreshed every K PPO iterations. Obs always from agent's color.
- Reward: 0 intermediate, ±1 at game end from agent's perspective. No shaping
  (or tiny capture shaping if value learning stalls — decide empirically).
- Policy: our arch wrapped with `forward_eval`. Train via `pufferl.train` with
  custom vecenv+policy, or a minimal hand-rolled PPO loop (~150 lines, CleanRL-style)
  for full control of snapshot updates and checkpointing.
- Cost of in-env batch-1 opponent inference: ~69 s per 16k-step rollout at 238/s
  → effective ~10–12M steps/day here. A custom rollout loop batching opponent
  inference (2× batch-64 forwards) recovers ~17M/day.
- Scaffolding estimate: 300–500 lines total, 1–2 days of focused work.

## Decision

- **Not the Ocean env** (wrong artifact: no self-play, wrong rules, wrong obs).
- **Here**: custom S1 env could run ~30–50M steps in 3 days — expect coherent
  beginner Go (legal, captures, finishes games), weak strategically. Honest but
  modest; borderline vs the 3-day rule.
- **Recommended**: run S1 on the user's hardware or a free Colab GPU. On even a
  modest GPU the tiny net's fwd+bwd goes ~30–50× faster, making 100–200M steps an
  hours-long run. That's where the real Autodidact should train.
