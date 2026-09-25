"""Self-play PPO training (Autodidact) — PLAN.md §5. No PufferLib.

Trains gotrain.net.GoNet from scratch: the learner plays both colors (alternating
per episode) against a frozen snapshot of its own policy, refreshed every K PPO
iterations. Pure +/-1 terminal reward; the value head learns win probability.
Opening exploration: Dirichlet noise is mixed into both sides' sampling for the
first --dirichlet-plies plies of each game (AlphaZero-style), which keeps the
opening from collapsing to a single memorized line.

Checkpoint discipline mirrors train_cloning.py:
  - rolling `latest.pt` (model + optimizer + opponent snapshot + step + RNG),
    every --ckpt-every PPO iterations;
  - frozen log-spaced snapshots `snap_{env_steps}.pt` in ENV STEPS
    (1k, 3k, 10k, 30k, ...) — never overwritten, feed the time-machine.
  - fp16 export of the final model at the end (gotrain.export).

An "env step" here = one learner move across one env (one PPO sample).
Opponent moves are environment dynamics, not counted.

Resume:  python -m gotrain.train_selfplay --out <dir> --resume <dir>/latest.pt
         (all other flags are re-read from the checkpoint's hparams)

Full runs (--device defaults to auto: cuda > mps > cpu; pass it explicitly
only to override, e.g. --device cpu when debugging a backend quirk):

  # Free Colab GPU (T4) — ~100-200M steps ≈ 4-8 h:
  python -m gotrain.train_selfplay --out runs/auto_v1 --num-envs 64 \\
      --total-steps 200000000 --rollout-steps 256

  # Apple Silicon (PyTorch MPS — no MLX port needed):
  python -m gotrain.train_selfplay --out runs/auto_v1 --num-envs 64 \\
      --total-steps 200000000 --rollout-steps 256

  # CPU pilot (this box, 2 vCPUs) — plumbing validation only, a few M steps:
  python -m gotrain.train_selfplay --out runs/auto_pilot --num-envs 32 \\
      --total-steps 2000000 --rollout-steps 128 --device cpu \\
      --opp-refresh-every 10 --eval-every 20 --ckpt-every 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .export import export_weights
from .net import GoNet, N_POINTS
from .net_aux import (GoNetAux, load_trunk_from_gonet, check_trunk_resume,
                      to_gonet_state_dict)
from .ppo import PPOConfig, compute_gae, ppo_update, explained_variance
from .selfplay import (SelfPlayGo, PASS, set_komi, ownership_labels,
                       margin_label, to_learner_perspective)

# Frozen museum snapshots, log-spaced in env steps (cf. train_cloning.SNAP_STEPS,
# which is in gradient steps — here the natural unit is env steps / PPO samples).
SNAP_STEPS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000,
              3000000, 10000000, 30000000, 100000000, 300000000]


# ---------------------------------------------------------------------------
# masked action sampling
# ---------------------------------------------------------------------------
def masked_dist(logits, masks):
    """Categorical over legal moves only (illegal logits -> -inf -> 0 mass)."""
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~masks, float("-inf")))


@torch.no_grad()
def dirichlet_noised_dist(dist, masks, noise_mask, alpha, eps):
    """AlphaZero-style opening exploration: P' = (1-eps)*P + eps*Dir(alpha).

    The noise is supported on legal moves only and is mixed in per-row for
    rows where noise_mask is True; other rows keep the policy's distribution.
    Works even on a fully collapsed (delta) policy, where temperature
    scaling would be a no-op, because the noise injects mass independently
    of the policy's output.
    """
    probs = dist.probs
    d = torch.distributions.Dirichlet(
        torch.full((probs.shape[-1],), alpha, device=probs.device)).sample((probs.shape[0],))
    d = d.masked_fill(~masks, 0.0)
    d = d / d.sum(-1, keepdim=True).clamp_min(1e-12)
    mixed = (1.0 - eps) * probs + eps * d
    mixed = mixed.masked_fill(~masks, 0.0)
    mixed = mixed / mixed.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.where(noise_mask.unsqueeze(-1), mixed, probs)
    return torch.distributions.Categorical(probs=out)


@torch.no_grad()
def sample_actions(policy, obs_t, masks_t, noise_mask=None,
                   dirichlet_alpha=0.05, dirichlet_eps=0.25):
    logits, values = policy(obs_t)
    dist = masked_dist(logits, masks_t)
    if noise_mask is not None and bool(noise_mask.any()):
        dist = dirichlet_noised_dist(dist, masks_t, noise_mask,
                                     dirichlet_alpha, dirichlet_eps)
    actions = dist.sample()
    # logps are under the *behavior* distribution (noise included), which is
    # what PPO's importance ratio requires.
    return actions, dist.log_prob(actions), values.view(-1)


@torch.no_grad()
def greedy_actions(policy, obs_t, masks_t):
    """Argmax over legal moves (for eval)."""
    logits, _ = policy(obs_t)
    masked = logits.masked_fill(~masks_t, float("-inf"))
    return masked.argmax(dim=-1)


def aux_update(policy, optimizer, cfg, obs, own_tgt, margin_tgt,
               own_w, margin_w, epochs):
    """Supervised auxiliary update on finished-game labels (KataGo-style).

    obs: (B,6,9,9); own_tgt: (B,81) in {-1,0,1} from the side-to-move's
    perspective; margin_tgt: (B,) = (my_score - opp_score)/81. Only steps
    whose game finished inside the rollout carry labels (the caller filters).
    Runs as a separate phase after the PPO update so the clipped trust
    region never sees the supervised gradients. Returns mean losses.
    """
    policy.train()
    B = obs.shape[0]
    acc_own = acc_margin = 0.0
    n = 0
    for _ in range(epochs):
        perm = torch.randperm(B, device=obs.device)
        for s in range(0, B, cfg.minibatch_size):
            idx = perm[s:s + cfg.minibatch_size]
            _, _, own, margin = policy.forward_aux(obs[idx])
            own_loss = F.mse_loss(own, own_tgt[idx])
            margin_loss = F.mse_loss(margin, margin_tgt[idx])
            loss = own_w * own_loss + margin_w * margin_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.max_grad_norm)
            optimizer.step()
            acc_own += own_loss.item()
            acc_margin += margin_loss.item()
            n += 1
    return {"aux_own": acc_own / max(1, n),
            "aux_margin": acc_margin / max(1, n)}


def random_opponent(obs, masks, game_plies=None):
    """Uniform over legal moves. obs unused; signature matches opponent_fn."""
    B = masks.shape[0]
    actions = np.empty(B, dtype=np.int64)
    for b in range(B):
        legal = np.flatnonzero(masks[b])
        actions[b] = np.random.choice(legal)
    return actions


def attach_finished_game_labels(env, i, ep_step_idxs, b_own, b_margin,
                                 b_aux_valid):
    """Compute ownership/margin labels for env i's finished game and attach
    them to the game's rollout steps.

    Must be called BEFORE env.reset(i): reads env.boards[i] (the terminal
    position) and env.learner_color[i] (still the finished episode's color).
    b_own/b_margin/b_aux_valid are the (T, N[, 81]) rollout buffers.
    """
    own_abs = ownership_labels(env.boards[i])
    mgn_abs = margin_label(env.boards[i])
    own, mgn = to_learner_perspective(own_abs, mgn_abs,
                                      int(env.learner_color[i]))
    idx = torch.tensor(list(ep_step_idxs), dtype=torch.long)
    b_own[idx, i] = torch.from_numpy(own)
    # normalize by board size so the margin target is O(1)
    b_margin[idx, i] = float(mgn) / N_POINTS
    b_aux_valid[idx, i] = True


def _captures_if(own, opp, r, c):
    """Stones captured by playing (r, c): adjacent opponent groups whose only
    liberty is (r, c). `own`/`opp` are (9,9) bool arrays from the side to move's
    perspective. Only called on legal moves (suicide/ko already masked out)."""
    caps = 0
    seen = np.zeros((9, 9), dtype=bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 9 and 0 <= nc < 9) or not opp[nr, nc] or seen[nr, nc]:
            continue
        stack, stones, libs = [(nr, nc)], 0, set()
        seen[nr, nc] = True
        while stack:
            sr, sc = stack.pop()
            stones += 1
            for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ar, ac = sr + dr2, sc + dc2
                if 0 <= ar < 9 and 0 <= ac < 9:
                    if not own[ar, ac] and not opp[ar, ac]:
                        if (ar, ac) != (r, c):
                            libs.add((ar, ac))
                    elif opp[ar, ac] and not seen[ar, ac]:
                        seen[ar, ac] = True
                        stack.append((ar, ac))
        if not libs:
            caps += stones
    return caps


def greedy_capture_opponent(obs, masks, game_plies=None):
    """1-ply greedy tactical bot: maximizes immediate stones captured (random
    tie-break, pass loses ties); with nothing to capture, plays a random
    non-pass move. First rung above random on the eval ladder."""
    B = masks.shape[0]
    actions = np.empty(B, dtype=np.int64)
    for b in range(B):
        own = obs[b, 0] > 0.5
        opp = obs[b, 1] > 0.5
        legal = np.flatnonzero(masks[b])
        best, best_caps = [], -1
        for i in legal:
            caps = -1 if i == 81 else _captures_if(own, opp, i // 9, i % 9)
            if caps > best_caps:
                best, best_caps = [i], caps
            elif caps == best_caps:
                best.append(i)
        actions[b] = np.random.choice(best)
    return actions


def make_snapshot_opponent(snapshot_net, device, greedy=False,
                           dirichlet_plies=0, dirichlet_alpha=0.05,
                           dirichlet_eps=0.25):
    """opponent_fn playing the frozen snapshot (sampled, or greedy for eval).

    When not greedy, Dirichlet noise is mixed into the first `dirichlet_plies`
    plies of each game (AlphaZero-style opening exploration); the env passes
    per-game ply counts as `game_plies`. Greedy eval never gets noise.
    """
    snapshot_net.eval()

    @torch.no_grad()
    def fn(obs, masks, game_plies=None):
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)
        if greedy:
            a = greedy_actions(snapshot_net, obs_t, masks_t)
        else:
            noise_mask = None
            if game_plies is not None and dirichlet_plies > 0:
                noise_mask = torch.from_numpy(
                    np.asarray(game_plies) < dirichlet_plies).to(device)
            a, _, _ = sample_actions(snapshot_net, obs_t, masks_t,
                                     noise_mask=noise_mask,
                                     dirichlet_alpha=dirichlet_alpha,
                                     dirichlet_eps=dirichlet_eps)
        return a.cpu().numpy().astype(np.int64)

    return fn


# ---------------------------------------------------------------------------
# evaluation: learner (greedy) vs an opponent_fn, alternating colors
# ---------------------------------------------------------------------------
def _tally_finished(done_idxs, rewards, wins, played, n_games):
    """Fold newly finished games into (wins, played), capping at n_games.

    Several envs can finish on the same step; without the cap, `played` can
    overshoot n_games and the win rate would average over more games than
    requested. Returns (wins, played, counted_idxs).
    """
    done_idxs = np.asarray(done_idxs)
    if played < n_games:
        done_idxs = done_idxs[: n_games - played]
    for i in done_idxs:
        played += 1
        if rewards[i] > 0:
            wins += 1
    return wins, played, done_idxs


@torch.no_grad()
def evaluate(policy, opponent_fn, n_games, device, seed=12345, max_plies=None):
    """Win rate of the greedy learner vs opponent_fn (learner alternates color)."""
    env = SelfPlayGo(num_envs=n_games, seed=seed,
                     max_plies=max_plies or 3 * 9 * 9, opponent_fn=opponent_fn)
    # stagger starting colors: without this every env's first game has the
    # learner as Black, and greedy-vs-greedy self-play is deterministic, so the
    # "win rate" would really be one game repeated n_games times.
    env.color_counter[:] = np.arange(n_games) % 2
    obs = env.reset()
    wins = 0
    played = 0
    while played < n_games:
        masks = env.legal_masks_learner()
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)
        actions = greedy_actions(policy, obs_t, masks_t).cpu().numpy()
        obs, rewards, dones, terms, ep_lens = env.step(actions)
        # snapshot before reset: env.reset() clears env.done; and cap the
        # count at exactly n_games (several envs can finish on one step).
        done_idxs = np.where(dones)[0]
        wins, played, counted = _tally_finished(done_idxs, rewards, wins,
                                               played, n_games)
        if len(counted):
            obs[counted] = env.reset(counted)
    return wins / max(1, played)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_ckpt(path, policy, optimizer, snapshot, step, ppo_iter, snap_ptr, hparams):
    torch.save({
        "step": step,               # env steps (learner moves) so far
        "ppo_iter": ppo_iter,
        "snap_ptr": snap_ptr,
        "model": policy.state_dict(),
        "opponent": snapshot.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "hparams": hparams,
    }, path)


def _load_model_flexible(model, sd):
    """Load a checkpoint model dict into a GoNetAux.

    Accepts plain GoNet dicts (e.g. the 15.5M run) via trunk mapping — aux
    heads keep their random init. Plain GoNet targets load plain dicts
    untouched (existing save/load round-trip tests). Returns True when
    trunk-mapped.
    """
    is_aux_target = any(k.startswith("trunk.") for k in model.state_dict())
    sd_is_plain = ("conv1.weight" in sd
                   and not any(k.startswith("trunk.") for k in sd))
    if is_aux_target and sd_is_plain:
        res = load_trunk_from_gonet(model, sd)
        check_trunk_resume(res.missing_keys)
        if res.unexpected_keys:
            raise RuntimeError(
                f"unexpected keys in trunk resume: {res.unexpected_keys}")
        return True
    model.load_state_dict(sd)
    return False


def load_ckpt(path, policy, optimizer, snapshot, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    trunk_mapped = _load_model_flexible(policy, ck["model"])
    _load_model_flexible(snapshot, ck["opponent"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    np.random.set_state(ck["numpy_rng"])
    if trunk_mapped:
        print("trunk-mapped resume from a plain GoNet checkpoint "
              "(e.g. the 15.5M run); aux heads randomly initialized",
              flush=True)
    return ck


def explicit_cli_flags(argv=None):
    """--flag names (underscore form) explicitly present on the command line."""
    out = set()
    for tok in (sys.argv[1:] if argv is None else argv):
        if tok.startswith("--"):
            out.add(tok[2:].split("=")[0].replace("-", "_"))
    return out


def apply_resumed_hparams(args, ck, explicit=None):
    """Restore a resumed run's hyperparameters from its checkpoint.

    --out and --device always stay as given on the CLI (run directory and
    launch environment are the resumer's choice). Any other flag explicitly
    passed on the CLI also wins -- e.g. --total-steps 3000000 extends the
    run instead of being silently reverted to the checkpoint's value (which
    made a resumed run exit immediately with "done at step 200704").
    Everything not explicitly given reverts to the checkpoint's values:
    keeping argparse defaults here corrupts the LR schedule (it can even go
    negative -> gradient ascent, destroying the policy in one iter).
    Returns the restored hparams dict.
    """
    if explicit is None:
        explicit = explicit_cli_flags()
    for k, v in ck.get("hparams", {}).items():
        if k in ("out", "device") or k in explicit:
            continue
        setattr(args, k, v)
    return {k: v for k, v in vars(args).items() if k != "resume"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def resolve_device(name):
    """Map a --device name to torch.device.

    'auto' picks cuda when available, else mps, else cpu. Explicit names are
    validated so a typo'd or unavailable backend fails fast instead of
    silently training on the wrong device.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    mps = getattr(torch.backends, "mps", None)
    if name == "mps" and not (mps is not None and mps.is_available()):
        raise SystemExit("--device mps requested but not available on this machine")
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir (checkpoints + train.log)")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--total-steps", type=int, default=2000000,
                    help="env steps (learner moves); 100-200M for the real run")
    ap.add_argument("--rollout-steps", type=int, default=128,
                    help="learner moves per env per PPO iteration")
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--dirichlet-plies", type=int, default=12,
                    help="opening plies per game with Dirichlet noise mixed into "
                         "both sides' sampling (0 disables)")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.05,
                    help="Dirichlet concentration for opening noise")
    ap.add_argument("--dirichlet-eps", type=float, default=0.25,
                    help="noise mixture weight for opening noise")
    ap.add_argument("--ownership", action="store_true",
                    help="enable KataGo-style auxiliary heads (Wu 2019): "
                         "per-point ownership + score-margin targets from "
                         "finished self-play games, trained as extra MSE "
                         "losses after each PPO update. Off = plain "
                         "policy/value PPO on the identical trunk.")
    ap.add_argument("--aux-own-w", type=float, default=0.5,
                    help="MSE weight for the ownership head (tanh outputs, "
                         "targets in {-1,0,1} from the side to move's "
                         "perspective)")
    ap.add_argument("--aux-margin-w", type=float, default=0.5,
                    help="MSE weight for the score-margin head (target = "
                         "(my_score - opp_score)/81)")
    ap.add_argument("--aux-epochs", type=int, default=2,
                    help="supervised passes over labeled rollout steps per "
                         "PPO iteration")
    ap.add_argument("--opp-refresh-every", type=int, default=10,
                    help="PPO iterations between opponent snapshot refreshes")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="PPO iterations between evals (0 = off)")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="PPO iterations between latest.pt writes")
    ap.add_argument("--max-plies", type=int, default=243)
    ap.add_argument("--train-komi", type=float, default=7.5,
                    help="komi for self-play game rewards (default 7.5). "
                         "6.5 approximates fair komi on 9x9 and gives Black "
                         "a balanced win rate; eval/benchmark komi is separate. "
                         "Explicitly passed value wins on --resume.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="compute device; 'auto' picks cuda > mps > cpu")
    ap.add_argument("--resume", default=None, help="path to latest.pt")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    cfg = PPOConfig(lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                    clip_coef=args.clip_coef, vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
                    update_epochs=args.update_epochs,
                    minibatch_size=args.minibatch_size)

    # Always GoNetAux (locked GoNet trunk + training-only heads): one code
    # path for both legs, so the ownership experiment isolates the learning
    # signal. With --ownership off the aux heads get no gradients and the
    # run is plain policy/value PPO on the identical trunk.
    policy = GoNetAux().to(device)
    snapshot = GoNetAux().to(device)   # frozen opponent; refreshed every K iters
    snapshot.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    hparams = {k: v for k, v in vars(args).items() if k != "resume"}

    step = 0          # env steps (learner moves) completed
    ppo_iter = 0
    snap_ptr = 0
    if args.resume:
        ck = load_ckpt(args.resume, policy, optimizer, snapshot, device)
        step = ck["step"]
        ppo_iter = ck["ppo_iter"]
        snap_ptr = ck["snap_ptr"]
        prev_komi = ck.get("hparams", {}).get("train_komi", 7.5)
        hparams = apply_resumed_hparams(args, ck)
        # rebuild the PPO config from the restored hyperparameters
        cfg = PPOConfig(lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        clip_coef=args.clip_coef, vf_coef=args.vf_coef,
                        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
                        update_epochs=args.update_epochs,
                        minibatch_size=args.minibatch_size)
        print(f"resumed from {args.resume} at step {step} (iter {ppo_iter})", flush=True)

    logf = open(os.path.join(args.out, "train.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"start: {json.dumps(hparams)} resuming_at={step}")
    log(f"device={device} params={policy.param_count()} "
        f"(trunk {policy.trunk.param_count()}, locked GoNet spec)")
    log(f"ownership_aux={args.ownership} aux_own_w={args.aux_own_w} "
        f"aux_margin_w={args.aux_margin_w} aux_epochs={args.aux_epochs}")

    set_komi(args.train_komi)
    log(f"train_komi={args.train_komi}")
    if args.resume and abs(args.train_komi - prev_komi) > 1e-9:
        log(f"NOTE: train komi changed {prev_komi} -> {args.train_komi} on resume; "
            f"value head was calibrated to {prev_komi} and will recalibrate")

    env = SelfPlayGo(num_envs=args.num_envs, seed=args.seed,
                     max_plies=args.max_plies,
                     opponent_fn=make_snapshot_opponent(
                         snapshot, device,
                         dirichlet_plies=args.dirichlet_plies,
                         dirichlet_alpha=args.dirichlet_alpha,
                         dirichlet_eps=args.dirichlet_eps))
    obs = env.reset()  # (N,6,9,9) float32, learner to move

    T, N = args.rollout_steps, args.num_envs
    # Annealing schedule anchored to the ORIGINAL run length, so it stays
    # consistent across resumes instead of being recomputed from remaining steps
    # (which would also let ppo_iter exceed total_iters -> negative LR).
    total_iters = max(1, (args.total_steps + T * N - 1) // (T * N))
    t0 = time.time()
    step0 = step  # for honest steps/sec across resumes
    policy.train()

    # Auxiliary labels only exist for games that finish inside the rollout;
    # when --ownership is off (or both weights are 0) skip the bookkeeping.
    aux_on = args.ownership and (args.aux_own_w > 0 or args.aux_margin_w > 0)

    while step < args.total_steps:
        # ---- rollout ------------------------------------------------------
        b_obs = torch.zeros(T, N, 6, 9, 9)
        b_masks = torch.zeros(T, N, PASS + 1, dtype=torch.bool)
        b_actions = torch.zeros(T, N, dtype=torch.int64)
        b_logps = torch.zeros(T, N)
        b_rewards = torch.zeros(T, N)
        b_terms = torch.zeros(T, N, dtype=torch.bool)
        b_truncs = torch.zeros(T, N, dtype=torch.bool)
        b_values = torch.zeros(T, N)
        # per-step aux targets, filled at game end (labels are game-final)
        b_own = torch.zeros(T, N, N_POINTS)
        b_margin = torch.zeros(T, N)
        b_aux_valid = torch.zeros(T, N, dtype=torch.bool)
        # rollout-step indices belonging to each env's current episode
        ep_steps = [[] for _ in range(N)]
        ep_rews, ep_lens = [], []
        last_obs, last_terms = None, None

        for t in range(T):
            for i in range(N):
                ep_steps[i].append(t)
            masks = env.legal_masks_learner()
            obs_t = torch.from_numpy(obs).to(device)
            masks_t = torch.from_numpy(masks).to(device)
            # Dirichlet opening noise for both colors' early plies: env.plies
            # is the upcoming move's ply index in each env (exact across
            # resets and color alternation, since the env owns the counters).
            noise_mask = None
            if args.dirichlet_plies > 0:
                noise_mask = torch.from_numpy(
                    env.plies < args.dirichlet_plies).to(device)
            actions, logps, values = sample_actions(
                policy, obs_t, masks_t, noise_mask=noise_mask,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps)

            b_obs[t] = obs_t.cpu()
            b_masks[t] = masks_t.cpu()
            b_actions[t] = actions.cpu()
            b_logps[t] = logps.cpu()
            b_values[t] = values.cpu()

            obs, rewards, dones, terms, lens = env.step(actions.cpu().numpy())
            b_rewards[t] = torch.from_numpy(rewards)
            b_terms[t] = torch.from_numpy(terms)
            b_truncs[t] = torch.from_numpy(dones & ~terms)
            for i in np.where(dones)[0]:
                ep_rews.append(float(rewards[i]))
                ep_lens.append(int(lens[i]))
                if aux_on:
                    # labels from the finished board — BEFORE the reset below.
                    # env.learner_color[i] is still the finished episode's
                    # color, so the perspective conversion is exact.
                    attach_finished_game_labels(env, i, ep_steps[i],
                                                b_own, b_margin, b_aux_valid)
                ep_steps[i] = []
            if t == T - 1:
                # keep the pre-reset final obs: the bootstrap value must come
                # from the actual final position, not a reset one. (Scored
                # max-ply endings are episodic terminals for GAE, so their
                # bootstrap is zeroed below along with true terminals.)
                last_obs, last_terms = obs.copy(), terms.copy()
                last_truncs = (dones & ~terms).copy()
            if np.any(dones):
                fresh = env.reset(np.where(dones)[0])
                obs[dones] = fresh

        # value bootstrap: V(final position) for live envs; 0 for true
        # terminals (two passes) and scored max-ply endings -- both are
        # episodic terminals for GAE (see gotrain.ppo.compute_gae).
        with torch.no_grad():
            next_values = policy(torch.from_numpy(last_obs).to(device))[1].cpu()
        final_done = torch.from_numpy(last_terms) | torch.from_numpy(last_truncs)
        next_values = next_values * (~final_done).float()

        advantages, returns = compute_gae(
            b_rewards, b_values, b_terms, b_truncs, next_values,
            torch.from_numpy(last_terms), torch.from_numpy(last_truncs),
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        ev = explained_variance(b_values.numpy(), returns.numpy())

        # ---- PPO update ----------------------------------------------------
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        # Clamp at 0: on a resumed run ppo_iter can reach total_iters, and a
        # negative LR is gradient ASCENT -- it destroys the policy in one step.
        lr_now = cfg.lr * max(0.0, 1.0 - ppo_iter / total_iters) if cfg.anneal_lr else cfg.lr
        stats = ppo_update(policy, optimizer, cfg,
                           flat(b_obs).to(device), flat(b_actions).to(device),
                           flat(b_logps).to(device), flat(advantages).to(device),
                           flat(returns).to(device), flat(b_values).to(device),
                           flat(b_masks).to(device), lr_now=lr_now)

        # ---- auxiliary ownership/margin update -------------------------------
        # Separate phase after PPO: the clipped trust region never sees the
        # supervised gradients. Only steps whose game finished in-rollout
        # carry labels (b_aux_valid); the final partial games are excluded.
        aux_str = ""
        if aux_on:
            valid = b_aux_valid.reshape(-1)
            if bool(valid.any()):
                vdev = valid.to(device)
                aux_stats = aux_update(
                    policy, optimizer, cfg,
                    flat(b_obs).to(device)[vdev],
                    flat(b_own).to(device)[vdev],
                    flat(b_margin).to(device)[vdev],
                    args.aux_own_w, args.aux_margin_w, args.aux_epochs)
                aux_str = (f" aux_own={aux_stats['aux_own']:.4f}"
                           f" aux_mgn={aux_stats['aux_margin']:.4f}")

        step += T * N
        ppo_iter += 1

        # ---- opponent refresh ----------------------------------------------
        if ppo_iter % args.opp_refresh_every == 0:
            snapshot.load_state_dict(policy.state_dict())

        # ---- snapshots -------------------------------------------------------
        while snap_ptr < len(SNAP_STEPS) and step >= SNAP_STEPS[snap_ptr]:
            sp = os.path.join(args.out, f"snap_{SNAP_STEPS[snap_ptr]:09d}.pt")
            save_ckpt(sp, policy, optimizer, snapshot, step, ppo_iter,
                      snap_ptr + 1, hparams)
            log(f"frozen snapshot -> {sp}")
            snap_ptr += 1
        if ppo_iter % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.out, "latest.pt"), policy, optimizer,
                      snapshot, step, ppo_iter, snap_ptr, hparams)

        # ---- eval ------------------------------------------------------------
        eval_str = ""
        if args.eval_every and ppo_iter % args.eval_every == 0:
            wr_rand = evaluate(policy, random_opponent, args.eval_games, device)
            wr_greedy = evaluate(policy, greedy_capture_opponent,
                                 args.eval_games, device)
            wr_snap = evaluate(policy, make_snapshot_opponent(snapshot, device,
                                                             greedy=True),
                               args.eval_games, device)
            eval_str = (f" eval_vs_random={wr_rand:.2f}"
                        f" eval_vs_greedy={wr_greedy:.2f}"
                        f" eval_vs_snapshot={wr_snap:.2f}")

        sps = (step - step0) / (time.time() - t0)
        ep_r = f"{np.mean(ep_rews):+.3f}" if ep_rews else "n/a"
        ep_l = f"{np.mean(ep_lens):.0f}" if ep_lens else "n/a"
        log(f"iter {ppo_iter}: step {step} sps={sps:.0f} ep_rew={ep_r} "
            f"ep_len={ep_l} pg={stats['pg_loss']:.4f} v={stats['v_loss']:.4f} "
            f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
            f"clipfrac={stats['clipfrac']:.3f} ev={ev:.3f}{aux_str}{eval_str}")

    save_ckpt(os.path.join(args.out, "latest.pt"), policy, optimizer, snapshot,
              step, ppo_iter, snap_ptr, hparams)
    final_bin = os.path.join(args.out, "autodidact-final.bin")
    # export the locked trunk only: the fp16 format (and the Rust/WASM demo
    # path) never sees the training-only aux heads.
    export_weights(policy.trunk.cpu(), final_bin)
    log(f"done at step {step}; fp16 export -> {final_bin} "
        f"({os.path.getsize(final_bin)} bytes)")
    logf.close()


if __name__ == "__main__":
    main()
