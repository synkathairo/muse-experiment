"""Self-play PPO training (Autodidact) — PLAN.md §5. No PufferLib.

Trains gotrain.net.GoNet from scratch: the learner plays both colors (alternating
per episode) against a frozen snapshot of its own policy, refreshed every K PPO
iterations. Pure +/-1 terminal reward; the value head learns win probability.

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
from .net import GoNet
from .ppo import PPOConfig, compute_gae, ppo_update, explained_variance
from .selfplay import SelfPlayGo, PASS

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
def sample_actions(policy, obs_t, masks_t):
    logits, values = policy(obs_t)
    dist = masked_dist(logits, masks_t)
    actions = dist.sample()
    return actions, dist.log_prob(actions), values.view(-1)


@torch.no_grad()
def greedy_actions(policy, obs_t, masks_t):
    """Argmax over legal moves (for eval)."""
    logits, _ = policy(obs_t)
    masked = logits.masked_fill(~masks_t, float("-inf"))
    return masked.argmax(dim=-1)


def random_opponent(obs, masks):
    """Uniform over legal moves. obs unused; signature matches opponent_fn."""
    B = masks.shape[0]
    actions = np.empty(B, dtype=np.int64)
    for b in range(B):
        legal = np.flatnonzero(masks[b])
        actions[b] = np.random.choice(legal)
    return actions


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


def greedy_capture_opponent(obs, masks):
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


def make_snapshot_opponent(snapshot_net, device, greedy=False):
    """opponent_fn playing the frozen snapshot (sampled, or greedy for eval)."""
    snapshot_net.eval()

    @torch.no_grad()
    def fn(obs, masks):
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)
        if greedy:
            a = greedy_actions(snapshot_net, obs_t, masks_t)
        else:
            a, _, _ = sample_actions(snapshot_net, obs_t, masks_t)
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


def load_ckpt(path, policy, optimizer, snapshot, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ck["model"])
    snapshot.load_state_dict(ck["opponent"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    np.random.set_state(ck["numpy_rng"])
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
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--opp-refresh-every", type=int, default=10,
                    help="PPO iterations between opponent snapshot refreshes")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="PPO iterations between evals (0 = off)")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="PPO iterations between latest.pt writes")
    ap.add_argument("--max-plies", type=int, default=243)
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

    policy = GoNet().to(device)
    snapshot = GoNet().to(device)   # frozen opponent; refreshed every K iters
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
    log(f"device={device} params={policy.param_count()}")

    env = SelfPlayGo(num_envs=args.num_envs, seed=args.seed,
                     max_plies=args.max_plies,
                     opponent_fn=make_snapshot_opponent(snapshot, device))
    obs = env.reset()  # (N,6,9,9) float32, learner to move

    T, N = args.rollout_steps, args.num_envs
    # Annealing schedule anchored to the ORIGINAL run length, so it stays
    # consistent across resumes instead of being recomputed from remaining steps
    # (which would also let ppo_iter exceed total_iters -> negative LR).
    total_iters = max(1, (args.total_steps + T * N - 1) // (T * N))
    t0 = time.time()
    step0 = step  # for honest steps/sec across resumes
    policy.train()

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
        ep_rews, ep_lens = [], []
        last_obs, last_terms = None, None

        for t in range(T):
            masks = env.legal_masks_learner()
            obs_t = torch.from_numpy(obs).to(device)
            masks_t = torch.from_numpy(masks).to(device)
            actions, logps, values = sample_actions(policy, obs_t, masks_t)

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
            f"clipfrac={stats['clipfrac']:.3f} ev={ev:.3f}{eval_str}")

    save_ckpt(os.path.join(args.out, "latest.pt"), policy, optimizer, snapshot,
              step, ppo_iter, snap_ptr, hparams)
    final_bin = os.path.join(args.out, "autodidact-final.bin")
    export_weights(policy.cpu(), final_bin)
    log(f"done at step {step}; fp16 export -> {final_bin} "
        f"({os.path.getsize(final_bin)} bytes)")
    logf.close()


if __name__ == "__main__":
    main()
