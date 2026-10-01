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
  python -m gotrain.train_selfplay --out runs/auto_v1 --num-envs 64 \\\\
      --total-steps 200000000 --rollout-steps 256

  # Apple Silicon (PyTorch MPS — no MLX port needed):
  python -m gotrain.train_selfplay --out runs/auto_v1 --num-envs 64 \\\\
      --total-steps 200000000 --rollout-steps 256

  # CPU pilot (this box, 2 vCPUs) — plumbing validation only, a few M steps:
  python -m gotrain.train_selfplay --out runs/auto_pilot --num-envs 32 \\\\
      --total-steps 2000000 --rollout-steps 128 --device cpu \\\\
      --opp-refresh-every 10 --eval-every 20 --ckpt-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from .export import export_weights, export_weights_tactical
from .net import N_POINTS, GoNetTactical
from .net_aux import (GoNetAux, load_trunk_from_gonet, check_trunk_resume)
from .ppo import PPOConfig, compute_gae, ppo_update, explained_variance
from .selfplay import (SelfPlayGo, PASS, set_komi, ownership_labels,
                       margin_label, to_learner_perspective, PositionArchive,
                       OpponentFn)

# Frozen museum snapshots, log-spaced in env steps (cf. train_cloning.SNAP_STEPS,
# which is in gradient steps — here the natural unit is env steps / PPO samples).
SNAP_STEPS: list[int] = [1000, 3000, 10000, 30000, 100000, 300000, 1000000,
              3000000, 10000000, 30000000, 100000000, 300000000]


# ---------------------------------------------------------------------------
# masked action sampling
# ---------------------------------------------------------------------------
def masked_dist(logits: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
    """Categorical over legal moves only (illegal logits -> -inf -> 0 mass)."""
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~masks, float("-inf")))


@torch.no_grad()
def dirichlet_noised_dist(dist: torch.distributions.Categorical,
                          masks: torch.Tensor, noise_mask: torch.Tensor,
                          alpha: float, eps: float) -> torch.distributions.Categorical:
    """AlphaZero-style opening exploration: P' = (1-eps)*P + eps*Dir(alpha).

    The noise is supported on legal moves only and is mixed in per-row for
    rows where noise_mask is True; other rows keep the policy's distribution.
    Works even on a fully collapsed (delta) policy, where temperature
    scaling would be a no-op, because the noise injects mass independently
    of the policy's output.
    """
    # Categorical.probs is a @lazy_property in torch: ty models instance
    # access as Tensor | lazy_property[...], but at runtime it always
    # materializes to a Tensor, so the cast is exact.
    probs = cast("torch.Tensor", dist.probs)
    d = torch.distributions.Dirichlet(
        torch.full((probs.shape[-1],), alpha, device=probs.device)).sample((probs.shape[0],))
    d = d.masked_fill(~masks, 0.0)
    d = d / d.sum(-1, keepdim=True).clamp_min(1e-12)
    mixed = (1.0 - eps) * probs + eps * d
    mixed = mixed.masked_fill(~masks, 0.0)
    mixed = mixed / mixed.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.where(noise_mask.unsqueeze(-1), mixed, probs)
    return torch.distributions.Categorical(probs=out)


# Precomputed dihedral group permutations for 9x9. _PERMS[s, old_idx] = new_idx.
# sym_idx = k*2 + flip (k=0..3 rotations, flip=0/1). Computed once at import.
def _build_dihedral_perms() -> tuple[torch.Tensor, torch.Tensor]:
    rr, cc = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
    perms = []
    for k in range(4):
        for f in range(2):
            r, c = rr.clone(), cc.clone()
            if f:
                c = 8 - c
            for _ in range(k):
                r, c = 8 - c, r
            perms.append((r * 9 + c).flatten())
    p = torch.stack(perms)  # (8, 81)
    return p, torch.argsort(p, dim=1)  # (perms, inverse perms)


_DIHEDRAL_PERMS: torch.Tensor
_DIHEDRAL_INV_PERMS: torch.Tensor
_DIHEDRAL_PERMS, _DIHEDRAL_INV_PERMS = _build_dihedral_perms()


def _symmetry_transforms(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a random dihedral symmetry per env. Returns sym_idx (B,) in 0..7."""
    k = torch.randint(0, 4, (batch_size,), device=device)
    flip = torch.randint(0, 2, (batch_size,), device=device)
    return k * 2 + flip


@torch.no_grad()
def _apply_symmetry(obs: torch.Tensor, masks: torch.Tensor,
                    sym_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor]:
    """Apply per-sample dihedral symmetry. Returns (obs_aug, masks_aug, perm, inv_perm).
    obs: (B, C, 9, 9), masks: (B, 82). perm[b, old]=new, inv_perm[b, new]=old.
    """
    device = obs.device
    B, C = obs.shape[0], obs.shape[1]
    # Index on CPU (perms live there), then move to device. Avoids device-sync
    # from indexing a CPU tensor with a device tensor.
    sym_cpu = sym_idx.cpu()
    perm = _DIHEDRAL_PERMS[sym_cpu].to(device)      # (B, 81)
    inv_perm = _DIHEDRAL_INV_PERMS[sym_cpu].to(device)

    # Obs: gather spatial dims through the INVERSE permutation.
    # We want obs_aug[b,:,new] = obs[b,:,old] where new=perm[old],
    # so obs_aug[b,:,i] = obs[b,:,inv_perm[i]].
    inv_perm_exp = inv_perm.unsqueeze(1).expand(B, C, 81)
    obs_aug = obs.view(B, C, 81).gather(2, inv_perm_exp).view(B, C, 9, 9)

    # Masks: new_mask[new] = old_mask[old], so gather with inverse
    masks_aug = torch.empty_like(masks)
    masks_aug[:, :81] = masks[:, :81].gather(1, inv_perm)
    masks_aug[:, 81] = masks[:, 81]  # pass unaffected

    return obs_aug, masks_aug, perm, inv_perm


@torch.no_grad()
def sample_actions(policy: torch.nn.Module, obs_t: torch.Tensor,
                   masks_t: torch.Tensor, noise_mask: torch.Tensor | None = None,
                   dirichlet_alpha: float = 0.05,
                   dirichlet_eps: float = 0.25) -> tuple[torch.Tensor, torch.Tensor,
                                                        torch.Tensor]:
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
def greedy_actions(policy: torch.nn.Module, obs_t: torch.Tensor,
                   masks_t: torch.Tensor) -> torch.Tensor:
    """Argmax over legal moves (for eval)."""
    logits, _ = policy(obs_t)
    masked = logits.masked_fill(~masks_t, float("-inf"))
    return masked.argmax(dim=-1)


def aux_update(policy: GoNetAux, optimizer: torch.optim.Optimizer, cfg: PPOConfig,
               obs: torch.Tensor, own_tgt: torch.Tensor, margin_tgt: torch.Tensor,
               own_w: float, margin_w: float, epochs: int) -> dict[str, float]:
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


def bc_update(policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
              cfg: PPOConfig, demo_x: torch.Tensor, demo_y: torch.Tensor,
              coef: float, epochs: int, batch_size: int) -> dict[str, float]:
    """Behavioral-cloning update on human demonstration positions.

    demo_x: (N,6,9,9) float positions; demo_y: (N,) long move indices.
    Runs as a separate phase after PPO (IN-RIL style): the clipped trust
    region never sees the supervised gradients. Each epoch does a fixed
    number of random minibatches (not a full pass over the demo set).
    The cross-entropy is scaled by coef. Returns mean BC loss (unscaled,
    for logging).
    """
    policy.train()
    N = demo_x.shape[0]
    # Fixed BC steps per epoch: a full 273K-position pass per PPO iter is
    # 500+ minibatches and 6x slower than PPO itself. 8 batches of 512 is
    # plenty to keep the imitation signal alive.
    steps_per_epoch = 8
    acc = 0.0
    n = 0
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=demo_x.device)
            logits, _ = policy(demo_x[idx])
            loss = F.cross_entropy(logits, demo_y[idx])
            optimizer.zero_grad()
            (coef * loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.max_grad_norm)
            optimizer.step()
            acc += loss.item()
            n += 1
    return {"bc_loss": acc / max(1, n)}


def random_opponent(obs: np.ndarray, masks: np.ndarray,
                    game_plies: np.ndarray | None = None) -> np.ndarray:
    """Uniform over legal moves. obs unused; signature matches opponent_fn."""
    B = masks.shape[0]
    actions = np.empty(B, dtype=np.int64)
    for b in range(B):
        legal = np.flatnonzero(masks[b])
        actions[b] = np.random.choice(legal)
    return actions


def attach_finished_game_labels(env: SelfPlayGo, i: int, ep_step_idxs: list[int],
                                b_own: torch.Tensor, b_margin: torch.Tensor,
                                b_aux_valid: torch.Tensor) -> None:
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


def _captures_if(own: np.ndarray, opp: np.ndarray, r: int | np.integer[Any],
                 c: int | np.integer[Any]) -> int:
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


def greedy_capture_opponent(obs: np.ndarray, masks: np.ndarray,
                            game_plies: np.ndarray | None = None) -> np.ndarray:
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


def make_snapshot_opponent(snapshot_net: torch.nn.Module, device: torch.device,
                           greedy: bool = False, dirichlet_plies: int = 0,
                           dirichlet_alpha: float = 0.05,
                           dirichlet_eps: float = 0.25) -> OpponentFn:
    """opponent_fn playing the frozen snapshot (sampled, or greedy for eval).

    When not greedy, Dirichlet noise is mixed into the first `dirichlet_plies`
    plies of each game (AlphaZero-style opening exploration); the env passes
    per-game ply counts as `game_plies`. Greedy eval never gets noise.
    """
    snapshot_net.eval()

    @torch.no_grad()
    def fn(obs: np.ndarray, masks: np.ndarray,
           game_plies: np.ndarray | None = None) -> np.ndarray:
        obs_t = torch.from_numpy(obs).to(device)
        masks_t = torch.from_numpy(masks).to(device)
        if greedy:
            a = greedy_actions(snapshot_net, obs_t, masks_t)
        else:
            noise_mask: torch.Tensor | None = None
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
def _tally_finished(done_idxs: np.ndarray, rewards: np.ndarray, wins: int,
                    played: int, n_games: int) -> tuple[int, int, np.ndarray]:
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
def evaluate(policy: torch.nn.Module, opponent_fn: OpponentFn, n_games: int,
             device: torch.device, seed: int = 12345,
             max_plies: int | None = None, tactical: bool = False) -> float:
    """Win rate of the greedy learner vs opponent_fn (learner alternates color)."""
    env = SelfPlayGo(num_envs=n_games, seed=seed,
                     max_plies=max_plies or 3 * 9 * 9, opponent_fn=opponent_fn,
                     tactical=tactical)
    # stagger starting colors: without this every env's first game has the
    # learner as Black, and greedy-vs-greedy self-play is deterministic, so the
    # "win rate" would really be one game repeated n_games times.
    env.color_counter[:] = np.arange(n_games) % 2
    obs = env.reset()
    wins = 0
    played = 0
    while played < n_games:
        masks = env.action_masks_learner()
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
def save_ckpt(path: str, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
              snapshot: torch.nn.Module, step: int, ppo_iter: int, snap_ptr: int,
              hparams: dict[str, Any]) -> None:
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


def _load_model_flexible(model: torch.nn.Module, sd: dict[str, Any]) -> bool:
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
        # is_aux_target means the model's state dict carries trunk.* keys,
        # which in this codebase is exactly GoNetAux (GoNetTactical and
        # plain GoNet store their conv weights unprefixed).
        res = load_trunk_from_gonet(cast("GoNetAux", model), sd)
        check_trunk_resume(res.missing_keys)
        if res.unexpected_keys:
            raise RuntimeError(
                f"unexpected keys in trunk resume: {res.unexpected_keys}")
        return True
    model.load_state_dict(sd)
    return False


def _migrate_optimizer_state_dict(ck_opt_sd: dict[str, Any],
                                  ck_model_sd: dict[str, Any],
                                  policy: torch.nn.Module) -> dict[str, Any]:
    """Rebuild an optimizer state dict saved over a plain GoNet so it loads
    into an optimizer over GoNetAux parameters.

    The checkpoint's optimizer was created as Adam(plain_gonet.parameters()):
    a single param group with IDs 0..K-1 in parameters() order, which matches
    the state_dict() key order of ck_model_sd (GoNet has no buffers). Each
    old param maps to its same-named trunk param ("trunk." + name) and keeps
    its Adam state (step, exp_avg, exp_avg_sq); aux-head params get no state
    entry, so Adam initializes them fresh on first use. Hyperparameters
    (lr, betas, eps, ...) carry over untouched.
    """
    groups = ck_opt_sd["param_groups"]
    if len(groups) != 1:
        raise RuntimeError(
            "cannot migrate optimizer state: expected 1 param group, "
            f"found {len(groups)}")
    old_group = groups[0]
    old_names = list(ck_model_sd.keys())  # parameters() order == state_dict order
    new_index = {n: i for i, (n, _) in enumerate(policy.named_parameters())}
    old_ids = list(old_group["params"])
    if len(old_ids) != len(old_names):
        raise RuntimeError(
            "cannot migrate optimizer state: "
            f"{len(old_ids)} optimizer params vs {len(old_names)} model params")
    new_state = {}
    for pos, old_id in enumerate(old_ids):
        new_name = "trunk." + old_names[pos]
        if new_name not in new_index:
            raise RuntimeError(
                "cannot migrate optimizer state: "
                f"no trunk param {new_name!r} in the new model")
        st = ck_opt_sd["state"].get(old_id)
        if st is None:
            st = ck_opt_sd["state"].get(str(old_id))
        if st is not None:
            new_state[new_index[new_name]] = st
    new_group = {k: v for k, v in old_group.items() if k != "params"}
    new_group["params"] = list(range(len(new_index)))
    return {"state": new_state, "param_groups": [new_group]}


def load_ckpt(path: str, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
              snapshot: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    ck = torch.load(path, map_location=device, weights_only=False)
    trunk_mapped = _load_model_flexible(policy, ck["model"])
    _load_model_flexible(snapshot, ck["opponent"])
    if trunk_mapped:
        # The checkpoint's optimizer was built over the plain GoNet's params;
        # migrate its Adam state onto the trunk, fresh state for aux heads.
        opt_sd = _migrate_optimizer_state_dict(ck["optimizer"], ck["model"],
                                               policy)
    else:
        opt_sd = ck["optimizer"]
    optimizer.load_state_dict(opt_sd)
    torch.set_rng_state(ck["torch_rng"].cpu())
    np.random.set_state(ck["numpy_rng"])
    if trunk_mapped:
        print("trunk-mapped resume from a plain GoNet checkpoint "
              "(e.g. the 15.5M run): trunk weights + Adam state migrated, "
              "aux heads randomly initialized",
              flush=True)
    return ck


def explicit_cli_flags(argv: list[str] | None = None) -> set[str]:
    """--flag names (underscore form) explicitly present on the command line."""
    out = set()
    for tok in (sys.argv[1:] if argv is None else argv):
        if tok.startswith("--"):
            out.add(tok[2:].split("=")[0].replace("-", "_"))
    return out


def apply_resumed_hparams(args: argparse.Namespace, ck: dict[str, Any],
                          explicit: set[str] | None = None) -> dict[str, Any]:
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
def resolve_device(name: str) -> torch.device:
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


def main() -> None:
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
    ap.add_argument("--bc-data", default=None,
                    help="directory with train_x.npy/train_y.npy demo "
                         "positions for a behavioral-cloning phase")
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="weight on the BC cross-entropy (0 = disabled)")
    ap.add_argument("--bc-epochs", type=int, default=1,
                    help="BC passes over the demo data per PPO iteration")
    ap.add_argument("--bc-batch-size", type=int, default=512,
                    help="minibatch size for the BC phase")
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
    ap.add_argument("--archive-restart-prob", type=float, default=0.0,
                    help="fraction of resets that restart from an archived midgame "
                         "position instead of a new game (0=disabled). Astra/Sol "
                         "recommendation: 0.5 for the kill test.")
    ap.add_argument("--archive-size", type=int, default=10000,
                    help="max archived positions (FIFO)")
    ap.add_argument("--archive-prob", type=float, default=0.02,
                    help="per-env per-step probability of archiving the current "
                         "position during rollout")
    ap.add_argument("--reward-mode", default="winloss", choices=["winloss", "score"],
                    help="terminal reward: +/-1 win/loss or tanh(score_margin/--reward-scale) "
                         "from the learner's perspective (KataGo-style graded signal)")
    ap.add_argument("--reward-scale", type=float, default=15.0,
                    help="score points at which tanh reaches ~0.76 in score reward mode")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="compute device; 'auto' picks cuda > mps > cpu")
    ap.add_argument("--resume", default=None, help="path to latest.pt")
    ap.add_argument("--tactical", action="store_true",
                    help="use the experimental 13-plane tactical observation "
                         "(gotrain.tactical) and GoNetTactical instead of the "
                         "locked 6-plane GoNet/GoNetAux")
    ap.add_argument("--no-anneal-lr", action="store_true",
                    help="disable linear LR annealing (constant LR); needed for "
                         "continued-training legs where the annealed schedule "
                         "would sit at ~0")
    args = ap.parse_args()

    # A tactical training checkpoint carries tactical=True in its hparams; if
    # the user didn't say --tactical explicitly, follow the checkpoint so the
    # policy architecture matches the resumed weights.
    if args.resume and "tactical" not in explicit_cli_flags():
        _ck0 = torch.load(args.resume, map_location="cpu", weights_only=False)
        if _ck0.get("hparams", {}).get("tactical"):
            args.tactical = True
            print("resume: tactical checkpoint detected; enabling --tactical",
                  flush=True)
        del _ck0

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    cfg = PPOConfig(lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                    clip_coef=args.clip_coef, vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
                    update_epochs=args.update_epochs,
                    minibatch_size=args.minibatch_size,
                    anneal_lr=not args.no_anneal_lr)

    # --tactical: experimental 13-plane GoNetTactical (gotrain.tactical).
    # Otherwise the locked GoNet trunk + training-only aux heads (GoNetAux):
    # one code path for both legs, so the ownership experiment isolates the
    # learning signal. With --ownership off the aux heads get no gradients and
    # the run is plain policy/value PPO on the identical trunk.
    if args.tactical:
        from .net import GoNetTactical
        policy = GoNetTactical().to(device)
        snapshot = GoNetTactical().to(device)  # frozen opponent
    else:
        policy = GoNetAux().to(device)
        snapshot = GoNetAux().to(device)   # frozen opponent; refreshed every K iters
    snapshot.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    hparams: dict[str, Any] = {k: v for k, v in vars(args).items() if k != "resume"}

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
                        minibatch_size=args.minibatch_size,
                        anneal_lr=not args.no_anneal_lr)
        print(f"resumed from {args.resume} at step {step} (iter {ppo_iter})", flush=True)

    # --seed takes precedence over RNG state restored from the checkpoint.
    # load_ckpt restores torch/numpy RNG (for bit-reproducible continuation),
    # but that nullifies --seed: two runs resuming from the same checkpoint
    # with different seeds would train identically. Re-seeding here ensures
    # the seed controls the training trajectory.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logf = open(os.path.join(args.out, "train.log"), "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"start: {json.dumps(hparams)} resuming_at={step}")
    if args.tactical:
        log(f"device={device} params={policy.param_count()} "
            f"(GoNetTactical, experimental 13-plane input)")
    else:
        # This branch constructs GoNetAux, so the cast is exact (and replaces
        # the type-ignore comment the union used to need here).
        log(f"device={device} params={policy.param_count()} "
            f"(trunk {cast('GoNetAux', policy).trunk.param_count()}, locked GoNet spec)")
    log(f"ownership_aux={args.ownership} aux_own_w={args.aux_own_w} "
        f"aux_margin_w={args.aux_margin_w} aux_epochs={args.aux_epochs}")

    set_komi(args.train_komi)
    log(f"train_komi={args.train_komi}")
    log(f"reward_mode={args.reward_mode} reward_scale={args.reward_scale}")
    if args.resume and abs(args.train_komi - prev_komi) > 1e-9:
        log(f"NOTE: train komi changed {prev_komi} -> {args.train_komi} on resume; "
            f"value head was calibrated to {prev_komi} and will recalibrate")

    env = SelfPlayGo(num_envs=args.num_envs, seed=args.seed,
                     max_plies=args.max_plies,
                     reward_mode=args.reward_mode,
                     reward_scale=args.reward_scale,
                     tactical=args.tactical,
                     opponent_fn=make_snapshot_opponent(
                         snapshot, device,
                         dirichlet_plies=args.dirichlet_plies,
                         dirichlet_alpha=args.dirichlet_alpha,
                         dirichlet_eps=args.dirichlet_eps))
    obs = env.reset()  # (N,6,9,9) float32, learner to move

    # Position archive for restarted self-play (0.0 = disabled).
    archive = PositionArchive(max_size=args.archive_size)
    archive_rng = np.random.default_rng(args.seed + 999)
    if args.archive_restart_prob > 0:
        log(f"position archive: size={args.archive_size} "
            f"restart_prob={args.archive_restart_prob} archive_prob={args.archive_prob}")

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
    if args.tactical and aux_on:
        raise ValueError(
            "--tactical and --ownership are incompatible: GoNetTactical has no "
            "auxiliary heads (no forward_aux), so the ownership/margin update "
            "cannot run. Use one or the other."
        )

    # ---- behavioral-cloning demo data --------------------------------------
    bc_on = args.bc_data is not None and args.bc_coef > 0
    demo_x: torch.Tensor | None = None
    demo_y: torch.Tensor | None = None
    if bc_on:
        dx = np.load(os.path.join(args.bc_data, "train_x.npy"))
        dy = np.load(os.path.join(args.bc_data, "train_y.npy"))
        demo_x = torch.from_numpy(dx).to(device)
        demo_y = torch.from_numpy(dy).long().to(device)
        log(f"bc demo data: {len(demo_x)} positions, coef={args.bc_coef} "
            f"epochs={args.bc_epochs} batch={args.bc_batch_size}")

    while step < args.total_steps:
        # ---- rollout ------------------------------------------------------
        b_obs = torch.zeros(T, N, env.n_planes, 9, 9)
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
        ep_steps: list[list[int]] = [[] for _ in range(N)]
        ep_rews: list[float] = []
        ep_lens: list[int] = []
        last_obs: np.ndarray | None = None
        last_terms: np.ndarray | None = None
        last_truncs: np.ndarray | None = None

        for t in range(T):
            for i in range(N):
                ep_steps[i].append(t)
            masks = env.action_masks_learner()
            obs_t = torch.from_numpy(obs).to(device)
            masks_t = torch.from_numpy(masks).to(device)
            # Dirichlet opening noise for both colors' early plies: env.plies
            # is the upcoming move's ply index in each env (exact across
            # resets and color alternation, since the env owns the counters).
            noise_mask: torch.Tensor | None = None
            if args.dirichlet_plies > 0:
                noise_mask = torch.from_numpy(
                    env.plies < args.dirichlet_plies).to(device)
            # Symmetry augmentation: random dihedral transform per env.
            # The net trains on the transformed (obs, action, mask); the env
            # steps with the action mapped back to the original frame.
            # This is ~8x free data: Go is invariant under all 8 symmetries.
            sym_idx = _symmetry_transforms(N, device)
            obs_aug, masks_aug, perm, inv_perm = _apply_symmetry(
                obs_t, masks_t, sym_idx)
            actions_aug, logps_aug, values_aug = sample_actions(
                policy, obs_aug, masks_aug, noise_mask=noise_mask,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps)
            # Map augmented actions back to original frame for env.step.
            # Pass (81) is unaffected by symmetries.
            not_pass = actions_aug < 81
            batch_idx = torch.arange(N, device=device)
            inv_actions = torch.where(
                not_pass,
                inv_perm[batch_idx, actions_aug.clamp(max=80)],
                actions_aug)
            # Store the AUGMENTED versions (consistent obs/action/mask/logp).
            b_obs[t] = obs_aug.cpu()
            b_masks[t] = masks_aug.cpu()
            b_actions[t] = actions_aug.cpu()
            b_logps[t] = logps_aug.cpu()
            b_values[t] = values_aug.cpu()

            obs, rewards, dones, terms, lens = env.step(inv_actions.cpu().numpy())
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
                done_idxs = np.where(dones)[0]
                if args.archive_restart_prob > 0 and len(archive) > 0:
                    # Split: some restart from archive, rest start fresh.
                    # (Archive empty at the start -> all fresh until it fills.)
                    n_restart = int(round(len(done_idxs) * args.archive_restart_prob))
                    # Random subset for archive restart (not just the first n).
                    perm = archive_rng.permutation(len(done_idxs))
                    restart_idxs = done_idxs[perm[:n_restart]]
                    fresh_idxs = done_idxs[perm[n_restart:]]
                    if len(restart_idxs) > 0:
                        fresh_restart, _ = env.reset_from_archive(restart_idxs, archive)
                        obs[restart_idxs] = fresh_restart
                    if len(fresh_idxs) > 0:
                        obs[fresh_idxs] = env.reset(fresh_idxs)
                else:
                    fresh = env.reset(done_idxs)
                    obs[dones] = fresh

            # Archive midgame positions for future restarts (after step, before
            # next iteration: board is at learner's turn, invariant holds).
            if args.archive_restart_prob > 0:
                for i in range(N):
                    if not dones[i] and archive_rng.random() < args.archive_prob:
                        archive.add(env.boards[i], env.learner_color[i],
                                    env.plies[i], env.consec_passes[i])

        # value bootstrap: V(final position) for live envs; 0 for true
        # terminals (two passes) and scored max-ply endings -- both are
        # episodic terminals for GAE (see gotrain.ppo.compute_gae).
        with torch.no_grad():
            next_values = policy(torch.from_numpy(
                cast("np.ndarray", last_obs)).to(device))[1].cpu()
        final_done = (torch.from_numpy(cast("np.ndarray", last_terms))
                      | torch.from_numpy(cast("np.ndarray", last_truncs)))
        next_values = next_values * (~final_done).float()

        advantages, returns = compute_gae(
            b_rewards, b_values, b_terms, b_truncs, next_values,
            torch.from_numpy(cast("np.ndarray", last_terms)),
            torch.from_numpy(cast("np.ndarray", last_truncs)),
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        ev = explained_variance(b_values.numpy(), returns.numpy())

        # ---- PPO update ----------------------------------------------------
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(T * N, *x.shape[2:])
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
                # aux_update needs the aux heads, i.e. GoNetAux. The --tactical +
                # --ownership combination is rejected at startup (ValueError),
                # so by the time aux_on is true, policy is always GoNetAux and
                # this cast is exact.
                aux_stats = aux_update(
                    cast("GoNetAux", policy), optimizer, cfg,
                    flat(b_obs).to(device)[vdev],
                    flat(b_own).to(device)[vdev],
                    flat(b_margin).to(device)[vdev],
                    args.aux_own_w, args.aux_margin_w, args.aux_epochs)
                aux_str = (f" aux_own={aux_stats['aux_own']:.4f}"
                           f" aux_mgn={aux_stats['aux_margin']:.4f}")

        # ---- behavioral-cloning phase --------------------------------------
        # Separate phase after PPO (IN-RIL style): supervised cross-entropy
        # on human demo positions, scaled by bc_coef. Keeps the imitation
        # signal alive during RL instead of using it only as a warm-start.
        bc_str = ""
        if bc_on:
            # bc_on implies the tensors were loaded above.
            bc_stats = bc_update(policy, optimizer, cfg,
                                 cast("torch.Tensor", demo_x),
                                 cast("torch.Tensor", demo_y),
                                 args.bc_coef, args.bc_epochs,
                                 args.bc_batch_size)
            bc_str = f" bc={bc_stats['bc_loss']:.4f}"

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
            wr_rand = evaluate(policy, random_opponent, args.eval_games, device,
                               tactical=args.tactical)
            wr_greedy = evaluate(policy, greedy_capture_opponent,
                                 args.eval_games, device,
                                 tactical=args.tactical)
            wr_snap = evaluate(policy, make_snapshot_opponent(snapshot, device,
                                                             greedy=True),
                               args.eval_games, device,
                               tactical=args.tactical)
            eval_str = (f" eval_vs_random={wr_rand:.2f}"
                        f" eval_vs_greedy={wr_greedy:.2f}"
                        f" eval_vs_snapshot={wr_snap:.2f}")

        sps = (step - step0) / (time.time() - t0)
        ep_r = f"{np.mean(ep_rews):+.3f}" if ep_rews else "n/a"
        ep_l = f"{np.mean(ep_lens):.0f}" if ep_lens else "n/a"
        log(f"iter {ppo_iter}: step {step} sps={sps:.0f} ep_rew={ep_r} "
            f"ep_len={ep_l} pg={stats['pg_loss']:.4f} v={stats['v_loss']:.4f} "
            f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
            f"clipfrac={stats['clipfrac']:.3f} ev={ev:.3f}{aux_str}{bc_str}{eval_str}")

    save_ckpt(os.path.join(args.out, "latest.pt"), policy, optimizer, snapshot,
              step, ppo_iter, snap_ptr, hparams)
    final_bin = os.path.join(args.out, "autodidact-final.bin")
    # export the locked trunk only: the fp16 format (and the Rust/WASM demo
    # path) never sees the training-only aux heads. Tactical runs export the
    # 13-channel net in the tactical layout (demo path pending phase 2).
    if args.tactical:
        # This branch constructs GoNetTactical, so the cast is exact.
        export_weights_tactical(cast("GoNetTactical", policy).cpu(), final_bin)
    else:
        # This branch constructs GoNetAux, so the cast is exact.
        export_weights(cast("GoNetAux", policy).trunk.cpu(), final_bin)
    log(f"done at step {step}; fp16 export -> {final_bin} "
        f"({os.path.getsize(final_bin)} bytes)")
    logf.close()


if __name__ == "__main__":
    main()
