"""Hand-rolled PPO (cleanrl-style) for the Autodidact leg. No PufferLib.

Implements the Schulman et al. 2017 PPO-Clip algorithm:
  - clipped surrogate policy objective,
  - Generalized Advantage Estimation (GAE),
  - clipped value-function loss,
  - entropy bonus,
  - minibatch SGD over several epochs per rollout,
  - gradient clipping, optional linear LR annealing.

Device-agnostic: works on cpu / cuda / mps — the caller moves the model and
passes tensors on the right device. Nothing here knows about Go; it operates on
flat batches of (obs, action, logprob, reward, term, trunc, value).

Conventions (shared with gotrain.train_selfplay):
  - `terms`: TRUE terminals (two passes -> game really ended, bootstrap 0).
  - `truncs`: truncations (max plies -> game scored, but value still bootstraps).
    dones = terms | truncs. next_nonterm = 1 - next_term.
  - rewards are +/-1 at game end, 0 elsewhere; the value head's tanh output
    already lives in [-1, 1], so no value rescaling is needed.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    anneal_lr: bool = True       # linear decay to 0 over total_iters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2       # PPO trust-region clip epsilon
    vf_coef: float = 0.5         # value loss weight
    ent_coef: float = 0.01       # entropy bonus weight
    max_grad_norm: float = 0.5
    update_epochs: int = 4       # passes over the rollout buffer per iteration
    minibatch_size: int = 256
    norm_adv: bool = True        # standardize advantages over the rollout batch
    clip_vloss: bool = True      # clip value loss like the policy loss
    target_kl: float = None      # if set, stop epochs early when approx KL exceeds this


def compute_gae(rewards, values, terms, truncs, next_value, next_term,
                gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation over a (T, N) rollout.

    rewards/values/terms/truncs: (T, N) tensors. next_value/next_term: (N,).
    terms=1 on TRUE terminals (bootstrap 0); truncations keep bootstrapping.
    Returns (advantages, returns) as (T, N) tensors.
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            next_nonterm = 1.0 - next_term.float()
        else:
            next_val = values[t + 1]
            next_nonterm = 1.0 - terms[t + 1].float()
        # truncations: reward was scored, but the value still bootstraps, so
        # next_nonterm stays 1 for them (only true terminals zero it)
        delta = rewards[t] + gamma * next_val * next_nonterm - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterm * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _masked_logps_entropy(logits, actions, masks):
    """Log-probs of taken actions + entropy under the legal-move mask.

    logits: (B, 82), actions: (B,) int64, masks: (B, 82) bool.
    Illegal logits are set to -inf before softmax so they get exactly 0 mass.
    """
    masked = logits.masked_fill(~masks, float("-inf"))
    logps = F.log_softmax(masked, dim=-1)
    taken = logps.gather(1, actions.unsqueeze(1)).squeeze(1)
    probs = masked.softmax(dim=-1)
    # Entropy over legal moves only. Naive probs*logps gives 0 * -inf = nan on
    # illegal entries; their true contribution is 0 * log 0 = 0, so substitute
    # a finite log value where the mask is False (probs are 0 there anyway).
    entropy = -(probs * logps.masked_fill(~masks, 0.0)).sum(-1)
    return taken, entropy


@torch.no_grad()
def evaluate_actions(model, obs, actions, masks):
    """Current policy's log-probs, entropy, and value for stored actions."""
    logits, values = model(obs)
    logps, entropy = _masked_logps_entropy(logits, actions, masks)
    return logps, entropy, values


def ppo_minibatch_update(model, optimizer, cfg, b_obs, b_actions, b_logps,
                         b_adv, b_ret, b_values, b_masks):
    """One SGD step on a minibatch. Returns dict of mean stats."""
    logits, new_values = model(b_obs)
    new_logps, entropy = _masked_logps_entropy(logits, b_actions, b_masks)

    log_ratio = new_logps - b_logps
    ratio = log_ratio.exp()
    # approx KL, the unbiased-ish estimator from the PPO / cleanrl lineage
    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    # --- clipped surrogate policy loss ---
    pg1 = ratio * b_adv
    pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * b_adv
    pg_loss = -torch.min(pg1, pg2).mean()

    # --- value loss (optionally clipped, as in cleanrl) ---
    new_values = new_values.view(-1)
    if cfg.clip_vloss:
        v_clipped = b_values + torch.clamp(new_values - b_values,
                                           -cfg.clip_coef, cfg.clip_coef)
        v_loss = 0.5 * torch.max((new_values - b_ret) ** 2,
                                 (v_clipped - b_ret) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_values - b_ret) ** 2).mean()

    ent_loss = -entropy.mean()  # maximize entropy => minimize -entropy
    loss = pg_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()

    # fraction of clipped policy updates (diagnostic: is the trust region binding?)
    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
    return {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
        "grad_norm": grad_norm.item(),
        "loss": loss.item(),
    }


def ppo_update(model, optimizer, cfg, obs, actions, logps, advantages, returns,
               values, masks, lr_now=None):
    """Full PPO update: `update_epochs` shuffled minibatch passes over the batch.

    All inputs are flat (B, ...) tensors on the model's device. Returns a dict
    of mean stats across minibatches (plus `early_stop` if target_kl fired).
    """
    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if lr_now is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

    B = obs.shape[0]
    stats_acc = {}
    n_mb = 0
    early_stop = False
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(B, device=obs.device)
        for s in range(0, B, cfg.minibatch_size):
            idx = perm[s:s + cfg.minibatch_size]
            st = ppo_minibatch_update(
                model, optimizer, cfg,
                obs[idx], actions[idx], logps[idx], advantages[idx],
                returns[idx], values[idx], masks[idx])
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v
            n_mb += 1
            if cfg.target_kl is not None and st["approx_kl"] > cfg.target_kl:
                early_stop = True
                break
        if early_stop:
            break
    stats = {k: v / max(1, n_mb) for k, v in stats_acc.items()}
    stats["early_stop"] = early_stop
    return stats


def explained_variance(values, returns):
    """1 - Var(returns - values)/Var(returns): is the value head learning?"""
    values = np.asarray(values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    var_ret = returns.var()
    if var_ret < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var() / var_ret)
