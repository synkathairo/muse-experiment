# Theory — what the two nets are learning and why

Two training runs, one architecture, two completely different learning
frameworks. This note records the theory behind both, for the exhibit page
and for anyone reproducing the work.

## The game

9×9 Go is a finite deterministic two-player zero-sum game with perfect
information. Provisional rules used here: modified Tromp–Taylor (area scoring
with no dead-stone agreement phase), positional superko (not simple ko), suicide prohibited, 7.5 komi, two
consecutive passes end the game. See `engine/` for the implementation and
its tests.

## Leg 1 — the Imitator (supervised behavioral cloning)

Framework: plain supervised learning. Given ~2.5M positions from ranked
human 9×9 games (OGS), the net maximizes the likelihood of the human's
move (cross-entropy on the policy head) and regresses the game outcome
(MSE on the value head, tanh output in [−1, +1]).

There is no game theory here — the net never reasons about winning, it
predicts what a human would do. Strength ceiling: roughly the average
strength of the humans in the corpus, minus cloning error.

## Leg 2 — the Autodidact (self-play PPO)

### Game-theoretic framing

The learner plays both colors against a periodically refreshed frozen
snapshot of its own policy. This is a form of **fictitious play**: each
side approximately best-responds to a mixture of the opponent's past
strategies. For finite zero-sum games, fictitious play converges toward
**Nash equilibrium** — i.e., toward optimal play.

Honest caveat: the convergence theorems assume exact best responses in
tabular games. We have *approximate* best responses via neural-network
function approximation, so no theorem strictly applies.

This is *not* the AlphaZero recipe, and the difference matters. AlphaZero
improves its policy through Monte Carlo tree search and trains the net to
imitate the search's visit counts, always playing the latest weights
against themselves. Here there is no search at all: PPO optimizes expected
return directly from game outcomes (pure policy play; adding MCTS is a
stretch goal), and the opponent is a periodically *frozen* snapshot rather
than the live policy — closer to fictitious play, or league training, than
to AlphaZero's pure self-play. Shared DNA: self-play from scratch,
two-headed policy+value net, zero human data. Different engine:
search-free policy gradients against snapshots instead of MCTS plus
distillation.

### RL algorithm: PPO

**Proximal Policy Optimization**, a policy-gradient actor-critic method
(CleanRL-style implementation, hand-rolled in `train/gotrain/ppo.py` —
no RL library).

- *Policy gradient theorem.* To improve expected return $J(\theta)$,
  move parameters along
  $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)\; A(s,a)]$,
  where $A(s,a)$ is the advantage of action $a$ in state $s$.
- *Clipped surrogate.* PPO keeps each update inside a trust region so
  one bad batch can't destroy the policy:
  $L = \mathbb{E}[\min(r(\theta)A,\ \mathrm{clip}(r(\theta), 1-\epsilon,
  1+\epsilon)A)]$,
  with $r(\theta) = \pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$ and
  $\epsilon = 0.2$.
- *GAE.* Advantages come from Generalized Advantage Estimation: the
  $\lambda$-weighted ($\lambda = 0.95$) blend of temporal-difference
  errors, trading bias against variance. Discount $\gamma = 0.99$.
- *Critic + exploration.* A value loss (regressing game outcomes,
  coefficient 0.5) trains the critic; an entropy bonus (coefficient
  0.01) keeps the policy exploring; gradients are clipped at norm 0.5
  and the learning rate anneals over the run.

Rewards are terminal only: +1/−1 from the game result (area score with
komi), from the perspective of the player to move — no shaped rewards.

### What the net is learning

Two mathematical objects:

- a policy $\pi(a|s)$ approximating a **best response** to the
  opponent distribution, and
- a value $V(s)$ approximating the **game-theoretic value** of the
  position (probability of eventually winning, from the side to move).

The exhibit's time-machine is literally these approximations sharpening
toward equilibrium: random flailing → knows the rules → captures greedily
→ positional play.

## Shared architecture (locked)

Both legs use the identical trunk and policy head — that is what makes
comparing them honest. Differences are confined to the value component
(PPO's critic vs the cloning value head), which is allowed.

- Input: 6×9×9 float planes (own stones, opponent stones, empty, last
  move, color-to-move, ko ban).
- Trunk: 4 padded 3×3 conv layers, 64 channels, ReLU, no batch norm.
- Policy head: 1×1 conv 64→2, flatten, linear 162→82 (81 points + pass).
- Value head: 1×1 conv 64→1, flatten, linear 81→32, ReLU, linear 32→1,
  tanh.
- 130,522 parameters; fp16 little-endian export, 261,044 bytes.

## Evaluation ladder (weakest to strongest signal)

1. **vs random** — sanity check: has it learned the rules at all?
2. **vs a greedy heuristic bot** — do basic tactics work?
3. **vs GNU Go** (GTP) — an interpretable kyu-level number; the
   exhibit's honest strength claim. Parked for later.
4. **vs its own past snapshots** — measures *progress*, the exhibit's
   actual story.
5. **vs the other leg's net** — Imitator vs Autodidact, the marquee
   cross-method match.

## Honest caveats

- Expected ceiling is weak-club-player level at best. The exhibit is the
  learning, not the Elo.
- RL from scratch may stall at "knows the rules, tactically naive" —
  that is a result too, and the time-machine still shows it learning.
- The engine implements simple ko and Tromp–Taylor scoring, not full
  superko / full Chinese rules with dead-stone agreement. Documented
  simplifications; they agree with official rules in essentially all
  practical 9×9 positions.
