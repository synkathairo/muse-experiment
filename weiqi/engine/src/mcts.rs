//! PUCT Monte-Carlo tree search over [`crate::rules::Game`], AlphaZero-style.
//!
//! The search is generic over an [`Evaluator`] (policy priors + leaf value),
//! so the same code serves the WASM demo (hand-rolled net) and any future
//! Rust-side evaluator. No rollouts: expanded leaves are scored by the
//! evaluator's value head, priors come from its policy head, and values back
//! up negamax-style (each ply flips the side-to-move perspective).
//!
//! Deterministic given the seed in [`SearchConfig`]: the only randomness is
//! root Dirichlet noise and optional temperature sampling.

use crate::rules::{Color, Game, Move, N_MOVES};

/// Policy priors (index 81 = pass) plus a value estimate, both from the
/// side-to-move's perspective; value in [-1, 1].
pub struct Eval {
    pub policy: [f32; N_MOVES],
    pub value: f32,
}

/// Source of priors and leaf values for the search.
pub trait Evaluator {
    fn evaluate(&self, game: &Game) -> Eval;
}

/// Search hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Number of select→expand→backup iterations.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Dirichlet noise shape at the root (AlphaZero used 0.03 on 19×19;
    /// 9×9 wants a larger value).
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight at the root; 0 disables noise.
    pub dirichlet_eps: f32,
    /// Final move selection temperature over visit counts; 0 = argmax.
    pub temperature: f32,
    /// RNG seed (noise + sampling). Fixed default = deterministic.
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            simulations: 100,
            c_puct: 1.5,
            dirichlet_alpha: 0.15,
            dirichlet_eps: 0.0,
            temperature: 0.0,
            seed: 0x9E3779B97F4A7C15,
        }
    }
}

/// One edge in the tree. `value_sum` accumulates leaf values from the
/// perspective of the side to move at the *child* node, so the parent's
/// selection score uses `-value_sum / visits`.
struct Child {
    action: u8, // 0..80 point, 81 pass
    node: u32,  // index into the arena; u32::MAX = unexpanded
    prior: f32,
    visits: u32,
    value_sum: f32,
}

impl Child {
    /// Mean value from the side-to-move-at-child's perspective.
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

struct Node {
    game: Game,
    children: Vec<Child>,
    expanded: bool,
}

/// Minimal deterministic RNG (xorshift64* + Marsaglia–Tsang gamma).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa uniform in [0, 1).
        const M: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / M
    }
    /// Gamma(alpha, 1) sample. Marsaglia–Tsang for alpha >= 1; the
    /// alpha+1 identity below that.
    fn gamma(&mut self, alpha: f64) -> f64 {
        debug_assert!(alpha > 0.0);
        if alpha < 1.0 {
            return self.gamma(alpha + 1.0) * self.next_f64().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            // Standard normal via Box–Muller.
            let (u1, u2) = (self.next_f64().max(1e-300), self.next_f64());
            let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

/// Run PUCT search from `game`'s current position and return the chosen move
/// index (0–80 point, 81 pass). Returns 81 immediately if the game is over.
///
/// Deterministic given `cfg.seed`. Each simulation performs exactly one
/// evaluator call (the expanded leaf), so `simulations` ≈ forward passes.
pub fn search<E: Evaluator>(game: &Game, eval: &E, cfg: &SearchConfig) -> usize {
    if game.is_over() {
        return 81;
    }
    let legal = game.legal_moves();
    let n_legal = legal.iter().filter(|&&b| b).count();
    if n_legal == 0 {
        return 81; // cannot happen (pass is always legal), but stay total
    }
    if n_legal == 1 {
        return legal.iter().position(|&b| b).unwrap();
    }

    let mut rng = Rng(cfg.seed);
    let mut arena: Vec<Node> = Vec::with_capacity(cfg.simulations as usize + 1);
    arena.push(Node {
        game: game.clone(),
        children: Vec::new(),
        expanded: false,
    });

    for _ in 0..cfg.simulations {
        // --- select: descend to a leaf, recording the path ---
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut node_idx: u32 = 0;
        let leaf_value: f32 = loop {
            // Terminal?
            let terminal_v = {
                let node = &arena[node_idx as usize];
                if node.game.is_over() {
                    let m = node.game.score().margin();
                    let v = if m > 0.0 {
                        1.0
                    } else if m < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    Some(if node.game.to_move() == Color::Black {
                        v
                    } else {
                        -v
                    })
                } else {
                    None
                }
            };
            if let Some(v) = terminal_v {
                break v;
            }
            // Expand unexpanded nodes (evaluates the position once).
            let fresh_eval = {
                let node = &arena[node_idx as usize];
                if !node.expanded {
                    Some(eval.evaluate(&node.game))
                } else {
                    None
                }
            };
            if let Some(ev) = fresh_eval {
                let node = &mut arena[node_idx as usize];
                let legal = node.game.legal_moves();
                let mut children = Vec::new();
                let mut prior_sum = 0.0f32;
                for (i, &ok) in legal.iter().enumerate() {
                    if ok {
                        let p = ev.policy[i].max(0.0);
                        prior_sum += p;
                        children.push(Child {
                            action: i as u8,
                            node: u32::MAX,
                            prior: p,
                            visits: 0,
                            value_sum: 0.0,
                        });
                    }
                }
                if prior_sum > 0.0 {
                    for ch in children.iter_mut() {
                        ch.prior /= prior_sum;
                    }
                } else {
                    let u = 1.0 / children.len() as f32;
                    for ch in children.iter_mut() {
                        ch.prior = u;
                    }
                }
                // Dirichlet noise at the root only.
                if node_idx == 0 && cfg.dirichlet_eps > 0.0 {
                    let k = children.len();
                    let mut sum = 0.0f64;
                    let mut noise = Vec::with_capacity(k);
                    for _ in 0..k {
                        let g = rng.gamma(cfg.dirichlet_alpha as f64);
                        sum += g;
                        noise.push(g);
                    }
                    for (ch, &g) in children.iter_mut().zip(noise.iter()) {
                        let n = (g / sum.max(1e-300)) as f32;
                        ch.prior =
                            (1.0 - cfg.dirichlet_eps) * ch.prior + cfg.dirichlet_eps * n;
                    }
                }
                node.children = children;
                node.expanded = true;
                break ev.value.clamp(-1.0, 1.0);
            }
            // Select the best child (PUCT). Ties (e.g. all-zero scores on the
            // first descent) break toward the higher prior.
            let (child_i, child_node) = {
                let node = &arena[node_idx as usize];
                let parent_visits: u32 = node.children.iter().map(|c| c.visits).sum();
                let pv = (parent_visits as f32).sqrt();
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                let mut best_prior = f32::NEG_INFINITY;
                for (i, ch) in node.children.iter().enumerate() {
                    // -q: the child's value is from the opponent's perspective.
                    let score =
                        -ch.q() + cfg.c_puct * ch.prior * pv / (1.0 + ch.visits as f32);
                    if score > best_score
                        || (score == best_score && ch.prior > best_prior)
                    {
                        best_score = score;
                        best_prior = ch.prior;
                        best = i;
                    }
                }
                (best, node.children[best].node)
            };
            path.push((node_idx, child_i));
            if child_node == u32::MAX {
                // First visit: materialize the child node, then loop back to
                // expand (and evaluate) it.
                let mut g = arena[node_idx as usize].game.clone();
                let mv = Move::from_index(
                    arena[node_idx as usize].children[child_i].action as usize,
                )
                .unwrap();
                debug_assert!(g.play(mv).is_ok());
                let new_idx = arena.len() as u32;
                arena[node_idx as usize].children[child_i].node = new_idx;
                arena.push(Node {
                    game: g,
                    children: Vec::new(),
                    expanded: false,
                });
                node_idx = new_idx;
            } else {
                node_idx = child_node;
            }
        };

        // --- backup: negate the value at every ply ---
        // `leaf_value` is from the side-to-move-at-leaf's perspective, which
        // is also the side-to-move-at-child perspective for the last edge.
        let mut v = leaf_value;
        for &(parent_idx, child_i) in path.iter().rev() {
            let child = &mut arena[parent_idx as usize].children[child_i];
            child.visits += 1;
            child.value_sum += v;
            v = -v;
        }
    }

    // --- final move: visits, with optional temperature ---
    let root = &arena[0];
    debug_assert!(root.expanded);
    if cfg.temperature <= 0.0 {
        // Argmax visits; ties broken by prior, then action index.
        let mut best = 0usize;
        for (i, ch) in root.children.iter().enumerate() {
            let b = &root.children[best];
            if ch.visits > b.visits
                || (ch.visits == b.visits
                    && (ch.prior > b.prior
                        || (ch.prior == b.prior && ch.action < b.action)))
            {
                best = i;
            }
        }
        root.children[best].action as usize
    } else {
        let inv_t = 1.0 / cfg.temperature;
        let mut sum = 0.0f64;
        let mut weights = Vec::with_capacity(root.children.len());
        for ch in root.children.iter() {
            let w = (ch.visits as f64).powf(inv_t as f64);
            sum += w;
            weights.push(w);
        }
        let mut r = rng.next_f64() * sum;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return root.children[i].action as usize;
            }
        }
        root.children.last().unwrap().action as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub evaluator: uniform priors, value = tanh(0.2 × stone-diff) from
    /// the side-to-move's perspective. Capturing strictly improves value, so
    /// search must find tactical captures the uniform prior is blind to.
    struct MaterialEval;
    impl Evaluator for MaterialEval {
        fn evaluate(&self, game: &Game) -> Eval {
            let mut mine = 0i32;
            let mut theirs = 0i32;
            for p in 0..81u8 {
                match game.stone_at(p) {
                    Some(c) if c == game.to_move() => mine += 1,
                    Some(_) => theirs += 1,
                    None => {}
                }
            }
            let policy = [1.0f32 / N_MOVES as f32; N_MOVES];
            Eval {
                policy,
                value: ((mine - theirs) as f32 * 0.2).tanh(),
            }
        }
    }

    /// Black to move; white stone at (4,4) has one liberty at (4,5).
    /// The capture is one of 80 legal moves under a uniform prior.
    fn capture_position() -> Game {
        let mut g = Game::new();
        // Surround (4,4)=40 with black at 31, 39, 49; white plays 40 last so
        // black is to move. Order: B31 W0 B39 W1 B49 W40.
        for &m in &[31usize, 0, 39, 1, 49, 40] {
            g.play(Move::from_index(m).unwrap()).unwrap();
        }
        assert_eq!(g.to_move(), Color::Black);
        g
    }

    #[test]
    fn finds_the_capture() {
        let g = capture_position();
        let cfg = SearchConfig {
            simulations: 200,
            ..Default::default()
        };
        // (4,5) = row 4, col 5 = index 41.
        assert_eq!(search(&g, &MaterialEval, &cfg), 41);
    }

    #[test]
    fn deterministic_given_seed() {
        let g = capture_position();
        let cfg = SearchConfig {
            simulations: 100,
            ..Default::default()
        };
        let a = search(&g, &MaterialEval, &cfg);
        let b = search(&g, &MaterialEval, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn never_returns_illegal_move() {
        // A few hundred plies of random-ish play, searching 20 sims each.
        let mut g = Game::new();
        let mut rng = Rng(12345);
        let cfg = SearchConfig {
            simulations: 20,
            ..Default::default()
        };
        for _ in 0..60 {
            if g.is_over() {
                break;
            }
            let m = search(&g, &MaterialEval, &cfg);
            let legal = g.legal_moves();
            assert!(legal[m], "search returned illegal move {m}");
            // Advance with a random legal move to vary positions.
            let legal_idxs: Vec<usize> =
                legal.iter().enumerate().filter(|&(_, &b)| b).map(|(i, _)| i).collect();
            let pick = legal_idxs[(rng.next_f64() * legal_idxs.len() as f64) as usize];
            g.play(Move::from_index(pick).unwrap()).unwrap();
        }
    }

    #[test]
    fn terminal_game_returns_pass() {
        let mut g = Game::new();
        g.play(Move::from_index(81).unwrap()).unwrap();
        g.play(Move::from_index(81).unwrap()).unwrap();
        assert!(g.is_over());
        let cfg = SearchConfig::default();
        assert_eq!(search(&g, &MaterialEval, &cfg), 81);
    }

    #[test]
    fn dirichlet_noise_keeps_priors_valid() {
        let g = Game::new();
        let cfg = SearchConfig {
            simulations: 50,
            dirichlet_eps: 0.25,
            ..Default::default()
        };
        let m = search(&g, &MaterialEval, &cfg);
        assert!(g.legal_moves()[m]);
    }

    #[test]
    fn temperature_sampling_stays_legal() {
        let g = capture_position();
        let cfg = SearchConfig {
            simulations: 100,
            temperature: 1.0,
            ..Default::default()
        };
        for _ in 0..20 {
            let m = search(&g, &MaterialEval, &cfg);
            assert!(g.legal_moves()[m]);
        }
    }
}
