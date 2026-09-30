//! Bot move selection directly from a policy vector (the non-search path).
//!
//! This lives engine-side rather than in page JS so every frontend shares
//! one move-selection implementation — the WASM demo today, a future GTP
//! binary (e.g. for Sabaki) tomorrow. The WASM binding is a thin wrapper;
//! a native GTP frontend would call [`policy_move`] directly.

use crate::rules::{Game, N_MOVES, N_POINTS};

/// Minimal deterministic RNG (xorshift64*); mirrors the one in `mcts`.
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
}

/// Choose the bot's move directly from a policy vector, without tree search.
///
/// - Illegal moves are masked out.
/// - Moves that fill the mover's own eye are excluded (see
///   [`Game::is_self_eye_fill`]); pass is always kept.
/// - `temperature <= 0.0` returns the argmax (first maximum wins ties).
/// - Otherwise samples proportionally to `max(p, 1e-9)^(1/temperature)`.
///
/// Deterministic given `seed`. Always returns a legal move index (0–80
/// point, 81 pass); pass is the fallback and is always legal.
pub fn policy_move(game: &Game, policy: &[f32; N_MOVES], temperature: f32, seed: u64) -> usize {
    let legal = game.legal_moves();
    let mut cands: Vec<usize> = Vec::new();
    for i in 0..N_MOVES {
        if !legal[i] {
            continue;
        }
        if i < N_POINTS && game.is_self_eye_fill(i as u8) {
            continue;
        }
        cands.push(i);
    }
    // Defensive: pass is always legal and never filtered, so this cannot
    // happen, but stay total.
    if cands.is_empty() {
        return N_POINTS;
    }
    if temperature <= 0.0 {
        let mut best = cands[0];
        let mut bp = f32::NEG_INFINITY;
        for &i in &cands {
            if policy[i] > bp {
                bp = policy[i];
                best = i;
            }
        }
        return best;
    }
    let inv_t = 1.0 / f64::from(temperature);
    let mut weights: Vec<f64> = Vec::with_capacity(cands.len());
    let mut sum = 0.0f64;
    for &i in &cands {
        let w = f64::from(policy[i].max(1e-9)).powf(inv_t);
        weights.push(w);
        sum += w;
    }
    let mut rng = Rng(seed);
    let mut r = rng.next_f64() * sum;
    for (k, &i) in cands.iter().enumerate() {
        r -= weights[k];
        if r <= 0.0 {
            return i;
        }
    }
    cands[cands.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{Color, Move};

    fn peaked_policy(at: usize, elsewhere: f32) -> [f32; N_MOVES] {
        let mut p = [elsewhere; N_MOVES];
        p[at] = 1.0;
        p
    }

    /// Black wall around tengen (40); black to move. 40 is black's eye.
    fn eye_position() -> Game {
        let mut g = Game::new();
        for &p in &[31u8, 49, 39, 41] {
            g.place_setup_stones(&[p], Color::Black, Color::Black);
        }
        assert_eq!(g.to_move(), Color::Black);
        g
    }

    #[test]
    fn argmax_picks_best_legal_move() {
        let g = Game::new();
        let m = policy_move(&g, &peaked_policy(40, 0.001), 0.0, 123);
        assert_eq!(m, 40);
    }

    #[test]
    fn argmax_breaks_ties_by_lowest_index() {
        let g = Game::new();
        let mut p = [0.0f32; N_MOVES];
        p[10] = 0.5;
        p[20] = 0.5;
        assert_eq!(policy_move(&g, &p, 0.0, 7), 10);
    }

    #[test]
    fn never_fills_own_eye() {
        let g = eye_position();
        // Greedy: must not pick 40 even with all mass on it.
        let m = policy_move(&g, &peaked_policy(40, 0.001), 0.0, 1);
        assert_ne!(m, 40);
        // Sampled: 500 seeds, still never 40 (it is not even a candidate).
        for seed in 0..500u64 {
            let m = policy_move(&g, &peaked_policy(40, 0.5), 1.0, seed);
            assert_ne!(m, 40);
            assert!(g.legal_moves()[m]);
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let g = eye_position();
        let p = peaked_policy(41, 0.3);
        let a = policy_move(&g, &p, 0.7, 999);
        let b = policy_move(&g, &p, 0.7, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn sampling_follows_policy_weights() {
        // Two legal moves, weights 3:1 at temperature 1. Over many seeds
        // both must appear and the heavier must win clearly.
        let g = Game::new();
        let mut p = [0.0f32; N_MOVES];
        p[0] = 0.75;
        p[1] = 0.25;
        // Mask everything else illegal by occupying the board is overkill;
        // instead zero the policy elsewhere (weight 1e-9 each, negligible).
        let (mut n0, mut n1) = (0, 0);
        for seed in 0..2000u64 {
            match policy_move(&g, &p, 1.0, seed) {
                0 => n0 += 1,
                1 => n1 += 1,
                _ => {}
            }
        }
        assert!(n1 > 0, "lighter move never sampled");
        assert!(n0 > n1 * 2, "heavier move should dominate: {n0} vs {n1}");
    }

    #[test]
    fn pass_is_the_fallback() {
        // Full board of black except 40: only pass is legal, and it must
        // be returned rather than panicking.
        let mut g = Game::new();
        let pts: Vec<u8> = (0..81u8).filter(|&p| p != 40).collect();
        g.place_setup_stones(&pts, Color::Black, Color::Black);
        assert_eq!(policy_move(&g, &peaked_policy(40, 0.5), 0.0, 3), 81);
        assert_eq!(policy_move(&g, &peaked_policy(40, 0.5), 1.0, 3), 81);
    }

    #[test]
    fn ignores_illegal_moves() {
        // Occupy 40, then peak the policy there: the argmax must go to a
        // legal move instead.
        let mut g = Game::new();
        g.play(Move::Play(40)).unwrap();
        let m = policy_move(&g, &peaked_policy(40, 0.9), 0.0, 5);
        assert_ne!(m, 40);
        assert!(g.legal_moves()[m]);
    }
}
