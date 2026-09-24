//! Feature encoder: the 6 LOCKED input planes from PLAN.md §3.
//!
//! Layout: 6 planes × 9×9, row-major, `out[(plane * 9 + row) * 9 + col]`.
//! Row 0 is the **top** of the board, col 0 the **left**
//! (SGF row 1 ↔ row 0, SGF column 'a' ↔ col 0).
//!
//! Planes:
//! - 0: own stones (side to move)
//! - 1: opponent stones
//! - 2: empty points
//! - 3: last move — 1.0 at the last played point; all zeros if there was no
//!   move yet or the last move was a pass
//! - 4: color to play — all 1.0 if Black to move, else all 0.0
//! - 5: ko-banned point — 1.0, else all zeros
//!
//! # Bit-exactness contract with `train/gotrain/features.py`
//!
//! Every value is exactly `0.0` or `1.0` (exactly representable in f32 on both
//! sides), planes are row-major, and row 0 is the top of the board. The two
//! implementations must agree on golden positions; the semantic risks are
//! which side counts as "own", the row orientation, and what counts as the
//! "last move" / ko point — all pinned down above and covered by tests.

use crate::rules::{Color, Game, Move, N_POINTS};

/// Number of input planes.
pub const N_PLANES: usize = 6;
/// Total feature values: 6 × 9 × 9.
pub const FEATURE_LEN: usize = N_PLANES * N_POINTS; // 486

/// Encode a game position as the 6 LOCKED planes, row-major.
pub fn encode(game: &Game) -> [f32; FEATURE_LEN] {
    let mut out = [0.0f32; FEATURE_LEN];
    let me = game.to_move();

    for i in 0..N_POINTS {
        match game.stone_at(i as u8) {
            Some(c) if c == me => out[i] = 1.0,                    // plane 0
            Some(_) => out[N_POINTS + i] = 1.0,                   // plane 1
            None => out[2 * N_POINTS + i] = 1.0,                  // plane 2
        }
    }

    // Plane 3: last move point (zeros if none or pass).
    if let Some(Move::Play(p)) = game.last_move() {
        out[3 * N_POINTS + p as usize] = 1.0;
    }

    // Plane 4: color to play.
    if me == Color::Black {
        for i in 0..N_POINTS {
            out[4 * N_POINTS + i] = 1.0;
        }
    }

    // Plane 5: ko-banned point.
    if let Some(k) = game.ko_point() {
        out[5 * N_POINTS + k as usize] = 1.0;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::idx;

    fn plane(feat: &[f32; FEATURE_LEN], p: usize) -> &[f32] {
        &feat[p * N_POINTS..(p + 1) * N_POINTS]
    }

    #[test]
    fn empty_board_black_to_move() {
        let g = Game::new();
        let f = encode(&g);
        assert!(plane(&f, 0).iter().all(|&x| x == 0.0));
        assert!(plane(&f, 1).iter().all(|&x| x == 0.0));
        assert!(plane(&f, 2).iter().all(|&x| x == 1.0));
        assert!(plane(&f, 3).iter().all(|&x| x == 0.0));
        assert!(plane(&f, 4).iter().all(|&x| x == 1.0));
        assert!(plane(&f, 5).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn after_black_corner_play() {
        // Black plays top-left (0,0); white to move.
        let mut g = Game::new();
        g.play(Move::Play(idx(0, 0) as u8)).unwrap();
        let f = encode(&g);
        let p0 = plane(&f, 0); // own (white): empty
        let p1 = plane(&f, 1); // opponent (black): stone at 0
        let p2 = plane(&f, 2); // empty: everywhere but 0
        let p3 = plane(&f, 3); // last move: point 0
        let p4 = plane(&f, 4); // white to move: zeros
        assert!(p0.iter().all(|&x| x == 0.0));
        assert_eq!(p1[0], 1.0);
        assert!(p1[1..].iter().all(|&x| x == 0.0));
        assert_eq!(p2[0], 0.0);
        assert!(p2[1..].iter().all(|&x| x == 1.0));
        assert_eq!(p3[0], 1.0);
        assert!(p3[1..].iter().all(|&x| x == 0.0));
        assert!(p4.iter().all(|&x| x == 0.0));
        assert!(plane(&f, 5).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pass_clears_last_move_plane() {
        let mut g = Game::new();
        g.play(Move::Play(idx(4, 4) as u8)).unwrap();
        g.play(Move::Pass).unwrap(); // white passes; black to move
        let f = encode(&g);
        // Last move was a pass -> plane 3 all zeros.
        assert!(plane(&f, 3).iter().all(|&x| x == 0.0));
        // Black to move: plane 0 = black stones, plane 4 = ones.
        assert_eq!(plane(&f, 0)[idx(4, 4)], 1.0);
        assert!(plane(&f, 4).iter().all(|&x| x == 1.0));
    }

    #[test]
    fn ko_position_planes() {
        // Rebuild the ko position from the rules tests: black captured at
        // (2,1), ko banned at (1,1) = index 10, white to move.
        let mut g = Game::new();
        g.play(Move::Play(idx(0, 1) as u8)).unwrap(); // B
        g.play(Move::Play(idx(1, 1) as u8)).unwrap(); // W
        g.play(Move::Play(idx(1, 0) as u8)).unwrap(); // B
        g.play(Move::Play(idx(3, 1) as u8)).unwrap(); // W
        g.play(Move::Play(idx(1, 2) as u8)).unwrap(); // B
        g.play(Move::Play(idx(2, 0) as u8)).unwrap(); // W
        g.play(Move::Play(idx(8, 8) as u8)).unwrap(); // B elsewhere
        g.play(Move::Play(idx(2, 2) as u8)).unwrap(); // W
        // Now black captures at (2,1).
        assert_eq!(g.to_move(), crate::rules::Color::Black);
        assert_eq!(g.play(Move::Play(idx(2, 1) as u8)), Ok(1));
        assert_eq!(g.ko_point(), Some(idx(1, 1) as u8));

        let f = encode(&g);
        // White to move: plane 0 = white stones, plane 1 = black stones.
        let p0 = plane(&f, 0);
        let p1 = plane(&f, 1);
        assert_eq!(p0[idx(1, 1)], 0.0); // captured
        assert_eq!(p0[idx(3, 1)], 1.0);
        assert_eq!(p0[idx(2, 0)], 1.0);
        assert_eq!(p0[idx(2, 2)], 1.0);
        for &b in &[idx(0, 1), idx(1, 0), idx(1, 2), idx(2, 1), idx(8, 8)] {
            assert_eq!(p1[b], 1.0);
        }
        // Plane 3: last move at (2,1) = index 19.
        assert_eq!(plane(&f, 3)[idx(2, 1)], 1.0);
        assert_eq!(plane(&f, 3).iter().sum::<f32>(), 1.0);
        // Plane 4: white to move -> zeros. Plane 5: ko at index 10.
        assert!(plane(&f, 4).iter().all(|&x| x == 0.0));
        assert_eq!(plane(&f, 5)[idx(1, 1)], 1.0);
        assert_eq!(plane(&f, 5).iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn values_are_exact_binary() {
        // Every emitted value must be exactly 0.0 or 1.0 (bit-exact vs Python).
        let mut g = Game::new();
        g.play(Move::Play(idx(4, 4) as u8)).unwrap();
        g.play(Move::Play(idx(4, 5) as u8)).unwrap();
        g.play(Move::Pass).unwrap();
        let f = encode(&g);
        assert!(f.iter().all(|&x| x == 0.0 || x == 1.0));
    }
}
