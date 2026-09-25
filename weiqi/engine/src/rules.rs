//! 9×9 Go rules: Tromp–Taylor-style legality, positional superko, Chinese area scoring.
//!
//! Point indexing is row-major, `idx = row * 9 + col`, with row 0 the **top** of the
//! board and col 0 the **left** (matching SGF coordinates: SGF row 1 ↔ row 0,
//! SGF column 'a' ↔ col 0). Use [`idx`] to build indices.
//!
//! Legality summary:
//! - A stone may be played on any empty point that is not ko-banned and does
//!   not repeat a previous board.
//! - Opponent groups left with no liberties are captured (removed).
//! - Suicide is illegal: a move that captures nothing and leaves the placed
//!   stone's own group with no liberties is rejected and the board is unchanged.
//! - Positional superko: a stone play whose resulting board layout matches any
//!   earlier board in this game is illegal ([`IllegalMove::Superko`]). The ban
//!   compares stone layouts only, not side to move. Passes are always legal:
//!   a pass creates no new board, so exempting it is what lets a game reach
//!   two consecutive passes instead of banning every pass as self-repeating.
//!   Repetition is detected with Zobrist hashing (one random `u64` per
//!   point/color from a fixed splitmix64 stream); every board hash is stored
//!   in a `HashSet`, so the check is O(1) per move.
//! - Simple ko is subsumed by superko (an immediate recapture recreates the
//!   board from two plies ago), but the one-move ko ban is kept as a fast
//!   path: it fires first with [`IllegalMove::Ko`], and [`Game::ko_point`]
//!   still reports the ko point for display.
//! - Pass is always legal. Two consecutive passes end the game.
//! - Scoring is Chinese area scoring on the final position as-is (Tromp–Taylor:
//!   no dead-stone removal disputes — everything on the board counts as alive).
//!   Each player's score = stones on board + empty points surrounded solely by
//!   that player; White adds [`KOMI`].

/// Board is 9×9.
pub const BOARD_SIZE: usize = 9;
/// Number of playable points.
pub const N_POINTS: usize = BOARD_SIZE * BOARD_SIZE; // 81
/// Number of distinct moves: 81 points + pass. Pass is move index 81.
pub const N_MOVES: usize = N_POINTS + 1;

/// Komi added to White's area score at game end.
///
/// Provisional per PLAN.md §9 open decision 1 — kept as a const so it is trivial
/// to change.
pub const KOMI: f32 = 7.5;

/// Row-major point index. Row 0 = top of the board, col 0 = left.
pub const fn idx(row: usize, col: usize) -> usize {
    row * BOARD_SIZE + col
}

/// Stone color / player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    Black,
    White,
}

impl Color {
    /// The opposing color.
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    fn index(self) -> usize {
        match self {
            Color::Black => 0,
            Color::White => 1,
        }
    }
}

/// A move: play a stone at a point index (0–80), or pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    /// Play at point `idx` (0–80, row-major).
    Play(u8),
    Pass,
}

impl Move {
    /// Move index in `[bool; 82]` / policy-vector convention: 0–80 points, 81 = pass.
    pub fn to_index(self) -> usize {
        match self {
            Move::Play(p) => p as usize,
            Move::Pass => N_POINTS,
        }
    }

    /// Inverse of [`Move::to_index`]; returns `None` for out-of-range indices.
    pub fn from_index(i: usize) -> Option<Move> {
        if i < N_POINTS {
            Some(Move::Play(i as u8))
        } else if i == N_POINTS {
            Some(Move::Pass)
        } else {
            None
        }
    }
}

/// Why a move was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IllegalMove {
    /// Point is outside 0–80.
    OutOfBounds,
    /// Point already occupied.
    Occupied,
    /// Point is ko-banned this turn (immediate simple-ko recapture).
    Ko,
    /// Positional superko: the resulting board repeats an earlier one.
    Superko,
    /// Suicide: captures nothing and leaves own group without liberties.
    Suicide,
    /// The game already ended (two passes).
    GameOver,
}

/// Final area score. White's value includes komi.
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub black: f32,
    pub white: f32,
}

impl Score {
    /// Winner under the area score (ties impossible with .5 komi).
    pub fn winner(&self) -> Color {
        if self.black > self.white {
            Color::Black
        } else {
            Color::White
        }
    }

    /// Point margin from Black's perspective (positive = Black leads).
    pub fn margin(&self) -> f32 {
        self.black - self.white
    }
}

/// A 9×9 game in progress.
#[derive(Clone, Debug)]
pub struct Game {
    board: [Option<Color>; N_POINTS],
    to_move: Color,
    ko: Option<u8>,
    last_move: Option<Move>,
    consecutive_passes: u8,
    captures: [u32; 2], // [black_captures, white_captures]
    /// Zobrist hashes of every board layout seen so far (including the
    /// initial empty board). Positional superko = "result already in here".
    seen: std::collections::HashSet<u64>,
}

/// SplitMix64: tiny deterministic PRNG for the fixed Zobrist stream.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Zobrist keys: one fixed random `u64` per (point, color).
fn zobrist_table() -> &'static [[u64; 2]; N_POINTS] {
    static TABLE: std::sync::OnceLock<[[u64; 2]; N_POINTS]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut t = [[0u64; 2]; N_POINTS];
        for p in 0..N_POINTS {
            for c in 0..2 {
                t[p][c] = splitmix64(&mut state);
            }
        }
        t
    })
}

/// Zobrist hash of a board: XOR of the keys of every stone.
/// Stones only — positional superko compares layouts, not side to move.
fn board_hash(board: &[Option<Color>; N_POINTS]) -> u64 {
    let t = zobrist_table();
    let mut h = 0u64;
    for (p, s) in board.iter().enumerate() {
        if let Some(c) = s {
            h ^= t[p][c.index()];
        }
    }
    h
}

impl Game {
    /// New empty game, Black to move.
    pub fn new() -> Game {
        Game {
            board: [None; N_POINTS],
            to_move: Color::Black,
            ko: None,
            last_move: None,
            consecutive_passes: 0,
            captures: [0, 0],
            seen: std::collections::HashSet::from([board_hash(&[None; N_POINTS])]),
        }
    }

    /// Side to move.
    pub fn to_move(&self) -> Color {
        self.to_move
    }

    /// Stone at point `p` (0–80), if any.
    pub fn stone_at(&self, p: u8) -> Option<Color> {
        self.board[p as usize]
    }

    /// The ko-banned point, if the previous move created a simple ko.
    pub fn ko_point(&self) -> Option<u8> {
        self.ko
    }

    /// The most recent move, or `None` before any move.
    pub fn last_move(&self) -> Option<Move> {
        self.last_move
    }

    /// Stones captured by `color` so far.
    pub fn captures(&self, color: Color) -> u32 {
        self.captures[color.index()]
    }

    /// Number of consecutive passes.
    pub fn consecutive_passes(&self) -> u8 {
        self.consecutive_passes
    }

    /// The game ends after two consecutive passes.
    pub fn is_over(&self) -> bool {
        self.consecutive_passes >= 2
    }

    /// Legality mask over all 82 moves (index 81 = pass, always legal).
    pub fn legal_moves(&self) -> [bool; N_MOVES] {
        let mut mask = [false; N_MOVES];
        if self.is_over() {
            return mask;
        }
        mask[N_POINTS] = true; // pass
        for p in 0..N_POINTS as u8 {
            let mut probe = self.clone();
            mask[p as usize] = probe.play(Move::Play(p)).is_ok();
        }
        mask
    }

    /// Play a move. Returns the number of stones captured, or why it was illegal.
    /// On error the game is unchanged.
    pub fn play(&mut self, m: Move) -> Result<u8, IllegalMove> {
        if self.is_over() {
            return Err(IllegalMove::GameOver);
        }
        match m {
            Move::Pass => {
                self.consecutive_passes += 1;
                self.ko = None;
                self.last_move = Some(Move::Pass);
                self.to_move = self.to_move.opponent();
                Ok(0)
            }
            Move::Play(p) => {
                if p as usize >= N_POINTS {
                    return Err(IllegalMove::OutOfBounds);
                }
                if self.board[p as usize].is_some() {
                    return Err(IllegalMove::Occupied);
                }
                if self.ko == Some(p) {
                    return Err(IllegalMove::Ko);
                }
                let me = self.to_move;
                let foe = me.opponent();

                // Tentatively place the stone.
                self.board[p as usize] = Some(me);

                // Remove opponent groups left without liberties.
                let mut captured: Vec<u8> = Vec::new();
                for nb in neighbors(p).into_iter().flatten() {
                    if self.board[nb as usize] == Some(foe) && !captured.contains(&nb) {
                        let (stones, libs) = self.group_liberties(nb);
                        if libs.is_empty() {
                            captured.extend(stones.iter());
                        }
                    }
                }
                for s in captured.iter() {
                    self.board[*s as usize] = None;
                }

                // Suicide check: no capture and own group has no liberties.
                let (own_stones, own_libs) = self.group_liberties(p);
                if own_libs.is_empty() {
                    self.board[p as usize] = None;
                    for s in captured.iter() {
                        self.board[*s as usize] = Some(foe);
                    }
                    return Err(IllegalMove::Suicide);
                }

                // Positional superko: the resulting board must not repeat any
                // earlier board in this game. (Passes are exempt — see module
                // docs — and never reach this branch.)
                let h = board_hash(&self.board);
                if self.seen.contains(&h) {
                    self.board[p as usize] = None;
                    for s in captured.iter() {
                        self.board[*s as usize] = Some(foe);
                    }
                    return Err(IllegalMove::Superko);
                }
                self.seen.insert(h);

                // Simple ko: exactly one stone captured, placed stone is alone,
                // and its group has exactly one liberty (the captured point).
                // Immediate recapture there would recreate the previous position.
                self.ko = if captured.len() == 1
                    && own_stones.len() == 1
                    && own_libs.len() == 1
                {
                    Some(captured[0])
                } else {
                    None
                };

                self.captures[me.index()] += captured.len() as u32;
                self.last_move = Some(Move::Play(p));
                self.consecutive_passes = 0;
                self.to_move = foe;
                Ok(captured.len() as u8)
            }
        }
    }

    /// Chinese area score of the current position: stones on board plus empty
    /// points surrounded solely by one color. All stones count as alive
    /// (Tromp–Taylor: no dead-stone removal). White's score includes [`KOMI`].
    pub fn score(&self) -> Score {
        let mut black = 0.0f32;
        let mut white = 0.0f32;
        for s in self.board.iter().flatten() {
            match s {
                Color::Black => black += 1.0,
                Color::White => white += 1.0,
            }
        }
        let mut visited = [false; N_POINTS];
        for i in 0..N_POINTS {
            if self.board[i].is_none() && !visited[i] {
                let mut stack = vec![i as u8];
                visited[i] = true;
                let mut size = 0u32;
                let mut touches_black = false;
                let mut touches_white = false;
                while let Some(q) = stack.pop() {
                    size += 1;
                    for nb in neighbors(q).into_iter().flatten() {
                        match self.board[nb as usize] {
                            None => {
                                if !visited[nb as usize] {
                                    visited[nb as usize] = true;
                                    stack.push(nb);
                                }
                            }
                            Some(Color::Black) => touches_black = true,
                            Some(Color::White) => touches_white = true,
                        }
                    }
                }
                match (touches_black, touches_white) {
                    (true, false) => black += size as f32,
                    (false, true) => white += size as f32,
                    _ => {} // dame / open area: no points
                }
            }
        }
        white += KOMI;
        Score { black, white }
    }

    /// Per-point ownership of the current position: 0 = neutral (dame, open
    /// area, or empty), 1 = black (stone or surrounded territory),
    /// 2 = white (stone or surrounded territory). Same Tromp–Taylor flood
    /// fill as [`Game::score`], so the counts agree with the area scores.
    /// Meant for end-of-game territory shading in the UI.
    pub fn territory(&self) -> [u8; N_POINTS] {
        let mut own = [0u8; N_POINTS];
        for (i, s) in self.board.iter().enumerate() {
            own[i] = match s {
                None => 0,
                Some(Color::Black) => 1,
                Some(Color::White) => 2,
            };
        }
        let mut visited = [false; N_POINTS];
        for i in 0..N_POINTS {
            if self.board[i].is_none() && !visited[i] {
                let mut stack = vec![i as u8];
                visited[i] = true;
                let mut region = vec![];
                let mut touches_black = false;
                let mut touches_white = false;
                while let Some(q) = stack.pop() {
                    region.push(q);
                    for nb in neighbors(q).into_iter().flatten() {
                        match self.board[nb as usize] {
                            None => {
                                if !visited[nb as usize] {
                                    visited[nb as usize] = true;
                                    stack.push(nb);
                                }
                            }
                            Some(Color::Black) => touches_black = true,
                            Some(Color::White) => touches_white = true,
                        }
                    }
                }
                let v = match (touches_black, touches_white) {
                    (true, false) => 1,
                    (false, true) => 2,
                    _ => 0,
                };
                for q in region {
                    own[q as usize] = v;
                }
            }
        }
        own
    }

    /// Stones and liberties of the group containing point `start`
    /// (which must hold a stone).
    fn group_liberties(&self, start: u8) -> (Vec<u8>, Vec<u8>) {
        let color = self.board[start as usize].expect("group_liberties on empty point");
        let mut stones = Vec::new();
        let mut libs = Vec::new();
        let mut seen = [false; N_POINTS];
        let mut stack = vec![start];
        seen[start as usize] = true;
        while let Some(q) = stack.pop() {
            stones.push(q);
            for nb in neighbors(q).into_iter().flatten() {
                match self.board[nb as usize] {
                    Some(c) if c == color => {
                        if !seen[nb as usize] {
                            seen[nb as usize] = true;
                            stack.push(nb);
                        }
                    }
                    None => {
                        if !libs.contains(&nb) {
                            libs.push(nb);
                        }
                    }
                    _ => {}
                }
            }
        }
        (stones, libs)
    }

    /// Test-only helper: place a stone directly, bypassing legality.
    #[cfg(test)]
    fn set(&mut self, color: Color, row: usize, col: usize) {
        self.board[idx(row, col)] = Some(color);
    }

    /// Test-only helper: set the side to move.
    #[cfg(test)]
    fn set_to_move(&mut self, color: Color) {
        self.to_move = color;
    }
}

impl Default for Game {
    fn default() -> Self {
        Game::new()
    }
}

/// The four orthogonal neighbors of point `p`, as `None` where off-board.
fn neighbors(p: u8) -> [Option<u8>; 4] {
    let r = p / BOARD_SIZE as u8;
    let c = p % BOARD_SIZE as u8;
    let up = if r > 0 { Some((r - 1) * BOARD_SIZE as u8 + c) } else { None };
    let down = if r + 1 < BOARD_SIZE as u8 {
        Some((r + 1) * BOARD_SIZE as u8 + c)
    } else {
        None
    };
    let left = if c > 0 { Some(r * BOARD_SIZE as u8 + (c - 1)) } else { None };
    let right = if c + 1 < BOARD_SIZE as u8 {
        Some(r * BOARD_SIZE as u8 + (c + 1))
    } else {
        None
    };
    [up, down, left, right]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_seq(moves: &[(Color, usize, usize)]) -> Game {
        let mut g = Game::new();
        for (color, r, c) in moves {
            assert_eq!(g.to_move(), *color, "wrong side to move");
            g.play(Move::Play(idx(*r, *c) as u8))
                .expect("setup move should be legal");
        }
        g
    }

    #[test]
    fn capture_two_stones() {
        // White stones at (0,0),(0,1); black surrounds and fills last liberty.
        let mut g = Game::new();
        g.set(Color::White, 0, 0);
        g.set(Color::White, 0, 1);
        g.set(Color::Black, 1, 0);
        g.set(Color::Black, 1, 1);
        g.set_to_move(Color::Black);
        assert_eq!(g.play(Move::Play(idx(0, 2) as u8)), Ok(2));
        assert_eq!(g.stone_at(idx(0, 0) as u8), None);
        assert_eq!(g.stone_at(idx(0, 1) as u8), None);
        assert_eq!(g.stone_at(idx(0, 2) as u8), Some(Color::Black));
        assert_eq!(g.captures(Color::Black), 2);
        assert_eq!(g.to_move(), Color::White);
    }

    #[test]
    fn snapback_is_not_ko() {
        // White throws in at (0,0); black captures it at (1,0); white recaptures
        // at (0,0) taking three black stones. The recapture takes MORE than the
        // single capturing stone, so it is not a ko — it must be legal.
        //
        //   . X O . .      (row 0)
        //   . X O . .      (row 1)
        //   O O . . .      (row 2)
        let mut g = Game::new();
        g.set(Color::Black, 0, 1);
        g.set(Color::Black, 1, 1);
        g.set(Color::White, 0, 2);
        g.set(Color::White, 1, 2);
        g.set(Color::White, 2, 1);
        g.set(Color::White, 2, 0);
        g.set_to_move(Color::White);

        // (i) throw-in
        assert_eq!(g.play(Move::Play(idx(0, 0) as u8)), Ok(0));
        // (ii) black captures the throw-in stone
        assert_eq!(g.play(Move::Play(idx(1, 0) as u8)), Ok(1));
        // not a ko: black's capturing group has 3 stones
        assert_eq!(g.ko_point(), None);
        // (iii) snapback recapture takes all three black stones
        assert_eq!(g.play(Move::Play(idx(0, 0) as u8)), Ok(3));
        assert_eq!(g.stone_at(idx(0, 1) as u8), None);
        assert_eq!(g.stone_at(idx(1, 1) as u8), None);
        assert_eq!(g.stone_at(idx(1, 0) as u8), None);
        assert_eq!(g.stone_at(idx(0, 0) as u8), Some(Color::White));
        assert_eq!(g.captures(Color::White), 3);
    }

    #[test]
    fn simple_ko_recapture_illegal_then_allowed_after_ko_threat() {
        // Classic ko:
        //   . X . . .      (row 0)
        //   X O X . .      (row 1)
        //   . X . . .      (row 2)
        //   . O . . .      (row 3)
        // Black fills white's last liberty at (2,1), capturing (1,1); the
        // capturing stone has exactly one liberty -> ko at (1,1).
        let mut g = Game::new();
        g.set(Color::Black, 0, 1);
        g.set(Color::Black, 1, 0);
        g.set(Color::Black, 1, 2);
        g.set(Color::White, 1, 1);
        g.set(Color::White, 3, 1);
        g.set(Color::White, 2, 0);
        g.set(Color::White, 2, 2);
        g.set_to_move(Color::Black);

        assert_eq!(g.play(Move::Play(idx(2, 1) as u8)), Ok(1));
        assert_eq!(g.ko_point(), Some(idx(1, 1) as u8));

        // Immediate recapture is illegal (ko) ...
        assert_eq!(g.play(Move::Play(idx(1, 1) as u8)), Err(IllegalMove::Ko));
        // ... but playing elsewhere is fine and lifts the ban.
        assert_eq!(g.play(Move::Play(idx(8, 8) as u8)), Ok(0));
        assert_eq!(g.ko_point(), None);
        // Black answers elsewhere; white recaptures, capturing one stone,
        // which creates a ko back the other way at (2,1).
        assert_eq!(g.play(Move::Play(idx(8, 0) as u8)), Ok(0));
        assert_eq!(g.play(Move::Play(idx(1, 1) as u8)), Ok(1));
        assert_eq!(g.ko_point(), Some(idx(2, 1) as u8));
        assert_eq!(g.play(Move::Play(idx(2, 1) as u8)), Err(IllegalMove::Ko));
    }

    #[test]
    fn suicide_illegal() {
        //   . X .
        //   X . .
        // White at (0,0): no liberties, captures nothing -> suicide.
        let mut g = Game::new();
        g.set(Color::Black, 0, 1);
        g.set(Color::Black, 1, 0);
        g.set_to_move(Color::White);
        assert_eq!(g.play(Move::Play(idx(0, 0) as u8)), Err(IllegalMove::Suicide));
        // Board unchanged.
        assert_eq!(g.stone_at(idx(0, 0) as u8), None);
        assert_eq!(g.stone_at(idx(0, 1) as u8), Some(Color::Black));
        assert_eq!(g.to_move(), Color::White);
    }

    #[test]
    fn occupied_and_pass_rules() {
        let mut g = play_seq(&[(Color::Black, 4, 4)]);
        assert_eq!(
            g.play(Move::Play(idx(4, 4) as u8)),
            Err(IllegalMove::Occupied)
        );
        assert_eq!(g.play(Move::Pass), Ok(0));
        assert!(!g.is_over());
        assert_eq!(g.play(Move::Pass), Ok(0));
        assert!(g.is_over());
        assert_eq!(g.play(Move::Pass), Err(IllegalMove::GameOver));
        assert_eq!(
            g.play(Move::Play(idx(0, 0) as u8)),
            Err(IllegalMove::GameOver)
        );
    }

    #[test]
    fn pass_resets_consecutive_counter() {
        let mut g = Game::new();
        g.play(Move::Pass).unwrap();
        g.play(Move::Play(idx(4, 4) as u8)).unwrap();
        g.play(Move::Pass).unwrap();
        assert!(!g.is_over());
        g.play(Move::Pass).unwrap();
        assert!(g.is_over());
    }

    #[test]
    fn score_corner_territories() {
        // Black owns (0,0), white owns (8,8); everything else is dame.
        let mut g = Game::new();
        g.set(Color::Black, 0, 1);
        g.set(Color::Black, 1, 0);
        g.set(Color::White, 8, 7);
        g.set(Color::White, 7, 8);
        let s = g.score();
        assert_eq!(s.black, 3.0); // 2 stones + 1 point
        assert_eq!(s.white, 2.0 + 1.0 + KOMI); // 2 stones + 1 point + komi
        assert_eq!(s.winner(), Color::White);
    }

    #[test]
    fn territory_matches_score() {
        // Black owns (0,0), white owns (8,8); everything else is dame.
        let mut g = Game::new();
        g.set(Color::Black, 0, 1);
        g.set(Color::Black, 1, 0);
        g.set(Color::White, 8, 7);
        g.set(Color::White, 7, 8);
        let t = g.territory();
        assert_eq!(t[idx(0, 0) as usize], 1);
        assert_eq!(t[idx(0, 1) as usize], 1); // black stones read back as 1
        assert_eq!(t[idx(1, 0) as usize], 1);
        assert_eq!(t[idx(8, 8) as usize], 2);
        assert_eq!(t[idx(8, 7) as usize], 2);
        assert_eq!(t[idx(7, 8) as usize], 2);
        assert_eq!(t[idx(4, 4) as usize], 0); // dame
        // Counts agree with the area score (minus komi, which is not spatial).
        let s = g.score();
        assert_eq!(t.iter().filter(|&&v| v == 1).count() as f32, s.black);
        assert_eq!(
            t.iter().filter(|&&v| v == 2).count() as f32,
            s.white - KOMI
        );
    }

    #[test]
    fn score_enclosed_corners() {
        // Black walls off rows 0-2, cols 0-2 (7 stones + 9 territory);
        // white mirrors in the opposite corner. The middle is dame.
        let mut g = Game::new();
        for r in 0..=2 {
            g.set(Color::Black, r, 3);
        }
        for c in 0..=3 {
            g.set(Color::Black, 3, c);
        }
        for r in 6..=8 {
            g.set(Color::White, r, 5);
        }
        for c in 5..=8 {
            g.set(Color::White, 5, c);
        }
        let s = g.score();
        assert_eq!(s.black, 16.0); // 7 stones + 9 points
        assert_eq!(s.white, 7.0 + 9.0 + KOMI); // 7 stones + 9 points + komi
        assert_eq!(s.winner(), Color::White);
        assert!((s.margin() + KOMI).abs() < 1e-6); // black trails by exactly komi
    }

    #[test]
    fn full_game_two_passes_then_score() {
        let mut g = play_seq(&[
            (Color::Black, 2, 2),
            (Color::White, 6, 6),
            (Color::Black, 2, 6),
            (Color::White, 6, 2),
        ]);
        g.play(Move::Pass).unwrap();
        g.play(Move::Pass).unwrap();
        assert!(g.is_over());
        let s = g.score();
        // 2 stones each, no enclosed territory, komi decides.
        assert_eq!(s.black, 2.0);
        assert_eq!(s.white, 2.0 + KOMI);
        assert_eq!(s.winner(), Color::White);
    }

    #[test]
    fn legal_moves_mask() {
        let g = Game::new();
        let mask = g.legal_moves();
        assert!(mask.iter().all(|&b| b)); // everything legal on empty board
        let mut g2 = Game::new();
        g2.set(Color::Black, 0, 1);
        g2.set(Color::Black, 1, 0);
        g2.set_to_move(Color::White);
        let mask2 = g2.legal_moves();
        assert!(!mask2[idx(0, 0)]); // suicide
        assert!(mask2[idx(0, 2)]);
        assert!(mask2[N_POINTS]); // pass always legal
    }

    #[test]
    fn move_index_roundtrip() {
        assert_eq!(Move::Play(40).to_index(), 40);
        assert_eq!(Move::Pass.to_index(), 81);
        assert_eq!(Move::from_index(0), Some(Move::Play(0)));
        assert_eq!(Move::from_index(81), Some(Move::Pass));
        assert_eq!(Move::from_index(82), None);
    }

    #[test]
    fn positional_superko_bans_triple_ko_cycle() {
        // Three independent ko shapes. Under simple ko alone, cycling takes
        // through them (B k1, W k2, B k3, W k1, B k2, W k3, ...) loops
        // forever — every take is at a different ko than the last. But the
        // 6th take recreates the exact initial board, so positional superko
        // must reject it.
        fn ko_shape(g: &mut Game, victim: Color, r: usize, c: usize) {
            // victim's stone at (r+1,c+1) has one liberty at (r+2,c+1);
            // the capturer takes it there, leaving the ko at (r+1,c+1).
            let wall = victim.opponent();
            g.set(wall, r, c + 1);
            g.set(wall, r + 1, c);
            g.set(wall, r + 1, c + 2);
            g.set(victim, r + 1, c + 1);
            g.set(victim, r + 3, c + 1);
            g.set(victim, r + 2, c);
            g.set(victim, r + 2, c + 2);
        }
        let mut g = Game::new();
        ko_shape(&mut g, Color::White, 0, 0); // k1: B takes (2,1), ko (1,1)
        ko_shape(&mut g, Color::Black, 0, 4); // k2: W takes (2,5), ko (1,5)
        ko_shape(&mut g, Color::White, 5, 0); // k3: B takes (7,1), ko (6,1)
        g.set_to_move(Color::Black);
        // set() bypasses move history; the built position counts as seen.
        g.seen.insert(board_hash(&g.board));

        let takes = [
            (Color::Black, 2, 1), // T1: B takes k1
            (Color::White, 2, 5), // T2: W takes k2
            (Color::Black, 7, 1), // T3: B takes k3
            (Color::White, 1, 1), // T4: W retakes k1
            (Color::Black, 1, 5), // T5: B retakes k2
        ];
        for (color, r, c) in takes {
            assert_eq!(g.to_move(), color, "wrong side before take at ({r},{c})");
            assert!(
                g.play(Move::Play(idx(r, c) as u8)).is_ok(),
                "take at ({r},{c}) should be legal"
            );
        }
        // T6: W retakes k3 — the board would exactly repeat the initial one.
        assert_eq!(g.to_move(), Color::White);
        assert_eq!(
            g.play(Move::Play(idx(6, 1) as u8)),
            Err(IllegalMove::Superko)
        );
        // Rejected move leaves the board unchanged.
        assert_eq!(g.stone_at(idx(6, 1) as u8), None);
        assert_eq!(g.stone_at(idx(7, 1) as u8), Some(Color::Black));
        assert_eq!(g.to_move(), Color::White);
    }

    #[test]
    fn zobrist_hash_order_independent() {
        // Same board reached via different move orders hashes identically.
        let mut a = Game::new();
        for (r, c) in [(0, 0), (8, 8), (0, 1), (8, 7)] {
            a.play(Move::Play(idx(r, c) as u8)).unwrap();
        }
        let mut b = Game::new();
        for (r, c) in [(0, 1), (8, 7), (0, 0), (8, 8)] {
            b.play(Move::Play(idx(r, c) as u8)).unwrap();
        }
        assert_eq!(board_hash(&a.board), board_hash(&b.board));
        assert_eq!(a.seen.len(), b.seen.len());
    }

    #[test]
    fn passes_stay_legal_under_superko() {
        // A pass creates no new board, so it must stay legal even though its
        // "resulting board" trivially repeats the current one.
        let mut g = Game::new();
        g.play(Move::Play(idx(4, 4) as u8)).unwrap();
        assert_eq!(g.play(Move::Pass), Ok(0));
        assert_eq!(g.play(Move::Pass), Ok(0));
        assert!(g.is_over());
    }
}
