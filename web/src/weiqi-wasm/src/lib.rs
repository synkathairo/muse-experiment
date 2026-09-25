//! wasm-bindgen shim over `weiqi-engine` for the browser demo.
//!
//! Two classes cross the boundary:
//!
//! - [`WasmGame`]: a 9x9 game (rules, legality, scoring). The page drives it
//!   directly for human moves and reads state back for rendering.
//! - [`WasmNet`]: the LOCKED GoNet arch with hand-rolled f32 inference.
//!   Constructed from a fetched fp16 weight blob; `policy`/`value` encode the
//!   position with the bit-exact feature planes and run the forward pass.
//!
//! Move-index convention everywhere: 0–80 = board points row-major
//! (row 0 = top), 81 = pass.

use wasm_bindgen::prelude::*;
use weiqi_engine::{
    features,
    infer::Net,
    mcts::{self, Eval, Evaluator, SearchConfig},
    Color, Game, Move,
};

/// A 9x9 game in progress.
#[wasm_bindgen]
pub struct WasmGame {
    inner: Game,
}

#[wasm_bindgen]
impl WasmGame {
    /// New empty game, Black to move.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGame {
        WasmGame { inner: Game::new() }
    }

    /// Side to move: 0 = black, 1 = white.
    pub fn to_move(&self) -> u8 {
        match self.inner.to_move() {
            Color::Black => 0,
            Color::White => 1,
        }
    }

    /// Stone at point `p` (0–80): 0 = empty, 1 = black, 2 = white.
    pub fn stone_at(&self, p: usize) -> u8 {
        match self.inner.stone_at(p as u8) {
            None => 0,
            Some(Color::Black) => 1,
            Some(Color::White) => 2,
        }
    }

    /// Ko-banned point this turn, or 81 when there is none.
    pub fn ko_point(&self) -> usize {
        self.inner.ko_point().map(|p| p as usize).unwrap_or(81)
    }

    /// Last move index (81 = pass), or 82 before any move.
    pub fn last_move(&self) -> usize {
        self.inner
            .last_move()
            .map(|m| m.to_index())
            .unwrap_or(82)
    }

    /// True once the game ended (two consecutive passes).
    pub fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// Play move `idx` (0–80 point, 81 pass). Returns false if illegal or
    /// out of range; the game is unchanged in that case.
    pub fn play(&mut self, idx: usize) -> bool {
        match Move::from_index(idx) {
            Some(m) => self.inner.play(m).is_ok(),
            None => false,
        }
    }

    /// 82 bytes, 1 = legal move (index 81 = pass).
    pub fn legal_mask(&self) -> Vec<u8> {
        self.inner.legal_moves().iter().map(|&b| b as u8).collect()
    }

    /// Stones captured by Black / White so far.
    pub fn captures_black(&self) -> u32 {
        self.inner.captures(Color::Black)
    }
    pub fn captures_white(&self) -> u32 {
        self.inner.captures(Color::White)
    }

    /// Final area scores (White includes this game's komi). Meaningful at
    /// game end.
    pub fn score_black(&self) -> f32 {
        self.inner.score().black
    }
    pub fn score_white(&self) -> f32 {
        self.inner.score().white
    }

    /// This game's komi (default 7.5).
    pub fn komi(&self) -> f32 {
        self.inner.komi()
    }

    /// Set this game's komi. Scoring only — the bot's moves never change.
    pub fn set_komi(&mut self, komi: f32) {
        self.inner.set_komi(komi);
    }

    /// Per-point ownership (81 bytes): 0 = neutral, 1 = black, 2 = white.
    /// Counts agree with the area scores (minus komi). For end-of-game
    /// territory shading.
    pub fn territory(&self) -> Vec<u8> {
        self.inner.territory().to_vec()
    }
}

/// The LOCKED GoNet with f32 weights, loaded from a fetched blob.
#[wasm_bindgen]
pub struct WasmNet {
    inner: Net,
}

#[wasm_bindgen]
impl WasmNet {
    /// Load from a flat fp16 little-endian blob in LOCKED tensor order
    /// (261,044 bytes). Throws on length mismatch.
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmNet, JsValue> {
        Net::from_fp16_le(bytes)
            .map(|inner| WasmNet { inner })
            .map_err(|e| JsValue::from_str(&format!("bad weight blob: {e:?}")))
    }

    /// Softmax policy for the side to move: 82 probabilities, index 81 = pass.
    pub fn policy(&self, game: &WasmGame) -> Vec<f32> {
        let feat = features::encode(&game.inner);
        self.inner.forward(&feat).policy.to_vec()
    }

    /// Win-probability estimate for the side to move, in [-1, 1].
    pub fn value(&self, game: &WasmGame) -> f32 {
        let feat = features::encode(&game.inner);
        self.inner.forward(&feat).value
    }

    /// PUCT search move for `game`'s current position: `simulations` tree
    /// iterations guided by this net's policy (priors) and value (leaves).
    /// Returns a move index (0–80 point, 81 pass), with a touch of root
    /// Dirichlet noise so repeated calls vary slightly.
    pub fn search(&self, game: &WasmGame, simulations: u32) -> usize {
        search_best_move(game, self, simulations, 0.05)
    }
}

impl Evaluator for WasmNet {
    /// Single forward pass: policy priors + leaf value together.
    fn evaluate(&self, game: &Game) -> Eval {
        let feat = features::encode(game);
        let out = self.inner.forward(&feat);
        Eval {
            policy: out.policy,
            value: out.value,
        }
    }
}

/// PUCT search over `game` using `net`, returning the move index (0–80
/// point, 81 pass). `dirichlet_eps` adds root exploration noise (0 disables).
/// Deterministic for a fixed position, net, and simulation count.
#[wasm_bindgen]
pub fn search_best_move(
    game: &WasmGame,
    net: &WasmNet,
    simulations: u32,
    dirichlet_eps: f32,
) -> usize {
    if game.inner.is_over() {
        return 81;
    }
    let cfg = SearchConfig {
        simulations: simulations.max(1),
        dirichlet_eps,
        ..SearchConfig::default()
    };
    mcts::search(&game.inner, net, &cfg)
}
