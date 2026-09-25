//! GTP engine: the Rust MCTS player, for benchmarking search strength with
//! `weiqi/train/eval_vs_gnugo.py --engine-cmd`.
//!
//! Usage:
//!   mcts_gtp --blob <weights.bin> --sims <n>
//!
//! Speaks GTP on stdin/stdout: protocol_version, name, version, boardsize,
//! clear_board, komi, play, genmove, quit. Deterministic for fixed blob,
//! position, and sim count.

use std::fs;
use std::io::{self, BufRead, Write};
use weiqi_engine::features;
use weiqi_engine::infer::Net;
use weiqi_engine::mcts::{self, Eval, Evaluator, SearchConfig};
use weiqi_engine::rules::{Game, Move};

struct NetEval<'a> {
    net: &'a Net,
}

impl Evaluator for NetEval<'_> {
    fn evaluate(&self, game: &Game) -> Eval {
        let feat = features::encode(game);
        let out = self.net.forward(&feat);
        Eval {
            policy: out.policy,
            value: out.value,
        }
    }
}

/// GTP vertex ("D4", "pass") -> move index (0..80, 81 pass).
fn parse_vertex(s: &str) -> Option<usize> {
    if s.eq_ignore_ascii_case("pass") {
        return Some(81);
    }
    let s = s.to_ascii_uppercase();
    let (col_s, row_s) = s.split_at(1);
    let col = match col_s {
        "A" => 0,
        "B" => 1,
        "C" => 2,
        "D" => 3,
        "E" => 4,
        "F" => 5,
        "G" => 6,
        "H" => 7,
        "J" => 8, // GTP skips 'I'
        _ => return None,
    };
    let row_num: usize = row_s.parse().ok()?;
    if !(1..=9).contains(&row_num) {
        return None;
    }
    Some((9 - row_num) * 9 + col)
}

/// Move index -> GTP vertex.
fn to_vertex(m: usize) -> String {
    if m == 81 {
        return "pass".to_string();
    }
    let cols = ["A", "B", "C", "D", "E", "F", "G", "H", "J"];
    format!("{}{}", cols[m % 9], 9 - m / 9)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut blob_path = None;
    let mut sims: u32 = 100;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--blob" => {
                i += 1;
                blob_path = Some(args[i].clone());
            }
            "--sims" => {
                i += 1;
                sims = args[i].parse().expect("--sims needs a number");
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let blob_path = blob_path.expect("--blob <weights.bin> required");
    let bytes = fs::read(&blob_path).expect("cannot read blob");
    let net = Net::from_fp16_le(&bytes).expect("bad weight blob");
    let eval = NetEval { net: &net };
    let cfg = SearchConfig {
        simulations: sims.max(1),
        dirichlet_eps: 0.15, // match the demo toggle: deliberate at low sims
        ..SearchConfig::default()
    };

    let mut game = Game::new();
    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let mut respond = |ok: bool, payload: &str| {
            if ok {
                write!(out, "= {payload}\n\n").unwrap();
            } else {
                write!(out, "? {payload}\n\n").unwrap();
            }
            out.flush().unwrap();
        };
        match parts[0] {
            "protocol_version" => respond(true, "2"),
            "name" => respond(true, "weiqi-mcts"),
            "version" => respond(true, "0.1.0"),
            "boardsize" => {
                if parts.get(1) == Some(&"9") {
                    respond(true, "")
                } else {
                    respond(false, "only 9x9 supported")
                }
            }
            "clear_board" => {
                let komi = game.komi();
                game = Game::new();
                game.set_komi(komi);
                respond(true, "")
            }
            "komi" => match parts.get(1).and_then(|k| k.parse::<f32>().ok()) {
                Some(k) => {
                    game.set_komi(k);
                    respond(true, "")
                }
                None => respond(false, "bad komi"),
            },
            "play" => {
                let mv = parts.get(2).and_then(|v| parse_vertex(v));
                match mv.and_then(|m| Move::from_index(m)) {
                    Some(mv) => match game.play(mv) {
                        Ok(_) => respond(true, ""),
                        Err(e) => respond(false, &format!("illegal move: {e:?}")),
                    },
                    None => respond(false, "bad vertex"),
                }
            }
            "genmove" => {
                if game.is_over() {
                    respond(true, "pass");
                } else {
                    let m = mcts::search(&game, &eval, &cfg);
                    let mv = Move::from_index(m).unwrap();
                    match game.play(mv) {
                        Ok(_) => respond(true, &to_vertex(m)),
                        Err(e) => respond(false, &format!("search illegal: {e:?}")),
                    }
                }
            }
            "quit" => {
                respond(true, "");
                return;
            }
            _ => respond(false, "unknown command"),
        }
    }
}
