//! Differential-fuzz driver for the Rust rules engine.
//!
//! Reads a fixture from stdin, plays it on `weiqi_engine::Game`, and prints a
//! canonical digest after every move so a Python harness can diff the Rust
//! engine against `gotrain.rules` and an independent naive reference.
//!
//! Protocol (one command per line):
//!   new                  -> start a new game, prints `game`
//!   play <0-80|pass>     -> attempt the move as the side to move;
//!                           prints `ok <digest>` or `err <IllegalMove>`
//!   score                -> prints `score <black> <white>`
//!
//! Digest format:
//!   to_move=<B|W> ko=<idx|-> caps=<black>,<white>
//!   stones=<81 chars, 0/1/2> legal=<hex of 82-bit mask, bit i = move i>
//!
//! Build: cargo build --example fuzz_driver
//! Use:   python3 fuzz_three_way.py < fixture.txt   (see train/tests/)

use std::io::{self, BufRead};
use weiqi_engine::{Color, Game, Move};

fn digest(g: &Game) -> String {
    let to_move = match g.to_move() {
        Color::Black => "B",
        Color::White => "W",
    };
    let ko = match g.ko_point() {
        Some(p) => p.to_string(),
        None => "-".to_string(),
    };
    let mut stones = String::with_capacity(81);
    for p in 0..81u8 {
        stones.push(match g.stone_at(p) {
            None => '0',
            Some(Color::Black) => '1',
            Some(Color::White) => '2',
        });
    }
    let mask = g.legal_moves();
    let mut bits: u128 = 0;
    for (i, m) in mask.iter().enumerate() {
        if *m {
            bits |= 1 << i;
        }
    }
    format!(
        "to_move={} ko={} caps={},{} stones={} legal={:x}",
        to_move,
        ko,
        g.captures(Color::Black),
        g.captures(Color::White),
        stones,
        bits
    )
}

fn main() {
    let mut game = Game::new();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next().unwrap() {
            "new" => {
                game = Game::new();
                println!("game");
            }
            "play" => {
                let arg = parts.next().unwrap_or("");
                let mv = if arg == "pass" {
                    Move::Pass
                } else {
                    match Move::from_index(arg.parse::<usize>().unwrap_or(999)) {
                        Some(m) => m,
                        None => {
                            println!("err OutOfBounds");
                            continue;
                        }
                    }
                };
                match game.play(mv) {
                    Ok(_) => println!("ok {}", digest(&game)),
                    Err(e) => println!("err {:?}", e),
                }
            }
            "score" => {
                let s = game.score();
                println!("score {} {}", s.black, s.white);
            }
            other => {
                println!("err unknown command: {}", other);
            }
        }
    }
}
