//! The UCI protocol, so chess GUIs such as Cute Chess or Arena, and other engines, can play a
//! run's network.
//!
//! Each `go` runs a fresh search from the position, evaluating one position at a time, so the GPU
//! does far less work per call than in self-play: about 150 simulations a second on the laptop.
//! The search stops at the node count or time its `go` asks for, or on `stop`, and plays the most
//! visited move, without noise or sampling. Without any limit it runs the default simulations
//! given at startup.

use crate::{
    chess::{moves::Move, state::START_FEN, State},
    engine::{
        mcts::{MctsConfig, Search},
        model::TransformerModel,
    },
};
use burn::tensor::backend::Backend;
use std::{
    io::{self, Write},
    sync::mpsc::Receiver,
    time::{Duration, Instant},
};

const NAME: &str = concat!("Cinder ", env!("CARGO_PKG_VERSION"));
// Room for the GUI's own overhead when the search has a time limit
const TIME_MARGIN: Duration = Duration::from_millis(50);
// Moves the remaining time is shared among when the GUI doesn't say
const DEFAULT_MOVES_TO_GO: u64 = 30;
// The most simulations a search runs, for `go infinite` among others: about 1 GB of tree
const MAX_SIMS: usize = 1_000_000;
const INFO_INTERVAL: Duration = Duration::from_secs(1);
const WARM_UP_SIMS: usize = 16;

/// Answers UCI commands from `input` on `output` until `quit` or the end of the input.
pub(crate) fn run<B: Backend>(
    model: &TransformerModel<B>,
    default_sims: usize,
    device: &B::Device,
    input: Receiver<String>,
    output: impl Write,
) -> io::Result<()> {
    let mut engine = Engine {
        model,
        device,
        default_sims,
        state: State::from_fen(START_FEN),
        input,
        output,
        warmed_up: false,
    };
    while let Ok(line) = engine.input.recv() {
        if !engine.handle(&line)? {
            break;
        }
    }
    Ok(())
}

struct Engine<'a, B: Backend, W: Write> {
    model: &'a TransformerModel<B>,
    device: &'a B::Device,
    default_sims: usize,
    state: State,
    input: Receiver<String>,
    output: W,
    warmed_up: bool,
}

/// What a `go` asks for: a node count, a time, or neither, which means the default simulations.
#[derive(Debug, Default, PartialEq)]
struct Limits {
    nodes: Option<usize>,
    time: Option<Duration>,
    infinite: bool,
}

impl<B: Backend, W: Write> Engine<'_, B, W> {
    // Returns false on `quit`
    fn handle(&mut self, line: &str) -> io::Result<bool> {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.first().copied() {
            Some("uci") => {
                self.send(&format!("id name {NAME}"))?;
                self.send("id author Kevin Chang")?;
                self.send("uciok")?;
            }
            Some("isready") => {
                self.warm_up();
                self.send("readyok")?
            }
            Some("ucinewgame") => self.state = State::from_fen(START_FEN),
            Some("position") => match parse_position(&words[1..]) {
                Ok(state) => self.state = state,
                Err(error) => self.send(&format!("info string {error}"))?,
            },
            Some("go") => {
                let limits = parse_go(&words[1..], &self.state);
                return self.go(&limits);
            }
            Some("quit") => return Ok(false),
            // `stop` outside a search, and anything unknown, which UCI says to ignore
            _ => {}
        }
        Ok(true)
    }

    // The first searches on a GPU spend seconds compiling kernels, so one short search runs before
    // the first `readyok`, which GUIs wait for, or else before the first move
    fn warm_up(&mut self) {
        if self.warmed_up {
            return;
        }
        let mut state = State::from_fen(START_FEN);
        let mut search = Search::new(MctsConfig::new().with_n_sims(WARM_UP_SIMS));
        while search.next_leaf(&mut state) {
            let evaluation = self
                .model
                .evaluate(&[&state], &[search.leaf_moves()], self.device)
                .remove(0);
            search.apply(&mut state, evaluation);
        }
        self.warmed_up = true;
    }

    // Searches the position and sends the best move. Returns false if `quit` came meanwhile.
    fn go(&mut self, limits: &Limits) -> io::Result<bool> {
        self.warm_up();
        let moves = self.state.generate_moves();
        if moves.is_empty() {
            self.send("bestmove 0000")?;
            return Ok(true);
        }

        let start = Instant::now();
        let n_sims = if limits.infinite || limits.time.is_some() && limits.nodes.is_none() {
            MAX_SIMS
        } else {
            limits.nodes.unwrap_or(self.default_sims).clamp(2, MAX_SIMS)
        };
        let mut search = Search::new(MctsConfig::new().with_n_sims(n_sims));
        // Searches return the position to where they started. It's taken out for the search, with
        // its history, which repetitions are found in, and put back after.
        let mut state = std::mem::replace(&mut self.state, State::from_fen(START_FEN));
        let mut last_info = start;
        let mut keep_going = true;
        let mut stop = false;

        while search.next_leaf(&mut state) {
            let evaluation = self
                .model
                .evaluate(&[&state], &[search.leaf_moves()], self.device)
                .remove(0);
            search.apply(&mut state, evaluation);

            // Commands that came during the search. It runs at least two simulations, so the
            // root's moves have visits to choose from, and remembers a stop that came sooner.
            while let Ok(line) = self.input.try_recv() {
                match line.split_whitespace().next() {
                    Some("stop") => stop = true,
                    Some("quit") => {
                        stop = true;
                        keep_going = false;
                    }
                    Some("isready") => self.send("readyok")?,
                    _ => {}
                }
            }
            let out_of_time = limits.time.is_some_and(|time| start.elapsed() >= time);
            if (stop || out_of_time) && search.n_sims_done() >= 2 {
                break;
            }
            if last_info.elapsed() >= INFO_INTERVAL {
                self.send_info(&search, start)?;
                last_info = Instant::now();
            }
        }

        self.state = state;
        self.send_info(&search, start)?;
        let best = search.result().best_move().unwrap_or(moves[0]);
        self.send(&format!("bestmove {best}"))?;
        Ok(keep_going)
    }

    fn send_info(&mut self, search: &Search, start: Instant) -> io::Result<()> {
        let line = search.principal_variation();
        let elapsed = start.elapsed();
        let n_sims = search.n_sims_done();
        let pv: Vec<String> = line.iter().map(Move::to_string).collect();
        self.send(&format!(
            "info depth {} nodes {n_sims} nps {:.0} time {} score cp {} pv {}",
            line.len().max(1),
            n_sims as f64 / elapsed.as_secs_f64().max(1e-3),
            elapsed.as_millis(),
            centipawns(search.root_value()),
            pv.join(" ")
        ))
    }

    fn send(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.output, "{line}")?;
        self.output.flush()
    }
}

// A value in [-1, 1] as centipawns, by the curve Leela uses, so GUIs show a familiar number
fn centipawns(value: f32) -> i32 {
    let value = value.clamp(-0.999, 0.999) as f64;
    (90.0 * (1.5637541897 * value).tan()).round() as i32
}

// `startpos` or `fen <fields>`, then optionally `moves` and moves in UCI notation
fn parse_position(words: &[&str]) -> Result<State, String> {
    let moves_at = words.iter().position(|&word| word == "moves");
    let (setup, moves) = words.split_at(moves_at.unwrap_or(words.len()));
    let mut state = match setup {
        ["startpos"] => State::from_fen(START_FEN),
        ["fen", fields @ ..] => State::parse_fen(&fields.join(" "))?,
        _ => return Err(format!("can't read position {:?}", setup.join(" "))),
    };

    for &text in moves.iter().skip(1) {
        let mv = state
            .generate_moves()
            .into_iter()
            .find(|mv| mv.to_string() == text)
            .ok_or_else(|| format!("{text} isn't a legal move in {}", state.to_fen()))?;
        state.make_move(mv);
    }
    Ok(state)
}

// The search's limits from `go`'s arguments, for the side to move in `state`
fn parse_go(words: &[&str], state: &State) -> Limits {
    let value = |name: &str| -> Option<u64> {
        let i = words.iter().position(|&word| word == name)?;
        words.get(i + 1)?.parse().ok()
    };
    let white = state.to_fen().split_whitespace().nth(1) == Some("w");
    let (time_left, increment) = if white {
        (value("wtime"), value("winc"))
    } else {
        (value("btime"), value("binc"))
    };

    // A share of the remaining time plus most of the increment, never more than half of what's
    // left
    let clock = time_left.map(|left| {
        let moves_to_go = value("movestogo").unwrap_or(DEFAULT_MOVES_TO_GO).max(1);
        let share = left / moves_to_go + increment.unwrap_or(0) * 3 / 4;
        Duration::from_millis(share.min(left / 2))
    });
    let time = value("movetime")
        .map(Duration::from_millis)
        .or(clock)
        .map(|time| {
            time.saturating_sub(TIME_MARGIN)
                .max(Duration::from_millis(10))
        });

    Limits {
        nodes: value("nodes").map(|nodes| nodes as usize),
        time,
        infinite: words.contains(&"infinite"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::tiny_model;
    use std::sync::mpsc;

    // Runs the engine on `commands` and returns what it sent
    fn session(commands: &[&str]) -> Vec<String> {
        let (sender, receiver) = mpsc::channel();
        for command in commands {
            sender.send(command.to_string()).unwrap();
        }
        drop(sender);
        let mut output = vec![];
        run(&tiny_model(), 8, &Default::default(), receiver, &mut output).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(String::from)
            .collect()
    }

    fn best_move(lines: &[String]) -> &str {
        let line = lines
            .iter()
            .rev()
            .find(|line| line.starts_with("bestmove "))
            .expect("a bestmove");
        line.strip_prefix("bestmove ").unwrap()
    }

    #[test]
    fn it_introduces_itself_and_says_when_ready() {
        let lines = session(&["uci", "isready", "quit"]);
        assert!(lines[0].starts_with("id name Cinder"));
        assert_eq!(lines.last().unwrap(), "readyok");
        assert!(lines.contains(&"uciok".to_string()));
    }

    #[test]
    fn it_plays_a_legal_move_in_the_position_given() {
        let lines = session(&["position startpos moves e2e4 e7e5", "go nodes 20"]);
        let state = parse_position(&["startpos", "moves", "e2e4", "e7e5"]).unwrap();
        let legal: Vec<String> = state
            .generate_moves()
            .iter()
            .map(|mv| mv.to_string())
            .collect();
        assert!(legal.contains(&best_move(&lines).to_string()), "{lines:?}");

        // The last report, after any sent while a slow search ran
        let info = lines
            .iter()
            .rev()
            .find(|line| line.starts_with("info depth"))
            .unwrap();
        assert!(info.contains("nodes 20 "), "{info}");
    }

    #[test]
    fn it_finds_a_mate_in_one() {
        let lines = session(&[
            "position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
            "go nodes 400",
        ]);
        assert_eq!(best_move(&lines), "a1a8");
    }

    #[test]
    fn stop_ends_an_infinite_search() {
        let lines = session(&["go infinite", "stop", "isready", "quit"]);
        assert_eq!(best_move(&lines).len(), 4);
        assert!(lines.contains(&"readyok".to_string()));
    }

    #[test]
    fn a_finished_game_has_no_move() {
        let lines = session(&[
            "position fen R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
            "go nodes 10",
        ]);
        assert_eq!(best_move(&lines), "0000");
    }

    #[test]
    fn bad_positions_are_reported_and_ignored() {
        let lines = session(&[
            "position startpos moves e2e5",
            "position fen 8/8/8/8/8/8/8/K5Rk w - - 0 1",
            "go nodes 10",
        ]);
        assert!(lines[0].contains("e2e5 isn't a legal move"), "{lines:?}");
        assert!(lines[1].contains("in check"), "{lines:?}");
        // The search was of the start position, the last one set
        let start: Vec<String> = State::from_fen(START_FEN)
            .generate_moves()
            .iter()
            .map(|mv| mv.to_string())
            .collect();
        assert!(start.contains(&best_move(&lines).to_string()));
    }

    #[test]
    fn go_turns_the_clock_into_a_time_limit() {
        let white = State::from_fen(START_FEN);
        let limits = parse_go(&["wtime", "60000", "btime", "1000", "winc", "1000"], &white);
        // 60 s / 30 moves + 3/4 of the 1 s increment, less the margin
        assert_eq!(limits.time, Some(Duration::from_millis(2_000 + 750 - 50)));
        let black = parse_position(&["startpos", "moves", "e2e4"]).unwrap();
        let limits = parse_go(&["wtime", "60000", "btime", "1000"], &black);
        // 33 ms less the margin, but at least 10 ms
        assert_eq!(limits.time, Some(Duration::from_millis(10)));
        assert_eq!(
            parse_go(&["movetime", "500"], &white).time,
            Some(Duration::from_millis(450))
        );
        assert_eq!(parse_go(&["nodes", "77"], &white).nodes, Some(77));
        assert_eq!(parse_go(&[], &white), Limits::default());
    }

    #[test]
    fn values_read_as_centipawns() {
        assert_eq!(centipawns(0.0), 0);
        assert!(centipawns(0.5) > 0 && centipawns(-0.5) == -centipawns(0.5));
        assert!(centipawns(1.0) > 1000);
    }
}
