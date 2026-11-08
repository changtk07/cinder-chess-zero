//! The UCI protocol, so chess GUIs such as Cute Chess or Arena, and other engines, can play a
//! run's network.
//!
//! A search evaluates positions in batches, as self-play does across games, but from one tree. A
//! GPU call costs about the same for one position as for 16, so batches run many times the
//! simulations a second. Each batch gathers up to an eighth of the root's visits so far, at most
//! 64, with virtual losses spreading its simulations over different moves; keeping batches small
//! next to the tree keeps them from wandering into moves a search one at a time would skip.
//!
//! The tree carries over between moves: when the position to search is the last search's, or one
//! or two moves on, as after its move and the reply, the search starts from that subtree. Node
//! limits count the root's visits, reused ones included, as self-play's do.
//!
//! `go ponder` searches the position after the reply the last `bestmove` expected while the
//! opponent thinks, and doesn't answer until `ponderhit`, which says the reply came and starts its
//! clock, or `stop`. A ponder search that reaches its node limit waits for either.
//!
//! The search stops at the node count or time its `go` asks for, or on `stop`, and plays the most
//! visited move, without noise or sampling, naming the reply it expects for GUIs that ponder.
//! Without any limit it runs the default simulations given at startup.

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
// Batches gather up to this share of the root's visits so far, and at most MAX_BATCH positions
const BATCH_SHARE: usize = 8;
const MAX_BATCH: usize = 64;

/// Answers UCI commands from `input` on `output` until `quit` or the end of the input.
pub(crate) fn run<B: Backend>(
    model: &TransformerModel<B>,
    default_sims: usize,
    device: &B::Device,
    input: Receiver<String>,
    output: impl Write,
) -> io::Result<()> {
    let mut engine = Engine::new(model, default_sims, device, input, output);
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
    search: Search,
    // The position the search's tree is of, if it has one
    tree_root: Option<State>,
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
    // Searching on the opponent's time, until `ponderhit` or `stop`
    ponder: bool,
}

// What commands that come during a search have asked for
struct Control {
    stop: bool,
    quit: bool,
    pondering: bool,
    // When the time limit started: at the `go`, or at `ponderhit`
    clock_start: Instant,
}

impl<'a, B: Backend, W: Write> Engine<'a, B, W> {
    fn new(
        model: &'a TransformerModel<B>,
        default_sims: usize,
        device: &'a B::Device,
        input: Receiver<String>,
        output: W,
    ) -> Self {
        Self {
            model,
            device,
            default_sims,
            state: State::from_fen(START_FEN),
            search: Search::new(MctsConfig::new().with_n_sims(default_sims)),
            tree_root: None,
            input,
            output,
            warmed_up: false,
        }
    }

    // Returns false on `quit`
    fn handle(&mut self, line: &str) -> io::Result<bool> {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.first().copied() {
            Some("uci") => {
                self.send(&format!("id name {NAME}"))?;
                self.send("id author Kevin Chang")?;
                // GUIs only ponder with engines that list this; the engine needn't know the value
                self.send("option name Ponder type check default false")?;
                self.send("uciok")?;
            }
            Some("isready") => {
                self.warm_up();
                self.send("readyok")?
            }
            Some("ucinewgame") => {
                self.state = State::from_fen(START_FEN);
                self.tree_root = None;
            }
            Some("position") => match parse_position(&words[1..]) {
                Ok(state) => self.state = state,
                Err(error) => self.send(&format!("info string {error}"))?,
            },
            Some("go") => {
                let limits = parse_go(&words[1..], &self.state);
                return self.go(&limits);
            }
            Some("quit") => return Ok(false),
            // `stop` or `ponderhit` outside a search, and anything unknown, which UCI says to ignore
            _ => {}
        }
        Ok(true)
    }

    // The first calls on a GPU for each batch size spend seconds compiling kernels, so every size
    // a search uses is evaluated once before the first `readyok`, which GUIs wait for, or else
    // before the first move
    fn warm_up(&mut self) {
        if self.warmed_up {
            return;
        }
        let state = State::from_fen(START_FEN);
        let moves = state.generate_moves();
        let mut size = 1;
        while size <= MAX_BATCH {
            self.model
                .evaluate(&vec![&state; size], &vec![&moves[..]; size], self.device);
            size *= 2;
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
        self.keep_tree();
        self.search.set_n_sims(n_sims);
        let reused_visits = self.search.root_visits();
        // Searches return the position to where they started. It's taken out for the search, with
        // its history, which repetitions are found in, and put back after.
        let mut state = std::mem::replace(&mut self.state, State::from_fen(START_FEN));
        let mut control = Control {
            stop: false,
            quit: false,
            pondering: limits.ponder,
            clock_start: start,
        };
        let mut last_info = start;

        loop {
            while let Ok(line) = self.input.try_recv() {
                self.during_search(&line, &mut control)?;
            }
            // A search stops only once the root's moves have visits to choose from, and remembers
            // a stop that came sooner
            let out_of_time = !control.pondering
                && limits
                    .time
                    .is_some_and(|time| control.clock_start.elapsed() >= time);
            let full = self.search.root_visits() >= n_sims;
            let can_stop = self.search.root_visits() >= 2;
            if can_stop && (control.stop || out_of_time) || full && !control.pondering {
                break;
            }
            if full {
                // A ponder search that has all its visits waits for the GUI to say how it ends
                match self.input.recv() {
                    Ok(line) => self.during_search(&line, &mut control)?,
                    Err(_) => control.quit = true,
                }
                control.stop |= control.quit;
                continue;
            }

            let batch = (self.search.root_visits() / BATCH_SHARE).clamp(1, MAX_BATCH);
            let leaves = self.search.gather_leaves(&mut state, batch);
            if !leaves.is_empty() {
                let states: Vec<&State> = leaves.iter().map(|leaf| &leaf.state).collect();
                let leaf_moves: Vec<&[Move]> = leaves.iter().map(|leaf| &leaf.moves[..]).collect();
                let evaluations = self
                    .model
                    .evaluate_padded(&states, &leaf_moves, self.device);
                self.search.apply_leaves(leaves, evaluations);
            }
            if last_info.elapsed() >= INFO_INTERVAL {
                self.send_info(start, reused_visits)?;
                last_info = Instant::now();
            }
        }

        self.state = state;
        self.tree_root = Some(self.state.clone());
        self.send_info(start, reused_visits)?;
        let best = self.search.result().best_move().unwrap_or(moves[0]);
        let ponder = match self.search.principal_variation()[..] {
            [first, reply, ..] if first == best => format!(" ponder {reply}"),
            _ => String::new(),
        };
        self.send(&format!("bestmove {best}{ponder}"))?;
        Ok(!control.quit)
    }

    // Acts on a command that came during a search
    fn during_search(&mut self, line: &str, control: &mut Control) -> io::Result<()> {
        match line.split_whitespace().next() {
            Some("stop") => control.stop = true,
            Some("quit") => {
                control.stop = true;
                control.quit = true;
            }
            Some("ponderhit") if control.pondering => {
                control.pondering = false;
                control.clock_start = Instant::now();
            }
            Some("isready") => self.send("readyok")?,
            _ => {}
        }
        Ok(())
    }

    // Keeps the subtree the last search's tree has of the position to search, if it's that
    // search's position or one or two moves on, or else starts afresh
    fn keep_tree(&mut self) {
        let line = self
            .tree_root
            .take()
            .and_then(|root| line_between(root, &self.state));
        match line {
            Some(line) => {
                for mv in line {
                    self.search.reuse_subtree(mv);
                }
            }
            None => self.search.reset(),
        }
    }

    // `nodes` counts the root's visits, reused ones included, and `nps` only this search's
    fn send_info(&mut self, start: Instant, reused_visits: usize) -> io::Result<()> {
        let line = self.search.principal_variation();
        let elapsed = start.elapsed();
        let visits = self.search.root_visits();
        let pv: Vec<String> = line.iter().map(Move::to_string).collect();
        self.send(&format!(
            "info depth {} nodes {visits} nps {:.0} time {} score cp {} pv {}",
            line.len().max(1),
            (visits - reused_visits) as f64 / elapsed.as_secs_f64().max(1e-3),
            elapsed.as_millis(),
            centipawns(self.search.root_value()),
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

// The moves, none to two, that lead from `root` to `target`'s position, if any do
fn line_between(mut root: State, target: &State) -> Option<Vec<Move>> {
    let goal = target.to_fen();
    if root.to_fen() == goal {
        return Some(vec![]);
    }
    for first in root.generate_moves() {
        root.make_move(first);
        if root.to_fen() == goal {
            return Some(vec![first]);
        }
        for reply in root.generate_moves() {
            root.make_move(reply);
            let found = root.to_fen() == goal;
            root.unmake_move();
            if found {
                return Some(vec![first, reply]);
            }
        }
        root.unmake_move();
    }
    None
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
        ponder: words.contains(&"ponder"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::tiny_model;
    use std::{
        sync::{mpsc, Arc, Mutex},
        thread,
    };

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

    // The last `bestmove` line's words after `bestmove`
    fn best_move_words(lines: &[String]) -> Vec<&str> {
        let line = lines
            .iter()
            .rev()
            .find(|line| line.starts_with("bestmove "))
            .expect("a bestmove");
        line.split_whitespace().skip(1).collect()
    }

    fn best_move(lines: &[String]) -> &str {
        best_move_words(lines)[0]
    }

    // Output tests can read while the engine still runs on another thread
    #[derive(Clone, Default)]
    struct SharedOutput(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedOutput {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl SharedOutput {
        fn has_bestmove(&self) -> bool {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .any(|line| line.starts_with("bestmove "))
        }

        fn wait_for_bestmove(&self) -> bool {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(30) {
                if self.has_bestmove() {
                    return true;
                }
                thread::sleep(Duration::from_millis(10));
            }
            false
        }
    }

    // Starts a ponder search with `go`, gives it `wait` to finish if it would answer by itself,
    // then sends `end`, checking it answers only after `end`
    fn ponder(go: &str, wait: Duration, end: &str) {
        let (sender, receiver) = mpsc::channel();
        let output = SharedOutput::default();
        let engine_output = output.clone();
        let engine = thread::spawn(move || {
            run(
                &tiny_model(),
                8,
                &Default::default(),
                receiver,
                engine_output,
            )
            .unwrap()
        });
        sender.send(go.to_string()).unwrap();
        thread::sleep(wait);
        assert!(!output.has_bestmove(), "answered before {end}");
        sender.send(end.to_string()).unwrap();
        assert!(output.wait_for_bestmove(), "no answer after {end}");
        sender.send("quit".to_string()).unwrap();
        engine.join().unwrap();
    }

    #[test]
    fn it_introduces_itself_and_says_when_ready() {
        let lines = session(&["uci", "isready", "quit"]);
        assert!(lines[0].starts_with("id name Cinder"));
        assert_eq!(lines.last().unwrap(), "readyok");
        assert!(lines.contains(&"uciok".to_string()));
        assert!(lines.contains(&"option name Ponder type check default false".to_string()));
    }

    #[test]
    fn it_names_the_reply_it_expects() {
        let lines = session(&["position startpos", "go nodes 100"]);
        let [best, "ponder", reply] = best_move_words(&lines)[..] else {
            panic!("{lines:?}");
        };
        let after = parse_position(&["startpos", "moves", best]).unwrap();
        assert!(after
            .generate_moves()
            .iter()
            .any(|mv| mv.to_string() == reply));
    }

    #[test]
    fn the_tree_carries_over_to_the_position_after_the_reply() {
        let model = tiny_model();
        let device = Default::default();
        let (_sender, receiver) = mpsc::channel();
        let mut engine = Engine::new(&model, 8, &device, receiver, vec![]);
        engine.handle("position startpos").unwrap();
        engine.handle("go nodes 200").unwrap();
        let line = engine.search.principal_variation();
        assert!(line.len() >= 2, "{line:?}");

        engine
            .handle(&format!("position startpos moves {} {}", line[0], line[1]))
            .unwrap();
        engine.handle("go nodes 200").unwrap();
        assert_eq!(engine.search.root_visits(), 200);
        assert!(engine.search.n_sims_done() < 200);

        // A position the tree doesn't reach starts afresh
        engine
            .handle("position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
            .unwrap();
        engine.handle("go nodes 200").unwrap();
        assert_eq!(engine.search.n_sims_done(), 200);
    }

    #[test]
    fn lines_of_up_to_two_moves_are_found() {
        let start = || State::from_fen(START_FEN);
        let after = |moves: &[&str]| {
            let words: Vec<&str> = ["startpos", "moves"].iter().chain(moves).copied().collect();
            parse_position(&words).unwrap()
        };
        let names = |line: Option<Vec<Move>>| {
            line.map(|line| line.iter().map(Move::to_string).collect::<Vec<_>>())
        };
        assert_eq!(names(line_between(start(), &start())), Some(vec![]));
        assert_eq!(
            names(line_between(start(), &after(&["g1f3"]))),
            Some(vec!["g1f3".into()])
        );
        assert_eq!(
            names(line_between(start(), &after(&["e2e4", "c7c5"]))),
            Some(vec!["e2e4".into(), "c7c5".into()])
        );
        assert_eq!(
            line_between(start(), &after(&["e2e4", "c7c5", "g1f3"])),
            None
        );
        assert_eq!(
            line_between(after(&["d2d4"]), &after(&["e2e4", "c7c5"])),
            None
        );
    }

    #[test]
    fn a_ponder_search_answers_only_after_ponderhit_or_stop() {
        // At its node limit it waits
        ponder("go ponder nodes 20", Duration::from_secs(2), "ponderhit");
        ponder("go ponder nodes 20", Duration::from_secs(2), "stop");
        // and its time limit starts at ponderhit
        ponder(
            "go ponder movetime 300",
            Duration::from_secs(1),
            "ponderhit",
        );
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
        assert!(parse_go(&["ponder", "wtime", "60000", "btime", "1000"], &white).ponder);
        assert_eq!(parse_go(&[], &white), Limits::default());
    }

    #[test]
    fn values_read_as_centipawns() {
        assert_eq!(centipawns(0.0), 0);
        assert!(centipawns(0.5) > 0 && centipawns(-0.5) == -centipawns(0.5));
        assert!(centipawns(1.0) > 1000);
    }
}
