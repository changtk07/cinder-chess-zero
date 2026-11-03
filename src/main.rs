mod bench;
mod chess;
mod engine;
mod sync;
mod uci;

use crate::sync::{pull_if_behind, Pusher, Remote};
use crate::{
    chess::State,
    engine::{
        mcts::MctsConfig,
        model::TransformerModelConfig,
        run::{self as training_run, Run, RunConfig},
        self_play::{
            self as self_play_driver, progress_line, GameResult, SelfPlayConfig, SelfPlayStats,
        },
    },
};
use burn::{
    backend::{Autodiff, Cuda, NdArray},
    tensor::{
        backend::{AutodiffBackend, Backend},
        bf16, f16,
    },
};
use std::{path::PathBuf, str::FromStr, time::Instant};

const USAGE: &str = "\
Usage: cinder <command> [--backend cpu|cuda|cuda-f16|cuda-bf16]

Commands:
  self-play [--parallel 512] [--sims 200] [--games 512] [--seed 0]
                      play games with untrained weights, reporting throughput
  train <dir> [--parallel 512] [--generations N] [--no-sync]
                      play and train generation after generation in the run in dir,
                      creating it if needed. Stop it any time, then run the same
                      command, here or on another machine, to continue. Training runs
                      in f32; --backend sets self-play's precision. A new run also takes
                      [--games 2048] [--sims 200] [--steps 1000] [--seed 0], and
                      [--sync remote] to keep a copy of itself there with rclone, such as
                      gdrive:cinder-runs; --sync also fetches a run this machine lacks
  match <dir> <generation> <generation> [--games 128] [--sims 50] [--parallel 128]
                      play two of a run's kept networks against each other, and
                      report the first's score
  uci <dir> <generation> [--sims 800]
                      play one of a run's kept networks over the UCI protocol, for chess
                      GUIs; a go without limits searches --sims simulations
  bench [max_batch]   network throughput by batch size, then search speed
  perft [depth]       count positions from the start, per root move

self-play, bench and new runs also take the network's size: [--blocks 8]
[--width 256], where the width is a multiple of 64";

// Settings a run fixes when it's created
const RUN_OPTIONS: [&str; 6] = [
    "--blocks", "--width", "--games", "--sims", "--steps", "--seed",
];

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = take_option(&mut args, "--backend").unwrap_or_else(|| "cpu".into());

    // Training runs in f32 on the same device, whatever precision self-play uses
    if args.first().is_some_and(|command| command == "train") {
        args.remove(0);
        match backend.as_str() {
            "cpu" => train::<NdArray, Autodiff<NdArray>>(args),
            "cuda" => train::<Cuda, Autodiff<Cuda>>(args),
            "cuda-f16" => train::<Cuda<f16>, Autodiff<Cuda>>(args),
            "cuda-bf16" => train::<Cuda<bf16>, Autodiff<Cuda>>(args),
            other => usage_error(&format!("unknown backend {other}")),
        }
        return;
    }

    match backend.as_str() {
        "cpu" => run::<NdArray>(args, &Default::default()),
        "cuda" => run::<Cuda>(args, &Default::default()),
        "cuda-f16" => run::<Cuda<f16>>(args, &Default::default()),
        "cuda-bf16" => run::<Cuda<bf16>>(args, &Default::default()),
        other => usage_error(&format!("unknown backend {other}")),
    }
}

fn run<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    if args.is_empty() {
        println!("{USAGE}");
        return;
    }

    match args.remove(0).as_str() {
        "perft" => {
            let depth = take_positional(&mut args, "depth", 5);
            reject_leftovers(&args);
            perft(depth);
        }
        "bench" => {
            let model_config = take_model_config(&mut args);
            let max_batch_size = take_positional(&mut args, "max_batch", 256);
            reject_leftovers(&args);
            bench::run::<B>(device, &model_config, max_batch_size);
        }
        "self-play" => self_play::<B>(args, device),
        "match" => play_match::<B>(args, device),
        "uci" => play_uci::<B>(args, device),
        other => usage_error(&format!("unknown command {other}")),
    }
}

// Removes `name` and the value after it from `args`, returning the value
fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == name)?;
    args.remove(index);

    match args.get(index) {
        Some(value) if !value.starts_with("--") => Some(args.remove(index)),
        _ => usage_error(&format!("{name} needs a value")),
    }
}

fn take_parsed_option<T: FromStr>(args: &mut Vec<String>, name: &str, default: T) -> T {
    take_option(args, name).map_or(default, |value| parse(name, &value))
}

// The network size from --blocks and --width. Heads keep their size, so the width sets how many
// there are.
fn take_model_config(args: &mut Vec<String>) -> TransformerModelConfig {
    let mut config = TransformerModelConfig::new();
    config.n_blocks = take_parsed_option(args, "--blocks", config.n_blocks);
    let width = take_parsed_option(args, "--width", config.d_model());

    if config.n_blocks == 0 || width == 0 || !width.is_multiple_of(config.head_dimension) {
        usage_error(&format!(
            "--blocks must be at least 1, and --width a positive multiple of {}",
            config.head_dimension
        ));
    }
    config.n_heads = width / config.head_dimension;
    config
}

// Removes `name` from `args`, returning whether it was there
fn take_flag(args: &mut Vec<String>, name: &str) -> bool {
    let index = args.iter().position(|arg| arg == name);
    if let Some(index) = index {
        args.remove(index);
    }
    index.is_some()
}

// Takes the first argument, which must be there and not an option
fn take_required<T: FromStr>(args: &mut Vec<String>, name: &str) -> T {
    match args.first() {
        Some(value) if !value.starts_with("--") => parse(name, &args.remove(0)),
        _ => usage_error(&format!("missing {name}")),
    }
}

// Takes the first argument if it isn't an option
fn take_positional<T: FromStr>(args: &mut Vec<String>, name: &str, default: T) -> T {
    match args.first() {
        Some(value) if !value.starts_with("--") => parse(name, &args.remove(0)),
        _ => default,
    }
}

fn parse<T: FromStr>(name: &str, value: &str) -> T {
    value
        .parse()
        .unwrap_or_else(|_| usage_error(&format!("{name} expects a number, got {value}")))
}

fn reject_leftovers(args: &[String]) {
    if !args.is_empty() {
        usage_error(&format!("unexpected arguments: {}", args.join(" ")));
    }
}

fn usage_error(message: &str) -> ! {
    eprintln!("error: {message}\n\n{USAGE}");
    std::process::exit(2);
}

fn perft(depth: u8) {
    let mut state = State::new();
    let start_time = Instant::now();

    println!("Running perft depth {} on starting position...", depth);
    state.divide(depth);

    let elapsed = start_time.elapsed();
    println!("Time taken: {:?}", elapsed);
}

// Self-play on backend P and training on T, on their default devices: the same GPU for CUDA
fn train<P: Backend, T: AutodiffBackend>(mut args: Vec<String>) {
    let (play_device, train_device) = (&Default::default(), &Default::default());
    let dir = match args.first() {
        Some(dir) if !dir.starts_with("--") => PathBuf::from(args.remove(0)),
        _ => usage_error("train needs a run directory"),
    };
    let n_parallel: usize = take_parsed_option(&mut args, "--parallel", 512);
    let n_generations: Option<usize> =
        take_option(&mut args, "--generations").map(|value| parse("--generations", &value));
    if n_parallel == 0 {
        usage_error("--parallel must be at least 1");
    }
    let sync_to = take_option(&mut args, "--sync");
    let no_sync = take_flag(&mut args, "--no-sync");

    // The run's copy: where its settings keep it, or for a run this machine doesn't have yet,
    // where --sync says to fetch it from or keep it
    let sync_root = if Run::exists(&dir) {
        if sync_to.is_some() {
            usage_error(&format!(
                "--sync is fixed when a run is created; edit {} to change it",
                dir.join("config.json").display()
            ));
        }
        Run::open(&dir)
            .unwrap_or_else(|error| fail(&error))
            .config
            .sync_to
    } else {
        sync_to.clone()
    };
    let remote = sync_root
        .filter(|_| !no_sync)
        .map(|root| Remote::new(&["rclone"], &root, &dir).unwrap_or_else(|error| fail(&error)));
    if let Some(remote) = &remote {
        pull_if_behind(remote, &dir).unwrap_or_else(|error| fail(&error));
    }

    let run = if Run::exists(&dir) {
        if let Some(option) = RUN_OPTIONS
            .iter()
            .find(|&&option| args.iter().any(|arg| arg == option))
        {
            usage_error(&format!(
                "{option} is fixed when a run is created; edit {} to change the run's settings",
                dir.join("config.json").display()
            ));
        }
        reject_leftovers(&args);
        Run::open(&dir)
    } else {
        let mut config = RunConfig::new(take_model_config(&mut args));
        config.games_per_generation =
            take_parsed_option(&mut args, "--games", config.games_per_generation);
        config.n_sims = take_parsed_option(&mut args, "--sims", config.n_sims);
        config.steps_per_generation =
            take_parsed_option(&mut args, "--steps", config.steps_per_generation);
        config.seed = take_parsed_option(&mut args, "--seed", config.seed);
        config.sync_to = sync_to;
        reject_leftovers(&args);

        if config.games_per_generation == 0 || config.steps_per_generation == 0 {
            usage_error("--games and --steps must be at least 1");
        }
        // The first simulation only expands the root, and samples store visit counts in 16 bits
        if !(2..=u16::MAX as usize).contains(&config.n_sims) {
            usage_error(&format!("--sims must be from 2 to {}", u16::MAX));
        }
        if config.full_search_share < 1.0 && config.n_sims <= config.fast_sims {
            usage_error(&format!(
                "--sims must be more than the {} simulations of the fast searches most moves get",
                config.fast_sims
            ));
        }
        println!("Starting a new run in {}", dir.display());
        Run::create::<T>(&dir, config, train_device)
    }
    .unwrap_or_else(|error| fail(&error));

    println!(
        "Self-play on {}, {n_parallel} games at a time; training on {}; network of {} blocks, \
         width {}",
        P::name(play_device),
        T::name(train_device),
        run.config.model.n_blocks,
        run.config.model.d_model()
    );
    let pusher = remote.map(|remote| {
        println!(
            "Syncing to {} after each checkpoint and when stopped",
            remote.path()
        );
        let pusher = Pusher::start(remote, &dir);
        pusher.push_on_stop().unwrap_or_else(|error| fail(&error));
        pusher
    });

    let trained = training_run::train::<P, T>(
        &run,
        n_parallel,
        n_generations,
        play_device,
        train_device,
        |_| {
            if let Some(pusher) = &pusher {
                pusher.push_soon();
            }
        },
    );
    // Pushes what the run got done even if it failed
    if let Some(pusher) = pusher {
        pusher.finish();
    }
    trained.unwrap_or_else(|error| fail(&error));
}

fn play_match<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    let dir: PathBuf = take_required(&mut args, "run directory");
    let first: usize = take_required(&mut args, "first generation");
    let second: usize = take_required(&mut args, "second generation");
    let games: usize = take_parsed_option(&mut args, "--games", 128);
    let n_sims: usize = take_parsed_option(&mut args, "--sims", 50);
    let n_parallel: usize = take_parsed_option(&mut args, "--parallel", 128);
    reject_leftovers(&args);
    if games == 0 || n_parallel == 0 || n_sims < 2 {
        usage_error("--games and --parallel must be at least 1, and --sims at least 2");
    }

    let run = Run::open(&dir).unwrap_or_else(|error| fail(&error));
    let start = Instant::now();
    let score = training_run::play_generations::<B>(
        &run,
        [first, second],
        games,
        n_sims,
        n_parallel.min(games),
        device,
    )
    .unwrap_or_else(|error| fail(&error));
    println!(
        "Generation {first} against {second}: {} wins, {} draws ({} at the ply limit), {} losses, \
         scoring {:.3}, {:+.0} Elo; {:.0} plies on average, in {:.0}s",
        score.wins,
        score.draws,
        score.stopped,
        score.losses,
        score.score(),
        score.elo(),
        score.average_plies(),
        start.elapsed().as_secs_f64()
    );
}

fn play_uci<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    let dir: PathBuf = take_required(&mut args, "run directory");
    let generation: usize = take_required(&mut args, "generation");
    let n_sims: usize = take_parsed_option(&mut args, "--sims", 800);
    reject_leftovers(&args);
    if n_sims < 2 {
        usage_error("--sims must be at least 2");
    }

    let run = Run::open(&dir).unwrap_or_else(|error| fail(&error));
    let model = run
        .load_network::<B>(generation, device)
        .unwrap_or_else(|error| fail(&error));

    // Lines go through a channel, so searches can take `stop` while they run
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    uci::run(&model, n_sims, device, receiver, std::io::stdout().lock())
        .unwrap_or_else(|error| fail(&error));
}

fn fail(error: &std::io::Error) -> ! {
    eprintln!("error: {error}");
    std::process::exit(1);
}

fn self_play<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    let model_config = take_model_config(&mut args);
    let n_parallel: usize = take_parsed_option(&mut args, "--parallel", 512);
    let n_sims: usize = take_parsed_option(&mut args, "--sims", 200);
    let n_games: usize = take_parsed_option(&mut args, "--games", 512);
    let seed: u64 = take_parsed_option(&mut args, "--seed", 0);
    reject_leftovers(&args);

    if n_parallel == 0 || n_games == 0 {
        usage_error("--parallel and --games must be at least 1");
    }
    // The first simulation only expands the root, and samples store visit counts in 16 bits
    if !(2..=u16::MAX as usize).contains(&n_sims) {
        usage_error(&format!("--sims must be from 2 to {}", u16::MAX));
    }

    // The seed fixes the untrained weights as well as the games
    B::seed(device, seed);
    let model = model_config.init::<B>(device);
    let mcts_config = MctsConfig::new().with_n_sims(n_sims);
    let config = SelfPlayConfig::new()
        .with_n_parallel(n_parallel)
        .with_n_games(n_games)
        .with_seed(seed);

    println!(
        "Backend: {}, network of {} blocks, width {}, {n_parallel} games at a time, {n_sims} \
         simulations per move",
        B::name(device),
        model_config.n_blocks,
        model_config.d_model()
    );

    let mut previous = SelfPlayStats::default();
    let stats = self_play_driver::run(
        &model,
        &mcts_config,
        &config,
        device,
        |_, _| {},
        |stats| {
            println!("{}", progress_line(stats, &previous));
            previous = stats.clone();
        },
    );

    let count = |result| {
        stats
            .games
            .iter()
            .filter(|record| record.result == result)
            .count()
    };
    let total_plies: usize = stats.games.iter().map(|record| record.plies).sum();
    println!(
        "\n{} games: {} white wins, {} black wins, {} draws, {} stopped at the ply limit, {:.0} plies on average",
        stats.games.len(),
        count(GameResult::WhiteWins),
        count(GameResult::BlackWins),
        count(GameResult::Draw),
        count(GameResult::PlyLimit),
        total_plies as f64 / stats.games.len() as f64,
    );
    println!(
        "Overall: {}",
        progress_line(&stats, &SelfPlayStats::default())
    );
}
