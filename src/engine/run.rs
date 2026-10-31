//! A training run: generation after generation of self-play and training, kept in one directory so
//! it can stop at any point and continue later, on the same machine or another.
//!
//! - `config.json`: the run's settings, written when it's created. They can be edited between
//!   sessions, except for the network size, which the checkpoints must match.
//! - `checkpoints/NNNNN/`: the network after NNNNN generations of training, with its optimizer
//!   state, at full precision whatever backend wrote it. The latest two are kept whole, and the
//!   networks of generation 0 and every `keep_every`-th generation stay without their optimizer
//!   state, as references to measure progress against with `cinder match`.
//! - `samples/NNNNN.bin`: the games generation NNNNN's network played, in the format described in
//!   `samples.rs`. The file is complete once checkpoint NNNNN + 1 exists, and never changes after.
//! - `history.csv`: a line per generation trained: its games' results, its training losses, and
//!   how the network it trained did against it.
//!
//! Games keep their slots across generations, so every batch of positions stays full: once a
//! generation has finished its games, the games still going wait while the network trains, and the
//! next generation's network finishes them. A game is recorded under the generation it finished in.
//!
//! Stopping the process at any point leaves the directory usable: games are appended whole, and
//! checkpoints are written under a temporary name and then renamed. Stopping loses the games still
//! going and at most the training of the generation in progress; the generation's finished games
//! are kept, and it plays only the rest when the run continues.

use crate::engine::{
    mcts::MctsConfig,
    model::TransformerModel,
    model::TransformerModelConfig,
    samples::{read_samples, Sample, SampleWriter},
    self_play::{play_match, progress_line, MatchScore, SelfPlay, SelfPlayConfig, SelfPlayStats},
    training::{load_model, remove_optimizer_state, Losses, Trainer},
};
use burn::{
    config::Config,
    tensor::backend::{AutodiffBackend, Backend},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const CONFIG_FILE: &str = "config.json";
const HISTORY_FILE: &str = "history.csv";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// A run's settings, fixed when it's created.
#[derive(Config, Debug)]
pub(crate) struct RunConfig {
    pub(crate) model: TransformerModelConfig,
    /// Games each generation's network plays before it's trained on them
    #[config(default = 512)]
    pub(crate) games_per_generation: usize,
    /// Simulations per move in self-play
    #[config(default = 200)]
    pub(crate) n_sims: usize,
    /// Games still going after this many plies are stopped and count as draws
    #[config(default = 512)]
    pub(crate) max_plies: usize,
    /// Optimizer steps per generation, each on positions drawn at random from the window
    #[config(default = 1000)]
    pub(crate) steps_per_generation: usize,
    /// Positions per optimizer step
    #[config(default = 256)]
    pub(crate) batch_size: usize,
    /// Training draws from the games of this many latest generations
    #[config(default = 4)]
    pub(crate) window_generations: usize,
    #[config(default = 3e-4)]
    pub(crate) learning_rate: f64,
    /// The learning rate rises linearly from zero over the run's first steps
    #[config(default = 500)]
    pub(crate) warmup_steps: usize,
    #[config(default = 1e-4)]
    pub(crate) weight_decay: f32,
    /// Games each newly trained network plays against the one it was trained from, to measure
    /// progress; 0 to skip. Early networks draw nearly all such games, so it's off by default.
    #[config(default = 0)]
    pub(crate) eval_games: usize,
    /// Simulations per move in those games, fewer than in self-play to keep them cheap
    #[config(default = 50)]
    pub(crate) eval_sims: usize,
    /// The network of every this many generations is kept for good, as a reference to measure
    /// later ones against; 0 keeps only generation 0's
    #[config(default = 10)]
    pub(crate) keep_every: usize,
    /// Fixes the initial weights, and with the generation, every game's random choices
    #[config(default = 0)]
    pub(crate) seed: u64,
}

pub(crate) struct Run {
    dir: PathBuf,
    pub(crate) config: RunConfig,
}

impl Run {
    /// Whether `dir` holds a run.
    pub(crate) fn exists(dir: &Path) -> bool {
        dir.join(CONFIG_FILE).exists()
    }

    /// Starts a run in `dir`, with untrained weights from the seed as generation 0.
    pub(crate) fn create<B: AutodiffBackend>(
        dir: &Path,
        config: RunConfig,
        device: &B::Device,
    ) -> io::Result<Self> {
        if Self::exists(dir) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{} already holds a run", dir.display()),
            ));
        }
        fs::create_dir_all(dir)?;

        B::seed(device, config.seed);
        let trainer = Trainer::<B>::new(config.model.init(device), config.weight_decay, device);
        let run = Self {
            dir: dir.to_path_buf(),
            config,
        };
        run.save_checkpoint(&trainer, 0)?;
        // Written last, so a run that failed to start can be started again
        run.config.save(dir.join(CONFIG_FILE))?;
        Ok(run)
    }

    /// Opens the run in `dir`. Settings added since its `config.json` was written take their
    /// defaults.
    pub(crate) fn open(dir: &Path) -> io::Result<Self> {
        let invalid = |error: serde_json::Error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("can't read {CONFIG_FILE}: {error}"),
            )
        };
        let saved =
            serde_json::from_str(&fs::read_to_string(dir.join(CONFIG_FILE))?).map_err(invalid)?;
        let defaults = RunConfig::new(TransformerModelConfig::new());
        let mut config = serde_json::to_value(&defaults).map_err(invalid)?;
        overlay(&mut config, saved);

        Ok(Self {
            dir: dir.to_path_buf(),
            config: serde_json::from_value(config).map_err(invalid)?,
        })
    }

    /// The generation of the latest checkpoint, whose network plays next.
    pub(crate) fn latest_generation(&self) -> io::Result<usize> {
        let mut latest = None;
        for entry in fs::read_dir(self.dir.join("checkpoints"))? {
            let name = entry?.file_name();
            // Temporary directories of interrupted saves don't parse
            if let Some(generation) = name.to_str().and_then(|name| name.parse().ok()) {
                latest = latest.max(Some(generation));
            }
        }
        latest.ok_or_else(|| io::Error::other("the run has no checkpoints"))
    }

    /// Loads generation `generation`'s network, if its checkpoint is still kept.
    pub(crate) fn load_network<B: Backend>(
        &self,
        generation: usize,
        device: &B::Device,
    ) -> io::Result<TransformerModel<B>> {
        load_model::<B>(&self.config.model, &self.checkpoint_dir(generation), device).map_err(
            |error| {
                io::Error::other(format!(
                    "can't load generation {generation}'s checkpoint: {error:?}"
                ))
            },
        )
    }

    fn checkpoint_dir(&self, generation: usize) -> PathBuf {
        self.dir
            .join("checkpoints")
            .join(format!("{generation:05}"))
    }

    fn samples_path(&self, generation: usize) -> PathBuf {
        self.dir
            .join("samples")
            .join(format!("{generation:05}.bin"))
    }

    // Writes the checkpoint under a temporary name and renames it, so a checkpoint either exists
    // whole or not at all. Then deletes older checkpoints but the latest two, keeping the networks
    // of reference generations without the optimizer state they no longer need.
    fn save_checkpoint<B: AutodiffBackend>(
        &self,
        trainer: &Trainer<B>,
        generation: usize,
    ) -> io::Result<()> {
        let dir = self.checkpoint_dir(generation);
        let temporary = dir.with_extension("tmp");
        if temporary.exists() {
            fs::remove_dir_all(&temporary)?;
        }
        fs::create_dir_all(&temporary)?;
        trainer
            .save(&temporary)
            .map_err(|error| io::Error::other(format!("can't save checkpoint: {error:?}")))?;
        fs::rename(&temporary, &dir)?;

        let keep_every = self.config.keep_every;
        for old in 0..generation.saturating_sub(1) {
            let dir = self.checkpoint_dir(old);
            if !dir.exists() {
                continue;
            }
            if old == 0 || (keep_every > 0 && old.is_multiple_of(keep_every)) {
                remove_optimizer_state(&dir)?;
            } else {
                fs::remove_dir_all(dir)?;
            }
        }
        Ok(())
    }
}

// Replaces `defaults`' values with those in `saved`, object by object, keeping defaults that
// `saved` lacks
fn overlay(defaults: &mut serde_json::Value, saved: serde_json::Value) {
    match (defaults, saved) {
        (serde_json::Value::Object(defaults), serde_json::Value::Object(saved)) => {
            for (key, value) in saved {
                match defaults.get_mut(&key) {
                    Some(default) => overlay(default, value),
                    None => {
                        defaults.insert(key, value);
                    }
                }
            }
        }
        (default, saved) => *default = saved,
    }
}

/// Plays and trains `n_generations` generations, or until stopped if `None`: each generation's
/// network plays its games on backend `P`, then trains on backend `T`, which should be full
/// precision, to become the next generation. `n_parallel` sets how many games play at once.
pub(crate) fn train<P: Backend, T: AutodiffBackend>(
    run: &Run,
    n_parallel: usize,
    n_generations: Option<usize>,
    play_device: &P::Device,
    train_device: &T::Device,
) -> io::Result<()> {
    let config = &run.config;
    let first = run.latest_generation()?;
    let end = n_generations.map(|n| first + n);

    // The games already played count towards the seed, so a resumed generation plays new games
    let n_played = count_games(&run.samples_path(first))?;
    let seed = derive_seed(config.seed, &[SELF_PLAY, first as u64, n_played as u64]);
    let mut games = SelfPlay::new(
        &MctsConfig::new().with_n_sims(config.n_sims),
        &SelfPlayConfig::new()
            .with_n_parallel(n_parallel)
            .with_n_games(usize::MAX)
            .with_max_plies(config.max_plies)
            .with_seed(seed),
    );

    let mut generation = first;
    while end.is_none_or(|end| generation < end) {
        play_generation::<P>(run, generation, &mut games, play_device)?;
        let mut summary = train_generation::<T>(run, generation, train_device)?;
        if config.eval_games > 0 {
            summary.eval = Some(evaluate::<P>(run, generation + 1, n_parallel, play_device)?);
        }
        append_history(run, &summary)?;
        generation += 1;
    }

    Ok(())
}

// Plays with `generation`'s network until it has finished its share of games, appending each to
// its sample file. The last step can finish a few more.
fn play_generation<P: Backend>(
    run: &Run,
    generation: usize,
    games: &mut SelfPlay,
    device: &P::Device,
) -> io::Result<()> {
    let config = &run.config;
    let path = run.samples_path(generation);
    let n_played = count_games(&path)?;
    let n_games = config.games_per_generation.saturating_sub(n_played);
    if n_games == 0 {
        return Ok(());
    }

    println!(
        "\nGeneration {generation}: playing {n_games} games, {} simulations per move{}",
        config.n_sims,
        if n_played > 0 {
            format!(" ({n_played} already played)")
        } else {
            String::new()
        }
    );
    let model = run.load_network::<P>(generation, device)?;
    let mut writer = SampleWriter::open(&path)?;

    let mut previous = SelfPlayStats::default();
    let stats = games.play(
        &model,
        n_games,
        device,
        |_, samples| {
            writer
                .write_game(generation as u32, &samples)
                .expect("writing samples");
        },
        |stats| {
            println!("{}", progress_line(stats, &previous));
            previous = stats.clone();
        },
    );
    println!(
        "Played: {}",
        progress_line(&stats, &SelfPlayStats::default())
    );
    Ok(())
}

/// What a generation played and how its training went, for `history.csv`.
struct GenerationSummary {
    generation: usize,
    games: GameCounts,
    losses: Losses,
    /// The entropy of the window's results: the value loss of predicting the same win, draw and
    /// loss shares for every position, which the network has to beat by telling positions apart
    result_entropy: f32,
    /// The trained network's score against the one it was trained from
    eval: Option<MatchScore>,
}

#[derive(Default)]
struct GameCounts {
    white_wins: usize,
    black_wins: usize,
    draws: usize,
    plies: usize,
}

impl GameCounts {
    fn total(&self) -> usize {
        self.white_wins + self.black_wins + self.draws
    }
}

// Trains `generation`'s network on the window's games, saving the result as the next generation
fn train_generation<T: AutodiffBackend>(
    run: &Run,
    generation: usize,
    device: &T::Device,
) -> io::Result<GenerationSummary> {
    let config = &run.config;
    let mut samples = vec![];
    let mut games = GameCounts::default();
    let window = (generation + 1).saturating_sub(config.window_generations)..=generation;
    for window_generation in window.clone() {
        let path = run.samples_path(window_generation);
        if !path.exists() {
            continue;
        }
        for game in read_samples(&path)?.games {
            if window_generation == generation {
                count_game(&mut games, &game.samples);
            }
            samples.extend(game.samples);
        }
    }
    if samples.is_empty() {
        return Err(io::Error::other("no games to train on"));
    }

    println!(
        "Generation {generation}: training {} steps on {} positions from generations {}-{}",
        config.steps_per_generation,
        samples.len(),
        window.start(),
        window.end()
    );
    let mut trainer = Trainer::<T>::load(
        &config.model,
        config.weight_decay,
        &run.checkpoint_dir(generation),
        device,
    )
    .map_err(|error| io::Error::other(format!("can't load checkpoint: {error:?}")))?;

    let mut rng = StdRng::seed_from_u64(derive_seed(config.seed, &[TRAINING, generation as u64]));
    let first_step = generation * config.steps_per_generation;
    let mut losses = Vec::with_capacity(config.steps_per_generation);
    let start = Instant::now();
    let mut last_report = (start, 0);

    for step in 0..config.steps_per_generation {
        let batch: Vec<&Sample> = (0..config.batch_size)
            .map(|_| &samples[rng.random_range(0..samples.len())])
            .collect();
        let warmup = (first_step + step + 1) as f64 / config.warmup_steps.max(1) as f64;
        let learning_rate = config.learning_rate * warmup.min(1.0);
        losses.push(trainer.step(&batch, learning_rate));

        let (report_time, report_step) = last_report;
        if report_time.elapsed() >= PROGRESS_INTERVAL || step + 1 == config.steps_per_generation {
            let recent = Losses::mean(&losses[report_step..]);
            println!(
                "{:>5.0}s  step {:>5}/{}  {}  {:.1} steps/s",
                start.elapsed().as_secs_f64(),
                step + 1,
                config.steps_per_generation,
                describe(&recent),
                (step + 1 - report_step) as f64 / report_time.elapsed().as_secs_f64(),
            );
            last_report = (Instant::now(), step + 1);
        }
    }

    run.save_checkpoint(&trainer, generation + 1)?;
    let result_entropy = [1, 0, -1]
        .map(|result| {
            let share = samples
                .iter()
                .filter(|sample| sample.result == result)
                .count() as f32
                / samples.len() as f32;
            if share > 0.0 {
                -share * share.ln()
            } else {
                0.0
            }
        })
        .iter()
        .sum();
    Ok(GenerationSummary {
        generation,
        games,
        losses: Losses::mean(&losses),
        result_entropy,
        eval: None,
    })
}

/// Sampling plies in evaluation games: enough for games to differ, few enough for both networks
/// to play their best moves after the opening
pub(crate) const EVAL_SAMPLING_PLIES: usize = 8;

/// Plays `games` games between generations `first` and `second`, without root noise, returning the
/// first's score.
pub(crate) fn play_generations<P: Backend>(
    run: &Run,
    [first, second]: [usize; 2],
    games: usize,
    n_sims: usize,
    n_parallel: usize,
    device: &P::Device,
) -> io::Result<MatchScore> {
    let networks = [
        run.load_network::<P>(first, device)?,
        run.load_network::<P>(second, device)?,
    ];
    let config = SelfPlayConfig::new()
        .with_n_parallel(n_parallel)
        .with_n_games(games)
        .with_max_plies(run.config.max_plies)
        .with_sampling_plies(EVAL_SAMPLING_PLIES)
        .with_noise_fraction(0.0)
        .with_seed(derive_seed(
            run.config.seed,
            &[EVALUATION, first as u64, second as u64],
        ));
    Ok(play_match(
        [&networks[0], &networks[1]],
        &MctsConfig::new().with_n_sims(n_sims),
        &config,
        device,
    ))
}

// Plays the newly trained `generation` against the one before it
fn evaluate<P: Backend>(
    run: &Run,
    generation: usize,
    n_parallel: usize,
    device: &P::Device,
) -> io::Result<MatchScore> {
    let config = &run.config;
    let start = Instant::now();
    let score = play_generations::<P>(
        run,
        [generation, generation - 1],
        config.eval_games,
        config.eval_sims,
        n_parallel.min(config.eval_games),
        device,
    )?;
    println!(
        "Generation {generation} against {}: {} wins, {} draws ({} at the ply limit), {} losses \
         at {} simulations per move, {:+.0} Elo; {:.0} plies on average, in {:.0}s",
        generation - 1,
        score.wins,
        score.draws,
        score.stopped,
        score.losses,
        config.eval_sims,
        score.elo(),
        score.average_plies(),
        start.elapsed().as_secs_f64()
    );
    Ok(score)
}

fn describe(losses: &Losses) -> String {
    format!(
        "policy loss {:.3} (targets' entropy {:.3}), value loss {:.3}",
        losses.policy, losses.target_entropy, losses.value
    )
}

fn count_games(path: &Path) -> io::Result<usize> {
    if path.exists() {
        Ok(read_samples(path)?.games.len())
    } else {
        Ok(0)
    }
}

// Games start from the initial position, so the first sample's result is white's
fn count_game(counts: &mut GameCounts, samples: &[Sample]) {
    match samples.first().map(|sample| sample.result) {
        Some(1) => counts.white_wins += 1,
        Some(-1) => counts.black_wins += 1,
        _ => counts.draws += 1,
    }
    counts.plies += samples.len();
}

fn append_history(run: &Run, summary: &GenerationSummary) -> io::Result<()> {
    let path = run.dir.join(HISTORY_FILE);
    let is_new = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if is_new {
        writeln!(
            file,
            "generation,games,white_wins,black_wins,draws,average_plies,\
             policy_loss,target_entropy,value_loss,result_entropy,\
             eval_wins,eval_draws,eval_losses,eval_elo"
        )?;
    }

    let GenerationSummary {
        generation,
        games,
        losses,
        result_entropy,
        eval,
    } = summary;
    let average_plies = games.plies as f64 / games.total().max(1) as f64;
    let eval = eval.map_or(",,,".to_string(), |score| {
        format!(
            "{},{},{},{:.0}",
            score.wins,
            score.draws,
            score.losses,
            score.elo()
        )
    });
    writeln!(
        file,
        "{generation},{},{},{},{},{average_plies:.1},{:.4},{:.4},{:.4},{result_entropy:.4},{eval}",
        games.total(),
        games.white_wins,
        games.black_wins,
        games.draws,
        losses.policy,
        losses.target_entropy,
        losses.value,
    )?;

    println!(
        "Generation {generation} done: {} white wins, {} black wins, {} draws, {average_plies:.0} \
         plies on average; {}, results' entropy {result_entropy:.3}",
        games.white_wins,
        games.black_wins,
        games.draws,
        describe(losses)
    );
    Ok(())
}

// Streams of random numbers the seed is split into
const SELF_PLAY: u64 = 0;
const TRAINING: u64 = 1;
const EVALUATION: u64 = 2;

// A seed for one part of the run, mixing in `parts` with SplitMix64's finalizer, so that no two
// generations or parts share a stream
fn derive_seed(seed: u64, parts: &[u64]) -> u64 {
    parts.iter().fold(seed, |hash, &part| {
        let mut z = (hash ^ part).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{model::tiny_model_config, samples::read_samples};
    use burn::backend::{Autodiff, NdArray};

    type T = Autodiff<NdArray>;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cinder-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn tiny_config() -> RunConfig {
        RunConfig::new(tiny_model_config())
            .with_games_per_generation(3)
            .with_n_sims(4)
            .with_max_plies(30)
            .with_steps_per_generation(2)
            .with_batch_size(8)
            .with_window_generations(2)
            .with_warmup_steps(1)
            .with_eval_games(2)
            .with_eval_sims(4)
            .with_keep_every(5)
    }

    fn train_generations(run: &Run, n: usize) {
        let device = Default::default();
        train::<NdArray, T>(run, 2, Some(n), &device, &device).unwrap();
    }

    fn checkpoints(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir.join("checkpoints"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    // At least the generation's 3 games, all recorded as its own; the last step can finish more
    fn assert_generation(path: &Path, generation: u32) {
        let games = read_samples(path).unwrap().games;
        assert!(games.len() >= 3, "{} games", games.len());
        assert!(games.iter().all(|game| game.generation == generation));
    }

    #[test]
    fn a_run_continues_after_stopping() {
        let dir = temp_dir("run");
        let run = Run::create::<T>(&dir, tiny_config(), &Default::default()).unwrap();
        assert_eq!(checkpoints(&dir), ["00000"]);
        train_generations(&run, 1);

        assert_eq!(checkpoints(&dir), ["00000", "00001"]);
        assert_generation(&dir.join("samples/00000.bin"), 0);

        // As if started again later, maybe on another machine
        let run = Run::open(&dir).unwrap();
        assert_eq!(run.latest_generation().unwrap(), 1);
        train_generations(&run, 2);

        // Generation 0 stays as a reference, without the optimizer state it no longer needs
        assert_eq!(checkpoints(&dir), ["00000", "00002", "00003"]);
        let reference: Vec<_> = fs::read_dir(dir.join("checkpoints/00000"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(reference, ["model.mpk"]);
        assert!(run.load_network::<NdArray>(0, &Default::default()).is_ok());
        assert_generation(&dir.join("samples/00001.bin"), 1);
        assert_generation(&dir.join("samples/00002.bin"), 2);
        let history = fs::read_to_string(dir.join(HISTORY_FILE)).unwrap();
        let lines: Vec<&str> = history.lines().collect();
        assert_eq!(lines.len(), 4);
        for (generation, line) in lines[1..].iter().enumerate() {
            assert!(line.starts_with(&format!("{generation},")), "{line}");
            // Each generation's evaluation played its 2 games
            let eval: Vec<usize> = line
                .split(',')
                .skip(10)
                .take(3)
                .map(|n| n.parse().unwrap())
                .collect();
            assert_eq!(eval.iter().sum::<usize>(), 2, "{line}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_interrupted_generation_plays_only_its_missing_games() {
        let dir = temp_dir("interrupted");
        let run = Run::create::<T>(&dir, tiny_config(), &Default::default()).unwrap();

        // As if stopped during generation 0's second game: one game, then part of another
        let path = run.samples_path(0);
        let start = crate::chess::State::new();
        let game = vec![Sample {
            fen: start.to_fen(),
            visits: start.generate_moves().iter().map(|&mv| (mv, 1)).collect(),
            result: 0,
        }];
        let mut writer = SampleWriter::open(&path).unwrap();
        writer.write_game(0, &game).unwrap();
        writer.write_game(0, &game).unwrap();
        let len = fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 2)
            .unwrap();
        // and while a checkpoint was being written
        fs::create_dir_all(dir.join("checkpoints/00001.tmp")).unwrap();
        assert_eq!(run.latest_generation().unwrap(), 0);

        train_generations(&run, 1);

        let file = read_samples(&path).unwrap();
        assert!(file.games.len() >= 3);
        assert_eq!(file.games[0].samples, game);
        assert_eq!(file.truncated_bytes, 0);
        assert_eq!(checkpoints(&dir), ["00000", "00001"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn settings_missing_from_an_older_config_take_their_defaults() {
        let dir = temp_dir("older");
        Run::create::<T>(&dir, tiny_config().with_eval_games(5), &Default::default()).unwrap();
        let path = dir.join(CONFIG_FILE);
        let mut saved: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        // As if written before these settings existed
        saved.as_object_mut().unwrap().remove("eval_sims");
        saved["model"]
            .as_object_mut()
            .unwrap()
            .remove("attention_dropout");
        fs::write(&path, saved.to_string()).unwrap();

        let config = Run::open(&dir).unwrap().config;
        assert_eq!(
            config.eval_sims,
            RunConfig::new(tiny_model_config()).eval_sims
        );
        assert_eq!(config.eval_games, 5);
        assert_eq!(config.model.n_blocks, 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_run_is_not_created_twice() {
        let dir = temp_dir("twice");
        Run::create::<T>(&dir, tiny_config(), &Default::default()).unwrap();
        assert!(Run::create::<T>(&dir, tiny_config(), &Default::default()).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn seeds_differ_by_every_part() {
        let seeds = [
            derive_seed(0, &[SELF_PLAY, 0, 0]),
            derive_seed(0, &[SELF_PLAY, 0, 1]),
            derive_seed(0, &[SELF_PLAY, 1, 0]),
            derive_seed(0, &[TRAINING, 0]),
            derive_seed(1, &[SELF_PLAY, 0, 0]),
        ];
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
