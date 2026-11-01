use crate::{
    chess::{moves::Move, types::Color, Outcome, State},
    engine::{
        mcts::{MctsConfig, Search},
        model::{Evaluation, TransformerModel},
        samples::Sample,
    },
};
use burn::{config::Config, tensor::backend::Backend};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Config, Debug)]
pub(crate) struct SelfPlayConfig {
    /// Games played at once, whose positions are evaluated together in one batch
    #[config(default = 256)]
    n_parallel: usize,
    #[config(default = 256)]
    n_games: usize,
    /// Moves are sampled in proportion to visit counts for this many plies, then chosen greedily
    #[config(default = 30)]
    sampling_plies: usize,
    /// Games still going after this many plies are stopped
    #[config(default = 512)]
    max_plies: usize,
    /// Dirichlet noise mixed into the root priors of every search, so games explore moves the
    /// network rates low: concentration alpha, and the noise's share of each prior
    #[config(default = 0.3)]
    noise_alpha: f32,
    #[config(default = 0.25)]
    noise_fraction: f32,
    /// Playout cap randomization, as in KataGo: each move's search is a full one of the search's
    /// `n_sims` visits with this chance, and a fast one of `fast_sims` visits otherwise. Only full
    /// searches become training samples, and only they get root noise, so most moves cost little
    /// and the games still finish. 1 searches every move in full.
    #[config(default = 1.0)]
    full_search_share: f64,
    #[config(default = 50)]
    fast_sims: usize,
    /// Seeds one generator that seeds each slot's own, so no two slots or seeds share a stream
    #[config(default = 0)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
    PlyLimit,
}

impl GameResult {
    /// 1 if white won, 0 for a draw, -1 if black won. Games stopped at the ply limit count as
    /// draws.
    pub(crate) fn for_white(self) -> i8 {
        match self {
            Self::WhiteWins => 1,
            Self::BlackWins => -1,
            Self::Draw | Self::PlyLimit => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GameRecord {
    pub(crate) result: GameResult,
    pub(crate) plies: usize,
}

#[derive(Clone, Default)]
pub(crate) struct SelfPlayStats {
    pub(crate) games: Vec<GameRecord>,
    /// Simulations run, including in games still going
    pub(crate) n_sims: usize,
    pub(crate) n_evaluations: usize,
    /// Batched evaluations, one per step
    pub(crate) n_steps: usize,
    pub(crate) elapsed: Duration,
    pub(crate) select_time: Duration,
    pub(crate) evaluate_time: Duration,
    pub(crate) apply_time: Duration,
    /// Time in `on_game`, such as writing samples, kept out of `select_time`
    pub(crate) on_game_time: Duration,
}

struct Game {
    state: State,
    search: Search,
    // Each game has its own generator, so its moves don't depend on the other games
    rng: StdRng,
    ply: usize,
    // Simulations in the searches already finished this game
    n_sims: usize,
    // Whether the current search is a full one, which becomes a training sample
    full_search: bool,
    // Positions played so far, waiting for the game's result
    pending: Vec<PendingSample>,
}

struct PendingSample {
    fen: String,
    visits: Vec<(Move, u16)>,
    turn: Color,
}

impl Game {
    fn new(mcts_config: &MctsConfig, config: &SelfPlayConfig, seed: u64) -> Self {
        let mut game = Self {
            state: State::new(),
            search: Search::new(mcts_config.clone()),
            rng: StdRng::seed_from_u64(seed),
            ply: 0,
            n_sims: 0,
            full_search: true,
            pending: vec![],
        };
        game.choose_search(config);
        game
    }

    fn restart(&mut self, config: &SelfPlayConfig) {
        self.state = State::new();
        self.search.reset();
        self.ply = 0;
        self.n_sims = 0;
        self.pending.clear();
        self.choose_search(config);
    }

    // Makes the next search a full or a fast one. Draws from the generator only when searches can
    // be fast, so games without fast searches stay the same for a seed.
    fn choose_search(&mut self, config: &SelfPlayConfig) {
        self.full_search =
            config.full_search_share >= 1.0 || self.rng.random::<f64>() < config.full_search_share;
        if !self.full_search {
            self.search.set_n_sims(config.fast_sims);
        }
    }

    /// Plays the move of each completed search until a search needs the network, leaving its
    /// position pending. Returns the record and training samples instead if the game ends. With
    /// `reuse_tree`, each search starts from the subtree of the move played before it, so it only
    /// tops that up to its simulations; only a game with one network can reuse its trees.
    fn advance(
        &mut self,
        config: &SelfPlayConfig,
        reuse_tree: bool,
    ) -> Option<(GameRecord, Vec<Sample>)> {
        while !self.search.next_leaf(&mut self.state) {
            self.n_sims += self.search.n_sims_done();
            let temperature = if self.ply < config.sampling_plies {
                1.0
            } else {
                0.0
            };
            let search_result = self.search.result();
            let mv = search_result
                .sample_move(temperature, &mut self.rng)
                .expect("only unfinished games are searched");

            if self.full_search {
                self.pending.push(PendingSample {
                    fen: self.state.to_fen(),
                    visits: search_result
                        .visits
                        .iter()
                        .map(|&(mv, n_visits)| {
                            let n_visits =
                                u16::try_from(n_visits).expect("visit counts fit in 16 bits");
                            (mv, n_visits)
                        })
                        .collect(),
                    turn: self.state.turn,
                });
            }

            self.state.make_move(mv);
            self.ply += 1;

            if let Some(result) = self.result(config) {
                let record = GameRecord {
                    result,
                    plies: self.ply,
                };
                return Some((record, self.take_samples(result)));
            }

            if reuse_tree {
                self.search.reuse_subtree(mv);
            } else {
                self.search.reset();
            }
            self.choose_search(config);
            // A reused root is already expanded, so its noise can't come with its evaluation
            if reuse_tree && self.full_search && config.noise_fraction > 0.0 {
                let rng = &mut self.rng;
                self.search.mix_into_root_priors(|priors| {
                    add_noise(priors, config.noise_alpha, config.noise_fraction, rng)
                });
            }
        }

        None
    }

    // Which of two networks the current search is for, given the one playing white: the side to
    // move at the search's root, which changes every ply
    fn network_to_move(&self, white: usize) -> usize {
        if self.ply.is_multiple_of(2) {
            white
        } else {
            1 - white
        }
    }

    fn apply(&mut self, mut evaluation: Evaluation, config: &SelfPlayConfig) {
        if self.search.leaf_is_root() && self.full_search && config.noise_fraction > 0.0 {
            add_noise(
                &mut evaluation.priors,
                config.noise_alpha,
                config.noise_fraction,
                &mut self.rng,
            );
        }
        self.search.apply(&mut self.state, evaluation);
    }

    // Gives each position played the game's result for its side to move
    fn take_samples(&mut self, result: GameResult) -> Vec<Sample> {
        let white_result = result.for_white();
        self.pending
            .drain(..)
            .map(|position| Sample {
                fen: position.fen,
                visits: position.visits,
                result: match position.turn {
                    Color::White => white_result,
                    Color::Black => -white_result,
                },
            })
            .collect()
    }

    fn result(&self, config: &SelfPlayConfig) -> Option<GameResult> {
        match self.state.outcome(&self.state.generate_moves()) {
            Some(Outcome::Checkmate) => Some(match self.state.turn {
                Color::White => GameResult::BlackWins,
                Color::Black => GameResult::WhiteWins,
            }),
            Some(Outcome::Draw) => Some(GameResult::Draw),
            None => (self.ply >= config.max_plies).then_some(GameResult::PlyLimit),
        }
    }
}

/// Plays `n_games` games, `n_parallel` at a time, with one network. See `SelfPlay::play`.
pub(crate) fn run<B: Backend>(
    model: &TransformerModel<B>,
    mcts_config: &MctsConfig,
    config: &SelfPlayConfig,
    device: &B::Device,
    on_game: impl FnMut(&GameRecord, Vec<Sample>),
    on_progress: impl FnMut(&SelfPlayStats),
) -> SelfPlayStats {
    SelfPlay::new(mcts_config, config).play(model, usize::MAX, device, on_game, on_progress)
}

/// Games in progress, one per slot. A finished game's slot starts a new one until `n_games` have
/// started, then empties. Games still going when `play` returns wait for the next call, which may
/// bring a newer network.
pub(crate) struct SelfPlay {
    config: SelfPlayConfig,
    games: Vec<Game>,
    n_started: usize,
    // Simulations in the games finished so far
    finished_sims: usize,
}

impl SelfPlay {
    pub(crate) fn new(mcts_config: &MctsConfig, config: &SelfPlayConfig) -> Self {
        // The first simulation only expands the root, so a search needs two to compare moves
        assert!(
            config.full_search_share >= 1.0 || config.fast_sims >= 2,
            "fast searches need at least 2 simulations"
        );
        let n_slots = config.n_parallel.min(config.n_games);
        let mut seeds = StdRng::seed_from_u64(config.seed);
        Self {
            config: config.clone(),
            games: (0..n_slots)
                .map(|_| Game::new(mcts_config, config, seeds.random()))
                .collect(),
            n_started: n_slots,
            finished_sims: 0,
        }
    }

    /// Plays with `model` until `n_finish` more games end or every game is done. Each step, every
    /// game's search advances to a position that needs the network, all of those positions are
    /// evaluated in one batch, and each search takes in its result. `on_game` receives each
    /// finished game with its training samples, and `on_progress` is called every few seconds with
    /// the stats of this call. Games stopped at the ply limit count as draws.
    ///
    /// The tree work runs on one thread. It's a small share of each step, and a thread pool measured
    /// slower overall: dispatching thousands of microsecond tasks costs more than it saves, and
    /// even the network calls got slower, likely from idle workers competing with the thread
    /// driving the GPU.
    pub(crate) fn play<B: Backend>(
        &mut self,
        model: &TransformerModel<B>,
        n_finish: usize,
        device: &B::Device,
        mut on_game: impl FnMut(&GameRecord, Vec<Sample>),
        mut on_progress: impl FnMut(&SelfPlayStats),
    ) -> SelfPlayStats {
        let start = Instant::now();
        let mut last_progress = start;
        let mut stats = SelfPlayStats::default();
        let sims_before = self.total_sims();
        let config = &self.config;

        // Stops after applying evaluations, so no game is left waiting for one
        while !self.games.is_empty() && stats.games.len() < n_finish {
            let select_start = Instant::now();
            let mut on_game_time = Duration::ZERO;
            let mut i = 0;
            while i < self.games.len() {
                let Some((record, samples)) = self.games[i].advance(config, true) else {
                    i += 1;
                    continue;
                };
                let on_game_start = Instant::now();
                on_game(&record, samples);
                on_game_time += on_game_start.elapsed();
                stats.games.push(record);
                self.finished_sims += self.games[i].n_sims;

                if self.n_started == config.n_games {
                    self.games.remove(i);
                    continue;
                }

                // The new game in this slot needs its first position evaluated too
                self.games[i].restart(config);
                self.n_started += 1;
                let record = self.games[i].advance(config, true);
                debug_assert!(
                    record.is_none(),
                    "a new game can't end before its first move"
                );
                i += 1;
            }
            stats.select_time += select_start.elapsed() - on_game_time;
            stats.on_game_time += on_game_time;

            if self.games.is_empty() {
                break;
            }

            let evaluate_start = Instant::now();
            let states: Vec<&State> = self.games.iter().map(|game| &game.state).collect();
            let moves: Vec<&[Move]> = self
                .games
                .iter()
                .map(|game| game.search.leaf_moves())
                .collect();
            let evaluations = model.evaluate(&states, &moves, device);
            stats.evaluate_time += evaluate_start.elapsed();
            stats.n_evaluations += evaluations.len();
            stats.n_steps += 1;

            let apply_start = Instant::now();
            for (game, evaluation) in self.games.iter_mut().zip(evaluations) {
                game.apply(evaluation, config);
            }
            stats.apply_time += apply_start.elapsed();

            if last_progress.elapsed() >= PROGRESS_INTERVAL {
                stats.n_sims = self.total_sims() - sims_before;
                stats.elapsed = start.elapsed();
                on_progress(&stats);
                last_progress = Instant::now();
            }
        }

        stats.n_sims = self.total_sims() - sims_before;
        stats.elapsed = start.elapsed();
        stats
    }

    // Simulations run so far, including in games still going
    fn total_sims(&self) -> usize {
        let ongoing: usize = self
            .games
            .iter()
            .map(|game| game.n_sims + game.search.n_sims_done())
            .sum();
        self.finished_sims + ongoing
    }
}

/// A match's result for its first network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MatchScore {
    pub(crate) wins: usize,
    pub(crate) draws: usize,
    pub(crate) losses: usize,
    /// Of the draws, games stopped at the ply limit
    pub(crate) stopped: usize,
    /// Plies played in all games together
    pub(crate) plies: usize,
}

impl MatchScore {
    pub(crate) fn games(&self) -> usize {
        self.wins + self.draws + self.losses
    }

    pub(crate) fn average_plies(&self) -> f64 {
        self.plies as f64 / self.games().max(1) as f64
    }

    /// Points per game, a draw counting half.
    pub(crate) fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.games().max(1) as f64
    }

    /// The Elo difference the score implies. A clean sweep would imply an infinite one, so the
    /// score is kept half a game away from 0 and 1.
    pub(crate) fn elo(&self) -> f64 {
        let margin = 0.5 / self.games().max(1) as f64;
        let score = self.score().clamp(margin, 1.0 - margin);
        400.0 * (score / (1.0 - score)).log10()
    }
}

/// Plays `config.n_games` games between two networks, `config.n_parallel` at a time, the first
/// network playing white in every other game, and returns the first network's score. Each search
/// uses only the network whose move it's for. Root noise and move sampling follow `config`, which
/// should keep a few plies of sampling so that the games differ.
pub(crate) fn play_match<B: Backend>(
    networks: [&TransformerModel<B>; 2],
    mcts_config: &MctsConfig,
    config: &SelfPlayConfig,
    device: &B::Device,
) -> MatchScore {
    let n_slots = config.n_parallel.min(config.n_games);
    let mut seeds = StdRng::seed_from_u64(config.seed);
    // Each game, with the index of the network playing white
    let mut games: Vec<(Game, usize)> = (0..n_slots)
        .map(|i| (Game::new(mcts_config, config, seeds.random()), i % 2))
        .collect();
    let mut n_started = n_slots;
    let mut score = MatchScore::default();

    loop {
        let mut i = 0;
        while i < games.len() {
            let Some((record, _)) = games[i].0.advance(config, false) else {
                i += 1;
                continue;
            };
            let white = games[i].1;
            match record.result {
                GameResult::WhiteWins if white == 0 => score.wins += 1,
                GameResult::BlackWins if white == 1 => score.wins += 1,
                GameResult::WhiteWins | GameResult::BlackWins => score.losses += 1,
                GameResult::Draw => score.draws += 1,
                GameResult::PlyLimit => {
                    score.draws += 1;
                    score.stopped += 1;
                }
            }
            score.plies += record.plies;

            if n_started == config.n_games {
                games.remove(i);
                continue;
            }
            let (game, white) = &mut games[i];
            game.restart(config);
            *white = n_started % 2;
            n_started += 1;
            let record = game.advance(config, false);
            debug_assert!(
                record.is_none(),
                "a new game can't end before its first move"
            );
            i += 1;
        }

        if games.is_empty() {
            return score;
        }

        // Each network evaluates the positions of the searches for its own moves. How many that
        // is changes every step, so the batches are padded to reuse a few sizes.
        for (network, model) in networks.iter().enumerate() {
            let searching: Vec<usize> = (0..games.len())
                .filter(|&i| games[i].0.network_to_move(games[i].1) == network)
                .collect();
            if searching.is_empty() {
                continue;
            }

            let states: Vec<&State> = searching.iter().map(|&i| &games[i].0.state).collect();
            let moves: Vec<&[Move]> = searching
                .iter()
                .map(|&i| games[i].0.search.leaf_moves())
                .collect();
            let evaluations = model.evaluate_padded(&states, &moves, device);
            for (&i, evaluation) in searching.iter().zip(evaluations) {
                games[i].0.apply(evaluation, config);
            }
        }
    }
}

/// Mixes Dirichlet(alpha) noise into `priors`, which keep summing to 1. The noise is drawn by
/// normalizing independent Gamma(alpha, 1) samples.
fn add_noise(priors: &mut [f32], alpha: f32, fraction: f32, rng: &mut impl Rng) {
    let gamma = Gamma::new(alpha, 1.0).expect("the noise alpha must be positive");
    let noise: Vec<f32> = priors.iter().map(|_| gamma.sample(rng)).collect();
    let total: f32 = noise.iter().sum();
    if total <= 0.0 {
        return;
    }

    for (prior, noise) in priors.iter_mut().zip(noise) {
        *prior = (1.0 - fraction) * *prior + fraction * noise / total;
    }
}

/// Rates over the interval since `previous`, and where that time went.
pub(crate) fn progress_line(current: &SelfPlayStats, previous: &SelfPlayStats) -> String {
    let seconds = (current.elapsed - previous.elapsed).as_secs_f64();
    let evaluations = (current.n_evaluations - previous.n_evaluations) as f64;
    let steps = (current.n_steps - previous.n_steps) as f64;
    let share = |now: std::time::Duration, before: std::time::Duration| {
        100.0 * (now - before).as_secs_f64() / seconds
    };

    format!(
        "{:>5.0}s  games {:>4}  {:>6.0} sims/s  {:>6.0} evals/s  batch {:>5.1}  \
         time: select {:>4.1}%, network {:>4.1}%, apply {:>4.1}%, write {:>4.1}%",
        current.elapsed.as_secs_f64(),
        current.games.len(),
        (current.n_sims - previous.n_sims) as f64 / seconds,
        evaluations / seconds,
        evaluations / steps,
        share(current.select_time, previous.select_time),
        share(current.evaluate_time, previous.evaluate_time),
        share(current.apply_time, previous.apply_time),
        share(current.on_game_time, previous.on_game_time),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::tiny_model;
    use burn::backend::NdArray;

    // More games than slots, so finished games are replaced and then slots retired
    fn test_config(seed: u64) -> SelfPlayConfig {
        SelfPlayConfig::new()
            .with_n_parallel(4)
            .with_n_games(6)
            .with_max_plies(40)
            .with_seed(seed)
    }

    // Returns the stats and each finished game with its samples
    fn play(
        model: &TransformerModel<NdArray>,
        seed: u64,
    ) -> (SelfPlayStats, Vec<(GameRecord, Vec<Sample>)>) {
        play_with(model, &test_config(seed))
    }

    fn play_with(
        model: &TransformerModel<NdArray>,
        config: &SelfPlayConfig,
    ) -> (SelfPlayStats, Vec<(GameRecord, Vec<Sample>)>) {
        let mcts_config = MctsConfig::new().with_n_sims(8);
        let mut games = vec![];
        let stats = run(
            model,
            &mcts_config,
            config,
            &Default::default(),
            |record, samples| games.push((record.clone(), samples)),
            |_| {},
        );
        (stats, games)
    }

    #[test]
    fn every_game_finishes() {
        let (stats, _) = play(&tiny_model(), 0);

        assert_eq!(stats.games.len(), 6);
        for record in &stats.games {
            assert!((1..=40).contains(&record.plies));
            // A game can also end by checkmate or a draw on the last allowed ply
            if record.result == GameResult::PlyLimit {
                assert_eq!(record.plies, 40);
            }
        }

        // Every search ran at least one simulation, and the reused subtrees saved some of the 8
        let total_plies: usize = stats.games.iter().map(|record| record.plies).sum();
        assert!((total_plies..total_plies * 8).contains(&stats.n_sims));
        // Each step evaluates one position per game still going, never more than the slots
        assert!(stats.n_steps <= stats.n_evaluations);
        assert!(stats.n_evaluations <= stats.n_steps * 4);
    }

    #[test]
    fn noise_keeps_priors_a_distribution() {
        let uniform = vec![0.05; 20];
        let noisy = |seed| {
            let mut priors = uniform.clone();
            add_noise(&mut priors, 0.3, 0.25, &mut StdRng::seed_from_u64(seed));
            priors
        };

        let priors = noisy(1);
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // Each prior keeps at least its unmixed share
        assert!(priors.iter().all(|&p| p >= 0.75 * 0.05 - 1e-6));
        assert_ne!(priors, uniform);
        assert_eq!(priors, noisy(1));
        assert_ne!(priors, noisy(2));
    }

    #[test]
    fn noise_changes_the_search() {
        // One game's first search with uniform priors, so only the noise can favour a move
        let first_search_visits = |config: &SelfPlayConfig| {
            let mut game = Game::new(&MctsConfig::new().with_n_sims(50), config, 0);
            while game.search.next_leaf(&mut game.state) {
                let n_moves = game.search.leaf_moves().len();
                let evaluation = Evaluation {
                    priors: vec![1.0 / n_moves as f32; n_moves],
                    value: 0.0,
                };
                game.apply(evaluation, config);
            }
            game.search.result().visits
        };

        let noisy = first_search_visits(&SelfPlayConfig::new());
        assert_eq!(noisy, first_search_visits(&SelfPlayConfig::new()));
        let plain = first_search_visits(&SelfPlayConfig::new().with_noise_fraction(0.0));
        assert_ne!(noisy, plain);
        // A fast search isn't a sample, so it explores no more than the network suggests
        let fast = SelfPlayConfig::new()
            .with_full_search_share(0.0)
            .with_fast_sims(50);
        assert_eq!(first_search_visits(&fast), plain);
    }

    #[test]
    fn a_reused_root_gets_noise_too() {
        // The root priors of a game's second search, which reuses the first one's subtree, when
        // every evaluation is uniform
        let second_root_priors = |config: &SelfPlayConfig| {
            let mut game = Game::new(&MctsConfig::new().with_n_sims(50), config, 0);
            while game.ply == 0 {
                assert!(game.advance(config, true).is_none());
                let n_moves = game.search.leaf_moves().len();
                let evaluation = Evaluation {
                    priors: vec![1.0 / n_moves as f32; n_moves],
                    value: 0.0,
                };
                game.apply(evaluation, config);
            }
            let mut priors = vec![];
            game.search
                .mix_into_root_priors(|root_priors| priors = root_priors.to_vec());
            priors
        };

        let plain = second_root_priors(&SelfPlayConfig::new().with_noise_fraction(0.0));
        assert_eq!(plain, vec![0.05; 20]);
        let noisy = second_root_priors(&SelfPlayConfig::new());
        assert_eq!(noisy.len(), 20);
        assert!((noisy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_ne!(noisy, plain);
        let all_fast = SelfPlayConfig::new()
            .with_full_search_share(0.0)
            .with_fast_sims(50);
        assert_eq!(second_root_priors(&all_fast), plain);
    }

    #[test]
    fn only_full_searches_become_samples() {
        let model = tiny_model();
        let with_share = |share| {
            play_with(
                &model,
                &test_config(0)
                    .with_full_search_share(share)
                    .with_fast_sims(3),
            )
        };

        // Fast searches of 3 simulations make moves cheaper than full ones of 8
        let sims_per_ply = |(stats, _): (SelfPlayStats, Vec<_>)| {
            let n_plies: usize = stats.games.iter().map(|record| record.plies).sum();
            stats.n_sims as f64 / n_plies as f64
        };
        assert!(sims_per_ply(with_share(0.5)) < 0.8 * sims_per_ply(with_share(1.0)));

        let (_, games) = with_share(0.5);
        let n_plies: usize = games.iter().map(|(record, _)| record.plies).sum();
        let n_samples: usize = games.iter().map(|(_, samples)| samples.len()).sum();
        assert!(
            0 < n_samples && n_samples < n_plies,
            "{n_samples} of {n_plies}"
        );
        // The samples' searches were full ones, 8 simulations with the first expanding the root
        for (_, samples) in &games {
            for sample in samples {
                let total_visits: u16 = sample.visits.iter().map(|(_, n)| n).sum();
                assert_eq!(total_visits, 7);
            }
        }
        assert_eq!(with_share(0.5).1, games);

        let (_, games) = with_share(0.0);
        assert!(games.iter().all(|(_, samples)| samples.is_empty()));
    }

    #[test]
    fn every_move_becomes_a_sample() {
        let (_, games) = play(&tiny_model(), 0);

        for (record, samples) in &games {
            assert_eq!(samples.len(), record.plies);
            assert_eq!(samples[0].fen, State::new().to_fen());

            for sample in samples {
                // The search ran 8 simulations, and the first one expanded the root
                let total_visits: u16 = sample.visits.iter().map(|(_, n)| n).sum();
                assert_eq!(total_visits, 7);

                // The FEN is the searched position, and its legal moves are all recorded
                let state = State::from_fen(&sample.fen);
                assert_eq!(state.to_fen(), sample.fen);
                let moves: Vec<Move> = sample.visits.iter().map(|(mv, _)| *mv).collect();
                assert_eq!(moves, state.generate_moves().as_slice());
            }
        }
    }

    #[test]
    fn results_are_for_the_side_to_move() {
        let results = |result| {
            let mut game = Game::new(&MctsConfig::new(), &SelfPlayConfig::new(), 0);
            for turn in [Color::White, Color::Black, Color::White] {
                game.pending.push(PendingSample {
                    fen: String::new(),
                    visits: vec![],
                    turn,
                });
            }
            let samples = game.take_samples(result);
            samples
                .iter()
                .map(|sample| sample.result)
                .collect::<Vec<_>>()
        };

        assert_eq!(results(GameResult::WhiteWins), [1, -1, 1]);
        assert_eq!(results(GameResult::BlackWins), [-1, 1, -1]);
        assert_eq!(results(GameResult::Draw), [0, 0, 0]);
        assert_eq!(results(GameResult::PlyLimit), [0, 0, 0]);
    }

    #[test]
    fn a_match_plays_every_game() {
        let (first, second) = (tiny_model(), tiny_model());
        let config = SelfPlayConfig::new()
            .with_n_parallel(4)
            .with_n_games(6)
            .with_max_plies(40)
            .with_noise_fraction(0.0)
            .with_sampling_plies(4);
        let mcts_config = MctsConfig::new().with_n_sims(8);
        let play = || {
            play_match(
                [&first, &second],
                &mcts_config,
                &config,
                &Default::default(),
            )
        };

        let score = play();
        assert_eq!(score.games(), 6);
        assert_eq!(play(), score);
    }

    #[test]
    fn match_scores_imply_elo_differences() {
        let score = |wins, draws, losses| MatchScore {
            wins,
            draws,
            losses,
            ..MatchScore::default()
        };
        assert_eq!(score(3, 4, 3).elo(), 0.0);
        assert!((score(3, 0, 1).elo() - 190.85).abs() < 0.01);
        assert!((score(1, 0, 3).elo() + 190.85).abs() < 0.01);
        // A sweep counts as half a game short of one
        assert_eq!(score(10, 0, 0).elo(), score(19, 0, 1).elo());
    }

    #[test]
    fn searches_alternate_between_the_networks() {
        let mut game = Game::new(&MctsConfig::new(), &SelfPlayConfig::new(), 0);
        assert_eq!(game.network_to_move(1), 1);
        game.ply = 1;
        assert_eq!(game.network_to_move(1), 0);
        assert_eq!(game.network_to_move(0), 1);
    }

    #[test]
    fn runs_with_the_same_seed_match() {
        let model = tiny_model();
        assert_eq!(play(&model, 7).1, play(&model, 7).1);
    }

    #[test]
    fn different_seeds_play_different_games() {
        let model = tiny_model();
        let first_games = |seed| {
            let (_, games) = play(&model, seed);
            games
                .into_iter()
                .map(|(_, samples)| samples)
                .collect::<Vec<_>>()
        };

        // Neighbouring seeds once shared all but one slot's games
        let seed_0 = first_games(0);
        let seed_1 = first_games(1);
        for game in &seed_1 {
            assert!(!seed_0.contains(game));
        }
    }
}
