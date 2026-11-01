use crate::{
    chess::{moves::Move, Outcome, State},
    engine::model::{Evaluation, TransformerModel},
};
use arrayvec::ArrayVec;
use burn::{config::Config, tensor::backend::Backend};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    Rng,
};
use std::{cmp::Ordering, ops::Range};

#[derive(Clone, Copy)]
struct Node {
    mv: Option<Move>,
    n_visits: usize,
    // From the perspective of the side to move at this node
    value_sum: f32,
    prior: f32,
    // Exact value of a finished game, so the model is never asked to guess it
    terminal: Option<f32>,
    // Children are pushed together on expansion, so they occupy a contiguous range of nodes
    first_child: u32,
    n_children: u8,
}

impl Node {
    fn new(mv: Option<Move>, prior: f32) -> Self {
        Self {
            mv,
            n_visits: 0,
            value_sum: 0.0,
            prior,
            terminal: None,
            first_child: 0,
            n_children: 0,
        }
    }

    fn children(&self) -> Range<usize> {
        let first = self.first_child as usize;
        first..first + self.n_children as usize
    }

    fn q_val(&self) -> f32 {
        if self.n_visits == 0 {
            0.0
        } else {
            self.value_sum / self.n_visits as f32
        }
    }

    fn u_val(&self, c_puct: f32, sqrt_parent_visits: f32) -> f32 {
        c_puct * self.prior * (sqrt_parent_visits / (1.0 + self.n_visits as f32))
    }

    // Q is negated: a child's value belongs to the opponent of the side choosing it
    fn score(&self, c_puct: f32, sqrt_parent_visits: f32) -> f32 {
        -self.q_val() + self.u_val(c_puct, sqrt_parent_visits)
    }

    fn record_visit(&mut self, val: f32) {
        self.n_visits += 1;
        self.value_sum += val;
    }
}

#[derive(Config, Debug)]
pub(crate) struct MctsConfig {
    /// How strongly selection favours moves with high priors and few visits over moves with good
    /// results so far
    #[config(default = 1.5)]
    c_puct: f32,
    /// Visits the root reaches in each search. A fresh search's first simulation expands the root,
    /// so the root's moves get `n_sims - 1` visits in total. A search that reuses a subtree starts
    /// with its visits and only runs the simulations still missing.
    #[config(default = 800)]
    n_sims: usize,
}

/// A search tree that pauses whenever a simulation reaches a position the network must evaluate,
/// so many searches can share one batched evaluation.
pub(crate) struct Search {
    nodes: Vec<Node>,
    config: MctsConfig,
    // Visits the root reaches in this search: `config.n_sims` unless lowered for a fast search
    n_sims: usize,
    n_sims_done: usize,
    // Nodes from the root to the current leaf; empty between simulations
    path: Vec<usize>,
    leaf_moves: ArrayVec<Move, 218>,
}

impl Search {
    pub(crate) fn new(config: MctsConfig) -> Self {
        let mut search = Self {
            nodes: vec![],
            n_sims: config.n_sims,
            config,
            n_sims_done: 0,
            path: vec![],
            leaf_moves: ArrayVec::new(),
        };
        search.reset();
        search
    }

    /// Discards the tree, ready to search a new position.
    pub(crate) fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::new(None, 1.0));
        self.n_sims = self.config.n_sims;
        self.n_sims_done = 0;
        self.path.clear();
    }

    /// Lowers the visits this search reaches from the configured `n_sims`, until the next `reset`
    /// or `reuse_subtree`.
    pub(crate) fn set_n_sims(&mut self, n_sims: usize) {
        self.n_sims = n_sims;
    }

    /// Keeps the subtree of the root's move `mv` as the new root, ready to search the position
    /// after `mv` with the visits it already has. Starts afresh if `mv` was never visited.
    pub(crate) fn reuse_subtree(&mut self, mv: Move) {
        let child = self.nodes[0]
            .children()
            .find(|&i| self.nodes[i].mv == Some(mv))
            .filter(|&i| self.nodes[i].n_visits > 0);
        let Some(child) = child else {
            self.reset();
            return;
        };

        // Copies the subtree breadth first, so each node's children stay contiguous
        let mut nodes = Vec::with_capacity(self.config.n_sims * 4);
        nodes.push(Node {
            mv: None,
            ..self.nodes[child]
        });
        let mut next = 0;
        while next < nodes.len() {
            let children = nodes[next].children();
            nodes[next].first_child = nodes.len() as u32;
            nodes.extend(children.map(|i| self.nodes[i]));
            next += 1;
        }

        self.nodes = nodes;
        self.n_sims = self.config.n_sims;
        self.n_sims_done = 0;
        self.path.clear();
    }

    /// Mixes into the priors of the root's moves with `mix`, if the root is already expanded, as a
    /// reused one is. A fresh root's priors come with its evaluation instead.
    pub(crate) fn mix_into_root_priors(&mut self, mix: impl FnOnce(&mut [f32])) {
        let children = self.nodes[0].children();
        if children.is_empty() {
            return;
        }
        let mut priors: Vec<f32> = children.clone().map(|i| self.nodes[i].prior).collect();
        mix(&mut priors);
        for (i, prior) in children.zip(priors) {
            self.nodes[i].prior = prior;
        }
    }

    /// Runs simulations until one reaches a position the network must evaluate, leaving `state`
    /// there with its legal moves in `leaf_moves`. Finished games are scored without the network.
    /// Returns false once the root has the search's visits, with `state` back at the root.
    pub(crate) fn next_leaf(&mut self, state: &mut State) -> bool {
        debug_assert!(
            self.path.is_empty(),
            "the pending leaf must be applied first"
        );

        while self.nodes[0].n_visits < self.n_sims {
            let leaf = self.select(state);

            if let Some(value) = self.nodes[leaf].terminal {
                self.backup(state, value);
                continue;
            }

            let moves = state.generate_moves();
            if let Some(outcome) = state.outcome(&moves) {
                let value = match outcome {
                    Outcome::Checkmate => -1.0,
                    Outcome::Draw => 0.0,
                };
                self.nodes[leaf].terminal = Some(value);
                self.backup(state, value);
                continue;
            }

            self.leaf_moves = moves;
            return true;
        }

        false
    }

    pub(crate) fn leaf_moves(&self) -> &[Move] {
        &self.leaf_moves
    }

    /// Whether the pending leaf is the root, which happens once per search, on its first simulation.
    pub(crate) fn leaf_is_root(&self) -> bool {
        self.path.len() == 1
    }

    /// Simulations run since the search started, not counting the visits of a reused subtree.
    pub(crate) fn n_sims_done(&self) -> usize {
        self.n_sims_done
    }

    /// Expands the pending leaf with the network's evaluation, backs its value up the tree and
    /// returns `state` to the root.
    pub(crate) fn apply(&mut self, state: &mut State, evaluation: Evaluation) {
        let leaf = *self.path.last().expect("no leaf is pending");
        debug_assert_eq!(evaluation.priors.len(), self.leaf_moves.len());

        self.nodes[leaf].first_child = self.nodes.len() as u32;
        self.nodes[leaf].n_children = self.leaf_moves.len() as u8;
        self.nodes.extend(
            self.leaf_moves
                .iter()
                .zip(evaluation.priors)
                .map(|(mv, prior)| Node::new(Some(*mv), prior)),
        );

        self.backup(state, evaluation.value);
    }

    pub(crate) fn result(&self) -> SearchResult {
        let root = &self.nodes[0];
        SearchResult {
            visits: root
                .children()
                .map(|i| (self.nodes[i].mv.unwrap(), self.nodes[i].n_visits))
                .collect(),
        }
    }

    // Descends from the root by PUCT score, making each move on `state`, until it reaches a node
    // that is unvisited or a finished game
    fn select(&mut self, state: &mut State) -> usize {
        let mut idx = 0;
        self.path.push(idx);

        while self.nodes[idx].n_visits > 0 && self.nodes[idx].terminal.is_none() {
            idx = self.best_child(idx);
            state.make_move(self.nodes[idx].mv.unwrap());
            self.path.push(idx);
        }

        idx
    }

    fn best_child(&self, idx: usize) -> usize {
        let sqrt_parent_visits = (self.nodes[idx].n_visits as f32).sqrt();

        self.nodes[idx]
            .children()
            .map(|i| {
                (
                    i,
                    self.nodes[i].score(self.config.c_puct, sqrt_parent_visits),
                )
            })
            .max_by(|(_, sa), (_, sb)| sa.partial_cmp(sb).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap()
    }

    // Records `value`, from the leaf's side to move, at each node on the path, flipping its sign
    // at every level up, and unmakes the path's moves
    fn backup(&mut self, state: &mut State, mut value: f32) {
        while let Some(idx) = self.path.pop() {
            self.nodes[idx].record_visit(value);
            value = -value;

            if !self.path.is_empty() {
                state.unmake_move();
            }
        }

        self.n_sims_done += 1;
    }
}

/// Runs a single search, evaluating one position at a time.
pub(crate) struct Mcts<'a, B: Backend> {
    search: Search,
    model: &'a TransformerModel<B>,
    device: B::Device,
}

impl<'a, B: Backend> Mcts<'a, B> {
    pub(crate) fn new(
        model: &'a TransformerModel<B>,
        config: MctsConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            search: Search::new(config),
            model,
            device: device.clone(),
        }
    }

    /// Builds a fresh tree from `state`, which is restored before returning.
    pub(crate) fn search(&mut self, state: &mut State) -> SearchResult {
        self.search.reset();

        while self.search.next_leaf(state) {
            let evaluation = self
                .model
                .evaluate(&[&*state], &[self.search.leaf_moves()], &self.device)
                .remove(0);
            self.search.apply(state, evaluation);
        }

        self.search.result()
    }
}

pub(crate) struct SearchResult {
    /// Root moves with their visit counts; empty if the root is a finished game
    pub(crate) visits: Vec<(Move, usize)>,
}

impl SearchResult {
    pub(crate) fn best_move(&self) -> Option<Move> {
        self.visits
            .iter()
            .max_by_key(|(_, n_visits)| *n_visits)
            .map(|(mv, _)| *mv)
    }

    /// Samples a move with probability proportional to N^(1 / temperature).
    pub(crate) fn sample_move(&self, temperature: f32, rng: &mut impl Rng) -> Option<Move> {
        let max_visits = self.visits.iter().map(|(_, n_visits)| *n_visits).max()?;
        if temperature <= 0.0 || max_visits == 0 {
            return self.best_move();
        }

        // Scaling by the max keeps N^(1 / temperature) finite at low temperatures
        let weights = self
            .visits
            .iter()
            .map(|(_, n_visits)| (*n_visits as f32 / max_visits as f32).powf(1.0 / temperature));
        let dist = WeightedIndex::new(weights).ok()?;

        Some(self.visits[dist.sample(rng)].0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chess::state::START_FEN, engine::model::tiny_model};
    use rand::{rngs::StdRng, SeedableRng};

    fn search(state: &mut State, n_sims: usize) -> SearchResult {
        let model = tiny_model();
        let config = MctsConfig::new().with_n_sims(n_sims);
        Mcts::new(&model, config, &Default::default()).search(state)
    }

    fn best_move(fen: &str, n_sims: usize) -> Option<String> {
        search(&mut State::from_fen(fen), n_sims)
            .best_move()
            .map(|mv| mv.to_string())
    }

    // An untrained model has no idea which move mates, so these only pass if
    // terminal values are exact and backed up with the correct sign
    #[test]
    fn finds_mate_in_one_for_white() {
        assert_eq!(
            best_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 400),
            Some("a1a8".into())
        );
    }

    #[test]
    fn finds_mate_in_one_for_black() {
        assert_eq!(
            best_move("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", 400),
            Some("a8a1".into())
        );
    }

    #[test]
    fn finished_game_has_no_moves() {
        let mated = search(
            &mut State::from_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"),
            10,
        );
        assert!(mated.visits.is_empty());
        assert_eq!(mated.best_move(), None);

        let stalemate = search(&mut State::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), 10);
        assert!(stalemate.visits.is_empty());
        assert_eq!(stalemate.best_move(), None);
    }

    #[test]
    fn search_restores_state() {
        let mut state = State::from_fen(START_FEN);
        let before = state.to_string();
        search(&mut state, 50);
        assert_eq!(state.to_string(), before);
        assert_eq!(state.perft(2), 400);
    }

    #[test]
    fn search_pauses_at_positions_needing_evaluation() {
        let n_sims = 30;
        let mut search = Search::new(MctsConfig::new().with_n_sims(n_sims));
        let mut state = State::from_fen(START_FEN);
        let root = state.to_string();
        let mut n_evaluations = 0;

        while search.next_leaf(&mut state) {
            // Only the first simulation stops at the root itself
            assert_eq!(state.to_string() == root, n_evaluations == 0);
            assert_eq!(search.leaf_is_root(), n_evaluations == 0);
            assert_eq!(search.leaf_moves(), state.generate_moves().as_slice());

            let n_moves = search.leaf_moves().len();
            let evaluation = Evaluation {
                priors: vec![1.0 / n_moves as f32; n_moves],
                value: 0.0,
            };
            search.apply(&mut state, evaluation);
            assert_eq!(state.to_string(), root);
            n_evaluations += 1;
        }

        // No game ends this close to the start, so every simulation needed the network
        assert_eq!(n_evaluations, n_sims);
        assert_eq!(state.to_string(), root);
        let total_visits: usize = search.result().visits.iter().map(|(_, n)| n).sum();
        assert_eq!(total_visits, n_sims - 1);
    }

    #[test]
    fn visits_account_for_every_simulation() {
        let n_sims = 50;
        let result = search(&mut State::from_fen(START_FEN), n_sims);

        // The first simulation expands the root, every later one descends into a child
        let total_visits: usize = result.visits.iter().map(|(_, n)| n).sum();
        assert_eq!(result.visits.len(), 20);
        assert_eq!(total_visits, n_sims - 1);
    }

    // Runs the search's remaining simulations with the tiny model, returning how many evaluations
    // they needed
    fn finish(search: &mut Search, state: &mut State) -> usize {
        let model = tiny_model();
        let mut n_evaluations = 0;
        while search.next_leaf(state) {
            let evaluation = model
                .evaluate(&[&*state], &[search.leaf_moves()], &Default::default())
                .remove(0);
            search.apply(state, evaluation);
            n_evaluations += 1;
        }
        n_evaluations
    }

    fn total_visits(search: &Search) -> usize {
        search.result().visits.iter().map(|(_, n)| n).sum()
    }

    #[test]
    fn a_reused_subtree_keeps_its_visits() {
        let n_sims = 60;
        let mut search = Search::new(MctsConfig::new().with_n_sims(n_sims));
        let mut state = State::from_fen(START_FEN);
        finish(&mut search, &mut state);
        let result = search.result();
        let mv = result.best_move().unwrap();
        let kept = result.visits.iter().find(|(m, _)| *m == mv).unwrap().1;
        assert!(
            kept > 1,
            "the test needs a move whose own moves were visited"
        );

        state.make_move(mv);
        search.reuse_subtree(mv);
        // The new root holds the old search's visits to its moves: all but its own expansion
        let moves: Vec<Move> = search.result().visits.iter().map(|(m, _)| *m).collect();
        assert_eq!(moves, state.generate_moves().as_slice());
        assert_eq!(total_visits(&search), kept - 1);

        // No game ends this close to the start, so every missing visit needs the network
        assert_eq!(finish(&mut search, &mut state), n_sims - kept);
        assert_eq!(search.n_sims_done(), n_sims - kept);
        assert_eq!(total_visits(&search), n_sims - 1);
    }

    #[test]
    fn reusing_an_unvisited_move_starts_afresh() {
        let mut search = Search::new(MctsConfig::new().with_n_sims(5));
        let mut state = State::from_fen(START_FEN);
        finish(&mut search, &mut state);
        let (mv, _) = *search
            .result()
            .visits
            .iter()
            .find(|(_, n)| *n == 0)
            .unwrap();

        state.make_move(mv);
        search.reuse_subtree(mv);
        assert!(search.result().visits.is_empty());
        assert_eq!(finish(&mut search, &mut state), 5);
    }

    #[test]
    fn mixing_into_a_reused_roots_priors_steers_its_search() {
        let n_sims = 200;
        let mut search = Search::new(MctsConfig::new().with_n_sims(n_sims));
        let mut state = State::from_fen(START_FEN);
        finish(&mut search, &mut state);
        let mv = search.result().best_move().unwrap();
        state.make_move(mv);
        search.reuse_subtree(mv);
        let before = search.result().visits;

        // All the prior on the last move, which then gets most of the new visits
        search.mix_into_root_priors(|priors| {
            assert_eq!(priors.len(), before.len());
            assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            priors.fill(0.0);
            *priors.last_mut().unwrap() = 1.0;
        });
        finish(&mut search, &mut state);
        let gained: Vec<usize> = search
            .result()
            .visits
            .iter()
            .zip(&before)
            .map(|((_, after), (_, before))| after - before)
            .collect();
        let n_new: usize = gained.iter().sum();
        assert!(2 * gained.last().unwrap() > n_new, "{gained:?}");
    }

    #[test]
    fn sample_move_respects_visits() {
        let result = search(&mut State::from_fen(START_FEN), 50);
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(result.sample_move(0.0, &mut rng), result.best_move());

        for _ in 0..100 {
            let mv = result.sample_move(1.0, &mut rng).unwrap();
            let (_, n_visits) = result.visits.iter().find(|(m, _)| *m == mv).unwrap();
            assert!(*n_visits > 0, "sampled unvisited move {mv}");
        }
    }
}
