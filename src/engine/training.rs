//! Training the network on self-play samples: the policy towards each search's visit
//! distribution, and the value towards each game's result.

use crate::{
    chess::{moves::Move, State},
    engine::{
        encoding::NetworkInput,
        model::{TransformerModel, TransformerModelConfig},
        samples::Sample,
    },
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder, RecorderError},
    tensor::{
        activation::log_softmax,
        backend::{AutodiffBackend, Backend},
        Tensor, TensorData,
    },
};
use std::{fs, io, path::Path};

/// Gradients are scaled down to at most this norm, so one unusual batch can't wreck the weights
const MAX_GRADIENT_NORM: f32 = 1.0;

/// Average losses over one or more batches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Losses {
    /// Cross-entropy of the priors against the visit distributions
    pub(crate) policy: f32,
    /// The visit distributions' own entropy, the least the policy loss can be
    pub(crate) target_entropy: f32,
    /// Cross-entropy of the win, draw and loss probabilities against the results
    pub(crate) value: f32,
}

impl Losses {
    /// Mean of `losses`, each weighted equally.
    pub(crate) fn mean(losses: &[Losses]) -> Losses {
        let n = losses.len().max(1) as f32;
        let sum = |field: fn(&Losses) -> f32| losses.iter().map(field).sum::<f32>() / n;
        Losses {
            policy: sum(|losses| losses.policy),
            target_entropy: sum(|losses| losses.target_entropy),
            value: sum(|losses| losses.value),
        }
    }
}

/// A network with its optimizer state, saved together as a checkpoint so training can continue
/// exactly where it stopped.
pub(crate) struct Trainer<B: AutodiffBackend> {
    model: TransformerModel<B>,
    optimizer: OptimizerAdaptor<AdamW, TransformerModel<B>, B>,
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub(crate) fn new(model: TransformerModel<B>, weight_decay: f32, device: &B::Device) -> Self {
        Self {
            model,
            optimizer: optimizer_config(weight_decay).init(),
            device: device.clone(),
        }
    }

    /// Takes one optimizer step on `batch`, returning its losses from before the step.
    pub(crate) fn step(&mut self, batch: &[&Sample], learning_rate: f64) -> Losses {
        let (input, policy_target, value_target, target_entropy) =
            encode_batch::<B>(batch, &self.device);
        let (logits, value) = self.model.forward(input);

        // The padding's logits are -1e9 and its targets 0, so it adds nothing to either
        let policy_loss = -(log_softmax(logits, 1) * policy_target).sum_dim(1).mean();
        let value_loss = -(log_softmax(value, 1) * value_target).sum_dim(1).mean();

        let loss = policy_loss.clone() + value_loss.clone();
        let gradients = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self
            .optimizer
            .step(learning_rate, self.model.clone(), gradients);

        let losses = Tensor::cat(vec![policy_loss, value_loss], 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        Losses {
            policy: losses[0],
            target_entropy,
            value: losses[1],
        }
    }

    /// Writes the network and the optimizer state into `dir`, which must exist.
    pub(crate) fn save(&self, dir: &Path) -> Result<(), RecorderError> {
        let recorder = recorder();
        self.model
            .clone()
            .save_file(dir.join(MODEL_FILE), &recorder)?;
        recorder.record(self.optimizer.to_record(), dir.join(OPTIMIZER_FILE))
    }

    /// Continues training from a checkpoint written by `save`.
    pub(crate) fn load(
        config: &TransformerModelConfig,
        weight_decay: f32,
        dir: &Path,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let model = load_model(config, dir, device)?;
        let optimizer = optimizer_config(weight_decay)
            .init()
            .load_record(recorder().load(dir.join(OPTIMIZER_FILE), device)?);
        Ok(Self {
            model,
            optimizer,
            device: device.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn model(&self) -> &TransformerModel<B> {
        &self.model
    }
}

const MODEL_FILE: &str = "model";
const OPTIMIZER_FILE: &str = "optimizer";

/// Loads the network from a checkpoint written by `Trainer::save`, on any backend and at any
/// precision: checkpoints store full-precision weights.
pub(crate) fn load_model<B: Backend>(
    config: &TransformerModelConfig,
    dir: &Path,
    device: &B::Device,
) -> Result<TransformerModel<B>, RecorderError> {
    config
        .init(device)
        .load_file(dir.join(MODEL_FILE), &recorder(), device)
}

/// Deletes a checkpoint's optimizer state, leaving a network that can play but not train further.
pub(crate) fn remove_optimizer_state(dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.file_stem().is_some_and(|stem| stem == OPTIMIZER_FILE) {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::new()
}

fn optimizer_config(weight_decay: f32) -> AdamWConfig {
    AdamWConfig::new()
        .with_weight_decay(weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(MAX_GRADIENT_NORM)))
}

// The network input for a batch, the visit distribution over each position's moves (padded with
// zeros like the moves), each game's result for the side to move as a one-hot win, draw or loss,
// and the distributions' mean entropy
fn encode_batch<B: Backend>(
    batch: &[&Sample],
    device: &B::Device,
) -> (NetworkInput<B>, Tensor<B, 2>, Tensor<B, 2>, f32) {
    let states: Vec<State> = batch
        .iter()
        .map(|sample| State::from_fen(&sample.fen))
        .collect();
    let states: Vec<&State> = states.iter().collect();
    let moves: Vec<Vec<Move>> = batch
        .iter()
        .map(|sample| sample.visits.iter().map(|&(mv, _)| mv).collect())
        .collect();
    let input = NetworkInput::new(&states, &moves, device);

    let width = moves.iter().map(Vec::len).max().unwrap_or(0);
    let mut policy = vec![0.0f32; batch.len() * width];
    let mut entropy = 0.0;
    for (row, sample) in policy.chunks_exact_mut(width).zip(batch) {
        let total: u32 = sample.visits.iter().map(|&(_, n)| n as u32).sum();
        for (target, &(_, n_visits)) in row.iter_mut().zip(&sample.visits) {
            *target = n_visits as f32 / total as f32;
            if *target > 0.0 {
                entropy -= *target * target.ln();
            }
        }
    }
    let values: Vec<f32> = batch
        .iter()
        .flat_map(|sample| [1, 0, -1].map(|result| (sample.result == result) as u8 as f32))
        .collect();

    (
        input,
        Tensor::from_data(TensorData::new(policy, [batch.len(), width]), device),
        Tensor::from_data(TensorData::new(values, [batch.len(), 3]), device),
        entropy / batch.len() as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chess::state::START_FEN, engine::model::tiny_model_config};
    use burn::{
        backend::{Autodiff, NdArray},
        module::AutodiffModule,
    };

    type B = Autodiff<NdArray>;

    // Positions with made-up visit counts, some moves unvisited, and results
    fn samples() -> Vec<Sample> {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];
        fens.iter()
            .zip([1, -1, 0, 1])
            .map(|(fen, result)| Sample {
                fen: fen.to_string(),
                visits: State::from_fen(fen)
                    .generate_moves()
                    .iter()
                    .enumerate()
                    .map(|(i, &mv)| (mv, (i % 3 * i) as u16))
                    .collect(),
                result,
            })
            .collect()
    }

    fn floats(tensor: Tensor<NdArray, 2>) -> Vec<f32> {
        tensor.into_data().to_vec().unwrap()
    }

    #[test]
    fn targets_follow_the_visits_and_results() {
        let samples = samples();
        let batch: Vec<&Sample> = samples.iter().collect();
        let (input, policy, value, _) = encode_batch::<NdArray>(&batch, &Default::default());

        let width = input.move_indices.dims()[1];
        let policy = floats(policy);
        for (row, sample) in policy.chunks_exact(width).zip(&samples) {
            let total: f32 = sample.visits.iter().map(|&(_, n)| n as f32).sum();
            for (i, &(_, n_visits)) in sample.visits.iter().enumerate() {
                assert_eq!(row[i], n_visits as f32 / total);
            }
            assert!(row[sample.visits.len()..].iter().all(|&p| p == 0.0));
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        // Results 1, -1, 0 and 1, as win, draw and loss
        assert_eq!(
            floats(value),
            [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.]
        );
    }

    #[test]
    fn training_fits_a_fixed_batch() {
        let device = Default::default();
        let mut trainer = Trainer::<B>::new(tiny_model_config().init(&device), 0.0, &device);
        let samples = samples();
        let batch: Vec<&Sample> = samples.iter().collect();

        let first = trainer.step(&batch, 1e-3);
        let last = (0..100).map(|_| trainer.step(&batch, 1e-3)).last().unwrap();

        // The policy loss can't fall below the targets' entropy, so compare what's above it
        let excess = |losses: Losses| losses.policy - losses.target_entropy;
        assert!(excess(last) < 0.2 * excess(first), "{first:?} -> {last:?}");
        assert!(last.value < 0.2 * first.value, "{first:?} -> {last:?}");
    }

    #[test]
    fn a_saved_trainer_continues_where_it_stopped() {
        let device = Default::default();
        let config = tiny_model_config();
        let samples = samples();
        let (first_half, second_half) = samples.split_at(2);
        let first_half: Vec<&Sample> = first_half.iter().collect();
        let second_half: Vec<&Sample> = second_half.iter().collect();

        let mut trainer = Trainer::<B>::new(config.init(&device), 1e-2, &device);
        for _ in 0..3 {
            trainer.step(&first_half, 1e-3);
        }
        let dir = std::env::temp_dir().join(format!("cinder-{}-trainer", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        trainer.save(&dir).unwrap();
        let mut loaded = Trainer::<B>::load(&config, 1e-2, &dir, &device).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        // Adam's next step depends on its moment estimates, so this only matches if the
        // optimizer state came back too
        let continued = trainer.step(&second_half, 1e-3);
        let resumed = loaded.step(&second_half, 1e-3);
        assert_eq!(continued, resumed);
        let after = |trainer: &Trainer<B>| {
            let state = State::new();
            let moves = state.generate_moves();
            trainer
                .model()
                .valid()
                .evaluate(&[&state], &[&moves[..]], &device)
                .remove(0)
        };
        assert_eq!(after(&trainer).priors, after(&loaded).priors);
        assert_eq!(after(&trainer).value, after(&loaded).value);
    }
}
