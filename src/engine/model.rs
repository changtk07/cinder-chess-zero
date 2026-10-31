use crate::{
    chess::{moves::Move, State},
    engine::encoding::{NetworkInput, N_MOVE_PLANES, N_PIECE_TOKENS, N_STATE_FEATURES},
};
use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm,
        RmsNormConfig,
    },
    tensor::{
        activation::{gelu, softmax},
        backend::Backend,
        Int, TensorData,
    },
    Tensor,
};

#[derive(Module, Debug)]
struct ChessEmbedding<B: Backend> {
    e_piece: Embedding<B>,
    e_pos: Embedding<B>,
    e_state: Linear<B>,
}

impl<B: Backend> ChessEmbedding<B> {
    // [batch_size, 64] piece tokens and [batch_size, N_STATE_FEATURES] -> [batch_size, 64, d_model],
    // one token per square
    fn forward(&self, pieces: Tensor<B, 2, Int>, state_features: Tensor<B, 2>) -> Tensor<B, 3> {
        let squares = TensorData::new((0i32..64).collect(), [1, 64]);
        let squares = Tensor::from_data(squares, &pieces.device());

        let encoded_pieces = self.e_piece.forward(pieces);
        let encoded_squares = self.e_pos.forward(squares);
        // Castling rights, en passant file and the halfmove clock go into every square's token
        // rather than a token of their own: attention over 64 tokens runs about a third faster
        // than over 65
        let encoded_state = self.e_state.forward(state_features).unsqueeze_dim(1);

        encoded_pieces + encoded_squares + encoded_state
    }
}

#[derive(Config, Debug)]
struct ChessEmbeddingConfig {
    #[config(default = 384)]
    d_model: usize,
}

impl ChessEmbeddingConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ChessEmbedding<B> {
        ChessEmbedding {
            e_piece: EmbeddingConfig::new(N_PIECE_TOKENS, self.d_model).init(device),
            e_pos: EmbeddingConfig::new(64, self.d_model).init(device),
            e_state: LinearConfig::new(N_STATE_FEATURES, self.d_model).init(device),
        }
    }
}

/// Multi-head self-attention over each position's tokens. It takes the batch's tokens stacked in
/// one 2D matrix, like the rest of the network, and projects queries, keys and values with a single
/// matrix multiply.
#[derive(Module, Debug)]
struct SelfAttention<B: Backend> {
    qkv: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
}

impl<B: Backend> SelfAttention<B> {
    // [batch_size * seq_len, d_model] -> [batch_size * seq_len, d_model]
    fn forward(&self, input: Tensor<B, 2>, batch_size: usize) -> Tensor<B, 2> {
        let [n_tokens, d_model] = input.dims();
        let seq_len = n_tokens / batch_size;
        let d_head = d_model / self.n_heads;

        // [batch_size, 3 * n_heads, seq_len, d_head]: every head's queries, then keys, then values
        let qkv = self
            .qkv
            .forward(input)
            .reshape([batch_size, seq_len, 3 * self.n_heads, d_head])
            .swap_dims(1, 2);
        let query = qkv.clone().narrow(1, 0, self.n_heads);
        let key = qkv.clone().narrow(1, self.n_heads, self.n_heads);
        let value = qkv.narrow(1, 2 * self.n_heads, self.n_heads);

        let scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((d_head as f32).sqrt());
        let weights = self.dropout.forward(softmax(scores, 3));
        let context = weights
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([n_tokens, d_model]);
        self.output.forward(context)
    }
}

#[derive(Module, Debug)]
struct AttentionBlock<B: Backend> {
    attention: SelfAttention<B>,
    norm1: RmsNorm<B>,
    norm2: RmsNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> AttentionBlock<B> {
    // [batch_size * 64, d_model]
    fn forward(&self, input: Tensor<B, 2>, batch_size: usize) -> Tensor<B, 2> {
        let x = self.norm1.forward(input.clone());
        let x = self.attention.forward(x, batch_size);
        let middle = input + x;
        let x = self.norm2.forward(middle.clone());
        let x = self.ff1.forward(x);
        let x = gelu(x);
        let x = self.ff2.forward(x);
        middle + x
    }
}

#[derive(Config, Debug)]
struct AttentionBlockConfig {
    #[config(default = 6)]
    n_heads: usize,
    #[config(default = 384)]
    d_model: usize,
    #[config(default = 768)]
    d_ff: usize,
    #[config(default = 0.0)]
    attention_dropout: f64,
}

impl AttentionBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> AttentionBlock<B> {
        AttentionBlock {
            attention: SelfAttention {
                qkv: LinearConfig::new(self.d_model, 3 * self.d_model).init(device),
                output: LinearConfig::new(self.d_model, self.d_model).init(device),
                dropout: DropoutConfig::new(self.attention_dropout).init(),
                n_heads: self.n_heads,
            },
            norm1: RmsNormConfig::new(self.d_model).init(device),
            norm2: RmsNormConfig::new(self.d_model).init(device),
            ff1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2: LinearConfig::new(self.d_ff, self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct PolicyHead<B: Backend> {
    ff: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    // [batch_size * 64, d_model] -> logits of the legal moves, [batch_size, max legal moves]
    fn forward(
        &self,
        input: Tensor<B, 2>,
        batch_size: usize,
        move_indices: Tensor<B, 2, Int>,
        padding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = self.ff.forward(input);
        let x = x.reshape([batch_size, 64 * N_MOVE_PLANES]);
        x.gather(1, move_indices) + padding
    }
}

#[derive(Config, Debug)]
struct PolicyHeadConfig {
    #[config(default = 384)]
    d_model: usize,
}

impl PolicyHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            ff: LinearConfig::new(self.d_model, N_MOVE_PLANES).init(device),
        }
    }
}

/// Reduces each square to a few features and reads the value from all of them together, as
/// Leela's transformer networks do, so the head knows which square each feature came from. The
/// value is a win, draw and loss probability for the side to move, also as Leela predicts, so a
/// surely drawn position and an unclear one look different.
#[derive(Module, Debug)]
struct ValueHead<B: Backend> {
    square_ff: Linear<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    // [batch_size * 64, d_model] -> win, draw and loss logits, [batch_size, 3]
    fn forward(&self, input: Tensor<B, 2>, batch_size: usize) -> Tensor<B, 2> {
        let x = gelu(self.square_ff.forward(input));
        let d_square = x.dims()[1];
        let x = x.reshape([batch_size, 64 * d_square]);
        let x = gelu(self.ff1.forward(x));
        self.ff2.forward(x)
    }
}

#[derive(Config, Debug)]
struct ValueHeadConfig {
    #[config(default = 384)]
    d_model: usize,
    /// Features kept per square
    #[config(default = 32)]
    d_square: usize,
    #[config(default = 128)]
    d_hidden: usize,
}

impl ValueHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            square_ff: LinearConfig::new(self.d_model, self.d_square).init(device),
            ff1: LinearConfig::new(64 * self.d_square, self.d_hidden).init(device),
            ff2: LinearConfig::new(self.d_hidden, 3).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TransformerModel<B: Backend> {
    embedding: ChessEmbedding<B>,
    attention_blocks: Vec<AttentionBlock<B>>,
    // The blocks normalize their inputs, so without this the residual stream would reach the
    // heads unnormalized
    final_norm: RmsNorm<B>,
    policy_head: PolicyHead<B>,
    value_head: ValueHead<B>,
}

/// Network output for one position: a prior per legal move, in the order the moves were given,
/// and the value for the side to move.
pub(crate) struct Evaluation {
    pub(crate) priors: Vec<f32>,
    /// The expected score in [-1, 1]: the win probability minus the loss probability
    pub(crate) value: f32,
}

impl<B: Backend> TransformerModel<B> {
    /// Returns legal-move logits, [batch_size, max legal moves], and the value's win, draw and
    /// loss logits for the side to move, [batch_size, 3]. Logits past a position's own move count
    /// are padding.
    pub(crate) fn forward(&self, input: NetworkInput<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let attn_input = self.embedding.forward(input.pieces, input.state_features);
        let [batch_size, seq_len, d_model] = attn_input.dims();

        // The network works on all positions' tokens stacked in one 2D matrix: burn multiplies 2D
        // matrices faster than the 3D form a linear layer would otherwise use
        let attn_input = attn_input.reshape([batch_size * seq_len, d_model]);
        let attn_output = self
            .attention_blocks
            .iter()
            .fold(attn_input, |x, block| block.forward(x, batch_size));
        let squares = self.final_norm.forward(attn_output);

        let policy = self.policy_head.forward(
            squares.clone(),
            batch_size,
            input.move_indices,
            input.padding,
        );
        let value = self.value_head.forward(squares, batch_size);

        (policy, value)
    }

    /// Evaluates a batch of positions, reading priors and values back from the device at once.
    pub(crate) fn evaluate(
        &self,
        states: &[&State],
        moves: &[impl AsRef<[Move]>],
        device: &B::Device,
    ) -> Vec<Evaluation> {
        let (logits, value) = self.forward(NetworkInput::new(states, moves, device));
        let width = logits.dims()[1];
        let output = Tensor::cat(vec![softmax(logits, 1), softmax(value, 1)], 1);
        let data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        data.chunks_exact(width + 3)
            .zip(moves)
            .map(|(row, moves)| Evaluation {
                priors: row[..moves.as_ref().len()].to_vec(),
                value: row[width] - row[width + 2],
            })
            .collect()
    }

    /// Like `evaluate`, but pads the batch to a power of two by repeating its last position. burn
    /// compiles kernels for every new batch size, which takes far longer than evaluating, so a
    /// batch whose size keeps changing is better off reusing a few sizes.
    pub(crate) fn evaluate_padded(
        &self,
        states: &[&State],
        moves: &[&[Move]],
        device: &B::Device,
    ) -> Vec<Evaluation> {
        let n = states.len();
        let size = n.next_power_of_two();
        let mut states = states.to_vec();
        let mut moves = moves.to_vec();
        states.resize(size, states[n - 1]);
        moves.resize(size, moves[n - 1]);

        let mut evaluations = self.evaluate(&states, &moves, device);
        evaluations.truncate(n);
        evaluations
    }
}

#[derive(Config, Debug)]
pub(crate) struct TransformerModelConfig {
    #[config(default = 8)]
    pub(crate) n_blocks: usize,
    /// The model width is the number of heads times the head dimension
    #[config(default = 4)]
    pub(crate) n_heads: usize,
    #[config(default = 2.0)]
    d_ff_scale: f64,
    #[config(default = 64)]
    pub(crate) head_dimension: usize,
    /// Share of attention weights dropped while training; inference never drops any. Off, rather
    /// than burn's default of 0.1: self-play keeps producing new positions, so the network is
    /// unlikely to overfit them, and AlphaZero regularized with weight decay alone.
    #[config(default = 0.0)]
    attention_dropout: f64,
}

impl TransformerModelConfig {
    pub(crate) fn d_model(&self) -> usize {
        self.n_heads * self.head_dimension
    }

    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        let d_model = self.d_model();
        let d_ff = (self.d_ff_scale * d_model as f64) as usize;

        TransformerModel {
            embedding: ChessEmbeddingConfig::new()
                .with_d_model(d_model)
                .init(device),
            attention_blocks: (0..self.n_blocks)
                .map(|_| {
                    AttentionBlockConfig::new()
                        .with_n_heads(self.n_heads)
                        .with_d_model(d_model)
                        .with_d_ff(d_ff)
                        .with_attention_dropout(self.attention_dropout)
                        .init(device)
                })
                .collect(),
            final_norm: RmsNormConfig::new(d_model).init(device),
            policy_head: PolicyHeadConfig::new().with_d_model(d_model).init(device),
            value_head: ValueHeadConfig::new().with_d_model(d_model).init(device),
        }
    }
}

/// A one-block network that keeps CPU tests fast. Untrained, its outputs serve as well as any to
/// test the code around it.
#[cfg(test)]
pub(crate) fn tiny_model() -> TransformerModel<burn::backend::NdArray> {
    tiny_model_config().init(&Default::default())
}

#[cfg(test)]
pub(crate) fn tiny_model_config() -> TransformerModelConfig {
    TransformerModelConfig::new()
        .with_n_blocks(1)
        .with_n_heads(2)
        .with_head_dimension(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chess::state::START_FEN,
        engine::encoding::{mirror_fen, mirror_move},
    };
    use burn::{
        backend::{Cuda, NdArray},
        module::Param,
        nn::attention::{MhaInput, MultiHeadAttentionConfig},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::{bf16, f16, Distribution},
    };

    const FENS: [&str; 3] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    fn evaluate_fens<B: Backend>(
        model: &TransformerModel<B>,
        fens: &[&str],
        device: &B::Device,
    ) -> Vec<Evaluation> {
        let states: Vec<State> = fens.iter().map(|fen| State::from_fen(fen)).collect();
        let states: Vec<&State> = states.iter().collect();
        let moves: Vec<_> = states.iter().map(|state| state.generate_moves()).collect();
        model.evaluate(&states, &moves, device)
    }

    // All priors in order, then all values
    fn flatten(evaluations: &[Evaluation]) -> (Vec<f32>, Vec<f32>) {
        let priors = evaluations.iter().flat_map(|e| e.priors.clone()).collect();
        let values = evaluations.iter().map(|e| e.value).collect();
        (priors, values)
    }

    // The positions have 20, 48 and 6 legal moves, so the shorter rows are padded
    #[test]
    fn values_are_the_win_probability_minus_the_loss_probability() {
        let model = tiny_model();
        let device = Default::default();
        let states: Vec<State> = FENS.iter().map(|fen| State::from_fen(fen)).collect();
        let states: Vec<&State> = states.iter().collect();
        let moves: Vec<_> = states.iter().map(|state| state.generate_moves()).collect();

        let (_, value) = model.forward(NetworkInput::new(&states, &moves, &device));
        let probabilities = softmax(value, 1).into_data().to_vec::<f32>().unwrap();
        let evaluations = model.evaluate(&states, &moves, &device);

        for (wdl, evaluation) in probabilities.chunks_exact(3).zip(&evaluations) {
            assert!((evaluation.value - (wdl[0] - wdl[2])).abs() < 1e-6);
        }
    }

    #[test]
    fn batched_evaluation_matches_single_positions() {
        let model = tiny_model();
        let device = Default::default();
        let batched = evaluate_fens(&model, &FENS, &device);

        for (fen, evaluation) in FENS.iter().zip(&batched) {
            let single = evaluate_fens(&model, &[fen], &device).remove(0);
            let n_moves = State::from_fen(fen).generate_moves().len();

            assert_eq!(evaluation.priors.len(), n_moves);
            assert!((evaluation.priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(max_diff(&evaluation.priors, &single.priors) < 1e-5);
            assert!((evaluation.value - single.value).abs() < 1e-5);
        }
    }

    #[test]
    fn padding_leaves_evaluations_unchanged() {
        let model = tiny_model();
        let device = Default::default();
        let states: Vec<State> = FENS.iter().map(|fen| State::from_fen(fen)).collect();
        let states: Vec<&State> = states.iter().collect();
        let moves: Vec<_> = states.iter().map(|state| state.generate_moves()).collect();
        let moves: Vec<&[Move]> = moves.iter().map(|moves| &moves[..]).collect();

        // Three positions are padded to four
        let padded = model.evaluate_padded(&states, &moves, &device);
        let plain = model.evaluate(&states, &moves, &device);
        assert_eq!(padded.len(), 3);
        for (padded, plain) in padded.iter().zip(&plain) {
            assert!(max_diff(&padded.priors, &plain.priors) < 1e-6);
            assert!((padded.value - plain.value).abs() < 1e-6);
        }
    }

    #[test]
    fn priors_follow_their_moves() {
        let model = tiny_model();
        let device = Default::default();
        let state = State::from_fen(FENS[1]);
        let moves = state.generate_moves();
        let reversed: Vec<Move> = moves.iter().rev().copied().collect();

        let forward = model.evaluate(&[&state], &[&moves[..]], &device).remove(0);
        let mut backward = model
            .evaluate(&[&state], &[&reversed[..]], &device)
            .remove(0);
        backward.priors.reverse();

        assert!(max_diff(&forward.priors, &backward.priors) < 1e-6);
    }

    // burn's multi-head attention projects queries, keys and values separately and works on 3D
    // inputs, so with the same weights it checks the fused projection and the head layout
    #[test]
    fn self_attention_matches_burns() {
        let device = Default::default();
        let (batch_size, seq_len, d_model, n_heads) = (3, 65, 32, 4);
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(0.0)
            .init::<NdArray>(&device);

        let projections = [&mha.query, &mha.key, &mha.value];
        let weights = projections.map(|linear| linear.weight.val());
        let biases = projections.map(|linear| linear.bias.as_ref().unwrap().val());
        let attention = SelfAttention {
            qkv: Linear {
                weight: Param::from_tensor(Tensor::cat(weights.to_vec(), 1)),
                bias: Some(Param::from_tensor(Tensor::cat(biases.to_vec(), 0))),
            },
            output: mha.output.clone(),
            dropout: DropoutConfig::new(0.0).init(),
            n_heads,
        };

        let input = Tensor::random(
            [batch_size, seq_len, d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let expected = mha.forward(MhaInput::self_attn(input.clone())).context;
        let actual = attention.forward(input.reshape([batch_size * seq_len, d_model]), batch_size);

        let expected = expected.into_data().to_vec::<f32>().unwrap();
        let actual = actual.into_data().to_vec::<f32>().unwrap();
        assert!(max_diff(&expected, &actual) < 1e-5);
    }

    #[test]
    fn mirrored_positions_evaluate_the_same() {
        let model = tiny_model();
        let device = Default::default();

        for fen in FENS {
            let state = State::from_fen(fen);
            let mirrored = State::from_fen(&mirror_fen(fen));
            let moves = state.generate_moves();
            let mirrored_moves: Vec<Move> = moves.iter().map(|&mv| mirror_move(mv)).collect();

            let evaluations = model.evaluate(
                &[&state, &mirrored],
                &[&moves[..], &mirrored_moves[..]],
                &device,
            );
            let (white, black) = (&evaluations[0], &evaluations[1]);
            assert!(max_diff(&white.priors, &black.priors) < 1e-6);
            assert!((white.value - black.value).abs() < 1e-6);
        }
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // Runs the same weights on the CPU and on backend B
    fn assert_matches_cpu<B: Backend>(tolerance: f32) {
        let cpu_device = Default::default();
        let cpu_model = TransformerModelConfig::new().init::<NdArray>(&cpu_device);
        let (cpu_policy, cpu_value) = flatten(&evaluate_fens(&cpu_model, &FENS, &cpu_device));

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(cpu_model.into_record(), ()).unwrap();
        let device = Default::default();
        let model = TransformerModelConfig::new()
            .init::<B>(&device)
            .load_record(recorder.load(bytes, &device).unwrap());
        let (policy, value) = flatten(&evaluate_fens(&model, &FENS, &device));

        // `f32::max` skips NaN, so check for it before comparing
        assert!(policy.iter().chain(&value).all(|x| x.is_finite()));
        assert!(max_diff(&policy, &cpu_policy) < tolerance);
        assert!(max_diff(&value, &cpu_value) < tolerance);
    }

    // Tolerances are about 10x the differences measured on an RTX 4090
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_matches_cpu() {
        assert_matches_cpu::<Cuda>(1e-3);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_f16_matches_cpu() {
        assert_matches_cpu::<Cuda<f16>>(5e-3);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_bf16_matches_cpu() {
        assert_matches_cpu::<Cuda<bf16>>(3e-2);
    }
}
