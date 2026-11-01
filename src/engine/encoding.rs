//! How positions and moves are encoded for the network: the pieces on each square and the rest of
//! the position's state as input, and each legal move as an index into the policy head's output.
//!
//! Both are seen from the side to move. For black, the board is mirrored top to bottom and the
//! colours swapped, so the network learns each pattern once rather than once per colour. Chess is
//! the same game under that change, so the side to move isn't an input: a position and its
//! mirror image, with the other side to move, encode identically.

use crate::chess::{
    board::CastlingRights,
    moves::{Move, MoveType},
    types::{Color, Direction, Piece, Position},
    State,
};
use burn::{
    tensor::{backend::Backend, Int, TensorData},
    Tensor,
};

/// Piece tokens: 0 for an empty square, 1-6 for the side to move's pawn, rook, knight, bishop,
/// queen and king, and 7-12 for the opponent's
pub(crate) const N_PIECE_TOKENS: usize = 13;
/// Inputs to the CLS token: castling rights, en passant file and the halfmove clock
pub(crate) const N_STATE_FEATURES: usize = 14;
pub(crate) const N_MOVE_PLANES: usize = 73;
const MAX_RAY_MOVE_DIST: usize = 7;
const KNIGHT_MOVE_PLANE_START: usize = 56;
const KNIGHT_PROMO_PLANE_START: i8 = 64;
const BISHOP_PROMO_PLANE_START: i8 = 67;
const ROOK_PROMO_PLANE_START: i8 = 70;

// The square as the side to move sees it: black's are mirrored top to bottom, so both sides' pawns
// move up the board
fn orient(square: Position, turn: Color) -> Position {
    match turn {
        Color::White => square,
        Color::Black => Position(square.0 ^ 56),
    }
}

fn piece_token(piece: Option<Piece>, turn: Color) -> i32 {
    // `Piece` lists white's six pieces, then black's in the same order
    piece.map_or(0, |piece| (piece as i32 + 6 * turn as i32) % 12 + 1)
}

impl Move {
    /// The move's index in the policy head's output, with `turn` to move.
    pub(crate) fn policy_index(&self, turn: Color) -> usize {
        let (from, to, mv_type) = self.unpack();
        let (from, to) = (orient(from, turn), orient(to, turn));
        let base = from.0 as usize * N_MOVE_PLANES;
        let (rank_diff, file_diff) = (
            to.rank() as i8 - from.rank() as i8,
            to.file() as i8 - from.file() as i8,
        );

        match mv_type {
            MoveType::PromotionKnight => {
                return base + (KNIGHT_PROMO_PLANE_START + file_diff + 1) as usize
            }
            MoveType::PromotionBishop => {
                return base + (BISHOP_PROMO_PLANE_START + file_diff + 1) as usize
            }
            MoveType::PromotionRook => {
                return base + (ROOK_PROMO_PLANE_START + file_diff + 1) as usize
            }
            _ => (),
        };

        let plane = match (rank_diff, file_diff) {
            (2, 1) => KNIGHT_MOVE_PLANE_START,
            (1, 2) => KNIGHT_MOVE_PLANE_START + 1,
            (-1, 2) => KNIGHT_MOVE_PLANE_START + 2,
            (-2, 1) => KNIGHT_MOVE_PLANE_START + 3,
            (-2, -1) => KNIGHT_MOVE_PLANE_START + 4,
            (-1, -2) => KNIGHT_MOVE_PLANE_START + 5,
            (1, -2) => KNIGHT_MOVE_PLANE_START + 6,
            (2, -1) => KNIGHT_MOVE_PLANE_START + 7,
            _ => {
                let (dir, dist) = match (rank_diff.signum(), file_diff.signum()) {
                    (1, 0) => (Direction::North, rank_diff),
                    (-1, 0) => (Direction::South, -rank_diff),
                    (0, 1) => (Direction::East, file_diff),
                    (0, -1) => (Direction::West, -file_diff),
                    (1, 1) => (Direction::NorthEast, rank_diff),
                    (1, -1) => (Direction::NorthWest, rank_diff),
                    (-1, 1) => (Direction::SouthEast, -rank_diff),
                    (-1, -1) => (Direction::SouthWest, -rank_diff),
                    _ => unreachable!(),
                };
                dir as usize * MAX_RAY_MOVE_DIST + dist as usize - 1
            }
        };

        base + plane
    }
}

/// A batch of positions with their legal moves, encoded for the network. Each tensor is built in
/// one host buffer and uploaded once.
pub(crate) struct NetworkInput<B: Backend> {
    /// Each square's piece token, [batch_size, 64]
    pub(crate) pieces: Tensor<B, 2, Int>,
    /// [batch_size, N_STATE_FEATURES]
    pub(crate) state_features: Tensor<B, 2>,
    /// Policy indices of each position's legal moves, padded with 0 to the longest row,
    /// [batch_size, max legal moves]
    pub(crate) move_indices: Tensor<B, 2, Int>,
    /// 0 for moves and -1e9 for padding, to add to the gathered logits before the softmax
    pub(crate) padding: Tensor<B, 2>,
}

impl<B: Backend> NetworkInput<B> {
    /// Encodes each position with its legal moves, whose logits the network returns in the order
    /// given.
    pub(crate) fn new(states: &[&State], moves: &[impl AsRef<[Move]>], device: &B::Device) -> Self {
        assert_eq!(states.len(), moves.len());
        let batch_size = states.len();

        let mut pieces = Vec::with_capacity(batch_size * 64);
        let mut state_features = Vec::with_capacity(batch_size * N_STATE_FEATURES);
        for state in states {
            pieces.extend((0..64).map(|square| {
                let square = orient(Position(square), state.turn);
                piece_token(state.board.mailbox[square], state.turn)
            }));
            state_features.extend(encode_state_features(state));
        }

        let (move_indices, padding) = legal_move_indices(states, moves, device);
        Self {
            pieces: Tensor::from_data(TensorData::new(pieces, [batch_size, 64]), device),
            state_features: Tensor::from_data(
                TensorData::new(state_features, [batch_size, N_STATE_FEATURES]),
                device,
            ),
            move_indices,
            padding,
        }
    }
}

// Castling rights, the side to move's then the opponent's, the en passant file one-hot (index 0
// for none, 1-8 for files a-h), and the halfmove clock scaled down
fn encode_state_features(state: &State) -> [f32; N_STATE_FEATURES] {
    let mut features = [0.0; N_STATE_FEATURES];
    let (us, them) = (state.turn, state.turn.flip());

    let rights = [
        CastlingRights::KING_SIDE[us],
        CastlingRights::QUEEN_SIDE[us],
        CastlingRights::KING_SIDE[them],
        CastlingRights::QUEEN_SIDE[them],
    ];
    for (feature, right) in features[..4].iter_mut().zip(rights) {
        *feature = state.castling_rights.has(right) as i8 as f32;
    }

    // Mirroring the board leaves files as they are
    features[4 + state.en_passant.map_or(0, |ep| ep.file() as usize + 1)] = 1.0;
    features[13] = state.halfmove_clock as f32 / 100.0;
    features
}

// Each position's legal-move policy indices and padding, one row per position and padded to the
// longest row, so the network can gather just those logits
fn legal_move_indices<B: Backend>(
    states: &[&State],
    moves: &[impl AsRef<[Move]>],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
    let width = moves
        .iter()
        .map(|moves| moves.as_ref().len())
        .max()
        .unwrap_or(0);
    assert!(
        width > 0,
        "positions without legal moves have nothing to evaluate"
    );

    let mut indices = vec![0i32; moves.len() * width];
    let mut padding = vec![-1e9f32; moves.len() * width];
    let rows = indices
        .chunks_exact_mut(width)
        .zip(padding.chunks_exact_mut(width));

    for (((index_row, padding_row), state), moves) in rows.zip(states).zip(moves) {
        for ((index, pad), mv) in index_row.iter_mut().zip(padding_row).zip(moves.as_ref()) {
            *index = mv.policy_index(state.turn) as i32;
            *pad = 0.0;
        }
    }

    let shape = [moves.len(), width];
    (
        Tensor::from_data(TensorData::new(indices, shape), device),
        Tensor::from_data(TensorData::new(padding, shape), device),
    )
}

/// `fen` mirrored top to bottom with the colours swapped: the same position for the other side,
/// which the network must see exactly as it sees `fen`.
#[cfg(test)]
pub(crate) fn mirror_fen(fen: &str) -> String {
    let swap_case = |text: &str| -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_uppercase() {
                    c.to_ascii_lowercase()
                } else {
                    c.to_ascii_uppercase()
                }
            })
            .collect()
    };

    let fields: Vec<&str> = fen.split_whitespace().collect();
    let board: Vec<String> = fields[0].split('/').rev().map(swap_case).collect();
    let turn = if fields[1] == "w" { "b" } else { "w" };
    // FEN lists white's rights first
    let mut castling: Vec<char> = swap_case(fields[2]).chars().collect();
    castling.sort_by_key(|c| c.is_ascii_lowercase());
    let castling: String = castling.into_iter().collect();
    let en_passant = match fields[3] {
        "-" => "-".to_string(),
        square => {
            let (file, rank) = square.split_at(1);
            format!("{file}{}", 9 - rank.parse::<u8>().unwrap())
        }
    };

    format!(
        "{} {turn} {castling} {en_passant} {} {}",
        board.join("/"),
        fields[4],
        fields[5]
    )
}

/// `mv` in the position `mirror_fen` gives.
#[cfg(test)]
pub(crate) fn mirror_move(mv: Move) -> Move {
    let (from, to, move_type) = mv.unpack();
    Move::new(Position(from.0 ^ 56), Position(to.0 ^ 56), move_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::state::START_FEN;
    use burn::backend::NdArray;

    const N_POTENTIAL_MOVES: usize = 64 * N_MOVE_PLANES;

    const FENS: [&str; 4] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    // Made up to set every kind of feature: black to move, white keeps only king-side castling and
    // black only queen-side, en passant on the e-file, which black's pawn on d4 can capture, and 7
    // halfmoves
    const MADE_UP: &str = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b Kq e3 7 1";

    // The test positions, then their mirror images with black to move
    fn both_colours() -> Vec<String> {
        let fens = FENS.iter().map(|fen| fen.to_string());
        fens.clone()
            .chain(fens.map(|fen| mirror_fen(&fen)))
            .collect()
    }

    // Encodes each position with its legal moves, in the order they're generated
    fn encode(fens: &[impl AsRef<str>]) -> (NetworkInput<NdArray>, Vec<State>, Vec<Vec<Move>>) {
        let states: Vec<State> = fens
            .iter()
            .map(|fen| State::from_fen(fen.as_ref()))
            .collect();
        let moves: Vec<Vec<Move>> = states
            .iter()
            .map(|state| state.generate_moves().to_vec())
            .collect();
        let state_refs: Vec<&State> = states.iter().collect();
        let input = NetworkInput::new(&state_refs, &moves, &Default::default());
        (input, states, moves)
    }

    fn ints(tensor: Tensor<NdArray, 2, Int>) -> Vec<i32> {
        tensor.into_data().convert::<i32>().to_vec().unwrap()
    }

    fn floats(tensor: Tensor<NdArray, 2>) -> Vec<f32> {
        tensor.into_data().to_vec().unwrap()
    }

    #[test]
    fn positions_are_encoded_from_the_side_to_move() {
        let (input, _, _) = encode(&[START_FEN, MADE_UP]);
        assert_eq!(input.pieces.dims(), [2, 64]);
        assert_eq!(input.state_features.dims(), [2, N_STATE_FEATURES]);

        // Tokens 1-6 are the side to move's pawn, rook, knight, bishop, queen and king, 7-12 the
        // opponent's
        let pieces = ints(input.pieces);
        let (start, made_up) = pieces.split_at(64);
        // White sees the board as it is: its rook on a1, its king on e1, the opponent's on e8
        assert_eq!([start[0], start[4], start[60]], [2, 6, 12]);
        assert_eq!(start.iter().filter(|&&token| token == 0).count(), 32);
        // Black sees it mirrored: its rook on a1, king on e1 and pawn on e2, the opponent's pawn
        // moved from e7 to e5, and the opponent's king on e8
        let squares = [0, 4, 12, 52, 36, 60];
        assert_eq!(squares.map(|square| made_up[square]), [2, 6, 1, 0, 7, 12]);

        let features = floats(input.state_features);
        let start_features = [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        assert_eq!(features[..N_STATE_FEATURES], start_features);
        // Black's own queen-side right comes before the opponent's king-side one
        let made_up_features = [0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.07];
        assert_eq!(features[N_STATE_FEATURES..], made_up_features);
    }

    #[test]
    fn mirrored_positions_encode_the_same() {
        assert_eq!(
            mirror_fen(MADE_UP),
            "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w Qk e6 7 1"
        );

        let sorted = |moves: &[Move]| {
            let mut moves = moves.to_vec();
            moves.sort_by_key(|mv| mv.0);
            moves
        };
        for fen in FENS.into_iter().chain([MADE_UP]) {
            let state = State::from_fen(fen);
            let mirrored = State::from_fen(&mirror_fen(fen));
            let moves = state.generate_moves().to_vec();
            let mirrored_moves: Vec<Move> = moves.iter().map(|&mv| mirror_move(mv)).collect();
            // The mirrored moves are exactly the mirrored position's legal moves
            assert_eq!(
                sorted(&mirrored_moves),
                sorted(&mirrored.generate_moves()),
                "{fen}"
            );

            let input = NetworkInput::<NdArray>::new(
                &[&state, &mirrored],
                &[moves, mirrored_moves],
                &Default::default(),
            );
            let pieces = ints(input.pieces);
            assert_eq!(pieces[..64], pieces[64..], "{fen}");
            let features = floats(input.state_features);
            let (features, mirrored_features) = features.split_at(N_STATE_FEATURES);
            assert_eq!(features, mirrored_features, "{fen}");
            let indices = ints(input.move_indices);
            let (indices, mirrored_indices) = indices.split_at(indices.len() / 2);
            assert_eq!(indices, mirrored_indices, "{fen}");
        }
    }

    #[test]
    fn legal_move_indices_are_padded_to_the_longest_row() {
        let fens = both_colours();
        let (input, states, moves) = encode(&fens);

        let width = moves.iter().map(|moves| moves.len()).max().unwrap();
        assert_eq!(input.move_indices.dims(), [fens.len(), width]);
        assert_eq!(input.padding.dims(), [fens.len(), width]);

        let indices = ints(input.move_indices);
        let padding = floats(input.padding);
        let rows = indices.chunks_exact(width).zip(padding.chunks_exact(width));
        for (((index_row, padding_row), state), moves) in rows.zip(&states).zip(&moves) {
            for (i, mv) in moves.iter().enumerate() {
                assert_eq!(index_row[i], mv.policy_index(state.turn) as i32);
                assert_eq!(padding_row[i], 0.0);
            }
            assert!(index_row[moves.len()..].iter().all(|&index| index == 0));
            assert!(padding_row[moves.len()..].iter().all(|&pad| pad == -1e9));
        }
    }

    #[test]
    fn test_move_indices_are_unique_and_in_bounds() {
        for fen in both_colours() {
            let state = State::from_fen(&fen);
            let moves = state.generate_moves();
            let mut indices = Vec::new();
            for mv in moves {
                let idx = mv.policy_index(state.turn);
                assert!(
                    idx < N_POTENTIAL_MOVES,
                    "Move index {} out of bounds for move {:?}",
                    idx,
                    mv
                );
                assert!(
                    !indices.contains(&idx),
                    "Duplicate move index {} for move {:?} in position:\n{}",
                    idx,
                    mv,
                    fen
                );
                indices.push(idx);
            }
        }
    }
}
