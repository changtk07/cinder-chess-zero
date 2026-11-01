use crate::chess::{
    bitmask::Bitmask,
    board::{Board, CastlingRights, MoveGenMasks},
    moves::{Move, MoveType},
    prng::{RAND_CASTLING, RAND_COLOR, RAND_EN_PASSANT, RAND_PLACEMENT},
    types::{Color, Direction, Piece, Position},
};
use arrayvec::ArrayVec;

pub(crate) const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Result of a finished game, from the perspective of the side to move.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum Outcome {
    Checkmate,
    Draw,
}

#[derive(Clone)]
struct UndoRecord {
    mv: Move,
    captured: Option<Piece>,
    en_passant: Option<Position>,
    castling_rights: CastlingRights,
    halfmove_clock: usize,
    hash: u64,
}

pub struct State {
    pub(crate) board: Board,
    pub(crate) turn: Color,
    pub(crate) en_passant: Option<Position>,
    pub(crate) castling_rights: CastlingRights,
    pub(crate) fullmove_number: usize,
    pub(crate) halfmove_clock: usize,
    hash: u64,
    history: Vec<UndoRecord>,
}

impl State {
    pub fn new() -> Self {
        Self::from_fen(START_FEN)
    }

    pub fn from_fen(fen: &str) -> Self {
        let parts: ArrayVec<&str, 6> = fen.split_whitespace().collect();
        let mut state = Self {
            board: Board::from_fen(parts[0]),
            turn: Color::from_fen(parts[1]),
            castling_rights: CastlingRights::from_fen(parts[2]),
            en_passant: Position::from_fen(parts[3]),
            halfmove_clock: parts[4].parse().unwrap(),
            fullmove_number: parts[5].parse().unwrap(),
            hash: 0,
            history: Vec::with_capacity(64),
        };
        state.generate_hash();
        state
    }

    pub fn to_fen(&self) -> String {
        let turn = match self.turn {
            Color::White => 'w',
            Color::Black => 'b',
        };
        let en_passant = self
            .en_passant
            .map_or_else(|| "-".to_string(), |position| position.to_string());

        format!(
            "{} {} {} {} {} {}",
            self.board.to_fen(),
            turn,
            self.castling_rights,
            en_passant,
            self.halfmove_clock,
            self.fullmove_number
        )
    }

    // ------------------------------------------------------------------------
    // Move Making
    // ------------------------------------------------------------------------

    pub fn make_move(&mut self, mv: Move) {
        let (from, to, move_type) = mv.unpack();
        let (moved, captured) = self.board.move_piece(from, to);

        self.history.push(UndoRecord {
            mv,
            captured,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
        });

        if let Some(piece) = moved {
            self.hash ^= RAND_PLACEMENT[piece][from];
            self.hash ^= RAND_PLACEMENT[piece][to];
        }
        if let Some(piece) = captured {
            self.hash ^= RAND_PLACEMENT[piece][to];
        }
        if let Some(pos) = self.en_passant {
            self.hash ^= RAND_EN_PASSANT[pos.file() as usize];
        }
        self.hash ^= RAND_CASTLING[self.castling_rights];

        match move_type {
            MoveType::DoublePush => self.make_move_double_push(from, to),
            MoveType::EnPassant => self.make_move_en_passant(from, to),
            MoveType::PromotionRook => self.make_move_promotion(to, Piece::rook(self.turn)),
            MoveType::PromotionKnight => self.make_move_promotion(to, Piece::knight(self.turn)),
            MoveType::PromotionBishop => self.make_move_promotion(to, Piece::bishop(self.turn)),
            MoveType::PromotionQueen => self.make_move_promotion(to, Piece::queen(self.turn)),
            MoveType::KingSideCastling => {
                self.make_move_castling(Position::KS_CASTLE_ROOK[self.turn])
            }
            MoveType::QueenSideCastling => {
                self.make_move_castling(Position::QS_CASTLE_ROOK[self.turn])
            }
            MoveType::Standard => self.make_move_standard(moved, captured),
        }

        self.castling_rights = self.castling_rights.update(from, to);
        self.fullmove_number += self.turn as usize;
        self.turn = self.turn.flip();
        self.hash ^= RAND_COLOR[Color::White];
        self.hash ^= RAND_COLOR[Color::Black];
        self.hash ^= RAND_CASTLING[self.castling_rights];
    }

    pub fn unmake_move(&mut self) {
        let Some(history) = self.history.pop() else {
            return;
        };

        self.hash = history.hash;
        self.en_passant = history.en_passant;
        self.castling_rights = history.castling_rights;
        self.halfmove_clock = history.halfmove_clock;
        self.turn = self.turn.flip();
        self.fullmove_number -= self.turn as usize;

        let from = history.mv.from();
        let to = history.mv.to();
        let move_type = history.mv.move_type();

        self.board.move_piece(to, from);

        if let Some(captured) = history.captured {
            self.board.set_piece(to, captured);
        }

        match move_type {
            MoveType::EnPassant => {
                let captured = Position::en_passant_captured(from, to);
                self.board
                    .set_piece(captured, Piece::pawn(self.turn.flip()));
            }
            MoveType::PromotionRook
            | MoveType::PromotionKnight
            | MoveType::PromotionBishop
            | MoveType::PromotionQueen => {
                self.board.set_piece(from, Piece::pawn(self.turn));
            }
            MoveType::KingSideCastling | MoveType::QueenSideCastling => {
                let (rook_from, rook_to) = if move_type == MoveType::KingSideCastling {
                    Position::KS_CASTLE_ROOK[self.turn]
                } else {
                    Position::QS_CASTLE_ROOK[self.turn]
                };
                self.board.move_piece(rook_to, rook_from);
            }
            _ => (),
        }
    }

    fn make_move_double_push(&mut self, from: Position, to: Position) {
        let en_passant = Position::middle_of(from, to);
        self.en_passant = Some(en_passant);
        self.halfmove_clock = 0;
        self.hash ^= RAND_EN_PASSANT[en_passant.file() as usize];
    }

    fn make_move_en_passant(&mut self, from: Position, to: Position) {
        let captured = Position::en_passant_captured(from, to);
        self.board.unset_piece(captured);
        self.en_passant = None;
        self.halfmove_clock = 0;
        self.hash ^= RAND_PLACEMENT[Piece::pawn(self.turn.flip())][captured];
    }

    // The pawn has already moved to `to`, so it's replaced there
    fn make_move_promotion(&mut self, to: Position, promoted: Piece) {
        self.board.set_piece(to, promoted);
        self.en_passant = None;
        self.halfmove_clock = 0;
        self.hash ^= RAND_PLACEMENT[Piece::pawn(self.turn)][to];
        self.hash ^= RAND_PLACEMENT[promoted][to];
    }

    // The king has already moved, so this moves the rook
    fn make_move_castling(&mut self, (rook_from, rook_to): (Position, Position)) {
        self.board.move_piece(rook_from, rook_to);
        self.en_passant = None;
        self.halfmove_clock += 1;
        self.hash ^= RAND_PLACEMENT[Piece::rook(self.turn)][rook_from];
        self.hash ^= RAND_PLACEMENT[Piece::rook(self.turn)][rook_to];
    }

    fn make_move_standard(&mut self, moved: Option<Piece>, captured: Option<Piece>) {
        self.en_passant = None;

        if matches!(moved, Some(Piece::WhitePawn | Piece::BlackPawn)) || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }
    }

    // ------------------------------------------------------------------------
    // Move Generation
    // ------------------------------------------------------------------------

    pub fn generate_moves(&self) -> ArrayVec<Move, 218> {
        let mut moves = ArrayVec::<Move, 218>::new();
        let masks = self.board.move_gen_masks(self.turn);

        if !masks.is_double_check() {
            self.board.pieces[Piece::pawn(self.turn)]
                .for_each(|position| self.generate_pawn_moves(position, &masks, &mut moves));
            let occupancy = self.board.occupancy;
            let pieces: [(Piece, fn(Position, Bitmask) -> Bitmask); 4] = [
                (Piece::knight(self.turn), |p, _| {
                    Bitmask::KNIGHT_ATTACK_MASKS[p]
                }),
                (Piece::bishop(self.turn), Bitmask::bishop_attack_mask),
                (Piece::rook(self.turn), Bitmask::rook_attack_mask),
                (Piece::queen(self.turn), Bitmask::queen_attack_mask),
            ];
            for (piece, attack_mask) in pieces {
                self.board.pieces[piece].for_each(|position| {
                    let targets = attack_mask(position, occupancy);
                    self.generate_piece_moves(position, targets, &masks, &mut moves)
                });
            }
        }

        self.board.pieces[Piece::king(self.turn)]
            .for_each(|position| self.generate_king_moves(position, &mut moves));

        moves
    }

    fn generate_pawn_moves(
        &self,
        position: Position,
        masks: &MoveGenMasks,
        moves: &mut ArrayVec<Move, 218>,
    ) {
        let legal_mask = masks.pin_rays[position] & masks.check_mask;
        let (forward_delta, start_rank_mask, end_rank_mask) = match self.turn {
            Color::White => (8i8, Bitmask::RANKS[1], Bitmask::RANKS[7]),
            Color::Black => (-8i8, Bitmask::RANKS[6], Bitmask::RANKS[0]),
        };

        'push: {
            let single_push_position = position.offset_unchecked(forward_delta);

            if self.board.is_occupied(single_push_position) {
                break 'push;
            }

            if legal_mask.contains(single_push_position) {
                push_pawn_move(position, single_push_position, end_rank_mask, moves);
            }

            if start_rank_mask.contains(position) {
                let double_push_position = single_push_position.offset_unchecked(forward_delta);

                if !self.board.is_occupied(double_push_position)
                    && legal_mask.contains(double_push_position)
                {
                    moves.push(Move::new(
                        position,
                        double_push_position,
                        MoveType::DoublePush,
                    ));
                }
            }
        }

        let attack_mask = Bitmask::PAWN_ATTACK_MASKS[self.turn][position];

        if let Some(en_passant) = self.en_passant {
            if attack_mask.contains(en_passant)
                && self.is_en_passant_legal(position, en_passant, masks)
            {
                moves.push(Move::new(position, en_passant, MoveType::EnPassant));
            }
        }

        let opponent_mask = self.board.colors[self.turn.flip()];
        (attack_mask & opponent_mask & legal_mask)
            .for_each(|p| push_pawn_move(position, p, end_rank_mask, moves));
    }

    // Moves to the squares a knight, bishop, rook or queen on `position` attacks, except the side's
    // own pieces and squares that leave its king in check
    fn generate_piece_moves(
        &self,
        position: Position,
        attack_mask: Bitmask,
        masks: &MoveGenMasks,
        moves: &mut ArrayVec<Move, 218>,
    ) {
        let legal_mask = masks.pin_rays[position] & masks.check_mask;
        let friendly_mask = self.board.colors[self.turn];
        (attack_mask & !friendly_mask & legal_mask)
            .for_each(|p| moves.push(Move::new(position, p, MoveType::Standard)));
    }

    fn generate_king_moves(&self, position: Position, moves: &mut ArrayVec<Move, 218>) {
        let attack_mask = Bitmask::KING_ATTACK_MASKS[position];
        let friendly_mask = self.board.colors[self.turn];
        let opponent_attack_mask = self
            .board
            .color_attack_mask(self.turn.flip(), self.board.pieces[Piece::king(self.turn)]);

        (attack_mask & !friendly_mask & !opponent_attack_mask)
            .for_each(|p| moves.push(Move::new(position, p, MoveType::Standard)));

        if self
            .castling_rights
            .has(CastlingRights::QUEEN_SIDE[self.turn])
            && Bitmask::QS_CASTLE_PATH[self.turn] & opponent_attack_mask == Bitmask::EMPTY
            && Bitmask::QS_CASTLE_GAP[self.turn] & self.board.occupancy == Bitmask::EMPTY
        {
            moves.push(Move::QUEEN_SIDE_CASTLING[self.turn]);
        }

        if self
            .castling_rights
            .has(CastlingRights::KING_SIDE[self.turn])
            && Bitmask::KS_CASTLE_PATH[self.turn] & opponent_attack_mask == Bitmask::EMPTY
            && Bitmask::KS_CASTLE_GAP[self.turn] & self.board.occupancy == Bitmask::EMPTY
        {
            moves.push(Move::KING_SIDE_CASTLING[self.turn]);
        }
    }

    fn is_en_passant_legal(
        &self,
        attacker: Position,
        en_passant: Position,
        masks: &MoveGenMasks,
    ) -> bool {
        if !masks.pin_rays[attacker].contains(en_passant) {
            return false;
        }

        let captured = Position::en_passant_captured(attacker, en_passant);

        if masks.check_mask & (en_passant.mask() | captured.mask()) == Bitmask::EMPTY {
            return false;
        }

        let king_pos = self.board.pieces[Piece::king(self.turn)].lsb();
        if king_pos.rank() != attacker.rank() {
            return true;
        }

        let opponent_sliders = self.board.pieces[Piece::rook(self.turn.flip())]
            | self.board.pieces[Piece::queen(self.turn.flip())];

        let occupancy_after = self.board.occupancy & !(attacker.mask() | captured.mask());

        let west_attackers = Bitmask::RAYS[Direction::West][king_pos] & opponent_sliders;
        if west_attackers != Bitmask::EMPTY {
            let between = Bitmask::BETWEEN_MASKS[west_attackers.msb()][king_pos];
            if between & occupancy_after == Bitmask::EMPTY {
                return false;
            }
        }

        let east_attackers = Bitmask::RAYS[Direction::East][king_pos] & opponent_sliders;
        if east_attackers != Bitmask::EMPTY {
            let between = Bitmask::BETWEEN_MASKS[east_attackers.lsb()][king_pos];
            if between & occupancy_after == Bitmask::EMPTY {
                return false;
            }
        }

        true
    }

    // ------------------------------------------------------------------------
    // Game Outcome
    // ------------------------------------------------------------------------

    /// `moves` must be the legal moves of the current position.
    pub(crate) fn outcome(&self, moves: &[Move]) -> Option<Outcome> {
        if moves.is_empty() {
            return Some(if self.is_in_check() {
                Outcome::Checkmate
            } else {
                Outcome::Draw
            });
        }

        let is_draw = self.halfmove_clock >= 100
            || self.repetitions() >= 2
            || self.is_insufficient_material();

        is_draw.then_some(Outcome::Draw)
    }

    fn is_in_check(&self) -> bool {
        self.board.move_gen_masks(self.turn).check_mask != Bitmask::FULL
    }

    /// Counts earlier occurrences of the current position. Only positions since the last
    /// irreversible move can match, and only every other ply has the same side to move.
    fn repetitions(&self) -> usize {
        self.history
            .iter()
            .rev()
            .take(self.halfmove_clock)
            .skip(1)
            .step_by(2)
            .filter(|record| record.hash == self.hash)
            .count()
    }

    /// Bare kings, or a single knight or bishop against a bare king.
    fn is_insufficient_material(&self) -> bool {
        let pieces = &self.board.pieces;
        let pawns_and_majors = |color| {
            pieces[Piece::pawn(color)] | pieces[Piece::rook(color)] | pieces[Piece::queen(color)]
        };

        (pawns_and_majors(Color::White) | pawns_and_majors(Color::Black)) == Bitmask::EMPTY
            && self.board.occupancy.0.count_ones() <= 3
    }

    // ------------------------------------------------------------------------
    // Zobrist Hashing
    // ------------------------------------------------------------------------

    fn generate_hash(&mut self) {
        self.hash = 0;

        self.board.mailbox.iter().enumerate().for_each(|(i, sqr)| {
            if let Some(piece) = *sqr {
                self.hash ^= RAND_PLACEMENT[piece][i];
            };
        });

        if let Some(en_passant) = self.en_passant {
            self.hash ^= RAND_EN_PASSANT[en_passant.file() as usize];
        }

        self.hash ^= RAND_CASTLING[self.castling_rights];
        self.hash ^= RAND_COLOR[self.turn];
    }

    // ------------------------------------------------------------------------
    // Performance Testing
    // ------------------------------------------------------------------------

    pub fn perft(&mut self, depth: u8) -> u64 {
        if depth == 0 {
            return 1;
        }

        let mut nodes = 0;
        let moves = self.generate_moves();

        for m in moves {
            self.make_move(m);
            nodes += self.perft(depth - 1);
            self.unmake_move();
        }

        nodes
    }

    pub fn divide(&mut self, depth: u8) {
        if depth == 0 {
            return;
        }

        println!("--- Divide Depth {} ---", depth);
        let mut total_nodes = 0;
        let moves = self.generate_moves();

        for m in moves {
            self.make_move(m);
            let nodes = self.perft(depth - 1);
            self.unmake_move();

            println!("{}: {}", m, nodes);
            total_nodes += nodes;
        }

        println!("\nTotal nodes: {}", total_nodes);
    }
}

// A pawn move to `to`, or its four promotions if `to` is on the last rank
fn push_pawn_move(
    from: Position,
    to: Position,
    end_rank_mask: Bitmask,
    moves: &mut ArrayVec<Move, 218>,
) {
    if end_rank_mask.contains(to) {
        moves.extend(
            [
                MoveType::PromotionQueen,
                MoveType::PromotionRook,
                MoveType::PromotionBishop,
                MoveType::PromotionKnight,
            ]
            .map(|promotion| Move::new(from, to, promotion)),
        );
    } else {
        moves.push(Move::new(from, to, MoveType::Standard));
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.board)?;
        writeln!(f, "Turn: {}", self.turn)?;
        writeln!(f, "Castling: {}", self.castling_rights)?;
        match self.en_passant {
            Some(p) => writeln!(f, "En Passant: {}", p)?,
            None => writeln!(f, "En Passant: -")?,
        };
        writeln!(f, "Halfmove Clock: {}", self.halfmove_clock)?;
        writeln!(f, "Fullmove Number: {}", self.fullmove_number)
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_depth_1() {
        let mut state = State::new();
        assert_eq!(state.perft(1), 20);
    }

    #[test]
    fn perft_depth_2() {
        let mut state = State::new();
        assert_eq!(state.perft(2), 400);
    }

    #[test]
    fn perft_depth_3() {
        let mut state = State::new();
        assert_eq!(state.perft(3), 8_902);
    }

    #[test]
    fn perft_depth_4() {
        let mut state = State::new();
        assert_eq!(state.perft(4), 197_281);
    }

    #[test]
    fn perft_depth_5() {
        let mut state = State::new();
        assert_eq!(state.perft(5), 4_865_609);
    }

    fn outcome_of(fen: &str) -> Option<Outcome> {
        let state = State::from_fen(fen);
        state.outcome(&state.generate_moves())
    }

    fn play(state: &mut State, moves: &[&str]) {
        for uci in moves {
            let mv = state
                .generate_moves()
                .into_iter()
                .find(|mv| mv.to_string() == *uci)
                .unwrap_or_else(|| panic!("illegal move {uci}"));
            state.make_move(mv);
        }
    }

    #[test]
    fn outcome_start_position() {
        assert_eq!(outcome_of(START_FEN), None);
    }

    #[test]
    fn outcome_checkmate() {
        // Fool's mate: white is mated
        assert_eq!(
            outcome_of("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
            Some(Outcome::Checkmate)
        );
        // Back rank: black is mated
        assert_eq!(
            outcome_of("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"),
            Some(Outcome::Checkmate)
        );
    }

    #[test]
    fn outcome_check_is_not_checkmate() {
        assert_eq!(outcome_of("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"), None);
        assert_eq!(outcome_of("R3k3/8/8/8/8/8/8/4K3 b - - 0 1"), None);
    }

    #[test]
    fn outcome_stalemate() {
        assert_eq!(
            outcome_of("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            Some(Outcome::Draw)
        );
    }

    #[test]
    fn outcome_fifty_move_rule() {
        assert_eq!(outcome_of("8/8/8/4k3/8/8/4K3/4R3 w - - 99 80"), None);
        assert_eq!(
            outcome_of("8/8/8/4k3/8/8/4K3/4R3 w - - 100 80"),
            Some(Outcome::Draw)
        );
    }

    #[test]
    fn outcome_checkmate_beats_fifty_move_rule() {
        assert_eq!(
            outcome_of("R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80"),
            Some(Outcome::Checkmate)
        );
    }

    #[test]
    fn outcome_insufficient_material() {
        assert_eq!(
            outcome_of("8/8/8/4k3/8/8/4K3/8 w - - 0 1"),
            Some(Outcome::Draw)
        );
        assert_eq!(
            outcome_of("8/8/8/4k3/8/8/4K3/5N2 w - - 0 1"),
            Some(Outcome::Draw)
        );
        assert_eq!(
            outcome_of("8/8/8/4k3/8/8/4K3/5b2 w - - 0 1"),
            Some(Outcome::Draw)
        );
        assert_eq!(outcome_of("8/8/8/4k3/8/8/4K3/4NN2 w - - 0 1"), None);
        assert_eq!(outcome_of("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"), None);
        assert_eq!(outcome_of("8/8/8/4k3/8/8/4KP2/8 w - - 0 1"), None);
    }

    #[test]
    fn fen_round_trips() {
        for fen in [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "8/8/8/4k3/8/8/4K3/4R3 b - - 99 80",
        ] {
            assert_eq!(State::from_fen(fen).to_fen(), fen);
        }
    }

    #[test]
    fn fen_of_a_played_position() {
        // Castling, captures of rights, a double push allowing en passant and the clocks
        let mut state = State::new();
        play(
            &mut state,
            &[
                "e2e4", "c7c5", "g1f3", "d7d6", "f1e2", "b8c6", "e1g1", "c5c4", "d2d4",
            ],
        );

        let fen = state.to_fen();
        assert_eq!(
            fen,
            "r1bqkbnr/pp2pppp/2np4/8/2pPP3/5N2/PPP1BPPP/RNBQ1RK1 b kq d3 0 5"
        );

        // The incrementally updated hash matches one computed from scratch
        let parsed = State::from_fen(&fen);
        assert_eq!(parsed.hash, state.hash);
        assert_eq!(parsed.generate_moves(), state.generate_moves());
    }

    #[test]
    fn outcome_threefold_repetition() {
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let mut state = State::new();

        play(&mut state, &shuffle);
        assert_eq!(state.repetitions(), 1);
        assert_eq!(state.outcome(&state.generate_moves()), None);

        play(&mut state, &shuffle);
        assert_eq!(state.repetitions(), 2);
        assert_eq!(state.outcome(&state.generate_moves()), Some(Outcome::Draw));

        state.unmake_move();
        assert_eq!(state.outcome(&state.generate_moves()), None);
    }

    #[test]
    fn repetition_requires_same_side_to_move() {
        // White triangulates, so the starting placement returns with black to move
        let mut state = State::from_fen("7k/8/8/8/8/8/7P/K7 w - - 0 1");
        play(&mut state, &["a1b1", "h8g8", "b1b2", "g8h8", "b2a1"]);
        assert_eq!(state.repetitions(), 0);
    }

    #[test]
    fn repetition_requires_same_castling_rights() {
        let shuffle = ["e1e2", "e8e7", "e2e1", "e7e8"];
        let mut state = State::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");

        play(&mut state, &shuffle);
        assert_eq!(state.repetitions(), 0);

        play(&mut state, &shuffle);
        assert_eq!(state.repetitions(), 1);
    }
}
