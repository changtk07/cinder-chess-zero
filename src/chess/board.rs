use crate::chess::{
    bitmask::Bitmask,
    types::{Color, Direction, Piece, Position},
};
use std::ops::{Index, IndexMut};

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) struct CastlingRights(pub(crate) u8);

impl CastlingRights {
    // Bit layout (4 bits total, LSB first):
    //   Bit 0 (0b0001): Black queen side  — cleared when A8 rook or E8 king moves
    //   Bit 1 (0b0010): Black king side   — cleared when H8 rook or E8 king moves
    //   Bit 2 (0b0100): White queen side  — cleared when A1 rook or E1 king moves
    //   Bit 3 (0b1000): White king side   — cleared when H1 rook or E1 king moves
    pub(crate) const WHITE_QUEEN_SIDE: Self = Self(0b0100);
    pub(crate) const BLACK_QUEEN_SIDE: Self = Self(0b0001);
    pub(crate) const QUEEN_SIDE: [Self; 2] = [Self::WHITE_QUEEN_SIDE, Self::BLACK_QUEEN_SIDE];
    pub(crate) const WHITE_KING_SIDE: Self = Self(0b1000);
    pub(crate) const BLACK_KING_SIDE: Self = Self(0b0010);
    pub(crate) const KING_SIDE: [Self; 2] = [Self::WHITE_KING_SIDE, Self::BLACK_KING_SIDE];

    const MASKS: [u8; 64] = [
        11, 15, 15, 15, 3, 15, 15, 7, // 0-7
        15, 15, 15, 15, 15, 15, 15, 15, // 8-15
        15, 15, 15, 15, 15, 15, 15, 15, // 16-23
        15, 15, 15, 15, 15, 15, 15, 15, // 24-31
        15, 15, 15, 15, 15, 15, 15, 15, // 32-39
        15, 15, 15, 15, 15, 15, 15, 15, // 40-47
        15, 15, 15, 15, 15, 15, 15, 15, // 48-55
        14, 15, 15, 15, 12, 15, 15, 13, // 56-63
    ];

    pub(crate) fn from_fen(fen: &str) -> Self {
        let mut bits = 0u8;
        fen.chars().for_each(|c| match c {
            'K' => bits |= Self::KING_SIDE[Color::White].0,
            'Q' => bits |= Self::QUEEN_SIDE[Color::White].0,
            'k' => bits |= Self::KING_SIDE[Color::Black].0,
            'q' => bits |= Self::QUEEN_SIDE[Color::Black].0,
            _ => (),
        });
        Self(bits)
    }

    pub(crate) const fn has(self, right: Self) -> bool {
        (self.0 & right.0) != 0
    }

    pub(crate) fn update(self, from: Position, to: Position) -> Self {
        Self(self.0 & Self::MASKS[from] & Self::MASKS[to])
    }
}

impl<T> Index<CastlingRights> for [T; 16] {
    type Output = T;
    #[inline]
    fn index(&self, index: CastlingRights) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl<T> IndexMut<CastlingRights> for [T; 16] {
    #[inline]
    fn index_mut(&mut self, index: CastlingRights) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl std::fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = String::with_capacity(4);
        if self.has(Self::KING_SIDE[Color::White]) {
            builder.push('K');
        }
        if self.has(Self::QUEEN_SIDE[Color::White]) {
            builder.push('Q');
        }
        if self.has(Self::KING_SIDE[Color::Black]) {
            builder.push('k');
        }
        if self.has(Self::QUEEN_SIDE[Color::Black]) {
            builder.push('q');
        }
        if builder.is_empty() {
            builder.push('-');
        }
        write!(f, "{}", builder)
    }
}

impl std::fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub(crate) struct MoveGenMasks {
    pub(crate) pin_rays: [Bitmask; 64],
    pub(crate) check_mask: Bitmask,
}

impl MoveGenMasks {
    const fn new() -> Self {
        Self {
            pin_rays: [Bitmask::FULL; 64],
            check_mask: Bitmask::FULL,
        }
    }

    pub(crate) const fn is_double_check(&self) -> bool {
        self.check_mask.0 == 0
    }
}

pub(crate) struct Board {
    pub(crate) pieces: [Bitmask; 12],
    pub(crate) colors: [Bitmask; 2],
    pub(crate) occupancy: Bitmask,
    pub(crate) mailbox: [Option<Piece>; 64],
}

impl Board {
    pub(crate) fn from_fen(fen: &str) -> Self {
        let mut board = Self {
            pieces: [Bitmask::EMPTY; 12],
            colors: [Bitmask::EMPTY; 2],
            occupancy: Bitmask::EMPTY,
            mailbox: [None; 64],
        };

        fen.split('/').rev().enumerate().for_each(|(i, rank)| {
            let mut file = 0u8;
            rank.chars().for_each(|c| {
                if c.is_ascii_digit() {
                    file += c.to_digit(10).unwrap() as u8;
                    return;
                }

                let pos = Position((i * 8) as u8 + file);
                let (piece, color) = match c {
                    'P' => (Piece::WhitePawn, Color::White),
                    'R' => (Piece::WhiteRook, Color::White),
                    'N' => (Piece::WhiteKnight, Color::White),
                    'B' => (Piece::WhiteBishop, Color::White),
                    'Q' => (Piece::WhiteQueen, Color::White),
                    'K' => (Piece::WhiteKing, Color::White),
                    'p' => (Piece::BlackPawn, Color::Black),
                    'r' => (Piece::BlackRook, Color::Black),
                    'n' => (Piece::BlackKnight, Color::Black),
                    'b' => (Piece::BlackBishop, Color::Black),
                    'q' => (Piece::BlackQueen, Color::Black),
                    'k' => (Piece::BlackKing, Color::Black),
                    _ => return,
                };
                board.pieces[piece].set_mut(pos);
                board.colors[color].set_mut(pos);
                board.occupancy.set_mut(pos);
                board.mailbox[pos] = Some(piece);
                file += 1;
            })
        });

        board
    }

    /// Piece placement, from rank 8 down to rank 1.
    pub(crate) fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(71);

        for rank in (0..8).rev() {
            let mut n_empty = 0;
            for file in 0..8 {
                match self.mailbox[rank * 8 + file] {
                    Some(piece) => {
                        if n_empty > 0 {
                            fen.push_str(&n_empty.to_string());
                            n_empty = 0;
                        }
                        fen.push_str(&piece.to_string());
                    }
                    None => n_empty += 1,
                }
            }
            if n_empty > 0 {
                fen.push_str(&n_empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen
    }

    pub(crate) const fn is_occupied(&self, position: Position) -> bool {
        self.occupancy.contains(position)
    }

    pub(crate) fn color_attack_mask(&self, color: Color, ignore: Bitmask) -> Bitmask {
        let mut mask = Bitmask::EMPTY;
        let occupancy_with_ignore = self.occupancy & !ignore;

        mask = self.pieces[Piece::pawn(color)]
            .map(|p| Bitmask::PAWN_ATTACK_MASKS[color][p])
            .fold(mask, |acc, attack| acc | attack);

        mask = self.pieces[Piece::knight(color)]
            .map(|p| Bitmask::KNIGHT_ATTACK_MASKS[p])
            .fold(mask, |acc, attack| acc | attack);

        mask = self.pieces[Piece::king(color)]
            .map(|p| Bitmask::KING_ATTACK_MASKS[p])
            .fold(mask, |acc, attack| acc | attack);

        mask = self.pieces[Piece::bishop(color)]
            .map(|p| Bitmask::bishop_attack_mask(p, occupancy_with_ignore))
            .fold(mask, |acc, attack| acc | attack);

        mask = self.pieces[Piece::rook(color)]
            .map(|p| Bitmask::rook_attack_mask(p, occupancy_with_ignore))
            .fold(mask, |acc, attack| acc | attack);

        mask = self.pieces[Piece::queen(color)]
            .map(|p| Bitmask::queen_attack_mask(p, occupancy_with_ignore))
            .fold(mask, |acc, attack| acc | attack);

        mask
    }

    pub(crate) fn move_gen_masks(&self, color: Color) -> MoveGenMasks {
        let mut masks = MoveGenMasks::new();
        let opponent = color.flip();
        let king_position = self.pieces[Piece::king(color)].lsb();

        let opponent_straight_slider =
            self.pieces[Piece::rook(opponent)] | self.pieces[Piece::queen(opponent)];
        let opponent_diagonal_slider =
            self.pieces[Piece::bishop(opponent)] | self.pieces[Piece::queen(opponent)];

        let rays = (Direction::STRAIGHT.map(|direction| (direction, opponent_straight_slider)))
            .into_iter()
            .chain(Direction::DIAGONAL.map(|direction| (direction, opponent_diagonal_slider)));
        for (direction, sliders) in rays {
            let attack_mask = Bitmask::RAYS[direction][king_position] & sliders;
            if attack_mask == Bitmask::EMPTY {
                continue;
            }

            let attacker_position = attack_mask.nearest(direction);

            let between_mask = Bitmask::BETWEEN_MASKS[attacker_position][king_position];
            let blocker_mask = between_mask & self.occupancy;

            if blocker_mask == Bitmask::EMPTY {
                masks.check_mask &= between_mask | attacker_position.mask();
                if masks.is_double_check() {
                    return masks;
                }
                continue;
            }

            let friendly_blocker_mask = between_mask & self.colors[color];
            if blocker_mask.has_only_one_set() && blocker_mask == friendly_blocker_mask {
                masks.pin_rays[blocker_mask.lsb()] = between_mask | attacker_position.mask();
            }
        }

        for attacker_position in
            Bitmask::KNIGHT_ATTACK_MASKS[king_position] & self.pieces[Piece::knight(opponent)]
        {
            masks.check_mask &= attacker_position.mask();
            if masks.is_double_check() {
                return masks;
            }
        }

        for attacker_position in
            Bitmask::PAWN_ATTACK_MASKS[color][king_position] & self.pieces[Piece::pawn(opponent)]
        {
            masks.check_mask &= attacker_position.mask();
            if masks.is_double_check() {
                return masks;
            }
        }

        masks
    }

    pub(crate) fn set_piece(&mut self, position: Position, piece: Piece) {
        if let Some(idx) = self.mailbox[position] {
            self.pieces[idx] = self.pieces[idx].unset(position);
            self.colors[idx.color()] = self.colors[idx.color()].unset(position);
        }
        self.pieces[piece] = self.pieces[piece].set(position);
        self.mailbox[position] = Some(piece);
        self.occupancy = self.occupancy.set(position);
        self.colors[piece.color()] = self.colors[piece.color()].set(position);
    }

    pub(crate) fn unset_piece(&mut self, position: Position) {
        let Some(idx) = self.mailbox[position] else {
            return;
        };
        self.pieces[idx] = self.pieces[idx].unset(position);
        self.mailbox[position] = None;
        self.occupancy = self.occupancy.unset(position);
        self.colors[idx.color()] = self.colors[idx.color()].unset(position);
    }

    pub(crate) fn move_piece(
        &mut self,
        from: Position,
        to: Position,
    ) -> (Option<Piece>, Option<Piece>) {
        let Some(from_piece) = self.mailbox[from] else {
            return (None, None);
        };

        let next_from_board = self.pieces[from_piece].unset(from).set(to);

        let captured = self.mailbox[to];
        if let Some(to_idx) = captured {
            self.pieces[to_idx] = self.pieces[to_idx].unset(to);
            self.colors[to_idx.color()] = self.colors[to_idx.color()].unset(to);
        }

        self.pieces[from_piece] = next_from_board;
        self.mailbox[to] = Some(from_piece);
        self.mailbox[from] = None;
        self.colors[from_piece.color()] = self.colors[from_piece.color()].unset(from).set(to);
        self.occupancy = self.occupancy.unset(from).set(to);
        (Some(from_piece), captured)
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let idx = rank * 8 + file;
                match self.mailbox[idx] {
                    Some(piece) => write!(f, " {}", piece)?,
                    None => write!(f, " .")?,
                };
            }
            writeln!(f)?;
        }
        writeln!(f, "   a b c d e f g h")
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
