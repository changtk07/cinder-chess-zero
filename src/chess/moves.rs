use crate::chess::types::Position;

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum MoveType {
    Standard,
    DoublePush,
    EnPassant,
    PromotionRook,
    PromotionKnight,
    PromotionBishop,
    PromotionQueen,
    KingSideCastling,
    QueenSideCastling,
}

impl MoveType {
    pub(crate) const fn new(val: u16) -> Self {
        match val {
            1 => Self::DoublePush,
            2 => Self::EnPassant,
            3 => Self::PromotionRook,
            4 => Self::PromotionKnight,
            5 => Self::PromotionBishop,
            6 => Self::PromotionQueen,
            7 => Self::KingSideCastling,
            8 => Self::QueenSideCastling,
            _ => Self::Standard,
        }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) struct Move(pub(crate) u16);

impl Move {
    pub(crate) const QUEEN_SIDE_CASTLING: [Self; 2] = [
        Self::new(Position::E1, Position::C1, MoveType::QueenSideCastling),
        Self::new(Position::E8, Position::C8, MoveType::QueenSideCastling),
    ];

    pub(crate) const KING_SIDE_CASTLING: [Self; 2] = [
        Self::new(Position::E1, Position::G1, MoveType::KingSideCastling),
        Self::new(Position::E8, Position::G8, MoveType::KingSideCastling),
    ];

    pub(crate) const fn new(from: Position, to: Position, move_type: MoveType) -> Self {
        let mut mv = 0u16;
        mv |= (from.0 as u16 & 0x3F) << 10;
        mv |= (to.0 as u16 & 0x3F) << 4;
        mv |= (move_type as u8 as u16) & 0x0F;
        Self(mv)
    }

    pub(crate) const fn unpack(self) -> (Position, Position, MoveType) {
        (self.from(), self.to(), self.move_type())
    }

    pub(crate) const fn from(self) -> Position {
        Position(((self.0 >> 10) & 0x3F) as u8)
    }

    pub(crate) const fn to(self) -> Position {
        Position(((self.0 >> 4) & 0x3F) as u8)
    }

    pub(crate) const fn move_type(self) -> MoveType {
        MoveType::new(self.0 & 0x0F)
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (from, to, mv_type) = self.unpack();
        write!(f, "{}{}", from, to)?;
        // UCI notation, as in a7a8q
        match mv_type {
            MoveType::PromotionQueen => write!(f, "q"),
            MoveType::PromotionRook => write!(f, "r"),
            MoveType::PromotionBishop => write!(f, "b"),
            MoveType::PromotionKnight => write!(f, "n"),
            _ => Ok(()),
        }
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
