use crate::chess::bitmask::Bitmask;
use std::ops::{Index, IndexMut};

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub(crate) fn from_fen(fen: &str) -> Self {
        match fen {
            "w" => Color::White,
            _ => Color::Black,
        }
    }

    pub(crate) const fn flip(self) -> Self {
        unsafe { std::mem::transmute(1 - (self as u8)) }
    }
}

impl<T> Index<Color> for [T; 2] {
    type Output = T;
    #[inline]
    fn index(&self, index: Color) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> IndexMut<Color> for [T; 2] {
    #[inline]
    fn index_mut(&mut self, index: Color) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = match self {
            Self::White => "White",
            Self::Black => "Black",
        };
        write!(f, "{}", c)
    }
}

impl std::fmt::Debug for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum Piece {
    WhitePawn = 0,
    WhiteRook = 1,
    WhiteKnight = 2,
    WhiteBishop = 3,
    WhiteQueen = 4,
    WhiteKing = 5,
    BlackPawn = 6,
    BlackRook = 7,
    BlackKnight = 8,
    BlackBishop = 9,
    BlackQueen = 10,
    BlackKing = 11,
}

impl Piece {
    pub(crate) const fn pawn(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6) }
    }

    pub(crate) const fn rook(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6 + 1) }
    }

    pub(crate) const fn knight(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6 + 2) }
    }

    pub(crate) const fn bishop(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6 + 3) }
    }

    pub(crate) const fn queen(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6 + 4) }
    }

    pub(crate) const fn king(color: Color) -> Self {
        unsafe { std::mem::transmute((color as u8) * 6 + 5) }
    }

    pub(crate) const fn color(self) -> Color {
        if (self as u8) < 6 {
            Color::White
        } else {
            Color::Black
        }
    }
}

impl<T> Index<Piece> for [T; 12] {
    type Output = T;
    #[inline]
    fn index(&self, index: Piece) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> IndexMut<Piece> for [T; 12] {
    #[inline]
    fn index_mut(&mut self, index: Piece) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl std::fmt::Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = match self {
            Self::WhitePawn => 'P',
            Self::WhiteKnight => 'N',
            Self::WhiteBishop => 'B',
            Self::WhiteRook => 'R',
            Self::WhiteQueen => 'Q',
            Self::WhiteKing => 'K',
            Self::BlackPawn => 'p',
            Self::BlackKnight => 'n',
            Self::BlackBishop => 'b',
            Self::BlackRook => 'r',
            Self::BlackQueen => 'q',
            Self::BlackKing => 'k',
        };
        write!(f, "{}", p)
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) struct Position(pub(crate) u8);

impl Position {
    pub(crate) const A1: Self = Self(0);
    pub(crate) const C1: Self = Self(2);
    pub(crate) const D1: Self = Self(3);
    pub(crate) const E1: Self = Self(4);
    pub(crate) const F1: Self = Self(5);
    pub(crate) const G1: Self = Self(6);
    pub(crate) const H1: Self = Self(7);
    pub(crate) const A8: Self = Self(56);
    pub(crate) const C8: Self = Self(58);
    pub(crate) const D8: Self = Self(59);
    pub(crate) const E8: Self = Self(60);
    pub(crate) const F8: Self = Self(61);
    pub(crate) const G8: Self = Self(62);
    pub(crate) const H8: Self = Self(63);
    pub(crate) const QS_CASTLE_ROOK: [(Self, Self); 2] =
        [(Self::A1, Self::D1), (Self::A8, Self::D8)];
    pub(crate) const KS_CASTLE_ROOK: [(Self, Self); 2] =
        [(Self::H1, Self::F1), (Self::H8, Self::F8)];

    /// The square named like `e4`, or None for anything else.
    pub(crate) fn from_fen(fen: &str) -> Option<Self> {
        let &[file, rank] = fen.as_bytes() else {
            return None;
        };
        let file = file.checked_sub(b'a').filter(|&file| file < 8)?;
        let rank = rank.checked_sub(b'1').filter(|&rank| rank < 8)?;
        Some(Self(rank * 8 + file))
    }

    pub(crate) const fn middle_of(a: Self, b: Self) -> Self {
        Self((a.0 + b.0) / 2)
    }

    pub(crate) const fn en_passant_captured(from: Self, to: Self) -> Self {
        Self(((from.0 >> 3) << 3) + (to.0 & 0b111)) // = (from/8)*8 + (to%8)
    }

    pub(crate) const fn mask(self) -> Bitmask {
        Bitmask(1u64 << self.0)
    }

    pub(crate) const fn offset_unchecked(self, delta: i8) -> Self {
        Self((self.0 as i8 + delta) as u8)
    }

    pub(crate) const fn rank(self) -> u8 {
        self.0 / 8
    }

    pub(crate) const fn file(self) -> u8 {
        self.0 % 8
    }
}

impl<T> Index<Position> for [T; 64] {
    type Output = T;
    #[inline]
    fn index(&self, index: Position) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl<T> IndexMut<Position> for [T; 64] {
    #[inline]
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (rank, file) = (self.0 / 8, self.0 % 8);
        write!(f, "{}{}", (b'a' + file) as char, rank + 1)
    }
}

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum Direction {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
    NorthEast = 4,
    NorthWest = 5,
    SouthEast = 6,
    SouthWest = 7,
}

impl Direction {
    pub(crate) const STRAIGHT: [Self; 4] = [Self::North, Self::South, Self::East, Self::West];
    pub(crate) const DIAGONAL: [Self; 4] = [
        Self::NorthEast,
        Self::NorthWest,
        Self::SouthEast,
        Self::SouthWest,
    ];

    /// Whether squares further along the direction have higher indices
    pub(crate) const fn ascends(self) -> bool {
        matches!(
            self,
            Self::North | Self::East | Self::NorthEast | Self::NorthWest
        )
    }
}

impl<T> Index<Direction> for [T; 8] {
    type Output = T;
    #[inline]
    fn index(&self, index: Direction) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> IndexMut<Direction> for [T; 8] {
    #[inline]
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        &mut self[index as usize]
    }
}
