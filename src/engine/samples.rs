//! Self-play training samples and their file format.
//!
//! All numbers are little-endian. A file starts with the bytes `CNDRSP` and a version byte, then
//! holds whole games:
//!
//! - **Game:** the u32 generation of the network that played it, or finished it in a training run,
//!   the game's result for white as an i8 (1 white won, 0 draw, -1 black won), the u16 number of
//!   plies played, the u16 number of samples, then the samples. Every move played is a sample,
//!   unless the run searches only some moves in full, so the result and length are kept apart.
//! - **Sample:** the u8 length of the position's FEN, the FEN in ASCII, the game's result for the
//!   side to move as an i8 (1 win, 0 draw, -1 loss), the u8 number of legal moves, then for each
//!   legal move its u16 bits and the u16 visit count its search gave it.
//! - **Move bits:** the from square in bits 10-15, the to square in bits 4-9 (squares numbered
//!   a1 = 0, b1 = 1, ..., h8 = 63), and the move type in bits 0-3: 0 normal, 1 double pawn push,
//!   2 en passant, 3-6 promotion to rook, knight, bishop or queen, 7 king-side castling, 8
//!   queen-side castling.
//!
//! Moves are stored as moves rather than network policy indices, so the data stays valid if the
//! network's move encoding changes.

use crate::chess::moves::Move;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::Path,
};

const MAGIC: &[u8; 6] = b"CNDRSP";
// 2 since games record their result and length, which a version 1 file would be misread without
const VERSION: u8 = 2;

/// One position from a self-play game, as a training target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sample {
    pub(crate) fen: String,
    /// Every legal move with its visit count from the position's search
    pub(crate) visits: Vec<(Move, u16)>,
    /// The game's result for the side to move: 1 for a win, 0 for a draw, -1 for a loss
    pub(crate) result: i8,
}

pub(crate) struct SampleWriter {
    file: File,
}

impl SampleWriter {
    /// Opens `path` for appending, creating it and its directory if needed. What's left of a game
    /// cut short by a killed process is removed first, so new games follow the last complete one.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        let len = file.metadata()?.len();
        if len == 0 {
            file.write_all(MAGIC)?;
            file.write_all(&[VERSION])?;
        } else {
            let contents = read_samples(path).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} isn't a version {VERSION} sample file", path.display()),
                )
            })?;
            file.set_len(len - contents.truncated_bytes as u64)?;
        }

        Ok(Self { file })
    }

    /// Appends one game with a single write, so stopping the run, or another process appending to
    /// the same file, doesn't split the game. A process killed during the write can still cut it
    /// short; readers skip what's left of it.
    pub(crate) fn write_game(&mut self, game: &RecordedGame) -> io::Result<()> {
        let n_samples =
            u16::try_from(game.samples.len()).expect("games are shorter than 65,536 plies");
        let mut bytes = vec![];
        bytes.extend_from_slice(&game.generation.to_le_bytes());
        bytes.push(game.white_result as u8);
        bytes.extend_from_slice(&game.plies.to_le_bytes());
        bytes.extend_from_slice(&n_samples.to_le_bytes());

        for sample in &game.samples {
            let fen = sample.fen.as_bytes();
            bytes.push(fen.len() as u8);
            bytes.extend_from_slice(fen);
            bytes.extend_from_slice(&[sample.result as u8, sample.visits.len() as u8]);

            for (mv, n_visits) in &sample.visits {
                bytes.extend_from_slice(&mv.0.to_le_bytes());
                bytes.extend_from_slice(&n_visits.to_le_bytes());
            }
        }

        self.file.write_all(&bytes)
    }
}

/// A game as stored in a sample file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RecordedGame {
    pub(crate) generation: u32,
    /// 1 if white won, 0 for a draw, -1 if black won
    pub(crate) white_result: i8,
    pub(crate) plies: u16,
    pub(crate) samples: Vec<Sample>,
}

/// The contents of a sample file.
pub(crate) struct SampleFile {
    pub(crate) games: Vec<RecordedGame>,
    /// Bytes at the end that don't form a complete game, left by a process killed mid-write
    pub(crate) truncated_bytes: usize,
}

/// Reads every complete game in a file.
pub(crate) fn read_samples(path: &Path) -> io::Result<SampleFile> {
    let data = fs::read(path)?;
    let mut bytes = data.as_slice();

    let is_sample_file = take(&mut bytes, MAGIC.len() + 1)
        .is_some_and(|header| header[..MAGIC.len()] == MAGIC[..] && header[MAGIC.len()] == VERSION);
    if !is_sample_file {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a sample file",
        ));
    }

    let mut games = vec![];
    while !bytes.is_empty() {
        let remaining = bytes.len();
        match read_game(&mut bytes)? {
            Some(game) => games.push(game),
            None => {
                return Ok(SampleFile {
                    games,
                    truncated_bytes: remaining,
                })
            }
        }
    }

    Ok(SampleFile {
        games,
        truncated_bytes: 0,
    })
}

// Returns None if the bytes run out before the game ends
fn read_game(bytes: &mut &[u8]) -> io::Result<Option<RecordedGame>> {
    let Some(&[g0, g1, g2, g3, white_result, p0, p1, n0, n1]) = take(bytes, 9) else {
        return Ok(None);
    };

    let mut samples = vec![];
    for _ in 0..u16::from_le_bytes([n0, n1]) {
        let Some(sample) = read_sample(bytes)? else {
            return Ok(None);
        };
        samples.push(sample);
    }

    Ok(Some(RecordedGame {
        generation: u32::from_le_bytes([g0, g1, g2, g3]),
        white_result: white_result as i8,
        plies: u16::from_le_bytes([p0, p1]),
        samples,
    }))
}

// Returns None if the bytes run out before the sample ends
fn read_sample(bytes: &mut &[u8]) -> io::Result<Option<Sample>> {
    let Some(&[fen_len]) = take(bytes, 1) else {
        return Ok(None);
    };
    let Some(fen) = take(bytes, fen_len as usize) else {
        return Ok(None);
    };
    let fen = String::from_utf8(fen.to_vec())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let Some(&[result, n_moves]) = take(bytes, 2) else {
        return Ok(None);
    };
    let Some(visits) = take(bytes, n_moves as usize * 4) else {
        return Ok(None);
    };

    let visits = visits
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[move_low, move_high, visits_low, visits_high]| {
            let mv = Move(u16::from_le_bytes([move_low, move_high]));
            (mv, u16::from_le_bytes([visits_low, visits_high]))
        })
        .collect();

    Ok(Some(Sample {
        fen,
        visits,
        result: result as i8,
    }))
}

fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = bytes.split_at_checked(n)?;
    *bytes = rest;
    Some(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::state::START_FEN;
    use crate::chess::State;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("cinder-{}-{name}.bin", std::process::id()));
        let _ = fs::remove_file(&path);
        path
    }

    fn sample(fen: &str, result: i8) -> Sample {
        let state = State::from_fen(fen);
        let visits = state
            .generate_moves()
            .iter()
            .enumerate()
            .map(|(i, mv)| (*mv, i as u16 * 7))
            .collect();
        Sample {
            fen: fen.to_string(),
            visits,
            result,
        }
    }

    fn game(generation: u32, white_result: i8, plies: u16, samples: Vec<Sample>) -> RecordedGame {
        RecordedGame {
            generation,
            white_result,
            plies,
            samples,
        }
    }

    #[test]
    fn games_round_trip_across_appends() {
        let path = temp_path("round-trip");
        let first_game = game(
            0,
            1,
            41,
            vec![
                sample(START_FEN, 1),
                sample(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    -1,
                ),
            ],
        );
        let second_game = game(
            3,
            0,
            300,
            vec![sample(
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                0,
            )],
        );
        // With searches in full only by chance, a game can have none
        let third_game = game(3, -1, 12, vec![]);

        SampleWriter::open(&path)
            .unwrap()
            .write_game(&first_game)
            .unwrap();
        // A later run appends to the same file
        let mut writer = SampleWriter::open(&path).unwrap();
        writer.write_game(&second_game).unwrap();
        writer.write_game(&third_game).unwrap();

        let file = read_samples(&path).unwrap();
        assert_eq!(file.games, [first_game, second_game, third_game]);
        assert_eq!(file.truncated_bytes, 0);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_cut_off_final_game_is_skipped() {
        let path = temp_path("cut-off");
        let complete = game(0, 1, 1, vec![sample(START_FEN, 1)]);
        let mut writer = SampleWriter::open(&path).unwrap();
        writer.write_game(&complete).unwrap();
        let complete_len = fs::metadata(&path).unwrap().len();
        let samples = vec![
            sample("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 0),
            sample("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1", 0),
        ];
        writer.write_game(&game(0, 0, 2, samples)).unwrap();

        // As if the process were killed during the second sample of the second game
        let full_len = fs::metadata(&path).unwrap().len();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full_len - 3)
            .unwrap();

        let file = read_samples(&path).unwrap();
        assert_eq!(file.games, [complete]);
        assert_eq!(file.truncated_bytes as u64, full_len - 3 - complete_len);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_cut_off_game_is_removed_before_appending() {
        let path = temp_path("repair");
        let complete = sample(START_FEN, 1);
        let cut_off = sample("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 0);
        let mut writer = SampleWriter::open(&path).unwrap();
        writer
            .write_game(&game(0, 1, 1, vec![complete.clone()]))
            .unwrap();
        writer.write_game(&game(0, 0, 1, vec![cut_off])).unwrap();
        let full_len = fs::metadata(&path).unwrap().len();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full_len - 3)
            .unwrap();

        // A later run appends after the last complete game, not after the cut-off one
        let appended = sample("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", -1);
        SampleWriter::open(&path)
            .unwrap()
            .write_game(&game(2, -1, 1, vec![appended.clone()]))
            .unwrap();

        let file = read_samples(&path).unwrap();
        let games: Vec<_> = file.games.iter().map(|game| &game.samples).collect();
        assert_eq!(games, [&vec![complete], &vec![appended]]);
        assert_eq!(file.truncated_bytes, 0);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn files_of_an_older_version_are_left_alone() {
        let path = temp_path("version-1");
        let mut old = MAGIC.to_vec();
        old.push(1);
        old.extend_from_slice(&[0, 0, 0, 0, 1, 0, 59]);
        fs::write(&path, &old).unwrap();

        assert!(read_samples(&path).is_err());
        assert!(SampleWriter::open(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), old);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn other_files_are_not_appended_to() {
        let path = temp_path("not-samples");
        fs::write(&path, b"some other data").unwrap();

        assert!(SampleWriter::open(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"some other data");
        fs::remove_file(&path).unwrap();
    }
}
