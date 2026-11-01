//! Keeps a copy of a training run in a remote folder, such as one on Google Drive, with rclone, so
//! the run can continue on another machine.
//!
//! A run created with `--sync <remote>` keeps its copy in `<remote>/<run directory's name>`. When
//! `train` starts, it first brings the local run up to date if the copy has got further, as when the
//! run last trained elsewhere, or fetches the whole run if this machine doesn't have it. While it
//! trains, it pushes the run after each new checkpoint, and once more when it stops: after its last
//! generation, or on Ctrl-C or a termination signal. Training doesn't wait for pushes, which run on
//! a thread of their own. A process that dies without a signal, as when the machine loses power,
//! leaves its current generation's finished games unpushed, and the run plays them again.
//!
//! Pushing doesn't pause the run. Every file in a run directory is written so the process can stop
//! at any moment, so a copy taken file by file mid-run is also a state the run can continue from.
//! rclone normally gives up on a file that changes while it uploads, which the current generation's
//! sample file does about once a second, so pushes pass `--local-no-check-updated`: rclone then
//! uploads an append-only file at the size it first saw, and a game cut off there is removed when
//! the run continues, like one cut off by a killed process.
//!
//! Only one machine should train a run at a time: the copy follows whichever pushed last.

use crate::engine::run::{latest_checkpoint, samples_file_name};
use std::{
    fs, io,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

// Interrupted checkpoint saves leave temporary directories, which the run cleans up itself
const EXCLUDE: &str = "/checkpoints/*.tmp/**";

// rclone's exit status when the folder it's asked about doesn't exist
const DIRECTORY_NOT_FOUND: i32 = 3;

/// How far a copy of a run has got: its latest checkpoint, then how much of that generation's
/// games it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Progress {
    generation: usize,
    samples_bytes: u64,
}

/// A run's copy in a remote folder, reached through rclone.
pub(crate) struct Remote {
    // The program to run and any arguments before rclone's own
    rclone: Vec<String>,
    path: String,
}

impl Remote {
    /// The copy of the run in `dir` under `root`, an rclone path such as `gdrive:cinder-runs`, in a
    /// folder named like `dir`. `rclone` is the command to run, normally just `["rclone"]`.
    pub(crate) fn new(rclone: &[&str], root: &str, dir: &Path) -> io::Result<Self> {
        let name = dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                io::Error::other(format!("{} has no name to sync it under", dir.display()))
            })?;
        Ok(Self {
            rclone: rclone.iter().map(|arg| arg.to_string()).collect(),
            path: format!("{}/{name}", root.trim_end_matches('/')),
        })
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    // The copy's progress, or None if there's no copy yet
    fn progress(&self) -> io::Result<Option<Progress>> {
        let output = self.run(&["lsf", "--recursive", "--format", "sp", &self.path])?;
        if output.status.code() == Some(DIRECTORY_NOT_FOUND) {
            return Ok(None);
        }
        self.check(&output)?;

        // Lines are `size;path`, and directories end in a slash
        let listing = String::from_utf8_lossy(&output.stdout);
        let entries: Vec<(&str, &str)> = listing
            .lines()
            .filter_map(|line| line.split_once(';'))
            .collect();
        let checkpoints = entries
            .iter()
            .filter_map(|(_, path)| path.strip_prefix("checkpoints/")?.strip_suffix('/'));
        let Some(generation) = latest_checkpoint(checkpoints) else {
            return Ok(None);
        };
        let samples = format!("samples/{}", samples_file_name(generation));
        let samples_bytes = entries
            .iter()
            .find(|(_, path)| *path == samples)
            .and_then(|(size, _)| size.parse().ok())
            .unwrap_or(0);
        Ok(Some(Progress {
            generation,
            samples_bytes,
        }))
    }

    fn push(&self, dir: &Path) -> io::Result<()> {
        let dir = dir.to_string_lossy();
        let output = self.run(&[
            "sync",
            &dir,
            &self.path,
            "--exclude",
            EXCLUDE,
            "--local-no-check-updated",
        ])?;
        self.check(&output)
    }

    fn pull(&self, dir: &Path) -> io::Result<()> {
        let dir = dir.to_string_lossy();
        let output = self.run(&["sync", &self.path, &dir, "--exclude", EXCLUDE])?;
        self.check(&output)
    }

    fn run(&self, args: &[&str]) -> io::Result<Output> {
        Command::new(&self.rclone[0])
            .args(&self.rclone[1..])
            .args(args)
            // A process group of its own, so a Ctrl-C meant for training doesn't cut a push short
            .process_group(0)
            .output()
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "can't run {}: {error}. Install rclone and set up the remote with \
                         `rclone config`, or pass --no-sync",
                        self.rclone.join(" ")
                    ),
                )
            })
    }

    fn check(&self, output: &Output) -> io::Result<()> {
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last_line = stderr.lines().rev().find(|line| !line.trim().is_empty());
        Err(io::Error::other(format!(
            "rclone failed with {} for {}: {}",
            output.status,
            self.path,
            last_line.unwrap_or("no message")
        )))
    }
}

/// Brings the run in `dir` up to date with its remote copy if the copy has got further, as when the
/// run last trained on another machine, or fetches it if `dir` holds no run yet. Returns whether it
/// pulled.
pub(crate) fn pull_if_behind(remote: &Remote, dir: &Path) -> io::Result<bool> {
    let Some(theirs) = remote.progress()? else {
        return Ok(false);
    };
    let ours = local_progress(dir)?;
    if ours >= Some(theirs) {
        return Ok(false);
    }

    println!(
        "Pulling the run from {}, which has got further: generation {}{}",
        remote.path,
        theirs.generation,
        ours.map_or(String::new(), |ours| format!(
            ", against {} here",
            ours.generation
        ))
    );
    remote.pull(dir)?;
    Ok(true)
}

fn local_progress(dir: &Path) -> io::Result<Option<Progress>> {
    let entries = match fs::read_dir(dir.join("checkpoints")) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut names = vec![];
    for entry in entries {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    let Some(generation) = latest_checkpoint(names.iter().map(String::as_str)) else {
        return Ok(None);
    };
    let samples = dir.join("samples").join(samples_file_name(generation));
    let samples_bytes = fs::metadata(samples).map_or(0, |metadata| metadata.len());
    Ok(Some(Progress {
        generation,
        samples_bytes,
    }))
}

/// Pushes a run to its remote copy from a background thread when asked to.
pub(crate) struct Pusher {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

struct Shared {
    remote: Remote,
    dir: PathBuf,
    // Held while pushing, so a final push waits for one in progress rather than overlapping it
    pushing: Mutex<()>,
    requests: Mutex<Requests>,
    wake: Condvar,
}

#[derive(Default)]
struct Requests {
    // A push was asked for, to start as soon as any in progress is done
    push: bool,
    stop: bool,
}

impl Shared {
    fn push(&self) {
        let _pushing = self.pushing.lock().unwrap();
        let start = Instant::now();
        match self.remote.push(&self.dir) {
            Ok(()) => println!(
                "Synced to {} in {:.0}s",
                self.remote.path,
                start.elapsed().as_secs_f64()
            ),
            Err(error) => println!("Couldn't sync: {error}"),
        }
    }
}

impl Pusher {
    /// Starts a thread to push the run in `dir` to `remote` when asked.
    pub(crate) fn start(remote: Remote, dir: &Path) -> Self {
        let shared = Arc::new(Shared {
            remote,
            dir: dir.to_path_buf(),
            pushing: Mutex::new(()),
            requests: Mutex::new(Requests::default()),
            wake: Condvar::new(),
        });

        let pusher = shared.clone();
        let thread = thread::spawn(move || loop {
            let requests = pusher.requests.lock().unwrap();
            let mut requests = pusher
                .wake
                .wait_while(requests, |requests| !requests.push && !requests.stop)
                .unwrap();
            if requests.stop {
                return;
            }
            requests.push = false;
            drop(requests);
            pusher.push();
        });

        Self {
            shared,
            thread: Some(thread),
        }
    }

    /// Also pushes once more when the process gets Ctrl-C or a termination signal, then exits. A
    /// second signal exits without waiting. Only one handler can be set per process.
    pub(crate) fn push_on_stop(&self) -> io::Result<()> {
        let shared = self.shared.clone();
        let stopping = AtomicBool::new(false);
        ctrlc::set_handler(move || {
            if stopping.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            println!(
                "\nStopping: syncing to {} first, or press Ctrl-C again to stop without waiting",
                shared.remote.path
            );
            shared.push();
            std::process::exit(130);
        })
        .map_err(io::Error::other)
    }

    /// Pushes as soon as any push in progress is done, without waiting for it.
    pub(crate) fn push_soon(&self) {
        self.shared.requests.lock().unwrap().push = true;
        self.shared.wake.notify_all();
    }

    /// Stops the thread and pushes once more.
    pub(crate) fn finish(mut self) {
        self.shared.requests.lock().unwrap().stop = true;
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the push thread doesn't panic");
        }
        self.shared.push();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Stands in for rclone with local folders as remotes, for the two commands this module uses,
    // logging each sync. Tests run it with bash rather than as a program: executing a file written
    // moments before fails now and then with "text file busy", when another test's child process,
    // between forking and starting its program, still holds the file open.
    const FAKE_RCLONE: &str = r#"#!/bin/bash
case "$1" in
  lsf)
    [ -d "$5" ] || exit 3
    cd "$5" && find . -mindepth 1 \( -type d -printf '-1;%P/\n' \) -o \( -type f -printf '%s;%P\n' \)
    ;;
  sync)
    rm -rf "$3" && mkdir -p "$(dirname "$3")" && cp -r "$2" "$3" && rm -rf "$3"/checkpoints/*.tmp
    echo "$@" >> "$(dirname "$0")/log"
    ;;
esac
"#;

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("cinder-{}-sync-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join("rclone"), FAKE_RCLONE).unwrap();
            Self { root }
        }

        fn local(&self) -> PathBuf {
            self.root.join("runs/long")
        }

        fn remote(&self) -> Remote {
            let rclone = self.root.join("rclone");
            let root = self.root.join("remote");
            Remote::new(
                &["bash", rclone.to_str().unwrap()],
                root.to_str().unwrap(),
                &self.local(),
            )
            .unwrap()
        }

        fn remote_dir(&self) -> PathBuf {
            self.root.join("remote/long")
        }

        fn syncs(&self) -> Vec<String> {
            fs::read_to_string(self.root.join("log"))
                .unwrap_or_default()
                .lines()
                .map(String::from)
                .collect()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    // A run directory at `generation` with `samples_bytes` of that generation's samples, and a
    // checkpoint save that was interrupted
    fn fake_run(dir: &Path, generation: usize, samples_bytes: usize) {
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir.join("samples")).unwrap();
        fs::write(dir.join("config.json"), "{}").unwrap();
        for kept in [0, generation.saturating_sub(1), generation] {
            fs::create_dir_all(dir.join(format!("checkpoints/{kept:05}"))).unwrap();
            fs::write(
                dir.join(format!("checkpoints/{kept:05}/model.mpk")),
                "weights",
            )
            .unwrap();
        }
        fs::create_dir_all(dir.join(format!("checkpoints/{:05}.tmp", generation + 1))).unwrap();
        fs::write(
            dir.join("samples").join(samples_file_name(generation)),
            vec![0; samples_bytes],
        )
        .unwrap();
    }

    #[test]
    fn copies_are_named_like_the_run_directory() {
        let remote =
            Remote::new(&["rclone"], "gdrive:cinder-runs/", Path::new("runs/long")).unwrap();
        assert_eq!(remote.path(), "gdrive:cinder-runs/long");
        assert!(Remote::new(&["rclone"], "gdrive:cinder-runs", Path::new("/")).is_err());
    }

    #[test]
    fn progress_reads_the_latest_checkpoint_and_its_samples() {
        let fixture = Fixture::new("progress");
        assert_eq!(fixture.remote().progress().unwrap(), None);

        fake_run(&fixture.remote_dir(), 12, 300);
        let expected = Progress {
            generation: 12,
            samples_bytes: 300,
        };
        assert_eq!(fixture.remote().progress().unwrap(), Some(expected));
        assert_eq!(
            local_progress(&fixture.remote_dir()).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn a_machine_without_the_run_fetches_it() {
        let fixture = Fixture::new("fetch");
        fake_run(&fixture.remote_dir(), 3, 10);

        assert!(pull_if_behind(&fixture.remote(), &fixture.local()).unwrap());
        assert!(fixture.local().join("checkpoints/00003/model.mpk").exists());
        assert!(fixture.local().join("config.json").exists());
    }

    #[test]
    fn a_copy_that_got_further_is_pulled() {
        let fixture = Fixture::new("behind");
        // An older generation, then the same generation with fewer of its games
        for (local_generation, local_bytes) in [(2, 500), (3, 5)] {
            fake_run(&fixture.remote_dir(), 3, 10);
            fake_run(&fixture.local(), local_generation, local_bytes);

            assert!(pull_if_behind(&fixture.remote(), &fixture.local()).unwrap());
            let local = local_progress(&fixture.local()).unwrap().unwrap();
            assert_eq!((local.generation, local.samples_bytes), (3, 10));
        }
    }

    #[test]
    fn a_run_ahead_of_its_copy_or_without_one_is_left_alone() {
        let fixture = Fixture::new("ahead");
        fake_run(&fixture.local(), 3, 10);
        assert!(!pull_if_behind(&fixture.remote(), &fixture.local()).unwrap());

        for (remote_generation, remote_bytes) in [(2, 500), (3, 5), (3, 10)] {
            fake_run(&fixture.remote_dir(), remote_generation, remote_bytes);
            assert!(!pull_if_behind(&fixture.remote(), &fixture.local()).unwrap());
        }
        assert!(fixture.syncs().is_empty());
    }

    #[test]
    fn the_pusher_pushes_when_asked_and_when_finished() {
        let fixture = Fixture::new("push");
        fake_run(&fixture.local(), 3, 10);

        let pusher = Pusher::start(fixture.remote(), &fixture.local());
        pusher.push_soon();
        // The push runs on the pusher's thread, without the caller waiting
        let start = Instant::now();
        while fixture.syncs().is_empty() {
            assert!(start.elapsed() < Duration::from_secs(10), "no push");
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            local_progress(&fixture.remote_dir()).unwrap(),
            local_progress(&fixture.local()).unwrap()
        );
        assert!(!fixture.remote_dir().join("checkpoints/00004.tmp").exists());

        // Games played since then reach the copy on the last push
        fs::write(fixture.local().join("samples/00003.bin"), vec![0; 20]).unwrap();
        pusher.finish();
        let syncs = fixture.syncs();
        assert_eq!(syncs.len(), 2, "{syncs:?}");
        assert!(syncs
            .iter()
            .all(|sync| sync.contains("--local-no-check-updated")));
        let remote = local_progress(&fixture.remote_dir()).unwrap().unwrap();
        assert_eq!(remote.samples_bytes, 20);
    }

    #[test]
    fn a_missing_rclone_says_what_to_do() {
        let remote = Remote::new(
            &["cinder-no-such-rclone"],
            "gdrive:x",
            Path::new("runs/long"),
        )
        .unwrap();
        let error = remote.progress().unwrap_err().to_string();
        assert!(error.contains("rclone config"), "{error}");
    }
}
