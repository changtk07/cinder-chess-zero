# Handoff: work in progress on 2026-11-08

Written when the desktop went out of use, so the work can carry on from the laptop. Delete this file once everything here is done or dropped.

All of it is on the branch `wip/uci-search`, in four commits on top of `0619303`:

1. `docs: record the learning rate test from generation 130`: finished, can go to main as is.
2. `docs: add search ideas to the roadmap`: finished, can go to main as is.
3. `wip: batched UCI search, tree reuse and pondering`: working and tested, but not yet measured on a free GPU, and the README's UCI paragraph still describes the old search. Redo it as scoped commits before it goes to main (see below).
4. This file.

The desktop's helper scripts and results lived outside the repository in `~/cinder-match`. They are copied to `gdrive:cinder-match`, with `runs/long.log` and `runs/long-210-lr3e-5.log`. Get them with `rclone copy gdrive:cinder-match ~/cinder-match`. The scripts hard-code the desktop's paths (`/home/kevin/cinder-chess-zero`, `CUDA_PATH=/opt/cuda`).

## The long run and the 3e-5 learning rate trial

**Where it stands.** `runs/long` trained at 1e-4 from checkpoint 140. The ladder below showed its gains slowing after 180, so a trial of 3e-5 against 1e-4 began at checkpoint 210, the same way as the trial at 130:

- **The split.** `runs/long` was copied to `runs/long-210-lr3e-5`, and the copy's `config.json` changed in only two lines: `learning_rate` 0.00003 and `sync_to` null.
  - The split came 35 games into generation 210's self-play. Both lines then played the rest of generation 210 and trained on it separately, so checkpoint 211 is the first network that differs between them.
- **The stop.** Both were stopped at checkpoint 211 when the desktop went out of use, partway into generation 211's games. Finished games are kept, and `train` picks up where each left off.
- **The 1e-4 line.** `runs/long`, at `gdrive:cinder-runs/long` as always.
- **The 3e-5 line.** `runs/long-210-lr3e-5`, uploaded to `gdrive:cinder-runs/long-210-lr3e-5`, with `sync_to` set back to `gdrive:cinder-runs` so it syncs itself like any run. Its config now differs from `runs/long`'s only in the learning rate.
  - The upload ran from a detached script after the runs stopped. Check that the upload finished: `gdrive:cinder-match/upload-for-laptop.txt` should end in "Done", or compare `rclone size` of the two remote runs.

**To finish the trial:**

1. **Train both to checkpoint 220,** one after the other or side by side, but each run on one machine only:

   ```sh
   cargo run --release -- train runs/long --generations 9 --backend cuda-bf16
   cargo run --release -- train runs/long-210-lr3e-5 --sync gdrive:cinder-runs --generations 9 --backend cuda-bf16
   ```

   - The laptop's old `runs/long` pulls the newer Drive copy on its own. `--sync` fetches the copy, which the laptop doesn't have yet.
   - The interrupted generation 211 counts as the first of the nine.
   - Side by side on the desktop's GPU, the pair took about 25 minutes a generation.

2. **Play the two generation 220 networks** in a scratch run. This is what `lr-trial-210.sh` did:
   - Copy `runs/long/config.json` to `runs/lr210-match/config.json` with `sync_to` null.
   - Copy 1e-4's `checkpoints/00220/model.mpk` into `runs/lr210-match/checkpoints/00001/`, and 3e-5's into `.../00002/`.
   - Run:

     ```sh
     cargo run --release -- match runs/lr210-match 2 1 --games 512 --backend cuda-bf16
     ```

   - Delete the scratch run afterwards.

3. **Decide.** The user's rule: if 3e-5 scores above 0.5, its checkpoints replace `runs/long`, and the run continues from the 3e-5 line's own checkpoint 220. Don't continue the 1e-4 line with a changed config: that was the mistake after the first trial, which threw away the winner's generations.
   - **If 3e-5 wins:**
     - Rename both locally and on Drive:

       ```sh
       mv runs/long runs/long-210-lr1e-4 && mv runs/long-210-lr3e-5 runs/long
       rclone moveto gdrive:cinder-runs/long gdrive:cinder-runs/long-210-lr1e-4
       rclone moveto gdrive:cinder-runs/long-210-lr3e-5 gdrive:cinder-runs/long
       ```

     - A run syncs to `<sync_to>/<its directory's name>`, so the renamed runs keep syncing to the right places.
     - Append the copy's log to `runs/long.log`.
   - **If 1e-4 wins or ties:** keep `runs/long` and delete the copy (`rclone purge gdrive:cinder-runs/long-210-lr3e-5 --drive-use-trash=false`, and the local directory).

4. **Write it up** in `docs/training.md` as "A lower learning rate again, from generation 210", like the section on generation 130 before it.
   - The section should have:
     - the split;
     - a table of every other generation's draws and losses for both rates, from each run's `history.csv`;
     - the match line with its standard deviations;
     - the decision.
   - Then restart `runs/long` with no generation limit.

**The 120–200 ladder** isn't in `docs/training.md` yet; the user hadn't said whether to add it. Each line is 256 games at 50 simulations against the network ten generations older:

| Match | Wins–draws–losses | Score | Elo | Learning rate |
|---|---|---|---|---|
| 130 v 120 | 85–115–56 | 0.557 | +40 | 3e-4 |
| 140 v 130 | 60–124–72 | 0.477 | −16 | 3e-4 |
| 150 v 140 | 85–122–49 | 0.570 | +49 | 1e-4 |
| 160 v 150 | 97–116–43 | 0.605 | +74 | 1e-4 |
| 170 v 160 | 99–113–44 | 0.607 | +76 | 1e-4 |
| 180 v 170 | 108–101–47 | 0.619 | +84 | 1e-4 |
| 190 v 180 | 74–124–58 | 0.531 | +22 | 1e-4 |
| 200 v 190 | 90–104–62 | 0.555 | +38 | 1e-4 |

## The UCI search (commit 3)

The user asked for the first two fixes to UCI search weaknesses, plus tree reuse and pondering. The other ideas went into the README roadmap (commit 2).

**What changed:**

- **Batched search.** `src/engine/mcts.rs` gains `gather_leaves` and `apply_leaves`.
  - `gather_leaves` collects up to `max` leaves for one network call. Each waiting leaf counts as a visit and a loss, from each node's own side to move, along its path (virtual loss). It stops at the first collision, a node waiting for an evaluation.
  - `apply_leaves` removes the virtual losses and expands and backs up each leaf.
  - `next_leaf` and `apply` share the new `simulate`, `record` and `unwind` with them. `State` and `Board` derive `Clone`.
  - In `src/uci.rs`, `go` gathers `root visits / 8` leaves at a time, from 1 to 64 (`BATCH_SHARE`, `MAX_BATCH`), and evaluates them with `evaluate_padded`.
  - The warm-up evaluates batches of 1, 2, 4, …, 64 so each shape's kernels are compiled before the first `go`.
- **Tree reuse.** The engine keeps its `Search` and the position it searched (`tree_root`).
  - At the next `go`, `line_between` looks for the new position up to two plies below the old root, comparing FENs. If it's there, `reuse_subtree` keeps that subtree. Otherwise the search starts fresh.
  - `ucinewgame` drops the tree. Node limits count the root's visits, reused ones included.
- **Pondering.**
  - `option name Ponder`, and `bestmove X ponder Y` with Y from the principal variation.
  - `go ponder` searches until `ponderhit` or `stop`. The time limit starts at `ponderhit`. A ponder search that reaches its node limit waits for the next command.
  - `isready` is answered during a search.

**Verified on the desktop:**

- `cargo test`, clippy and fmt all pass. The new tests cover batched visit accounting, batch spread, mate in one, continuing a reused subtree, line finding, the expected reply, and ponderhit or stop.
- Seeded CPU self-play (`self-play --blocks 1 --width 64 --parallel 8 --games 16 --sims 32 --seed 3`) gives identical output before and after, so training is unchanged.
- With a game holding the GPU at 100%, so the numbers are only relative, using `uci-check.py BINARY 800` on `runs/long 200`:

  | | Old | New |
  |---|---|---|
  | Warm-up | 11.5 s | 29–34 s (44 s the very first time) |
  | 800 nodes from the start position | 553 nodes in 194 s | 800 nodes in 14 s |
  | 800-node searches continuing the tree | — | about 6 s each |
  | Answer after `ponderhit` | — | at once |

  Both play h2h4 from the start position.

**Still to do:**

1. **Measure on a free GPU:** warm-up time and real simulations a second, old binary against new.
   - Build the new one with `CARGO_TARGET_DIR=target/uci-dev cargo build --release`, so a training run using `target/release` isn't disturbed.
   - `uci-check.py` drives a binary through it.
2. **Maybe shorten the warm-up.** It compiles seven batch shapes. Padding to powers of 4 or 8 instead of 2 would compile fewer, at the cost of more padding per call. A call costs about the same up to 16 positions (`docs/performance.md`).
3. **Check that batching doesn't weaken the search,** with new against old at equal nodes, and again at equal time, for example with fastchess or Cute Chess. Virtual loss spreads a batch over several lines, which can cost a little strength per node.
4. **Update the docs with the measured numbers:**
   - README: the `uci` paragraph (line 22 still says each `go` runs a fresh search, a position at a time, at about 160 simulations a second) and the search bullets in How it works.
   - A UCI section with the machine's numbers in `docs/performance.md`.
   - Roadmap entries, marked done, for batched UCI search, tree reuse and pondering. No machine names in the README.
5. **Split the work into scoped commits on main,** each with its docs:
   - batched search: the `mcts.rs` methods, the `Clone` derives, batching in `go`, and the warm-up;
   - tree reuse: `tree_root`, `keep_tree`, `line_between`, and `ucinewgame`;
   - pondering.
6. **Rebuild `target/release` on the desktop afterwards,** since `scripts/cinder-uci.bat` runs that binary.

## Smaller open items

- **The desktop's `scripts/cinder-uci.bat`** has the user's own uncommitted edit, `NETWORK=runs/long 130`. Leave it out of commits.
  - Offered, not yet answered: point it at a newer network, for example `runs/long 200` or the trial's winner at 220, with `--sims 1600`.
- **The endgame test** (`conversion-test.py`, results in `conversion-400.txt`) is written up in the README roadmap items on forced mates and a moves-left head.
  - In it, generation 130 at 400 simulations a move played against Stockfish's defence. It mated with KQ v K in 43 plies and KR v K in 41, about twice what's needed. With Q+R+4P v 2P it didn't mate in 80 plies.
