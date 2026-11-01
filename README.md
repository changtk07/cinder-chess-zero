# Cinder

Cinder is an AlphaZero-style chess engine in Rust: bitboard move generation, a transformer policy/value network built with [burn](https://burn.dev), and Monte Carlo tree search guided by the network.

The network isn't trained yet. The engine plays complete, legal games, but with random weights its moves are close to random.

## Usage

```sh
cargo run --release -- self-play --backend cuda-bf16   # play 512 games at once with untrained weights
cargo run --release -- train runs/first --backend cuda-bf16   # train, resumably
cargo run --release -- bench     # network throughput by batch size (up to 256), and search speed
cargo run --release -- perft 5   # count positions at depth 5 from the start, per root move
cargo test                       # move generation, game outcome, search and self-play tests
```

`self-play` measures how fast games play, with untrained weights. It takes `--parallel` (games at once, 512), `--sims` (simulations per move, 200), `--games` (total, 512) and `--seed`. Both `self-play` and `bench` take `--blocks` (8) and `--width` (256, a multiple of 64) to try other network sizes. Every few seconds `self-play` and `train` report simulations and network evaluations per second, the average batch size, and how the time splits between tree search, the network, applying its results, and writing samples.

`train` plays and trains generation after generation in a run directory, creating it on first use. Each generation's network plays `--games` games (512) at `--sims` simulations per move (200), then trains for `--steps` steps (1,000) of 256 positions drawn from the last four generations' games, and becomes the next generation. Each game's samples go to the run's `samples/` directory: one per move searched in full, with the position, every legal move's visit count from its search, and the game's result for the side to move, where games stopped at the ply limit count as draws. The format is described in `src/engine/samples.rs`. Stop it at any point and run the same command again, on this machine or another, to continue: the directory holds everything the run needs, and costs at most the games in progress and the current generation's training. Copy or sync the whole directory to move a run between machines. `--parallel` (512) and `--backend` can differ per machine; the rest of the settings are fixed when the run is created, in its `config.json`. Training always runs in f32, whatever precision self-play uses. `history.csv` gets a line per generation with its games' results and the training losses; the value loss is worth comparing with `result_entropy`, what predicting the same win, draw and loss shares for every position would score. A run keeps the networks of generation 0 and every tenth generation (`keep_every`), without their optimizer state, so `cinder match runs/first 30 20 --backend cuda-bf16` measures progress between them. Setting `eval_games` makes each new network also play the one it was trained from after training, with the result in `history.csv`; it's off by default, since early networks draw nearly all those games.

Commands run on the CPU by default. Add `--backend cuda` to run on an NVIDIA GPU, or `cuda-f16` / `cuda-bf16` for half precision, e.g. `cargo run --release -- bench 4096 --backend cuda-f16`. The first run on a GPU spends a minute or more compiling kernels.

The CUDA backend needs the CUDA 13 toolkit, for its runtime compiler NVRTC. `.cargo/config.toml` pins cudarc to its CUDA 13.1 bindings, since cudarc 0.18 rejects newer 13.x toolkits. To check that the GPU computes the same results as the CPU:

```sh
cargo test --release -- --ignored
```

burn's wgpu backend isn't enabled, so the GPU path is CUDA.

## How it works

**Chess core** (`src/chess/`)

- Bitboards, with attack tables and Zobrist keys generated at compile time.
- Legal move generation from pin and check masks, without trying moves and undoing them.
- Make/unmake with incremental Zobrist hashing. A double push records an en passant square only if an enemy pawn attacks it, so positions that differ only by an unusable square count as the same.
- Game outcomes: checkmate, stalemate, the 50-move rule, threefold repetition and insufficient material.
- FEN parsing and perft.

**Network** (`src/engine/model.rs`)

- Input is 64 tokens, one per square: a piece embedding, a square embedding, and an embedding of the castling rights, en passant file and halfmove clock, added together. A separate 65th token for those would make attention about a third slower.
- The board is seen from the side to move: for black it's mirrored top to bottom with the colors swapped, and moves are mirrored to match. The network learns each pattern once rather than once per color. Chess is the same game mirrored, so the side to move isn't an input, and a position and its mirror image evaluate identically.
- 8 pre-norm transformer blocks: 4 heads, model width 256, RMSNorm, GELU. A final RMSNorm normalizes the residual stream before the heads.
- The blocks keep every position's tokens in one 2D matrix, which burn multiplies faster than the 3D form, and attention projects queries, keys and values with a single matrix multiply.
- Policy head: 73 move planes per square, 4,672 moves in total, using the AlphaZero encoding in `src/engine/encoding.rs`. Only the legal moves' logits are gathered on the device, so the softmax and the data read back cover just those moves.
- Value head: reduces each square to 32 features, then reads all 64 squares' features together through a 128-unit layer to win, draw and loss probabilities for the side to move, as Leela's transformer networks do. The search uses the win probability minus the loss probability, a score in [−1, 1].

**Search** (`src/engine/mcts.rs`)

- PUCT selection, with the network supplying move priors and position values at the leaves, and exact values for finished games.
- Each search returns root visit counts. The move is either the most visited or sampled with a temperature.
- A search pauses whenever it reaches a position the network must evaluate, so many searches can share one batch.
- Self-play keeps the subtree of the move played as the next search's root, and tops it up to `--sims` visits rather than running `--sims` new simulations. Matches start each search afresh, since each side's network searches only its own moves.

**Self-play** (`src/engine/self_play.rs`)

- Many games run at once. Each step, every game's search advances to its next position needing the network, and all of those positions are evaluated in one batch on the GPU.
- Moves are sampled by visit count for the first 30 plies, then chosen greedily. Each game has its own random generator, seeded from `--seed` (which also fixes `self-play`'s untrained weights) or a run's seed, so the same seed plays the same games, on the GPU as well.
- Dirichlet noise (alpha 0.3, a quarter of each prior) is mixed into every search's root priors, so games explore moves the network rates low. A reused root gets fresh noise in its existing priors.
- Every move played becomes a training sample, written out when its game ends. A run can instead give each move a full search only by chance (`full_search_share` in its `config.json`), and a fast one of `fast_sims` visits otherwise, as KataGo's playout cap randomization does: then only the full searches become samples, and only they get root noise.

**Training** (`src/engine/run.rs`, `src/engine/training.rs`)

- Each generation's network plays its games, then trains on positions drawn at random from the latest generations' games. The policy learns each search's visit distribution, by cross-entropy, and the value each game's result as a win, draw or loss, also by cross-entropy, with AdamW and gradient clipping.
- Self-play keeps every slot busy across generations: games still going when a generation has finished its share wait while the network trains, and the next generation's network finishes them. Without that, the batch would shrink as the last games finish, and burn recompiles its kernels for every new batch size.
- Checkpoints store the weights and optimizer state at full precision, so a run continues on any backend. Every file is written so that the process can be stopped at any point.

## Roadmap

- [x] Batched leaf evaluation on a GPU backend
- [x] Self-play data: Dirichlet noise at the root, and recording (position, visit distribution, result) samples
- [x] Network changes before any weights are trained: inputs encoded apart from the network in `src/engine/encoding.rs`, the board seen from the side to move, a final RMSNorm before the heads, and attention dropout off
- [ ] Faster network, now about 95% of self-play time:
  - [x] 2D linear layers and a fused query/key/value projection.
  - [x] 64 tokens instead of 65.
  - [x] A default network size, chosen from self-play speed across sizes: 8 blocks of width 256, with 512 games at once.
  - Maybe: attention's matrix multiplies, a thousand or more tiny products per call, still run far below the GPU's speed.
- [x] Reuse the search tree between moves:
  - [x] Measure how many of each search's visits the chosen move's subtree holds: 36% with a network trained for ten generations, so reuse saves about that share of evaluations.
  - [x] Self-play reuses the subtree, with fresh noise mixed into the reused root's priors, and `--sims` counts the root's total visits, so samples keep recording the same number of visits.
  - [x] Self-play finishes games about 1.6× as fast with reuse.
  - Maybe: cache evaluations by position and halfmove clock. 11–16% of evaluations repeat a position from the same game, 15–22% one from any game, so self-play would run about 1.1–1.25× as fast, at up to about 2 GB of memory unless the cache is bounded.
- [ ] Training:
  - [x] A training loop alternating self-play and training, with runs that stop at any point and continue on either machine.
  - [x] A throwaway run: the network learns, and generation 10 beat generation 0 by 35 wins to 9, with 84 draws.
  - [x] Win, draw and loss probabilities from the value head instead of a single score, so a surely drawn position and an unclear one look different.
  - [x] Learning rate and training steps: three trial runs found neither 3× the steps nor a 3× higher learning rate better than the defaults of 3e-4 and 1,000 steps.
  - [x] Matches between a run's networks with `cinder match`, against references kept every ten generations, or each generation against the one before.
  - [x] Playout cap randomization (`full_search_share` and `fast_sims` in a run's `config.json`): for the same GPU time, searching a quarter of moves in full and the rest at 50 visits, with 4× the games per generation, made a clearly stronger network than searching every move in full.
  - [ ] Make playout cap randomization the run default.
  - [ ] A long run.
- [ ] Once the network is trained, score unvisited moves with first-play urgency (the parent's value minus a reduction, as Leela does) instead of as draws.
- [ ] UCI interface:
  - Print promotions as `a7a8q` rather than `a7a8=q`, together with the `play()` test helper, which matches moves by that string.
  - Validate the positions it's given: `State::from_fen` accepts impossible positions, where the side not to move is in check, and `Position::from_fen` panics on malformed squares.
- [x] Cleanups: one helper each for promotions, castling, piece moves and sliding attacks, with perft 5 as fast as before; 24-byte search tree nodes; and en passant squares only where a pawn can capture.

## License

Apache 2.0. See [LICENSE](LICENSE).
