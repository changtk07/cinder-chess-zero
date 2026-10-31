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

`self-play` takes `--parallel` (games at once, 512), `--sims` (simulations per move, 800), `--games` (total, 512) and `--seed`. Both `self-play` and `bench` take `--blocks` (8) and `--width` (256, a multiple of 64) to try other network sizes. Every few seconds it reports simulations and network evaluations per second, the average batch size, and how the time splits between tree search, the network, applying its results, and writing samples.

With `--out data/selfplay.bin`, each finished game's training samples are appended to that file: one per move played, with the position, every legal move's visit count from its search, and the game's result for the side to move. Games stopped at the ply limit count as draws. The format is described in `src/engine/samples.rs`.

`train` plays and trains generation after generation in a run directory, creating it on first use. Each generation's network plays `--games` games (512) at `--sims` simulations per move (200), then trains for `--steps` steps (1,000) of 256 positions drawn from the last four generations' games, and becomes the next generation. Stop it at any point and run the same command again, on this machine or another, to continue: the directory holds everything the run needs, and costs at most the games in progress and the current generation's training. Copy or sync the whole directory to move a run between machines. `--parallel` (512) and `--backend` can differ per machine; the rest of the settings are fixed when the run is created, in its `config.json`. Training always runs in f32, whatever precision self-play uses. `history.csv` gets a line per generation with its games' results and the training losses; the value loss is worth comparing with `result_entropy`, what predicting the same win, draw and loss shares for every position would score. A run keeps the networks of generation 0 and every tenth generation (`keep_every`), without their optimizer state, so `cinder match runs/first 30 20 --backend cuda-bf16` measures progress between them. Setting `eval_games` makes each new network also play the one it was trained from after training, with the result in `history.csv`; it's off by default, since early networks draw nearly all those games.

Commands run on the CPU by default. Add `--backend cuda` to run on an NVIDIA GPU, or `cuda-f16` / `cuda-bf16` for half precision, e.g. `cargo run --release -- bench 4096 --backend cuda-f16`. The first run on a GPU spends a minute or more compiling kernels.

The CUDA backend needs the CUDA 13 toolkit, for its runtime compiler NVRTC. `.cargo/config.toml` pins cudarc to its CUDA 13.1 bindings, since cudarc 0.18 rejects newer 13.x toolkits. To check that the GPU computes the same results as the CPU:

```sh
cargo test --release -- --ignored
```

burn's wgpu backend isn't enabled: under WSL2 it finds no Vulkan adapter, so the GPU path is CUDA.

## How it works

**Chess core** (`src/chess/`)

- Bitboards, with attack tables and Zobrist keys generated at compile time.
- Legal move generation from pin and check masks, without trying moves and undoing them.
- Make/unmake with incremental Zobrist hashing.
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

**Self-play** (`src/engine/self_play.rs`)

- Many games run at once. Each step, every game's search advances to its next position needing the network, and all of those positions are evaluated in one batch on the GPU.
- Moves are sampled by visit count for the first 30 plies, then chosen greedily. `--seed` fixes the untrained network's weights and seeds every game's own random generator, so runs with the same seed write the same samples, on the GPU as well.
- Dirichlet noise (alpha 0.3, a quarter of each prior) is mixed into every search's root priors, so games explore moves the network rates low.
- Every move played becomes a training sample, written out when its game ends.

**Training** (`src/engine/run.rs`, `src/engine/training.rs`)

- Each generation's network plays its games, then trains on positions drawn at random from the latest generations' games. The policy learns each search's visit distribution, by cross-entropy, and the value each game's result as a win, draw or loss, also by cross-entropy, with AdamW and gradient clipping.
- Self-play keeps every slot busy across generations: games still going when a generation has finished its share wait while the network trains, and the next generation's network finishes them. Without that, the batch would shrink as the last games finish, and burn recompiles its kernels for every new batch size.
- Checkpoints store the weights and optimizer state at full precision, so a run continues on any backend. Every file is written so that the process can be stopped at any point.

## Roadmap

- [x] Batched leaf evaluation on a GPU backend: see [docs/performance.md](docs/performance.md)
- [x] Self-play data: Dirichlet noise at the root, and recording (position, visit distribution, result) samples
- [x] Network changes before any weights are trained: inputs encoded apart from the network in `src/engine/encoding.rs`, the board seen from the side to move, a final RMSNorm before the heads, and attention dropout off
- [ ] Faster network, now about 98% of self-play time:
  - [x] 2D linear layers and a fused query/key/value projection: self-play about 1.7× faster on the laptop.
  - [x] 64 tokens instead of 65: about 25% faster again at the current size, more at smaller ones.
  - [x] A default network size for both the RTX 4090 desktop and the RTX 5080 laptop: 8 blocks of width 256, with 512 games at once on both. Self-play runs about 1.35× as fast as 8 × 384 on the desktop and 2× on the laptop (`docs/performance.md`, "Network size").
  - Maybe: attention's matrix multiplies, a thousand or more tiny products per call, still run far below the GPU's speed.
- [ ] Reuse the search tree between moves:
  - First measure how many of each search's visits the chosen move's subtree holds: that's the share of evaluations reuse would save.
  - A reused root is already expanded, so noise must be mixed into its existing priors. Decide whether `--sims` means new simulations per move or total visits at the root, since samples record visits.
  - Caching evaluations by Zobrist hash and halfmove clock is an alternative or a complement.
- [ ] Training:
  - [x] A training loop alternating self-play and training, with runs that stop at any point and continue on either machine.
  - [x] A throwaway run on the laptop: the network learns, and generation 10 beat generation 0 by 35 wins to 9, with 84 draws. Results and findings in [docs/training.md](docs/training.md).
  - [x] Win, draw and loss probabilities from the value head instead of a single score, so a surely drawn position and an unclear one look different.
  - [ ] Before a long run: choose the learning rate and training steps, as `docs/training.md` explains.
  - [x] Matches between a run's networks with `cinder match`, against references kept every ten generations, or each generation against the one before.
- [ ] Once the network is trained, score unvisited moves with first-play urgency (the parent's value minus a reduction, as Leela does) instead of as draws.
- [ ] UCI interface:
  - Print promotions as `a7a8q` rather than `a7a8=q`, together with the `play()` test helper, which matches moves by that string.
  - Validate the positions it's given: `State::from_fen` accepts impossible positions, where the side not to move is in check, and `Position::from_fen` panics on malformed squares.
- [ ] Cleanups, when convenient:
  - Merge the four promotion helpers and the two castling helpers in `State::make_move`, checked with perft.
  - Share one ray-attack helper between the rook and bishop attack masks and the pin and check masks, checking that perft 5 isn't slower.
  - Shrink search tree nodes from 40 to 24 bytes, with `u32` visit counts and a flag for finished games. Only needed for much larger `--parallel`.
  - Maybe set the en passant square only when a capture is possible. Today it's set after every double push, so a repetition can be detected up to a move late, and the network sees a flag that changes nothing.

## License

Apache 2.0. See [LICENSE](LICENSE).
