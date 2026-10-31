# Training notes

Runs of `cinder train`, what they showed, and what to settle before a long run.

## Throwaway run on the laptop

Run from 2026-10-28 to 2026-10-30 at `bebdc21`, on the RTX 5080 Laptop GPU, to check that the network learns. Its directory, `runs/throwaway`, stays on the laptop outside git.

Settings: the run defaults with 8 blocks of width 256. Each generation played 512 games at 200 simulations per move, 512 at once in bf16, then trained in f32 for 1,000 steps of 256 positions drawn from the last four generations' games, with AdamW at a learning rate of 3e-4 after 500 warmup steps and weight decay 1e-4.

| generation | white wins | black wins | draws | plies | policy loss | targets' entropy | value loss | draw baseline |
|---|---|---|---|---|---|---|---|---|
| 0 | 250 | 262 | 0 | 77 | 3.276 | 3.068 | 0.456 | 1.000 |
| 1 | 137 | 164 | 211 | 126 | 3.012 | 2.390 | 0.207 | 0.658 |
| 2 | 111 | 101 | 300 | 224 | 2.723 | 2.059 | 0.160 | 0.460 |
| 3 | 114 | 113 | 285 | 211 | 2.629 | 1.982 | 0.149 | 0.410 |
| 4 | 104 | 137 | 271 | 208 | 2.509 | 1.848 | 0.141 | 0.331 |
| 5 | 110 | 110 | 292 | 207 | 2.458 | 1.853 | 0.132 | 0.310 |
| 6 | 126 | 128 | 258 | 191 | 2.474 | 1.913 | 0.138 | 0.330 |
| 7 | 108 | 129 | 275 | 203 | 2.471 | 1.937 | 0.137 | 0.339 |
| 8 | 109 | 127 | 276 | 198 | 2.481 | 1.956 | 0.134 | 0.339 |
| 9 | 124 | 115 | 273 | 195 | 2.485 | 1.959 | 0.135 | 0.342 |

Each line is one generation's own games and the training that followed them. The value head then gave a single score, trained by squared error, and the draw baseline is the value loss of always predicting a draw over the training window. Runs now predict win, draw and loss probabilities, with a cross-entropy value loss and the results' entropy as its baseline, so their value losses don't compare with these.

Generation 10 against generation 0, 128 games at 50 simulations per move without root noise: 35 wins, 84 draws (12 at the ply limit) and 9 losses, a score of 0.602 or +72 Elo. Equal networks would split 44 decisive games about 22 to 22, give or take 3, so 35 to 9 is far outside chance.

How self-play games ended, estimated from each game's last position: from generation 3 on, about 46% were decisive, 35% drawn by repetition (or the occasional stalemate), 15–18% by the 50-move rule, and 2% stopped at the 512-ply limit. Untrained play in generation 0 ended every game in checkmate, mostly early blunders a 200-simulation search punishes.

### Findings

- **The network learns.** The value loss fell well under the draw baseline from generation 1, and generation 10 beat generation 0.
- **The losses flattened from about generation 5**, with draws holding at 50–55% and games at about 200 plies. Ten generations of 512 games is tiny by AlphaZero standards, so that's not yet a ceiling.
- **Draws come from not knowing how to make progress**, not from the ply limit: repetitions and the 50-move rule end about half of all games, and the limit rarely matters.
- **Evaluating each new network against the one before cost a quarter of each generation and showed nothing:** generation 1 against 0 drew 120 of 128 games at 50 simulations without noise, with 8 or 30 sampled opening plies. It's now off by default, and runs keep a reference network every ten generations for `cinder match` instead.
- **The laptop caps its GPU at 33 W in standby**, about a fifth of its speed, while the run keeps going. Keep it awake while it trains.

### Timing and size

At full power a generation took about 20–25 minutes, almost all of it self-play at about 23,000 simulations per second. Training ran at 11.4 steps per second, about 90 seconds a generation, after a first step that compiled kernels for about 3 minutes in a new process, or about 25 seconds once autotune results were cached.

A checkpoint with its optimizer state took 52 MB, a reference network 18 MB, and each generation's samples about 12 MB.

### To settle before a long run

- **The learning rate and training steps.** The policy loss stays about 0.5 above the targets' entropy, so the network may be underfitting what the searches teach. Training is about 5% of each generation, so more steps per generation, or a schedule, costs little.
- **Cheaper data.** Self-play is nearly all of the time. Reusing the search tree between moves, and fewer simulations for most moves (playout cap randomization, as in KataGo), would both multiply games per hour.
