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

## Learning rate and training steps, on the desktop

Run from 2026-11-01 at `5243abd`, on the RTX 4090, after the value head changed to win, draw and loss probabilities. Three runs in `runs/lr-trials`, outside git, all at the defaults of 8 blocks of width 256 and 512 games of 200 simulations per generation, with the same seed, so they started from the same weights and played the same generation 0 games. They differed only in training:

| run | learning rate | steps per generation | time for 10 generations |
|---|---|---|---|
| a | 3e-4 | 1,000 | 310 minutes |
| b | 3e-4 | 3,000 | 330 minutes |
| c | 1e-3 | 1,000 | 295 minutes |

The three ran side by side on the one GPU, so their times are each about three times what a run alone takes.

The last generation's training, with the value loss as a share of the results' entropy, and the policy loss above the targets' entropy:

| run | value loss | results' entropy | share | policy loss above entropy |
|---|---|---|---|---|
| a | 0.318 | 0.792 | 40% | 0.54 |
| b | 0.165 | 0.856 | 19% | 0.58 |
| c | 0.377 | 0.788 | 48% | 0.51 |

These were about the same from generation 2 on.

Generation 10 against generation 0, 128 games at 50 simulations per move without root noise:

| run | wins | draws | losses | score | Elo |
|---|---|---|---|---|---|
| a | 24 | 97 | 7 | 0.566 | +46 |
| b | 44 | 77 | 7 | 0.645 | +103 |
| c | 35 | 90 | 3 | 0.625 | +89 |

The generation 10 networks against each other, 256 games per pair, in a scratch run holding copies of their checkpoints:

| pair | first's wins | draws | losses | score | decisive games from an even split |
|---|---|---|---|---|---|
| a against b | 55 | 157 | 44 | 0.521 | +1.1 standard deviations |
| a against c | 56 | 159 | 41 | 0.529 | +1.5 |
| b against c | 50 | 160 | 46 | 0.508 | +0.4 |

### Findings

- **Neither more steps nor a higher learning rate made a stronger network in ten generations.** Head to head, no pair differs significantly. Run a, the defaults, won both of its pairings narrowly: 111 to 85 in decisive games against the other two together, 1.9 standard deviations from an even split.
- **Matches against generation 0 disagree with the matches between the runs.** Run b scored best against generation 0 and run a worst, yet run a did at least as well as both head to head. With 30 to 50 decisive games in 128, scores against a weak opponent are too noisy to rank similar networks; compare them directly instead.
- **Three times the steps cut the value loss in half without stronger play.** Run b saw each position in its window about three times as often, so its lower value loss is mostly a closer fit to positions it trained on.
- **The policy loss stays about 0.5 above the targets' entropy whatever the training.** More steps or a higher learning rate didn't close it, so it isn't underfitting. A 200-simulation search's visit counts carry noise no network can predict, which likely accounts for most of it.
- **With the new value head, run a scored 0.566 against generation 0**, against 0.602 for the throwaway run with the same settings and the old head. The difference is under one standard deviation.

So the defaults stay: a learning rate of 3e-4 and 1,000 steps per generation, the cheapest of the three. A learning rate schedule is worth a look once a long run's losses and match results flatten.

### Since then

- **Cheaper data.** Self-play is nearly all of the time. Reusing the search tree between moves now plays games about 1.55–1.8× as fast (`docs/performance.md`, "Tree reuse"). Fewer simulations for most moves, KataGo's playout cap randomization, is a run option whose trial is below.

## Playout cap randomization, on the desktop

Run on 2026-11-01 from 17:10 to 21:10 at `c901b6a`, on the RTX 4090. Two runs in `runs/pcr-trial`, outside git and only on the desktop, side by side for 4 hours each from the same seed, so they started from the same weights:

| run | searches | games per generation | generations in 4 hours |
|---|---|---|---|
| base | every move in full, 200 visits | 512 | 14 |
| pcr | a quarter of moves in full (`full_search_share` 0.25), the rest 50 visits (`fast_sims`) | 2,048 | 8 |

So both trained on about the same number of samples per generation, from 512 games' full searches against a quarter of 2,048 games' moves, with the run defaults otherwise. pcr's value loss stayed about 70% of its results' entropy against about 40–45% for base, and its policy loss closer to its targets' entropy, about 0.37 above it against about 0.48.

The binary predates `7dc4dcf`, so pcr's `history.csv` splits decisive games between white and black wrongly, and its `average_plies` is samples per game, about a quarter of the real length. Draws, decisive totals and losses are right. Neither run can be resumed by later binaries.

The latest networks against each other, 512 games at 50 simulations per move without root noise, in a scratch run holding copies of their checkpoints:

| match | base 14's wins | draws | pcr 8's wins | pcr's score | decisive games from an even split |
|---|---|---|---|---|---|
| base 14 against pcr 8 | 76 | 284 | 152 | 0.574, +52 Elo | 5.0 standard deviations |

Each against their common generation 0, 128 games at 50 simulations per move without root noise:

| run | wins | draws | losses | score | Elo |
|---|---|---|---|---|---|
| base 14 | 41 | 83 | 4 | 0.645 | +103 |
| pcr 8 | 71 | 54 | 3 | 0.766 | +206 |

### Findings

- **For the same GPU time, playout cap randomization made the clearly stronger network**, with fewer generations: pcr won 152 of the 228 decisive games, where equal networks would split them about 114 each, give or take 8.
- **Its higher value loss didn't mean a worse network.** pcr's positions come from 4× as many games, so they're harder to fit than base's, and the loss compares runs poorly, as in the learning rate trial.
- **Against generation 0 the two agree with the head-to-head**, this time: pcr scored 0.766 to base's 0.645.

So runs now default to it, with a quarter of moves in full and 2,048 games per generation as in this trial. A share or fast search size other than 0.25 and 50 is untested.

## The long run

Started on 2026-11-02 at 01:18 at `83821f7`, on the RTX 5080 laptop, as `runs/long` with the run defaults: 8 blocks of width 256; 2,048 games per generation with a quarter of moves searched at 200 simulations and the rest at 50; 1,000 training steps of 256 positions per generation from the last four generations; 512 games at once in bf16. It syncs to `gdrive:cinder-runs/long` (`--sync gdrive:cinder-runs`), so either machine can continue it.

A generation takes about 20 minutes on the laptop at full power: 18.5 minutes of self-play at about 26,400 simulations per second, 1 minute 45 seconds of training, and a push to Google Drive of about 40 seconds that training doesn't wait for. The laptop dropped to its 33 W cap a few times while unattended.

| generation | white wins | black wins | draws | plies | policy loss | targets' entropy | value loss | results' entropy |
|---|---|---|---|---|---|---|---|---|
| 0 | 714 | 727 | 607 | 152 | 3.117 | 2.956 | 0.747 | 1.060 |
| 1 | 276 | 283 | 1,489 | 227 | 2.909 | 2.592 | 0.468 | 0.820 |
| 2 | 448 | 404 | 1,196 | 186 | 2.857 | 2.526 | 0.473 | 0.796 |
| 4 | 437 | 422 | 1,189 | 184 | 2.728 | 2.361 | 0.517 | 0.708 |
| 6 | 421 | 431 | 1,196 | 167 | 2.695 | 2.347 | 0.584 | 0.800 |
| 8 | 433 | 444 | 1,171 | 159 | 2.691 | 2.317 | 0.575 | 0.824 |
| 10 | 467 | 463 | 1,118 | 154 | 2.665 | 2.264 | 0.568 | 0.846 |
| 12 | 497 | 489 | 1,062 | 154 | 2.632 | 2.211 | 0.577 | 0.894 |
| 14 | 506 | 417 | 1,125 | 157 | 2.619 | 2.186 | 0.565 | 0.906 |
| 16 | 508 | 448 | 1,092 | 159 | 2.610 | 2.165 | 0.535 | 0.869 |
| 18 | 477 | 454 | 1,117 | 162 | 2.601 | 2.141 | 0.521 | 0.846 |
| 19 | 504 | 449 | 1,095 | 158 | 2.598 | 2.137 | 0.535 | 0.864 |
| 20 | 477 | 485 | 1,086 | 160 | 2.605 | 2.146 | 0.552 | 0.874 |
| 22 | 578 | 540 | 930 | 161 | 2.585 | 2.103 | 0.599 | 0.943 |
| 24 | 552 | 550 | 946 | 168 | 2.557 | 2.054 | 0.596 | 0.968 |
| 26 | 529 | 509 | 1,010 | 170 | 2.519 | 1.993 | 0.589 | 0.972 |
| 28 | 483 | 521 | 1,044 | 177 | 2.476 | 1.951 | 0.581 | 0.979 |
| 29 | 581 | 472 | 995 | 169 | 2.473 | 1.952 | 0.579 | 0.990 |
| 30 | 548 | 480 | 1,020 | 175 | 2.465 | 1.952 | 0.569 | 0.991 |
| 32 | 562 | 568 | 918 | 171 | 2.442 | 1.930 | 0.569 | 1.012 |
| 34 | 526 | 521 | 1,001 | 168 | 2.432 | 1.917 | 0.567 | 1.021 |
| 36 | 543 | 514 | 991 | 175 | 2.418 | 1.902 | 0.557 | 1.029 |
| 38 | 576 | 507 | 965 | 173 | 2.405 | 1.886 | 0.541 | 1.032 |
| 39 | 589 | 488 | 971 | 177 | 2.396 | 1.874 | 0.538 | 1.034 |
| 40 | 588 | 538 | 922 | 180 | 2.387 | 1.860 | 0.541 | 1.041 |
| 41 | 655 | 607 | 786 | 171 | 2.372 | 1.844 | 0.544 | 1.062 |
| 43 | 654 | 578 | 816 | 174 | 2.362 | 1.828 | 0.538 | 1.079 |
| 45 | 629 | 618 | 801 | 175 | 2.346 | 1.809 | 0.530 | 1.082 |
| 47 | 631 | 593 | 824 | 177 | 2.313 | 1.763 | 0.516 | 1.084 |
| 49 | 612 | 604 | 832 | 178 | 2.292 | 1.739 | 0.510 | 1.086 |
| 51 | 655 | 620 | 773 | 178 | 2.269 | 1.728 | 0.511 | 1.087 |
| 53 | 654 | 654 | 740 | 180 | 2.242 | 1.712 | 0.501 | 1.094 |
| 55 | 672 | 653 | 723 | 181 | 2.214 | 1.686 | 0.489 | 1.097 |
| 57 | 712 | 632 | 704 | 179 | 2.188 | 1.665 | 0.476 | 1.098 |
| 59 | 663 | 651 | 734 | 187 | 2.150 | 1.636 | 0.476 | 1.098 |
| 61 | 680 | 644 | 724 | 184 | 2.113 | 1.604 | 0.478 | 1.098 |
| 63 | 696 | 657 | 695 | 180 | 2.110 | 1.603 | 0.472 | 1.098 |
| 65 | 674 | 694 | 680 | 182 | 2.113 | 1.610 | 0.468 | 1.099 |
| 67 | 691 | 656 | 701 | 184 | 2.106 | 1.603 | 0.467 | 1.099 |
| 69 | 668 | 680 | 700 | 182 | 2.106 | 1.597 | 0.465 | 1.098 |
| 71 | 663 | 665 | 720 | 181 | 2.110 | 1.594 | 0.470 | 1.098 |
| 73 | 642 | 715 | 691 | 183 | 2.111 | 1.593 | 0.471 | 1.098 |
| 75 | 704 | 727 | 617 | 171 | 2.119 | 1.603 | 0.465 | 1.099 |
| 77 | 689 | 732 | 627 | 190 | 2.106 | 1.594 | 0.478 | 1.098 |
| 78 | 713 | 713 | 622 | 187 | 2.093 | 1.583 | 0.469 | 1.098 |
| 79 | 783 | 706 | 559 | 187 | 2.079 | 1.565 | 0.460 | 1.098 |

Generation 10 against generation 0, 128 games at 50 simulations per move without root noise: 61 wins, 65 draws (11 at the ply limit) and 2 losses, a score of 0.730 or +173 Elo, close to the playout cap trial's +206 for its generation 8. Against an untrained network the score will soon saturate, so later references play the one before.

| match | games | wins | draws | losses | score | Elo | decisive games from an even split |
|---|---|---|---|---|---|---|---|
| 10 against 0 | 128 | 61 | 65 | 2 | 0.730 | +173 | +10.4 standard deviations |
| 20 against 10 | 256 | 68 | 160 | 28 | 0.578 | +55 | +4.1 |
| 30 against 20 | 256 | 76 | 154 | 26 | 0.598 | +69 | +5.0 |
| 40 against 30 | 256 | 70 | 161 | 25 | 0.588 | +62 | +4.6 |
| 50 against 40 | 256 | 87 | 144 | 25 | 0.621 | +86 | +5.9 |
| 60 against 50 | 256 | 103 | 124 | 29 | 0.645 | +103 | +6.4 |
| 70 against 60 | 256 | 79 | 141 | 36 | 0.584 | +59 | +4.0 |
| 80 against 70 | 256 | 78 | 145 | 33 | 0.588 | +62 | +4.3 |

All at 50 simulations per move without root noise, each match's Elo uncertain by about ±15. After the first ten generations, each ten gained 55–103 Elo, so generation 80 is roughly 670 Elo above generation 0, though chaining adds up each match's error. At generation 80 the run directory took 1.2 GB, growing by about 15 MB a generation.

The run was stopped after generation 74's training and continued from checkpoint 75 to 80 at `13d3178`, whose only change stops a run without a CUDA panic on exit. From generation 10 to 20 the draw rate held at 52–59% and games at about 155 plies, while the policy loss kept falling slowly. From 20 to 30 the draws fell to 45–53% and games lengthened to about 170 plies, with the policy loss falling faster again, and progress didn't slow: 20 to 30 gained a little more than 10 to 20. From 30 to 40, draws stayed at 45–52% and games at about 175 plies, and the value loss fell to its lowest share of the results' entropy, 52%. From generation 41 the draws dropped to 38–42% and stayed there, both losses fell faster, and 40 to 50 was the largest gain since the first ten generations. From 50 to 60 the draws kept falling, to 33–38%, with games lengthening to about 180 plies, the value loss under half the results' entropy, and a larger gain again. From 61 to 69 everything held still: draws at 33–36%, games at about 183 plies, the policy loss at 2.11 against targets' entropy of 1.60, and the value loss barely moving. 60 to 70 gained +59, down from +103, a drop of about two standard deviations, so progress has probably slowed at the constant learning rate of 3e-4. From 75 on, draws fell to 27–32% and both losses started falling again by generation 79, and 70 to 80 gained +62, about the same as 60 to 70.

### A lower learning rate, from generation 80

Run on 2026-11-03 at `abf3d0d`, on the laptop. With losses flat from generation 61 to 75 and 60 to 70 gaining less than 50 to 60, the run was copied at checkpoint 80 to `runs/long-lr1e-4`, outside git and not synced, with only the learning rate changed from 3e-4 to 1e-4. Both trained 10 generations side by side, from the same seed, so generation 80's games were the same in both.

| generation | 3e-4 draws | 3e-4 policy loss | 3e-4 value loss | 1e-4 draws | 1e-4 policy loss | 1e-4 value loss |
|---|---|---|---|---|---|---|
| 81 | 29% | 2.066 | 0.468 | 25% | 2.047 | 0.467 |
| 83 | 27% | 2.046 | 0.463 | 26% | 2.017 | 0.478 |
| 85 | 31% | 2.021 | 0.458 | 25% | 1.989 | 0.468 |
| 87 | 28% | 2.022 | 0.459 | 26% | 1.990 | 0.474 |
| 89 | 28% | 2.035 | 0.454 | 25% | 1.995 | 0.483 |

The two generation 90 networks against each other, 512 games at 50 simulations per move without root noise: 1e-4 won 128, drew 274 and lost 110, a score of 0.518 or +12 Elo, 1.2 standard deviations from an even split in decisive games.

- **The lower learning rate made no clear difference in ten generations.** Its policy loss fell about 0.04 further and its games drew a little less, but its value loss rose, and head to head it was within chance of the 3e-4 network.
- So the run continues at 3e-4 from `runs/long`'s checkpoint 90. A decay is worth trying again once the ten-generation matches stop showing gains.
