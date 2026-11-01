#!/bin/bash
# Self-play throughput for each network size and number of games at once, to choose a network size
# and --parallel for the machine it runs on. Builds first, then plays each combination until its
# kernels are compiled and tuned, and averages the next REPORTS 5-second reports.
#
# A machine's first run of each shape autotunes its kernels, which can stall self-play for minutes;
# later runs only recompile, in under a minute. So rather than skipping a fixed number of reports,
# each run waits for the first report faster than 1,000 simulations/s and skips it and the next,
# since a report straddling the stall still includes part of it.
#
# Usage: scripts/bench-sizes.sh [BLOCKSxWIDTH ...]    default: 6x256 8x256 10x256 12x256 8x384
# Environment: PARALLEL (default "256 512 1024"), REPORTS (default 24, two minutes)
#
# Check nothing else is using the GPU first: other load skews every number.

cd "$(dirname "$0")/.." || exit 1

sizes=("$@")
[ ${#sizes[@]} -eq 0 ] && sizes=(6x256 8x256 10x256 12x256 8x384)
read -ra parallels <<< "${PARALLEL:-256 512 1024}"
reports=${REPORTS:-24}

cargo build --release --quiet || exit 1
nvidia-smi --query-gpu=name,enforced.power.limit --format=csv,noheader

for size in "${sizes[@]}"; do
  blocks=${size%x*}
  width=${size#*x}
  for parallel in "${parallels[@]}"; do
    # The timeout only guards against a run that never warms up
    rate=$(timeout 1200 target/release/cinder self-play --backend cuda-bf16 --games 1000000 \
        --parallel "$parallel" --blocks "$blocks" --width "$width" 2>&1 \
      | awk -v want="$reports" '
          /sims\/s/ && $4 > 1000 { warm++ }
          /sims\/s/ && warm > 2 { sum += $4; n++; start = start ? start : $1 }
          n == want { exit }
          END { if (n == want) printf "%6.0f simulations/s (%d reports from %s)", sum / n, n, start
                else printf "only %d reports after warming up", n }')
    printf '%3s x %-4s %4s games: %s\n' "$blocks" "$width" "$parallel" "$rate"
  done
done
