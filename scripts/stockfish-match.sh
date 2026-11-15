#!/bin/bash
# Plays one of a run's kept networks against Stockfish at a fixed level, over UCI with fastchess,
# to measure strength on a scale from outside the run.
#
# Cinder searches NODES nodes a move, so the result doesn't depend on the GPU's speed or what else
# is using it, and several games can share the GPU. Stockfish plays on one thread, either at a
# fixed node count a move (nodes=N), or limited to a rating (elo=N, 1320 to 3190) with 60+0.6 on
# the clock, the time control its ratings were calibrated at. Each opening in
# scripts/openings.epd is played with both colours, and games are played out without
# adjudication, so a won endgame has to be won on the board.
#
# Usage: scripts/stockfish-match.sh <run dir> <generation> nodes=<N>|elo=<N> [games]
#        games defaults to 100, from the first 50 openings
# Environment: NODES (default 800), CONCURRENCY (4), BACKEND (cuda-bf16)
# Needs fastchess and stockfish on the PATH. Writes the games and fastchess's log to
# target/stockfish/, and prints the result.

cd "$(dirname "$0")/.." || exit 1

if [ $# -lt 3 ]; then
  echo "usage: scripts/stockfish-match.sh <run dir> <generation> nodes=<N>|elo=<N> [games]" >&2
  exit 1
fi
run=$1 generation=$2 level=$3 games=${4:-100}
nodes=${NODES:-800}
concurrency=${CONCURRENCY:-4}
backend=${BACKEND:-cuda-bf16}

case $level in
  nodes=*) stockfish=(name=sf-n${level#nodes=} "$level" tc=10000+10) ;;
  elo=*) stockfish=(name=sf-elo${level#elo=} option.UCI_LimitStrength=true
                    option.UCI_Elo=${level#elo=} tc=60+0.6) ;;
  *) echo "the Stockfish level is nodes=N or elo=N, not $level" >&2; exit 1 ;;
esac

out=target/stockfish
mkdir -p "$out"
name=$(basename "$run")-$generation-n$nodes-vs-${stockfish[0]#name=}
cinder=$(pwd)/target/release/cinder
[ -x "$cinder" ] || { echo "build it first: cargo build --release" >&2; exit 1; }

fastchess \
  -engine name=cinder-$generation dir="$(pwd)" cmd="$cinder" \
    args="uci $run $generation --backend $backend" tc=10000+10 nodes="$nodes" \
  -engine cmd=stockfish "${stockfish[@]}" \
  -openings file=scripts/openings.epd format=epd order=sequential \
  -rounds $((games / 2)) -games 2 -repeat -concurrency "$concurrency" \
  -pgnout file="$out/$name.pgn" append=false -config outname="$out/$name.json" \
  > "$out/$name.log" 2>&1
status=$?

echo "$name:"
grep -A4 '^Results of' "$out/$name.log" | tail -5
exit $status
