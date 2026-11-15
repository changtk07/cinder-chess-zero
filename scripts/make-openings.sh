#!/bin/bash
# Writes balanced 8-ply openings as EPD, for matches between deterministic engines.
#
# Each ply picks at random among Stockfish's MultiPV moves within MARGIN centipawns of its best at a
# shallow depth; the final position is kept if a deeper search scores it within KEEP centipawns.
# scripts/openings.epd, for scripts/stockfish-match.sh, is 160 of them. Stockfish's search on
# several threads varies from run to run, so a rerun with the same seed picks different ones.
#
# Usage: scripts/make-openings.sh <count> <out.epd> [seed]
#        seed defaults to 1
# Needs stockfish on the PATH.

if [ $# -lt 2 ]; then
  echo "usage: scripts/make-openings.sh <count> <out.epd> [seed]" >&2
  exit 1
fi
count=$1 out=$2
RANDOM=${3:-1}
PLIES=8 MULTIPV=5 DEPTH=9 MARGIN=35 CHECK_DEPTH=14 KEEP=60

coproc SF { stockfish; }

send() { printf '%s\n' "$1" >&"${SF[1]}"; }

# Reads Stockfish's output into `lines` up to and including the first line starting with $1
read_until() {
  lines=()
  local line
  while IFS= read -r line <&"${SF[0]}"; do
    lines+=("$line")
    if [[ $line == "$1"* ]]; then return; fi
  done
  echo "stockfish exited" >&2
  exit 1
}

# Fills `scores` with each move's centipawns for the side to move after `moves`, from the MultiPV
# lines of the search to depth $1
score_moves() {
  local line cp
  send "position startpos moves ${moves[*]}"
  send "go depth $1"
  read_until bestmove
  scores=()
  for line in "${lines[@]}"; do
    if [[ $line =~ ^info\ depth\ $1\ .*\ multipv\ .*\ score\ (cp|mate)\ (-?[0-9]+)\ .*\ pv\ ([a-h1-8qrbn]+) ]]; then
      cp=${BASH_REMATCH[2]}
      if [ "${BASH_REMATCH[1]}" = mate ]; then
        if (( cp > 0 )); then cp=10000; else cp=-10000; fi
      fi
      scores[${BASH_REMATCH[3]}]=$cp
    fi
  done
}

# Sets `best` to the highest score in `scores`, or 10000 if there are none
best_score() {
  local move
  best=
  for move in "${!scores[@]}"; do
    if [ -z "$best" ] || (( scores[$move] > best )); then best=${scores[$move]}; fi
  done
  best=${best:-10000}
}

send uci
read_until uciok
send "setoption name MultiPV value $MULTIPV"
send "setoption name Threads value 4"
send isready
read_until readyok

declare -A scores seen
kept=()
while (( ${#kept[@]} < count )); do
  moves=()
  for (( ply = 0; ply < PLIES; ply++ )); do
    score_moves $DEPTH
    if (( ${#scores[@]} == 0 )); then break; fi
    best_score
    # Sorted, so a seed picks the same moves from the same scores
    candidates=()
    for move in $(printf '%s\n' "${!scores[@]}" | sort); do
      if (( scores[$move] >= best - MARGIN )); then candidates+=("$move"); fi
    done
    moves+=("${candidates[RANDOM % ${#candidates[@]}]}")
  done
  if (( ${#moves[@]} < PLIES )); then continue; fi

  send "position startpos moves ${moves[*]}"
  send d
  read_until Checkers
  for line in "${lines[@]}"; do
    if [[ $line == "Fen: "* ]]; then read -r board side castling en_passant _ <<< "${line#Fen: }"; fi
  done
  position="$board $side $castling $en_passant"
  if [ -n "${seen[$position]}" ]; then continue; fi

  send "setoption name MultiPV value 1"
  score_moves $CHECK_DEPTH
  send "setoption name MultiPV value $MULTIPV"
  best_score
  if (( best >= -KEEP && best <= KEEP )); then
    seen[$position]=1
    kept+=("$position")
    echo "${#kept[@]} $position $best"
  fi
done

printf '%s\n' "${kept[@]}" > "$out"
send quit
wait
