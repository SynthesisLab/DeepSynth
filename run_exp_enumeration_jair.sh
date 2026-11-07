#!/bin/bash
# Enumeration experiments of the JAIR version (Figures 4-6).
#
# Every (scaling, seed, algorithm, size) is a separate process, so that
# --max-memory (peak RSS of the process) applies to one run only, and runs are
# resumable: a run whose growth file exists is skipped.
#
#   ./run_exp_enumeration_jair.sh jobs                       # print the job list
#   ./run_exp_enumeration_jair.sh run <scaling> <seed> <algo> <size>
#   ./run_exp_enumeration_jair.sh merge                      # merged CSVs for the plots
#   ./run_exp_enumeration_jair.sh [N]                        # all jobs, N in parallel (default 2), then merge
#
# Scalings: distance (R_k), nonterminals (N_k), derivations (D_k).
# Heap Search runs on R_k only up to size 10, Bee Search on R_k only for size 4.
# Outputs: enumeration_jair/raw/<scaling>_<seed>_<algo>_<size>_{growth,detailed}.csv
# and, after merge, enumeration_jair/results_<scaling>_{growth,detailed}.csv
# (the detailed throughput file is the one of size 4).
set -euo pipefail

SEEDS="${SEEDS:-1 2 3 4 5}"
SIZES="${SIZES:-4 6 8 10 12 14 16 18 20 22}"
SCALINGS="${SCALINGS:-distance nonterminals derivations}"
ALGOS="${ALGOS:-beap_search cd4 cd16 cd64 heap_search bee_search}"
PROGRAMS="${PROGRAMS:-1000000000}"
TIME="${TIME:-120}"
MEMORY="${MEMORY:-14}"
OUT="${OUT:-enumeration_jair}"
PYTHON="${PYTHON:-uv run python}"

cd "$(dirname "$0")"
mkdir -p "$OUT/raw" "$OUT/logs"

jobs() {
    for scaling in $SCALINGS; do
        for seed in $SEEDS; do
            for algo in $ALGOS; do
                for size in $SIZES; do
                    if [ "$scaling" = distance ]; then
                        if [ "$algo" = heap_search ] && [ "$size" -gt 10 ]; then continue; fi
                        if [ "$algo" = bee_search ] && [ "$size" -gt 4 ]; then continue; fi
                    fi
                    echo "$scaling $seed $algo $size"
                done
            done
        done
    done
}

run() {
    local scaling=$1 seed=$2 algo=$3 size=$4
    local name="$OUT/raw/${scaling}_${seed}_${algo}_${size}"
    if [ -s "${name}_growth.csv" ]; then
        return 0
    fi
    $PYTHON examples/compare_enumeration.py "$PROGRAMS" "$size" "$scaling" \
        -s "$seed" -t "$TIME" --algos "$algo" --sizes "$size" --max-memory "$MEMORY" \
        -o "${name}.csv" > "$OUT/logs/${scaling}_${seed}_${algo}_${size}.log" 2>&1
}

merge() {
    for scaling in $SCALINGS; do
        for kind in growth detailed; do
            local out="$OUT/results_${scaling}_${kind}.csv"
            local pattern="$OUT/raw/${scaling}_*_${kind}.csv"
            if [ "$kind" = detailed ]; then
                pattern="$OUT/raw/${scaling}_*_4_${kind}.csv"
            fi
            # shellcheck disable=SC2086
            set -- $pattern
            if [ ! -e "$1" ]; then continue; fi
            head -n 1 "$1" > "$out"
            for f in "$@"; do tail -n +2 "$f" >> "$out"; done
            echo "$out: $(($(wc -l < "$out") - 1)) rows"
        done
    done
}

case "${1:-}" in
    jobs) jobs ;;
    run) shift; run "$@" ;;
    merge) merge ;;
    *)
        export -f run
        export OUT PROGRAMS TIME MEMORY PYTHON
        jobs | xargs -P "${1:-2}" -L 1 bash -c 'run "$@"' _
        merge
        ;;
esac
