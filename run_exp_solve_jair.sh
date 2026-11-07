#!/bin/bash
# FlashFill (SyGuS strings) and DeepCoder experiments of the JAIR version (Figures 2-3).
#
#   ./run_exp_solve_jair.sh data <beesearch_tasks_folder> <deepcoder_dataset_folder>
#   ./run_exp_solve_jair.sh train <domain> <seed> [loss]   # training set, model, audit, PCFGs of the test set
#   ./run_exp_solve_jair.sh jobs-train                     # one line per (domain, seed)
#   ./run_exp_solve_jair.sh solve <domain> <seed> <algo> <shard> [loss]
#   ./run_exp_solve_jair.sh jobs-solve                     # one line per (domain, seed, algo, shard)
#   ./run_exp_solve_jair.sh merge                          # merge the shards of solve.py
#
# Domains: string (the 205 SyGuS string tasks of Bee Search), deepcoder (the
# 792 tasks of T=1..5_test of the DeepCoder archive). Every step is resumable.
set -euo pipefail

SEEDS="${SEEDS:-1 48 35132 849 98465}"
DOMAINS="${DOMAINS:-string deepcoder}"
ALGOS="${ALGOS:-heap_search bee_search beap_search cd_search bee_search_buckets cd_search_eager}"
TIMEOUT="${TIMEOUT:-300}"
MEMORY="${MEMORY:-16}"
PYTHON="${PYTHON:-uv run python}"
GEN_TAGS="--inputs 2 --programs 1000 --uniform"
EPOCHS="${EPOCHS:-2}"

cd "$(dirname "$0")"

config() {
    case "$1" in
        string)
            DSL=string
            TEST_NAME=sygus_strings
            TEST_FILE=string/sygus_strings.pickle
            BASE_FILE=$TEST_FILE
            MODEL_FLAGS="--b 8 --max-depth -1 --ngram 1"
            SOLVER=naive
            SHARDS=${STRING_SHARDS:-10}
            ;;
        deepcoder)
            DSL=deepcoder
            TEST_NAME=test_deepcoder
            TEST_FILE=deepcoder/test_deepcoder.pickle
            BASE_FILE=deepcoder/deepcoder.pickle
            MODEL_FLAGS="--b 8 --max-depth -1 --ngram 2"
            SOLVER=cutoff
            SHARDS=${DEEPCODER_SHARDS:-40}
            ;;
        *) echo "unknown domain $1" >&2; exit 1 ;;
    esac
}

# model name for a seed and a loss (mse is the recipe of the submitted version)
model_name() {
    local seed=$1 loss=${2:-mse}
    if [ "$loss" = mse ]; then echo "seed_${seed}"; else echo "seed_${seed}_${loss}"; fi
}

data() {
    local bee=$1 dc=$2
    mkdir -p string deepcoder
    [ -s string/sygus_strings.pickle ] || \
        $PYTHON -m examples.pbe.sygus.convert_sygus_string "$bee" -o string/sygus_strings.pickle -v
    [ -s deepcoder/deepcoder.pickle ] || \
        $PYTHON examples/pbe/deepcoder/convert_deepcoder.py "$dc/T=3_train.json" -o deepcoder/deepcoder.pickle
    [ -s deepcoder/test_deepcoder.pickle ] || \
        $PYTHON examples/pbe/deepcoder/convert_deepcoder.py "$dc"/T={1,2,3,4,5}_test.json -o deepcoder/test_deepcoder.pickle
}

train() {
    local domain=$1 seed=$2 loss=${3:-mse}
    config "$domain"
    local train_file=$DSL/train_${DSL}_seed_${seed}.pickle
    local model; model=$(model_name "$seed" "$loss")
    local model_file=$DSL/$model.pt
    mkdir -p "$DSL/logs"
    if [ ! -s "$train_file" ]; then
        $PYTHON examples/pbe/dataset_generator_unique.py --dsl "$DSL" --dataset "$BASE_FILE" \
            -o "$train_file.tmp" --seed "$seed" $GEN_TAGS > "$DSL/logs/gen_seed_${seed}.log" 2>&1
        mv "$train_file.tmp" "$train_file"
    fi
    if [ ! -s "$model_file" ]; then
        $PYTHON examples/pbe/model_trainer.py --dsl "$DSL" --dataset "$train_file" --seed "$seed" \
            -o "$model_file.tmp" $MODEL_FLAGS -e "$EPOCHS" --loss "$loss" > "$DSL/logs/train_${model}.log" 2>&1
        mv "$model_file.tmp" "$model_file"
    fi
    if [ ! -s "$DSL/audit_${model}.csv" ]; then
        $PYTHON examples/pbe/model_audit.py --dsl "$DSL" --dataset "$TEST_FILE" --model "$model_file" \
            --support "$train_file" $MODEL_FLAGS -o "$DSL/audit_${model}.csv" > "$DSL/logs/audit_${model}.log" 2>&1
    fi
    if [ ! -s "$DSL/pcfgs_${TEST_NAME}_${model}.pickle" ]; then
        $PYTHON examples/pbe/model_prediction.py --dsl "$DSL" --dataset "$TEST_FILE" --model "$model_file" \
            --support "$train_file" $MODEL_FLAGS > "$DSL/logs/predict_${model}.log" 2>&1
    fi
}

solve() {
    local domain=$1 seed=$2 algo=$3 shard=$4 loss=${5:-mse}
    config "$domain"
    local train_file=$DSL/train_${DSL}_seed_${seed}.pickle
    local model; model=$(model_name "$seed" "$loss")
    local out=$DSL/results
    local done_file="$out/shards/.done_${model}_${algo}_${shard}of${SHARDS}"
    if [ -e "$done_file" ]; then return 0; fi
    mkdir -p "$out/shards" "$DSL/logs/solve"
    $PYTHON examples/pbe/solve.py --dsl "$DSL" --dataset "$TEST_FILE" -o "$out" --support "$train_file" \
        --pcfg "$DSL/pcfgs_${TEST_NAME}_${model}.pickle" --solver "$SOLVER" --search "$algo" \
        -t "$TIMEOUT" --pruning obs-eq --memory --max-memory "$MEMORY" --shard "$shard/$SHARDS" \
        > "$DSL/logs/solve/${model}_${algo}_${shard}of${SHARDS}.log" 2>&1
    touch "$done_file"
}

jobs_train() {
    for domain in $DOMAINS; do
        for seed in $SEEDS; do echo "$domain $seed"; done
    done
}

jobs_solve() {
    for domain in $DOMAINS; do
        config "$domain"
        for seed in $SEEDS; do
            for algo in $ALGOS; do
                for ((shard = 0; shard < SHARDS; shard++)); do
                    echo "$domain $seed $algo $shard"
                done
            done
        done
    done
}

merge() {
    for domain in $DOMAINS; do
        config "$domain"
        [ -d "$DSL/results/shards" ] && $PYTHON examples/pbe/merge_solve_shards.py "$DSL/results"
    done
}

cmd=$1; shift
case "$cmd" in
    data) data "$@" ;;
    train) train "$@" ;;
    solve) solve "$@" ;;
    jobs-train) jobs_train ;;
    jobs-solve) jobs_solve ;;
    merge) merge ;;
    *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
