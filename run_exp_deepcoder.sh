#!/usr/bin/bash
# ============================================================
# PARAMETERS ==================================================
# ============================================================
DSL="deepcoder"
TEST_FILENAME="test_deepcoder"
TEST_FILE="./$DSL/$TEST_FILENAME.pickle"
BASE_FILE="./$DSL/$DSL.pickle"
ALL_SEEDS="1 48 35132 849 98465" # 77984812 2798494 618421762 69871020" 
METHODS="beap_search bee_search cd_search"
SOLVERS="cutoff"
# ============================================================
# FLAGS =======================================================
# ============================================================
MODEL_FLAGS="--b 8 --max-depth -1 --ngram 2"
GEN_TAGS="--inputs 2 --programs 1000 --uniform"
TRAIN_TAGS="$MODEL_FLAGS -e 2"
EVAL_TAGS="-t 300 --pruning obs-eq"
# ============================================================
# CODE =======================================================
# ============================================================
function abort_on_failure(){
    out=$?
    if [ $out != 0 ]; then
        echo "An error has occured"
        exit 1
    fi
}
function gen_data(){
    dsl=$1
    seed=$2
    train_file=./$DSL/train_${dsl}_seed_${seed}.pickle
    if [ ! -f "$train_file" ]; then
        python examples/pbe/dataset_generator_unique.py --dsl $dsl --dataset $BASE_FILE -o $train_file --seed $seed $GEN_TAGS
        abort_on_failure
    fi
}

function do_exp(){
    dsl=$1
    seed=$2
    train_file=./$DSL/train_${dsl}_seed_${seed}.pickle
    model_name=seed_${seed}
    model_file="./$DSL/$model_name.pt"
    if   [ ! -f "$model_file" ]; then
        echo "Training model..."
        python examples/pbe/model_trainer.py --dsl $dsl --dataset $train_file --seed $seed -o $model_file $TRAIN_TAGS
        abort_on_failure
    fi
    echo "Predicting PCFGs..."
    python examples/pbe/model_prediction.py --dsl $dsl --dataset $TEST_FILE --model $model_file --support $train_file ${MODEL_FLAGS}
    abort_on_failure
    
    pcfg_file="./$DSL/pcfgs_${TEST_FILENAME}_$model_name.pickle"
    echo "Solve..."
    for solver in $SOLVERS
    do
        for method in $METHODS
        do
            echo "  solver: $solver search: $method"
            python examples/pbe/solve.py --dsl $dsl --dataset $TEST_FILE -o "./$dsl" --support $train_file --pcfg $pcfg_file --solver $solver --search $method ${EVAL_TAGS} &
            abort_on_failure
        done
    done
    wait
} 

# Make folder
if [ ! -d "./$DSL" ]; then
    mkdir "./$DSL"
fi
# Generate test dataset
if [ ! -f "$TEST_FILE" ]; then
    echo "Generating test file..."
    python examples/pbe/dataset_generator_unique.py --dsl $DSL --dataset $DSL.pickle -o $TEST_FILE --seed 2410 --inputs 2 --programs 100
    abort_on_failure
fi
# Generate train datasets
for my_seed in $ALL_SEEDS
do 
    echo "Generating traing for seed $my_seed"
    gen_data $DSL $my_seed
    echo "Running experiment for seed $my_seed"
    do_exp $DSL $my_seed ""
done
