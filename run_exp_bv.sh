#!/usr/bin/bash
# ============================================================
# PARAMETERS ==================================================
# ============================================================
DSL="bitvectors"
TEST_FILENAME="sygus_bitvectors"
TEST_FILE="./$DSL/$TEST_FILENAME.pickle"
METHODS="beap_search heap_search bee_search cd_search"
SOLVERS="cutoff"
# ============================================================
# FLAGS =======================================================
# ============================================================
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


# Make folder
if [ ! -d "./$DSL" ]; then
    mkdir "./$DSL"
fi
for solver in $SOLVERS
    do
        for method in $METHODS
        do
            echo "  solver: $solver search: $method"
            python examples/pbe/solve.py --dsl $DSL --dataset $TEST_FILE -o "./$DSL" --solver $solver --search $method ${EVAL_TAGS} &
            abort_on_failure
        done
    done
    wait
done