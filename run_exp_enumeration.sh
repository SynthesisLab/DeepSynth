


SEEDS="1 2 3 4 5"
mkdir -p enumeration
touch enumeration/results_detailed_distance.csv
touch enumeration/results_growth_distance.csv
TIME="120"
# DO 22 for beap, CD (REMOVE HEAP AND BEE)
# for seed in $SEEDS; do
#     file="enumeration/results_${seed}_distance_detailed.csv"
#     if [ ! -f $file ]; then
#         python examples/compare_enumeration.py 1000000000 22 -s $seed -t $TIME -o "enumeration/results_${seed}_distance.csv"
#     fi
#     tail -n +2 "enumeration/results_${seed}_distance_detailed.csv" >> enumeration/results_distance_detailed.csv
#     tail -n +2 "enumeration/results_${seed}_distance_growth.csv" >> enumeration/results_distance_growth.csv
# done

# # DO 10 for heap (STILL BEE REMOVED)
for seed in $SEEDS; do
    file="enumeration/results_slow_${seed}_distance_detailed.csv"
    if [ ! -f $file ]; then
        python examples/compare_enumeration.py 1000000000 10 distance -s $seed -t $TIME -o "enumeration/results_slow_${seed}_distance.csv"
    fi
    tail -n +2 "enumeration/results_slow_${seed}_distance_detailed.csv" >> enumeration/results_distance_detailed.csv
    tail -n +2 "enumeration/results_slow_${seed}_distance_growth.csv" >> enumeration/results_distance_growth.csv
done


# for seed in $SEEDS; do
#     file="enumeration/results_${seed}_nonterminals_detailed.csv"
#     if [ ! -f $file ]; then
#         python examples/compare_enumeration.py 1000000000 20 nonterminals -s $seed -t $TIME -o "enumeration/results_${seed}_nonterminals.csv"
#     fi
#     tail -n +2 "enumeration/results_${seed}_nonterminals_detailed.csv" >> enumeration/results_nonterminals_detailed.csv
#     tail -n +2 "enumeration/results_${seed}_nonterminals_growth.csv" >> enumeration/results_nonterminals_growth.csv
# done

# for seed in $SEEDS; do
#     file="enumeration/results_${seed}_derivations_detailed.csv"
#     if [ ! -f $file ]; then
#         python examples/compare_enumeration.py 1000000000 20 derivations -s $seed -t $TIME -o "enumeration/results_${seed}_derivations.csv"
#     fi
#     tail -n +2 "enumeration/results_${seed}_derivations_detailed.csv" >> enumeration/results_derivations_detailed.csv
#     tail -n +2 "enumeration/results_${seed}_derivations_growth.csv" >> enumeration/results_derivations_growth.csv
# done