# JAIR revision: code changes and checks

The code changes are on the branch `journal_experiments`, based on `main`.
The same diff applies cleanly to `eco_search_aaai`: the enumeration code, `solve.py` and `plot_helper.py` are identical on both branches.

Not available on this machine: `JAIR/` (paper, response letter), `lean/`, the FlashFill and DeepCoder datasets, the trained models.
So no paper edits, and no rerun of Figures 2-6. The runs below use the experiment DSLs with random PCFGs as a proxy for the trained ones.

## 1. Heap Search

### Two bugs, both in `HSEnumerator` (`heap_search.py`)

1. **Order of pop and expansion** (the one of the TODO). Fixed as proposed: `query(S, p)` expands `p`, then pops.
   The bare `except` is replaced by emptiness tests, `if program:` by `is not None` (also in `__add_successors__`).
   The deleted-program loop keeps its expansion. The popped program is no longer expanded.
   `BucketSearch` inherits the fix. `u_heap_search.py` (`UHSEnumerator.query`) had the same old order and is fixed the same way.
2. **Initialisation (new, not in the TODO).** The heap of `S` must start with `f(max(S1), ..., max(Sn))` for each rule.
   With cycles, `max_priority[(S, P)]` was frozen at its first value: `__init_non_terminal__` returns as soon as every rule of `S` has an entry, so `_reevaluate_` never improved it.
   Example: `T0 -> f0(T1,T1) 4 | f1(T1) 3 | f2 2`, `T1 -> f3(T0) 1 | f4 5 | f5 4`: the heap of `T0` starts with `f1(f5)` (cost 7) instead of `f1(f3(f2))` (cost 6), which is then never output.
   In addition, the recursion of `__init_non_terminal__` recomputes non-terminals an exponential number of times with cycles: on DeepCoder (`ngram 2`, 48 non-terminals) **the first program never comes (no output in 600 s), with the old code**.
   This is probably why `run_exp_deepcoder.sh` on `eco_search_aaai` has `METHODS="beap_search bee_search cd_search"`, without `heap_search`.
   Fix: `_reevaluate_` computes the best programs as a fixpoint over all non-terminals, without the recursion. On DeepCoder the first program now comes after 0.03 s, 10^5 programs in 2.1 s.

### Tests (1.3)

**Counterexample** `W -> k(U) 1, U -> u1 4, U -> q(W) 0.5`: the old code outputs only `k(u1)` from `W` and stops (from `U`: `u1`, `q(k(u1))`, stops). Even worse than the TODO predicts: the enumeration stops, not only misses `q(k(q(k(u1))))`. The new code, Eco, Bee and Beap output exactly the brute-force programs.

**Random grammars**, 2-3 non-terminals with cycles, all programs of cost <= 12 compared with brute force (300 grammars, costs 1-5; and 300 with costs in 10^4..10^5, bound 250000, as with the `-log p * 10^5` costs of the experiments):

| algorithm | wrong (small costs) | wrong (large costs) |
|---|---|---|
| old Heap Search | 149 / 300 (50%) | 143 / 300 |
| old order + fixed init | 148 | 139 |
| new order + old init | 1 | 5 |
| **new Heap Search** | **0** | **0** |
| Eco Search (k=5 and k=20) | 0 (after fixing a crash, see 5) | 0 |
| Bee Search | 0 wrong; stops yielding (timeout) on 8 grammars, in the one I inspected the language is finite | 0 wrong, 13 timeouts |
| Beap Search (Eco w/o buckets) | 0 | 0 |

The Lean helper found about 2% for the old order; here it is about 50%. The difference may come from the grammar distribution (here every grammar has cycles, arities up to 2), worth checking.

**Experiment grammars** (first 10^5 programs, random PCFGs: `ProbDetGrammar.random`, as in `run_exp_enumeration.sh`). "Wrong" = misses programs output by new Heap Search and Eco below the common cost bound, or outputs them out of order.

| grammar | seeds | old Heap Search | new Heap Search vs Eco |
|---|---|---|---|
| D_4 | 1-5 | correct on 5/5 (as you expected) | agree |
| N_4 | 1-5 | **wrong on 4/5**: seed 1 out of order (6764 inversions) and misses 96127 programs, first one `(cast4 (+4 1))` of cost 3.36; seed 2 stops after 5 programs; seeds 3, 4 miss 97593 and 80955 | agree |
| R_4 | 1-5 | **wrong on 2/5** (seeds 2, 3: miss 52209 and 72255, from cost 4.1 and 4.3) | agree |
| FlashFill DSL (`ngram 1`), `string -> string`, `string -> string -> string` | 1-3 each | **wrong on 1/6** (misses 9184, from cost 12.7) | agree |
| DeepCoder DSL (`ngram 2`), `int -> int list -> int list`, `int list -> int` | 1-3 each | **does not start** (init); old order with the new init: wrong on 5/6 (misses 13745 to 46285, from cost 7.5) | agree |

So the published Heap Search curves are affected on N_k, R_k, and probably FlashFill; Section 1.4 is needed.
For DeepCoder, if Figure 3 has a Heap Search curve, check where it comes from: with this code Heap Search cannot even initialise on these grammars.

### Pre-existing, not fixed
- Heap Search on TTCFGs (`TTCFG.size_constraint`, test `test_unicity_heapSearch[cfg1]`, depends on the hash seed): replacing argument `i` changes the non-terminals of the following arguments, the new program can be invalid, `compute_priority` raises `KeyError`, and the successor and everything below it are dropped. Fails as often before and after. Not used in the paper (all grammars are CFGs).
- `solve.py` writes `solved=False` even for solved tasks (the `send(True)` raises `StopIteration`); the plots use `len(solution) > 1`, so this does not matter.

## 2. Ablations

- **Bee Search with bucket queues**: `BeeSearchBuckets` in `bee_search.py`, `--search bee_search_buckets` in `solve.py` (k=20).
  A single `CDQueue` for all non-terminals: Bee pushes delayed combinations whose cost can be below the cheapest queued combination of their own non-terminal, which a per-non-terminal bucket queue cannot hold. The window is not known beforehand: it doubles when needed (see 5). The initial combinations are pushed in increasing order.
- **Eco Search with eager expansion**: `CDSearchEager` in `constant_delay.py`, `--search cd_search_eager`.
  Doing literally what the TODO says (calling `query(S, l+1)` at the end of `query(S, l)`) does not terminate, and even one level ahead it does not terminate if the levels requested by an eager generation are expanded eagerly too: on the counterexample, generating `W_1` requests `U_1`, whose next level `U_2` requests `W_1`, whose next level `W_2`... This is the cascade of Section 4.3, worth one sentence in Section 4.7.
  It is also incorrect if done inside the generation of a level (it can read a level of another non-terminal that is being generated).
  Implemented: every level requested by the frugal algorithm (start levels, argument levels, derivation levels) has its next level generated, after the current level of the start symbol, and without cascade. Counter `programs_beyond_current_cost`.
- Both agree with brute force on 100 random cyclic grammars.
- Throughput on the proxies (first 10^5 programs, seed 1, this Mac, not the Xeon):

| grammar | Eco | Eco eager | Bee | Bee buckets |
|---|---|---|---|---|
| D_4 | 0.14 s | 0.19 s | 0.20 s | 0.14 s |
| N_4 | 0.21 s | 0.27 s | 0.50 s | 0.51 s |
| R_4 | 0.41 s | 0.49 s | timeout (11256 programs in 300 s) | timeout (44415 programs in 300 s) |
| FlashFill DSL | 0.25 s | 0.28 s | 10.5 s | 7.0 s |
| DeepCoder `int -> int list -> int list` | 1.36 s | 2.12 s | timeout (81543 programs in 300 s) | 242 s |
| DeepCoder `int list -> int` | 1.57 s | 2.51 s | 234 s | 182 s |

  Eco eager: almost every output program was generated one level before it was needed (~10^5 "beyond the current cost" at generation time), but at the end of the run it has generated only a few dozen programs more than Eco (D_4: 98, FlashFill: 37, DeepCoder: 0-1). Cost: 1.1x to 1.6x the time.
  Bee buckets: as fast as Bee to 4x faster (R_4: 44415 programs in 300 s against 11256), still one to two orders of magnitude slower than Eco. Bee generates 2x (FlashFill: 3.7x) more programs than Eco, and on DeepCoder 4x10^6 programs in its banks for 10^5 output, against 2x10^5 for Eco.
- To do on the Xeon: FlashFill and DeepCoder, five seeds, timeout 300 s, `--search bee_search_buckets` and `--search cd_search_eager`.

## 3. Peak memory

- `solve.py --memory`: each task in a forked child process; the child reports its peak RSS (`ru_maxrss`, converted to kB on macOS) and its RSS when it started. Columns `peak_memory_kb`, `baseline_memory_kb`, `memory_error` (`MemoryError`, or `killed (exit code -9)` for the OOM killer). Such a run is a failure. On Linux a forked child starts with the RSS of the parent: subtract `baseline_memory_kb`.
- Tested on a small DeepCoder dataset with `cd_search`, `heap_search`, `bee_search_buckets`, `cd_search_eager`.
- Sanity check of Section 4.6 (cost tuples pushed <= #rules + m x #programs generated), first 10^5 programs, seed 1:

| grammar | rules | m | programs generated | cost tuples pushed | bound |
|---|---|---|---|---|---|
| D_4 | 11 | 2 | 100000 | 13421 | 200011 |
| N_4 | 11 | 1 | 200001 | 6911 | 200012 |
| R_4 | 13 | 3 | 200030 | 43056 | 600103 |
| FlashFill DSL (1) | 26 | 3 | 105863 | 31199 | 317615 |
| FlashFill DSL (2) | 27 | 3 | 105049 | 37295 | 315174 |
| DeepCoder (1) | 1341 | 2 | 199198 | 306275 | 399737 |
| DeepCoder (2) | 1320 | 2 | 227363 | 353197 | 456046 |

## 4. Figures

`plot_helper.py`: each algorithm has a fixed color, line style, marker (and hatch for the bar plots), the same in all figures, whatever its name on `main` or `eco_search_aaai` (`cd_search`/`eco search`, `beap_search`/`eco search w/o buckets`, `cd4`/`eco search (bucket size= 4 )`, ...), including the two ablations. Markers every 1/8 of the curve, lighter confidence bands. Log axes are labelled "(log scale)". Checked in grayscale on dummy data. The figures themselves need the result CSVs.

## 5. Eco Search implementation

1. **Per-non-terminal queues are binary heaps**: confirmed. `_queue_nt[S]` is a `heapq` list of `Derivation`, only `_queue_derivation` (one per tuple of argument non-terminals, shared by rules with the same arguments) are `CDQueue`. But `_queue_nt[S]` holds at most one entry per rule of `S` (popping `(P, i)` pushes `(P, i+1)`), so its operations take O(log |rules(S)|), a constant of the grammar. Suggest describing this in Section 5 rather than changing the code.
2. **The window is heuristic**: confirmed, and worse: `self.M` is not used in the end, `__compute_bounds__` replaces it by the spread of the initial costs of each non-terminal, propagated to the arguments (k = M buckets if M <= 1000). On random grammars with large costs, it was **too small in 1 run out of 202**, and the `assert` fired (`AssertionError`). Implemented the doubling strategy in `CDQueue.push` (same check as the assert, so no cost when the window is large enough; independent of `python -O`). Also found two bugs in `CDQueue`:
   - an element at distance exactly `maxi` landed in the current cell (the assert used `<=`);
   - **floating point**: the window start `mini = start + maxi * n / k` drifts, an element on the upper boundary of a cell can be pushed into the previous cell with an offset equal to the width, its sub-cell index is then `k` and wraps to 0: it is popped too early. Found with a randomized test (about 1% of the traces with non-integer cell widths, as with k=20 and the experiment costs). Fixed by clamping the index.
   - a push below the window start (does not happen in Eco, happens with the initialisation of Bee) now moves the window instead of misplacing the element.
   - `M = 0` (all rules with the same cost, e.g. a uniform PCFG with the same number of rules per non-terminal) crashed with `ZeroDivisionError`: the window is now at least 1.
3. **Duplicate tuples**: the rule implemented is "the parent of a tuple is obtained by decrementing its first non-zero coordinate" (a tuple pushes `t + e_i` for all `i` up to its first non-zero coordinate). Every tuple has exactly one parent, of smaller or equal cost, so every valid tuple keeps an ancestor in the queue. The "finite" case (`combination[i] + 1 >= len(cl)`) is consistent with this rule. Also checked by brute force (above).
4. `CDQueue` merges tuples whose costs differ by at most 1 (`abs(val.cost - element.cost) > 1` to split): with the costs `-log p * 10^5`, the order is exact up to 10^-5 per merge. Worth a sentence in Section 5 if not already there.

## 6. Paper

Not accessible here. On `experiments.tex` line 108 vs 140: the scripts use `-t 300` for FlashFill and DeepCoder (`EVAL_TAGS="-t 300 ..."`) and `TIME="120"` per grammar for the enumeration experiments (`run_exp_enumeration.sh`).

## Proposed text (to adapt)

Response letter, Lean item:
> The implementation of Heap Search used in the submitted version had the order of the original presentation (pop, then expand the popped program), which is incorrect for grammars with cycles between non-terminals, as well as a second issue in its initialisation on such grammars. On the grammars of the experiments, it agreed with Eco Search on D_k (a single non-terminal) but missed programs on N_k, R_k and FlashFill, and could not initialise on DeepCoder. We corrected both issues, checked the corrected implementation against a brute-force enumeration and against Eco Search on all the grammars of the experiments, and reran the Heap Search experiments with it.

`enumeration_algorithms.tex`, "Comparison with the original presentation":
> The experiments of Section 5 use the corrected order.
