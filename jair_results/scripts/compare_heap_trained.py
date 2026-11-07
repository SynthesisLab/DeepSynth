"""Old Heap Search (main) vs corrected Heap Search vs Eco Search on trained PCFGs.

usage: python jair_results/scripts/compare_heap_trained.py <dir with old_heap_search.py> <dataset> <pcfgs> <n>
where old_heap_search.py = git show origin/main:synth/syntax/grammars/enumeration/heap_search.py
"""
import sys, math, itertools
sys.path.insert(0, sys.argv[1])
import old_heap_search
from synth import Dataset
from synth.utils import load_object
from synth.specification import PBEWithConstants
from synth.syntax.grammars.enumeration.heap_search import enumerate_prob_grammar as new_hs
from synth.syntax.grammars.enumeration.constant_delay import enumerate_prob_grammar as cd

dataset = Dataset.load(sys.argv[2])
pcfgs = load_object(sys.argv[3])
N = int(sys.argv[4])
summary = {"old": 0, "new": 0}
seen_tr = {}
for task, pcfg in zip(dataset.tasks, pcfgs):
    # a few tasks per type request
    if seen_tr.get(task.type_request, 0) >= 3:
        continue
    seen_tr[task.type_request] = seen_tr.get(task.type_request, 0) + 1
    if isinstance(task.specification, PBEWithConstants):
        pcfg = pcfg.instantiate_constants(task.specification.constants)
    outs = {}
    for name, f in [("eco", lambda g: cd(g, 20)), ("new", new_hs), ("old", old_heap_search.enumerate_prob_grammar)]:
        progs = []
        for p in itertools.islice(f(pcfg).generator(), N):
            progs.append((p, pcfg.probability(p)))
        outs[name] = progs
    # common bound: the smallest of the last probabilities output
    bound = max(min(pr for _, pr in v) for v in outs.values() if v)
    sets = {k: {p for p, pr in v if pr > bound * (1 + 1e-9)} for k, v in outs.items()}
    line = []
    for k in ["new", "old"]:
        missing = len(sets["eco"] - sets[k])
        inversions = sum(1 for a, b in zip(outs[k], outs[k][1:]) if b[1] > a[1] * (1 + 1e-9))
        wrong = missing > 0 or inversions > 0 or len(outs[k]) < N
        summary[k] += wrong
        line.append(f"{k}: out={len(outs[k])} missing={missing} inversions={inversions}")
    print(task.metadata.get("name"), task.type_request, "|", " | ".join(line), flush=True)
print("tasks:", sum(seen_tr.values()), "wrong:", summary)
