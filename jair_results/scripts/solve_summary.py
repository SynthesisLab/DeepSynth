"""Summary of the solve runs: tasks solved, time, peak memory, per algorithm and seed.

usage: python jair_results/scripts/solve_summary.py jair_results/solve
Tasks dropped by solve.py (type request not in the training set) count as failures.
"""
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

TOTAL = {"string": 205, "deepcoder": 792}
NAMES = {
    "cd_search": "Eco Search (k=20)",
    "cd_search_eager": "Eco Search, eager expansion",
    "beap_search": "Eco Search w/o buckets",
    "bee_search": "Bee Search",
    "bee_search_buckets": "Bee Search with buckets",
    "heap_search": "Heap Search (corrected)",
}
ORDER = list(NAMES)
folder = sys.argv[1]
for domain in ["string", "deepcoder"]:
    data = defaultdict(dict)
    for f in glob.glob(os.path.join(folder, domain, "*.csv")):
        base = os.path.basename(f)
        algo = next(a for a in sorted(NAMES, key=len, reverse=True) if f"_{a}_seed_" in base)
        seed = int(base.split("_seed_")[1].split("_")[0])
        rows = list(csv.DictReader(open(f)))
        solved = [r for r in rows if len(r["solution"]) > 1]
        peaks = [
            (int(r["peak_memory_kb"]) - int(r["baseline_memory_kb"])) / 1024**2
            for r in rows
            if r["peak_memory_kb"] and r["baseline_memory_kb"]
        ]
        data[algo][seed] = dict(
            solved=len(solved),
            rows=len(rows),
            time_solved=sum(float(r["time"]) for r in solved),
            mean_peak=st.mean(peaks),
            max_peak=max(peaks),
            memory_failures=sum(1 for r in rows if r["memory_error"]),
        )
    seeds = sorted({s for d in data.values() for s in d})
    total = TOTAL[domain]
    print(f"\n### {domain}: tasks solved out of {total} (seeds {', '.join(map(str, seeds))})\n")
    print("| algorithm | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | time on solved (s, mean) | peak memory mean / max (GB) | memory failures |")
    print("|---" * (len(seeds) + 5) + "|")
    for algo in ORDER:
        d = data[algo]
        solved = [d[s]["solved"] for s in seeds]
        dropped = sum(total - d[s]["rows"] for s in seeds)
        print(
            f"| {NAMES[algo]} | "
            + " | ".join(str(x) for x in solved)
            + f" | {st.mean(solved):.1f} | {st.mean(d[s]['time_solved'] for s in seeds):.0f}"
            + f" | {st.mean(d[s]['mean_peak'] for s in seeds):.2f} / {max(d[s]['max_peak'] for s in seeds):.1f}"
            + f" | {sum(d[s]['memory_failures'] for s in seeds)} |"
        )
    print(f"\n(dropped tasks counted as failures: {sum(total - data[ORDER[0]][s]['rows'] for s in seeds)} per algorithm)")
