"""
Merge the shards written by solve.py --shard I/N (in <folder>/shards/) into
the file solve.py writes without --shard (in <folder>/), with the tasks in
their original order.

    python examples/pbe/merge_solve_shards.py <folder>
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List

parser = argparse.ArgumentParser(description="Merge the shards of solve.py")
parser.add_argument("folder", type=str, help="output folder of solve.py")
parameters = parser.parse_args()

shard_folder = os.path.join(parameters.folder, "shards")
pattern = re.compile(r"^(.*)\.(\d+)of(\d+)\.csv$")
groups: Dict[str, Dict[int, str]] = defaultdict(dict)
counts: Dict[str, int] = {}
for file in os.listdir(shard_folder):
    match = pattern.match(file)
    if match is None:
        continue
    base, index, count = match.group(1), int(match.group(2)), int(match.group(3))
    groups[base][index] = os.path.join(shard_folder, file)
    assert counts.setdefault(base, count) == count, f"{base}: different shard counts"

for base in sorted(groups):
    count = counts[base]
    missing = [i for i in range(count) if i not in groups[base]]
    if missing:
        print(f"{base}: skipped, missing shards {missing}")
        continue
    shards: List[List[List[str]]] = []
    header = None
    for i in range(count):
        with open(groups[base][i]) as fd:
            rows = list(csv.reader(fd))
        if header is None:
            header = rows[0]
        assert rows[0] == header, f"{base}: shard {i} has a different header"
        shards.append(rows[1:])
    merged = [header]
    # task k of shard i is task i + k * count
    for k in range(max(len(rows) for rows in shards)):
        for i in range(count):
            if k < len(shards[i]):
                merged.append(shards[i][k])
    sizes = [len(rows) for rows in shards]
    with open(os.path.join(parameters.folder, base + ".csv"), "w") as fd:
        csv.writer(fd).writerows(merged)
    print(f"{base}: {len(merged) - 1} tasks (shards: {min(sizes)}-{max(sizes)} tasks)")
