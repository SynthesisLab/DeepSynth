"""
Audit of a trained model: does the predicted grammar make the solutions cheaper?

For each task with a known solution, compare the cost -log P(solution) under
- the grammar predicted by the model for this task,
- the uniform grammar,
- the mean of the grammars predicted for the tasks of the same type request
  (if it is as good as the predicted grammar, the model does not use the examples).
For tasks with constants (PBEWithConstants), the cost includes the choice of
each constant among the constants of the task, as in solve.py.

Also prints the coverage of the type requests of the dataset by the support
(the training set): tasks of a type request not in the support are dropped by
model_prediction.py and solve.py.
"""

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from model_loader import add_model_choice_arg, instantiate_predictor

from synth import Dataset, Task
from synth.specification import PBEWithConstants
from synth.syntax import CFG, ProbDetGrammar, Program, Type

parser = argparse.ArgumentParser(description="Audit a trained model")
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument("-m", "--model", type=str, required=True, help="model file")
add_model_choice_arg(parser)
parser.add_argument(
    "--support",
    type=str,
    default=None,
    help="train dataset to get the set of supported type requests",
)
parser.add_argument(
    "-b", "--batch-size", type=int, default=16, help="batch size (default: 16)"
)
parser.add_argument(
    "-o", "--output", type=str, default=None, help="CSV file with one row per task"
)
parameters = parser.parse_args()

dsl_module = load_DSL(parameters.dsl)
dsl, lexicon = dsl_module.dsl, dsl_module.lexicon
constant_types = getattr(dsl_module, "constant_types", set())
dataset: Dataset = load_dataset(parameters.dsl, parameters.dataset)
support: Optional[Dataset] = (
    Dataset.load(parameters.support) if parameters.support else None
)

# ================================
# Coverage of the type requests
# ================================
dataset_trs = dataset.type_requests()
supported = support.type_requests() if support is not None else dataset_trs
by_tr: Dict[Type, int] = defaultdict(int)
for task in dataset:
    by_tr[task.type_request] += 1
covered = sum(n for tr, n in by_tr.items() if tr in supported)
print(
    f"Coverage: {len([tr for tr in dataset_trs if tr in supported])}/{len(dataset_trs)} type requests, "
    f"{covered}/{len(dataset)} tasks"
)
for tr, n in sorted(by_tr.items(), key=lambda x: -x[1]):
    print(f"\t{'ok' if tr in supported else 'MISSING'}\t{n}\t{tr}")

# ================================
# Model
# ================================
cfgs = [
    CFG.depth_constraint(
        dsl,
        tr,
        parameters.max_depth,
        constant_types=constant_types,
        min_variable_depth=0,
        n_gram=parameters.ngram,
    )
    for tr in supported
]
device = "cuda" if not parameters.cpu and torch.cuda.is_available() else "cpu"
predictor = instantiate_predictor(parameters, cfgs, lexicon)
predictor.load_state_dict(torch.load(parameters.model, map_location=device))
predictor = predictor.to(device)
predictor.eval()
layer = predictor.bigram_layer
uniform = {cfg.type_request: ProbDetGrammar.uniform(cfg) for cfg in cfgs}

tasks: List[Task] = [
    t for t in dataset if t.solution is not None and t.type_request in supported
]
print(f"{len(tasks)} tasks with a solution and a supported type request")


def unassigned(program: Program) -> Program:
    """The solution with its constants reset, as derived by the grammar."""
    p = program.clone()
    for c in p.constants():
        if c is not None:
            c.reset()
    return p


def constants_cost(task: Task) -> float:
    if not isinstance(task.specification, PBEWithConstants):
        return 0
    cost = 0.0
    for c in task.solution.constants():
        if c is not None:
            cost += math.log(max(1, len(task.specification.constants[c.type])))
    return cost


# ================================
# Predictions
# ================================
outputs: List[torch.Tensor] = []
with torch.no_grad():
    for i in tqdm.trange(0, len(tasks), parameters.batch_size, desc="prediction"):
        batch = tasks[i : i + parameters.batch_size]
        out = predictor(batch)
        normed = torch.zeros_like(out)
        layer.__normalize__(out, normed)
        outputs += list(normed)

mean_by_tr: Dict[Type, torch.Tensor] = {}
for tr in set(t.type_request for t in tasks):
    probs = [
        torch.exp(o) for o, t in zip(outputs, tasks) if t.type_request == tr
    ]
    mean_by_tr[tr] = torch.log(torch.stack(probs).mean(dim=0))

rows = []
with torch.no_grad():
    for task, out in tqdm.tqdm(list(zip(tasks, outputs)), desc="costs"):
        sol = unassigned(task.solution)
        extra = constants_cost(task)
        tr = task.type_request
        pred = -float(layer.tensor2log_prob_grammar(out.clone(), tr).log_probability(sol))
        mean = -float(
            layer.tensor2log_prob_grammar(mean_by_tr[tr].clone(), tr).log_probability(sol)
        )
        p_uni = uniform[tr].probability(sol)
        uni = -math.log(p_uni) if p_uni > 0 else math.inf
        rows.append(
            {
                "name": task.metadata.get("name", ""),
                "type_request": str(tr),
                "size": task.solution.size(),
                "predicted": pred + extra,
                "mean_predicted": mean + extra,
                "uniform": uni + extra,
            }
        )


def summary(key: str) -> str:
    values = np.array([r[key] for r in rows])
    finite = values[np.isfinite(values)]
    return f"mean {np.mean(finite):.2f}, median {np.median(finite):.2f}, infinite {len(values) - len(finite)}"


print("Cost -log P(solution) (natural log):")
for key in ["predicted", "mean_predicted", "uniform"]:
    print(f"\t{key:15s} {summary(key)}")
n = len(rows)
if n > 0:
    better_uni = sum(1 for r in rows if r["predicted"] < r["uniform"])
    better_mean = sum(1 for r in rows if r["predicted"] < r["mean_predicted"])
    print(f"predicted cheaper than uniform on {better_uni}/{n} tasks")
    print(f"predicted cheaper than mean predicted on {better_mean}/{n} tasks")
if parameters.output:
    with open(parameters.output, "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
