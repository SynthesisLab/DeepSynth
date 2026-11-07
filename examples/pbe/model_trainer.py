from typing import List
import sys
import os
import random

import tqdm

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL
from model_loader import (
    add_model_choice_arg,
    instantiate_predictor,
)


from synth import Dataset, PBE, Task
from synth.nn import print_model_summary
from synth.syntax import CFG, UCFG
from synth.utils import chrono
from synth.filter import add_dfta_constraints

DREAMCODER = "dreamcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"


import argparse

parser = argparse.ArgumentParser(description="Evaluate model prediction")
add_dataset_choice_arg(parser)
add_dsl_choice_arg(parser)
add_model_choice_arg(parser)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="seed_{seed}_model.pt",
    help="model file name, should respect format 'seed_X_Y' where X is the seed and Y is the name of the model (default: seed_{seed}_model.pt)",
)

parser.add_argument(
    "--no-clean",
    action="store_true",
    default=False,
    help="do not delete intermediary model files",
)
parser.add_argument(
    "--no-stats",
    action="store_true",
    default=False,
    help="do not produce stats increasing speed",
)
g = parser.add_argument_group("training parameters")
g.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=16,
    help="batch size to compute PCFGs (default: 16)",
)
g.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=1,
    help="number of epochs (default: 1)",
)
g.add_argument(
    "--no-shuffle",
    action="store_true",
    default=False,
    help="do not shuffle dataset between epochs",
)
g.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=1e-3,
    help="learning rate (default: 1e-3)",
)
g.add_argument(
    "-wd",
    "--weight-decay",
    type=float,
    default=1e-4,
    help="weight decay (default: 1e-4)",
)
g.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
g.add_argument(
    "--loss",
    choices=["mse", "nll"],
    default="mse",
    help="mse: squared error between the log-probabilities and log(1e-5 + indicator of the rules used); nll: negative log-likelihood of the solution, normalised by its size (default: mse)",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
seed: int = parameters.seed
output_file: str = parameters.output.format(seed=seed)
batch_size: int = parameters.batch_size
epochs: int = parameters.epochs
lr: float = parameters.learning_rate
weight_decay: float = parameters.weight_decay
cpu_only: bool = parameters.cpu
no_clean: bool = parameters.no_clean
no_shuffle: bool = parameters.no_shuffle
no_stats: bool = parameters.no_stats
loss_name: str = parameters.loss
constrained: bool = parameters.constrained
max_depth: int = parameters.max_depth
ngram: int = parameters.ngram

random.seed(seed)
torch.manual_seed(seed)
# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
dsl_constant_types = set()
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon
if dsl_name == DREAMCODER:
    max_list_length = 10
elif dsl_name == REGEXP:
    max_list_length = 10

if hasattr(dsl_module, "constant_types"):
    dsl_constant_types = dsl_module.constant_types
constraints = []
if hasattr(dsl_module, "constraints"):
    constraints = dsl_module.constraints
else:
    constrained = False

# ================================
# Load dataset & Task Generator
# ================================

if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)
# Load dataset

full_dataset: Dataset[PBE] = load_dataset(dsl_name, dataset_file)


# ================================
# Misc init
# ================================
# Get device
device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Logging
writer = SummaryWriter()
# ================================
# Neural Network creation
# ================================
# Generate the CFG dictionnary
all_type_requests = full_dataset.type_requests()
print("max depth:", max_depth)
cfgs = [
    CFG.depth_constraint(
        dsl,
        t,
        max_depth,
        constant_types=dsl_constant_types,
        min_variable_depth=0,
        n_gram=ngram,
    )
    for t in all_type_requests
]
type2cfg = {
    cfg.type_request: UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, constraints, progress=False), ngram
    )
    if constrained
    else cfg
    for cfg in cfgs
}
cfgs = list(type2cfg.values())
print(f"{len(all_type_requests)} type requests supported.")
print(f"Lexicon: [{min(lexicon)};{max(lexicon)}]")


predictor = instantiate_predictor(parameters, cfgs, lexicon).to(device)
print_model_summary(predictor)
optim = torch.optim.AdamW(predictor.parameters(), lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min")
dataset_index = 0
losses: List[float] = []


@chrono.clock(prefix="train.do_batch")
def get_batch_of_tasks() -> List[Task[PBE]]:
    global dataset_index
    batch = full_dataset[dataset_index : dataset_index + batch_size]
    dataset_index += batch_size
    copy = []
    for task in batch:
        if task.solution is not None:
            copy.append(task)
    return copy


def do_batch(iter_number: int) -> None:
    batch = get_batch_of_tasks()
    batch_programs = [task.solution for task in batch]
    batch_tr = [task.type_request for task in batch]
    # Logging
    if all([p is not None for p in batch_programs]):
        writer.add_scalar(
            "program/depth", np.mean([p.depth() for p in batch_programs]), iter_number
        )
        mean_length = np.mean([p.size() for p in batch_programs])
        writer.add_scalar("program/length", mean_length, iter_number)
    with chrono.clock("train.do_batch.inference"):
        batch_outputs: Tensor = predictor(batch)

    # Gradient descent
    with chrono.clock("train.do_batch.loss"):
        optim.zero_grad()
        with chrono.clock("train.do_batch.loss.compute"):
            if loss_name == "mse":
                loss = predictor.bigram_layer.loss_mse(
                    batch_programs, batch_tr, batch_outputs
                )
            else:
                loss = predictor.bigram_layer.loss_negative_log_prob(
                    batch_programs,
                    [
                        predictor.bigram_layer.tensor2log_prob_grammar(
                            batch_outputs[i].clone(), tr
                        )
                        for i, tr in enumerate(batch_tr)
                    ],
                )
        with chrono.clock("train.do_batch.loss.backprop"):
            loss.backward()
            optim.step()
            # Should be called on val_loss but we don't have one here
            scheduler.step(loss.item())
    # Logging
    writer.add_scalar("train/loss", loss.item(), iter_number)
    writer.add_scalar("train/lr", optim.param_groups[0]["lr"], iter_number)
    losses.append(loss.item())
    if not no_stats:
        with chrono.clock("train.do_batch.stats"):
            with torch.no_grad():
                batch_logprobs = torch.stack(
                    [
                        predictor.bigram_layer.tensor2log_prob_grammar(
                            batch_outputs[i], task.type_request
                        ).log_probability(task.solution)
                        for i, task in enumerate(batch)
                    ]
                )
                writer.add_scalar(
                    "train/program_probability",
                    torch.mean(torch.exp(batch_logprobs)),
                    iter_number,
                )


def do_epoch(j: int) -> int:
    global dataset_index
    dataset_index = 0
    if not no_shuffle:
        random.shuffle(full_dataset.tasks)
    nb_batch_per_epoch = int(np.ceil(len(full_dataset) / batch_size))
    i = j
    for _ in tqdm.trange(nb_batch_per_epoch, desc="batchs"):
        do_batch(i)
        i += 1
    return i


def train() -> None:
    j = 0
    for ep in tqdm.trange(epochs, desc="epochs"):
        j = do_epoch(j)
        print(
            f"epoch {ep}: mean loss {np.mean(losses):.4f} (last 20 batches: {np.mean(losses[-20:]):.4f}), lr {optim.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        losses.clear()
        torch.save(predictor.state_dict(), f"{output_file}_epoch{ep}.tmp")


train()
torch.save(predictor.state_dict(), output_file)
if not no_clean:
    for ep in range(epochs):
        os.remove(f"{output_file}_epoch{ep}.tmp")
