import argparse
from collections import defaultdict
from typing import Any, Callable, Dict, Set, Tuple, List, Union

import tqdm

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL

from synth import Dataset, PBE
from synth.pbe.task_generator import TaskGenerator
from synth.semantic import DSLEvaluator
from synth.utils import chrono
from synth.syntax import CFG, Type, Program

DREAMCODER = "dreamcoder"
REGEXP = "regexp"
CALCULATOR = "calculator"
TRANSDUCTION = "transduction"


parser = argparse.ArgumentParser(
    description="Generate a dataset copying the original distribution of another dataset"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="dataset.pickle",
    help="output file (default: dataset.pickle)",
)
parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
parser.add_argument(
    "--programs", type=int, default=50, help="generated programs (default: 100)"
)
parser.add_argument(
    "--inputs", type=int, default=1, help="generated inputs (default: 1)"
)
parser.add_argument(
    "--max-depth", type=int, default=5, help="solutions max depth (default: 5)"
)
parser.add_argument(
    "--max-examples",
    type=int,
    default=5,
    help="max number of examples per task (default: 5)",
)
parser.add_argument(
    "--test-examples",
    type=int,
    default=0,
    help="number of test examples per task (default: 0)",
)
parser.add_argument(
    "--uniform", action="store_true", default=False, help="use uniform PCFGs"
)
parser.add_argument(
    "--constrained",
    action="store_true",
    default=False,
    help="tries to add constraints of the DSL to the grammar",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="verbose generation",
)
parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
output_file: str = parameters.output
seed: int = parameters.seed
max_depth: int = parameters.max_depth
max_examples: int = parameters.max_examples
test_examples: int = parameters.test_examples
nb_programs: int = parameters.programs
nb_inputs: int = parameters.inputs
uniform: bool = parameters.uniform
constrained: bool = parameters.constrained
verbose: bool = parameters.verbose

if constrained and not uniform:
    raise NotImplementedError(
        "Constrained grammars when uniform=False is currently not implemented!"
    )

# ================================
# Load constants specific to DSL
# ================================
max_list_length = None
dsl_module = load_DSL(dsl_name)
dsl, evaluator, lexicon = dsl_module.dsl, dsl_module.evaluator, dsl_module.lexicon

if hasattr(dsl_module, "reproduce_dataset"):
    reproduce_dataset = dsl_module.reproduce_dataset
else:
    from synth.pbe.task_generator import reproduce_int_dataset as reproduce_dataset

constraints = []
if hasattr(dsl_module, "constraints") and constrained:
    constraints = dsl_module.constraints

if dsl_name == DREAMCODER:
    max_list_length = 10

# ================================
# Load dataset & Task Generator
# ================================
# Load dataset
full_dataset: Dataset[PBE] = load_dataset(dsl_name, dataset_file)

# Reproduce dataset distribution
print("Reproducing dataset...", end="", flush=True)
with chrono.clock("dataset.reproduce") as c:
    task_generator, lexicon = reproduce_dataset(
        full_dataset,
        dsl,
        evaluator,
        seed,
        max_list_length=max_list_length,
        default_max_depth=max_depth,
        uniform_pgrammar=uniform,
        constraints=constraints,
        verbose=verbose,
    )
    cfgs = task_generator.type2pgrammar
    if constrained:
        cfgs = {
            t: CFG.depth_constraint(dsl, t, max_depth, min_variable_depth=0)
            for t in task_generator.type2pgrammar
        }
    print("done in", c.elapsed_time(), "s")
# Add some exceptions that are ignored during task generation
task_generator.skip_exceptions.add(TypeError)
task_generator.uniques = True
task_generator.verbose = True


def generate_programs_and_samples_for(
    tr: Type,
    nb_programs: int,
    nb_inputs: int,
    task_generator: TaskGenerator,
    threshold: int = 1000,
):
    # 3 Phases algorithm to generate m programs
    # Phase 1 generate n programs
    # Phase 2 generate k examples to differentiate as much as possible programs in n
    # Phase 3 try to generate new programs to get n as close as possible to m
    # Phase 1
    programs = set()
    for _ in tqdm.trange(nb_programs * 2, desc="1: generation"):
        prog, unique = task_generator.generate_program(tr)
        while not unique:
            prog, unique = task_generator.generate_program(tr)
        if isinstance(task_generator.evaluator, DSLEvaluator):
            prog = task_generator.evaluator.compress(prog)
        programs.add(prog)
    # Phase 2
    samples, equiv = generate_samples_for(
        programs,
        lambda: task_generator.sample_input(tr.arguments()),
        task_generator.eval_input,
        task_generator.evaluator.clear_cache,
        examples=max_examples,
        threshold=threshold,
    )
    # Phase 3
    pbar = tqdm.tqdm(total=nb_programs - len(equiv), desc="3: improvement")
    tries = 0
    while len(equiv) < nb_programs:
        prog, unique = task_generator.generate_program(tr)
        while not unique:
            prog, unique = task_generator.generate_program(tr)
        if isinstance(task_generator.evaluator, DSLEvaluator):
            prog = task_generator.evaluator.compress(prog)
        # Compute semantic hash
        cl = None
        has_none = False
        for x in samples:
            o = task_generator.eval_input(prog, x)
            if o is None:
                has_none = True
                break
            if isinstance(o, List):
                o = tuple(o)
            cl = (o, cl)
        # Check
        if cl not in equiv and not has_none:
            equiv[cl].append(prog)
            tries = 0
            pbar.update(1)
        else:
            tries += 1
            if tries > 1000:
                break
    pbar.close()
    rel_programs = [
        min([(p.size(), p) for p in v], key=lambda x: x[0])[1] for v in equiv.values()
    ]
    out = [samples]
    if nb_inputs > 1:
        for _ in tqdm.trange(nb_inputs - 1, desc="4: additional examples"):
            samples, equiv = generate_samples_for(
                rel_programs,
                lambda: task_generator.sample_input(tr.arguments()),
                task_generator.eval_input,
                task_generator.evaluator.clear_cache,
                examples=max_examples,
                threshold=-threshold,
            )
            if len(samples) == 0:
                samples, equiv = generate_samples_for(
                    rel_programs,
                    lambda: task_generator.sample_input(tr.arguments()),
                    task_generator.eval_input,
                    task_generator.evaluator.clear_cache,
                    examples=max_examples,
                    threshold=-threshold,
                )
                if len(samples) > 0:
                    out.append(samples)
            else:
                out.append(samples)

    if test_examples > 0:
        out = [
            x
            + [
                task_generator.sample_input(tr.arguments())
                for _ in range(test_examples)
            ]
            for x in out
        ]
    del task_generator.type2pgrammar[tr]
    return out, rel_programs


#
def generate_samples_for(
    programs: Union[List[Program], Set[Program]],
    input_sampler: Callable[[], Any],
    eval_prog: Callable[[Program, Any], Any],
    clear_cache: Callable,
    threshold: int = 1000,
    examples: int = 5,
) -> Tuple[List, Dict[Any, List[Program]]]:
    samples = []
    equiv_classes = {None: programs}
    nb_examples = 0
    nb_tested = 0
    pbar = tqdm.tqdm(total=examples * abs(threshold), desc="2: sem. unicity")
    best = None
    best_score = 0
    while nb_examples < examples:
        next_equiv_classes = defaultdict(list)
        clear_cache()
        thres_reached = nb_tested * nb_tested > threshold * threshold
        ui = best if thres_reached and best is not None else input_sampler()
        failed_ratio = 0
        for cl, prog in equiv_classes.items():
            for p in prog:
                o = eval_prog(p, ui)
                if not task_generator.output_validator(o):
                    failed_ratio += 1
                if isinstance(o, List):
                    o = tuple(o)
                next_equiv_classes[(o, cl)].append(p)
        ratio = len(programs) / len(equiv_classes)
        if len(next_equiv_classes) > best_score and failed_ratio / len(programs) < 0.2:
            best = ui
            best_score = len(next_equiv_classes)
        # Early stopping if no new examples is interesting
        if thres_reached and best_score == len(equiv_classes):
            break
        # If timeout or good example
        if thres_reached or (
            threshold > 0
            and len(next_equiv_classes) * (ratio ** (examples - nb_examples))
            >= len(programs) / 2
        ):
            nb_examples += 1
            nb_tested = 0
            pbar.update(1)
            equiv_classes = next_equiv_classes
            samples.append(ui)
            best_score = len(next_equiv_classes)
            pbar.n = nb_examples * abs(threshold)
            pbar.refresh()
        pbar.set_postfix_str(
            f"{len(equiv_classes)}->{best_score} | {best_score/len(programs):.0%}"
        )
        pbar.update(1)
        nb_tested += 1
    clear_cache()
    pbar.close()
    return samples, equiv_classes


#
print("Computing task type distribution...", end="", flush=True)
programs_by_tr = defaultdict(int)
with chrono.clock("dataset.generate.distribution") as c:
    for i in tqdm.trange(nb_programs):
        tr = task_generator.generate_type_request()
        programs_by_tr[tr] += 1
    print("done in", c.elapsed_time(), "s")
tasks = []

print("Generating tasks by type...", flush=True)
for tr, count in programs_by_tr.items():
    print("\t", tr)
    list_samples, programs = generate_programs_and_samples_for(
        tr, count, nb_inputs, task_generator
    )
    for samples in list_samples:
        for program in programs:
            tasks.append(
                task_generator.make_task(
                    tr,
                    program,
                    samples,
                    [task_generator.eval_input(program, x) for x in samples],
                    test_examples=test_examples,
                )
            )
gen_dataset = Dataset(tasks)
print("Saving dataset...", end="", flush=True)
with chrono.clock("dataset.save") as c:
    gen_dataset.save(output_file)
    print("done in", c.elapsed_time(), "s")

# ================================
# Print some stats
# ================================
# Generate the CFG dictionnary
all_type_requests = gen_dataset.type_requests()
print(f"{len(all_type_requests)} type requests supported.")
print(f"Lexicon: {len(lexicon)} elements")
