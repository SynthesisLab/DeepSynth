import os
import sys
import multiprocessing
import resource
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, Union
import csv

import tqdm

from dataset_loader import add_dataset_choice_arg, load_dataset
from dsl_loader import add_dsl_choice_arg, load_DSL


from synth import Dataset, PBE
from synth.filter.filter import Filter
from synth.semantic.evaluator import DSLEvaluator
from synth.specification import PBEWithConstants
from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    DSL,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    hs_enumerate_bucket_prob_grammar,
    hs_enumerate_bucket_prob_u_grammar,
    cd_enumerate_prob_grammar,
    as_enumerate_prob_grammar,
    ProgramEnumerator,
    Type,
    CFG,
)
from synth.filter import DFTAFilter, ObsEqFilter
from synth.filter.constraints import add_dfta_constraints
from synth.syntax.program import Program
from synth.syntax.grammars.enumeration.bee_search import enumerate_prob_grammar_buckets
from synth.syntax.grammars.enumeration.constant_delay import (
    enumerate_prob_grammar_eager as cd_enumerate_prob_grammar_eager,
)
from synth.task import Task
from synth.utils import load_object
from synth.utils.data_storage import legacy_load_object
from synth.utils.import_utils import import_file_function
from synth.pbe.solvers import (
    NaivePBESolver,
    PBESolver,
    CutoffPBESolver,
    RestartPBESolver,
)


import argparse


SOLVERS = {solver.name(): solver for solver in [NaivePBESolver, CutoffPBESolver]}
base_solvers = {x: y for x, y in SOLVERS.items()}
for meta_solver in [RestartPBESolver]:
    for name, solver in base_solvers.items():
        SOLVERS[f"{meta_solver.name()}.{name}"] = lambda *args, **kwargs: meta_solver(
            *args, solver_builder=solver, **kwargs
        )

SEARCH_ALGOS = {
    "cd_search": (lambda x: cd_enumerate_prob_grammar(x, 20), None),
    "beap_search": (bps_enumerate_prob_grammar, None),
    "heap_search": (hs_enumerate_prob_grammar, hs_enumerate_prob_u_grammar),
    "bucket_search": (
        lambda x: hs_enumerate_bucket_prob_grammar(x, 3),
        lambda x: hs_enumerate_bucket_prob_u_grammar(x, 3),
    ),
    "bee_search": (bs_enumerate_prob_grammar, None),
    # ablations: Bee Search with the bucket queue of Eco Search, Eco Search with eager expansion
    "bee_search_buckets": (lambda x: enumerate_prob_grammar_buckets(x, k=20), None),
    "cd_search_eager": (lambda x: cd_enumerate_prob_grammar_eager(x, 20), None),
    "a_star": (as_enumerate_prob_grammar, None),
}

PRUNING = {"dfta", "obs-eq"}

parser = argparse.ArgumentParser(
    description="Solve program synthesis tasks", fromfile_prefix_chars="@"
)
add_dsl_choice_arg(parser)
add_dataset_choice_arg(parser)
parser.add_argument(
    "--pcfg", type=str, default=None, help="files containing the predicted PCFGs"
)
parser.add_argument(
    "-s",
    "--search",
    choices=SEARCH_ALGOS.keys(),
    default=list(SEARCH_ALGOS.keys())[0],
    help=f"enumeration algorithm (default: {list(SEARCH_ALGOS.keys())[0]})",
)
parser.add_argument(
    "--solver",
    choices=list(SOLVERS.keys()),
    default="naive",
    help=f"used solver (default: naive)",
)
parser.add_argument(
    "-o", "--output", type=str, default="./", help="output folder (default: './')"
)
parser.add_argument(
    "--constrained",
    action="store_true",
    default=False,
    help="use unambigous grammar to include constraints in the grammar if available",
)
parser.add_argument(
    "--support",
    type=str,
    default=None,
    help="train dataset to get the set of supported type requests",
)
parser.add_argument(
    "-t", "--timeout", type=float, default=300, help="task timeout in s (default: 300)"
)

parser.add_argument(
    "-p",
    "--pruning",
    nargs="*",
    choices=list(x for x in PRUNING),
    help="runtime pruning",
)
parser.add_argument(
    "--memory",
    action="store_true",
    help="solve each task in a child process and record its peak resident memory",
)
parser.add_argument(
    "--max-memory",
    type=float,
    default=-1,
    help="with --memory, stop a task when the resident memory of its process exceeds this value in GB; it is a failure (default: -1, no limit)",
)
parser.add_argument(
    "--shard",
    type=str,
    default=None,
    help="I/N: solve only the tasks of index I modulo N (0 <= I < N); the output goes to <output>/shards/ (merge with merge_solve_shards.py)",
)
parser.add_argument(
    "--filter",
    nargs="*",
    type=str,
    help="load the given files and call their get_filter functions to get a Filter[Program]",
)

parameters = parser.parse_args()
dsl_name: str = parameters.dsl
dataset_file: str = parameters.dataset
pcfg_file: Optional[str] = parameters.pcfg
search_algo: str = parameters.search
method: Callable[[Any], PBESolver] = SOLVERS[parameters.solver]
output_folder: str = parameters.output
task_timeout: float = parameters.timeout
constrained: bool = parameters.constrained
support: Optional[str] = (
    None if not parameters.support else parameters.support.format(dsl_name=dsl_name)
)
pruning: List[str] = parameters.pruning or []
filter_files: List[str] = parameters.filter or []
measure_memory: bool = parameters.memory
max_memory_kb: float = parameters.max_memory * 1024**2
shard: Optional[Tuple[int, int]] = None
if parameters.shard is not None:
    shard_index, shard_count = (int(x) for x in parameters.shard.split("/"))
    assert 0 <= shard_index < shard_count, "--shard I/N needs 0 <= I < N"
    shard = (shard_index, shard_count)

if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
    print("Dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)
elif support is not None and (
    not os.path.exists(support) or not os.path.isfile(support)
):
    print("Support dataset must be a valid dataset file!", file=sys.stderr)
    sys.exit(1)

det_search, u_search = SEARCH_ALGOS[search_algo]
custom_enumerate = u_search if constrained else det_search
if custom_enumerate is None:
    txt = "det-CFG" if not constrained else "UCFG"
    print(
        f"search algorithm {search_algo} does not support enumeration for {txt}!",
        file=sys.stderr,
    )
    sys.exit(1)

start_index = (
    0
    if not os.path.sep in dataset_file
    else (len(dataset_file) - dataset_file[::-1].index(os.path.sep))
)
dataset_name = dataset_file[start_index : dataset_file.index(".", start_index)]

supported_type_requests = Dataset.load(support).type_requests() if support else None

# ================================
# Load dftas files
# ================================
filter_pot_funs = [
    import_file_function(file[:-3].replace("/", "."), ["get_filter"])()
    for file in filter_files
]
filter_funs = [x.get_filter for x in filter_pot_funs if x is not None]

# ================================
# Load constants specific to dataset
# ================================


def load_dsl_and_dataset() -> (
    Tuple[Dataset[PBE], DSL, DSLEvaluator, List[str], Set[Type]]
):
    dsl_module = load_DSL(dsl_name)
    dsl, evaluator = dsl_module.dsl, dsl_module.evaluator
    # ================================
    # Load dataset
    # ================================
    full_dataset = load_dataset(dsl_name, dataset_file)

    return (
        full_dataset,
        dsl,
        evaluator,
        getattr(dsl_module, "constraints", []),
        getattr(dsl_module, "constant_types", set()),
    )


# Produce PCFGS ==========================================================


def save(trace: Iterable, file: str) -> None:
    with open(file, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(trace)


# Enumeration methods =====================================================
def setup_filters(
    task: Task[PBE], constant_types: Set[Type]
) -> Optional[Filter[Program]]:
    out = None
    # Dynamic DFTA filters
    filters = [f(task.type_request, constant_types) for f in filter_funs]
    for filter in filters:
        out = filter if out is None else out.intersection(filter)
    if "dfta" in pruning:
        base_grammar = CFG.infinite(
            dsl, task.type_request, constant_types=constant_types
        )
        filter = DFTAFilter(
            add_dfta_constraints(base_grammar, constraints, progress=False)
        )
        out = filter if out is None else out.intersection(filter)
    if "obs-eq" in pruning:
        filter = ObsEqFilter(
            solver.evaluator, [ex.inputs for ex in task.specification.examples]
        )
        out = filter if out is None else out.intersection(filter)
    return out


def solve_task(
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    solver: PBESolver,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    constant_types: Set[Type],
    stats_name: List[str],
) -> Tuple[bool, Optional[Program], List[Any]]:
    task_solved = False
    solution = None
    try:
        enumerator = custom_enumerate(pcfg)
        enumerator.filter = setup_filters(task, constant_types)
        sol_generator = solver.solve(task, enumerator, timeout=task_timeout)
        solution = next(sol_generator)
        sol_generator.send(True)
        task_solved = True
    except StopIteration:
        pass
    stats = [solver.get_stats(name) for name in stats_name]
    solver.reset_stats()
    return task_solved, solution, stats


def __max_rss_kb__() -> int:
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if sys.platform == "darwin" else rss


MEMORY_LIMIT_EXIT_CODE = 77


def __current_rss_kb__() -> int:
    try:
        with open("/proc/self/statm") as fd:
            return int(fd.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except OSError:
        return __max_rss_kb__()


def __memory_watchdog__(connection: Any, baseline: int) -> None:
    while True:
        rss = __current_rss_kb__()
        if rss > max_memory_kb:
            connection.send(
                (
                    (False, None, [], f"memory limit ({parameters.max_memory} GB)"),
                    __max_rss_kb__(),
                    baseline,
                )
            )
            connection.close()
            os._exit(MEMORY_LIMIT_EXIT_CODE)
        time.sleep(0.2)


def __child_solve__(connection: Any, *args: Any) -> None:
    baseline = __max_rss_kb__()
    if max_memory_kb > 0:
        threading.Thread(
            target=__memory_watchdog__, args=(connection, baseline), daemon=True
        ).start()
    try:
        task_solved, solution, stats = solve_task(*args)
        result = (task_solved, str(solution) if solution else None, stats, "")
    except MemoryError:
        result = (False, None, [], "MemoryError")
    connection.send((result, __max_rss_kb__(), baseline))
    connection.close()


def solve_in_child_process(
    task: Task[PBE],
    pcfg: Union[ProbDetGrammar, ProbUGrammar],
    solver: PBESolver,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    constant_types: Set[Type],
    stats_name: List[str],
) -> Tuple[bool, Optional[str], List[Any], List[Any]]:
    """
    Solve the task in a forked child process, and return in addition
    [peak resident memory of the child, its resident memory when it started, memory error].
    A run that raises MemoryError or that is killed (e.g. by the OOM killer, exit code 137) is a failure.
    """
    ctx = multiprocessing.get_context("fork")
    parent_connection, child_connection = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=__child_solve__,
        args=(
            child_connection,
            task,
            pcfg,
            solver,
            custom_enumerate,
            constant_types,
            stats_name,
        ),
    )
    started = time.perf_counter()
    process.start()
    child_connection.close()
    try:
        (task_solved, solution, stats, error), peak, baseline = parent_connection.recv()
    except EOFError:
        # the child died without answering
        task_solved, solution, stats, error, peak, baseline = (
            False,
            None,
            [],
            "",
            None,
            None,
        )
    process.join()
    if process.exitcode == MEMORY_LIMIT_EXIT_CODE and not error:
        error = f"memory limit ({parameters.max_memory} GB)"
    if process.exitcode != 0 and not error:
        error = f"killed (exit code {process.exitcode})"
    if not stats:
        # the child was stopped: the plots need a time and a number of programs
        elapsed = time.perf_counter() - started
        stats = [
            elapsed if name == "time" else 0 if name == "programs" else None
            for name in stats_name
        ]
    return task_solved, solution, stats, [peak, baseline, error]


def enumerative_search(
    dataset: Dataset[PBE],
    evaluator: DSLEvaluator,
    pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]],
    trace: List[Tuple[bool, float]],
    solver: PBESolver,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    save_file: str,
    constant_types: Set[Type],
) -> None:
    start = max(0, len(trace) - 1)
    pbar = tqdm.tqdm(total=len(pcfgs) - start, desc="Tasks", smoothing=0)
    i = 0
    solved = 0
    total = 0
    tasks = [
        t
        for t in dataset.tasks
        if supported_type_requests is None or t.type_request in supported_type_requests
    ]
    if shard is not None:
        tasks = tasks[shard[0] :: shard[1]]
        pcfgs = pcfgs[shard[0] :: shard[1]]
    stats_name = solver.available_stats()
    memory_columns = (
        ["peak_memory_kb", "baseline_memory_kb", "memory_error"]
        if measure_memory
        else []
    )
    if start == 0:
        trace.append(["solved", "solution"] + stats_name + memory_columns)
    for task, pcfg in zip(tasks[start:], pcfgs[start:]):
        if task.metadata.get("name", None) is not None:
            pbar.set_description_str(task.metadata["name"])
        total += 1
        if isinstance(task.specification, PBEWithConstants):
            pcfg = pcfg.instantiate_constants(task.specification.constants)
        try:
            if measure_memory:
                task_solved, solution, stats, memory = solve_in_child_process(
                    task, pcfg, solver, custom_enumerate, constant_types, stats_name
                )
            else:
                task_solved, solution, stats = solve_task(
                    task, pcfg, solver, custom_enumerate, constant_types, stats_name
                )
                memory = []
        except KeyboardInterrupt:
            break
        solved += int(task_solved)
        out = [task_solved, solution] + stats + memory
        trace.append(out)
        pbar.update(1)
        evaluator.clear_cache()
        # print("Cache hit:", evaluator.cache_hit_rate)
        # print("Programs tried:", trace[len(trace) - 1][2])
        if i % 10 == 0:
            pbar.set_postfix_str("Saving...")
            save(trace, save_file)
        pbar.set_postfix_str(f"Solved {solved}/{total}")

    pbar.close()


def load_pcfgs(
    pcfg_file: Optional[str],
) -> Union[List[ProbDetGrammar], List[ProbUGrammar]]:
    if pcfg_file is not None:
        if constrained:
            return legacy_load_object(pcfg_file)
        else:
            return load_object(pcfg_file)
    pcfgs = []
    for task in full_dataset:
        constant_types = set()
        if isinstance(task.specification, PBEWithConstants):
            constant_types = set(task.specification.constants.keys())
        g = CFG.infinite(dsl, task.type_request, 1, constant_types=constant_types)
        pcfgs.append(ProbDetGrammar.uniform(g))
    return pcfgs


# Main ====================================================================

if __name__ == "__main__":
    (full_dataset, dsl, evaluator, constraints, constant_types) = load_dsl_and_dataset()

    solver: PBESolver = method(evaluator=evaluator)

    pcfgs = load_pcfgs(pcfg_file)
    if pcfg_file is None:
        model_name = "uniform"
    else:
        model_name = os.path.split(pcfg_file)[1][
            len(f"pcfgs_{dataset_name}_") : -len(".pickle")
        ]
    pruning_suffix = "_".join(pruning)
    file = os.path.join(
        output_folder,
        f"{dataset_name}_{search_algo}_{model_name}_{solver.full_name()}{pruning_suffix}.csv",
    )
    if shard is not None:
        os.makedirs(os.path.join(output_folder, "shards"), exist_ok=True)
        file = os.path.join(
            output_folder,
            "shards",
            os.path.basename(file)[: -len(".csv")] + f".{shard[0]}of{shard[1]}.csv",
        )
    trace = []
    if os.path.exists(file):
        with open(file, "r") as fd:
            reader = csv.reader(fd)
            trace = [tuple(row) for row in reader]
            print(
                "\tLoaded",
                len(trace) - 1,
                "/",
                len(full_dataset),
                "(",
                int((len(trace) - 1) * 100 / len(full_dataset)),
                "%)",
            )
    enumerative_search(
        full_dataset,
        evaluator,
        pcfgs,
        trace,
        solver,
        custom_enumerate,
        file,
        constant_types,
    )
    save(trace, file)
    print("csv file was saved as:", file)
