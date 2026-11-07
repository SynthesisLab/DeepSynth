from typing import Callable, Iterable, List, Optional, Tuple, Union
import csv
import resource
import sys
import time
import argparse

from synth.syntax import (
    ProbDetGrammar,
    ProbUGrammar,
    CFG,
    DSL,
    hs_enumerate_prob_grammar,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    as_enumerate_prob_grammar,
    ProgramEnumerator,
    auto_type,
)
from synth.syntax.grammars.enumeration.constant_delay import (
    enumerate_prob_grammar as cd,
)
import tqdm
import timeout_decorator

SEARCH_ALGOS = {
    "a_star": as_enumerate_prob_grammar,
    "bee_search": bs_enumerate_prob_grammar,
    "beap_search": bps_enumerate_prob_grammar,
    "heap_search": hs_enumerate_prob_grammar,
    "cd4": lambda x: cd(x, k=4),
    "cd16": lambda x: cd(x, k=16),
    "cd64": lambda x: cd(x, k=64),
    "cd_search": lambda x: cd(x, k=20),
}

parser = argparse.ArgumentParser(
    description="Compare search algorithms", fromfile_prefix_chars="@"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="./enumeration.csv",
    help="output file (default: './enumeration.csv')",
)
parser.add_argument(
    dest="n", metavar="programs", type=int, help="number of programs to be enumerated"
)
# parser.add_argument(
#     dest="max_rules", type=int, help="maximum number of derivation rules"
# )
parser.add_argument(
    dest="max_non_terminals", type=int, help="maximum number of non terminals"
)
parser.add_argument(
    dest="scaling",
    choices=["distance", "nonterminals", "derivations"],
    help="maximum number of non terminals",
)
parser.add_argument(
    "-t", "--timeout", type=int, default=300, help="timeout in seconds (default: 300)"
)
parser.add_argument(
    "-s", "--seed", type=int, default=-1, help="seed (default: -1, uniform)"
)
parser.add_argument(
    "--algos",
    nargs="+",
    choices=list(SEARCH_ALGOS.keys()),
    default=None,
    help="enumeration algorithms to run (default: all)",
)
parser.add_argument(
    "--sizes",
    nargs="+",
    type=int,
    default=None,
    help="sizes of the grammars (default: 4, 6, 8, 10, 12, 22, ... up to max_non_terminals)",
)
parser.add_argument(
    "--max-memory",
    type=float,
    default=-1,
    help="stop an enumeration when the peak memory of the process exceeds this value in GB (default: -1, no limit)",
)


parameters = parser.parse_args()
output_file: str = parameters.output
programs: int = parameters.n
timeout: int = parameters.timeout
seed: int = parameters.seed
scaling: str = parameters.scaling
max_non_terminals: int = parameters.max_non_terminals
max_memory: float = parameters.max_memory
if parameters.algos is not None:
    SEARCH_ALGOS = {name: SEARCH_ALGOS[name] for name in parameters.algos}
file_name = output_file[: -len(".csv")] if output_file.endswith(".csv") else output_file


# ================================
# Load constants specific to dataset
# ================================


def peak_memory_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # kilobytes on Linux, bytes on macOS
    if sys.platform == "darwin":
        return peak / 1024**3
    return peak / 1024**2


def memory_exceeded(name: str, n: int) -> bool:
    if max_memory <= 0:
        return False
    peak = peak_memory_gb()
    if peak > max_memory:
        print(
            f"[max-memory] {name}: stopped after {n} programs, peak memory {peak:.2f} GB > {max_memory} GB",
            file=sys.stderr,
        )
        return True
    return False


def save(trace: Iterable, name: str) -> None:
    with open(name, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(trace)


# Enumeration methods =====================================================
def summary_enumerative_search(
    pcfg: ProbDetGrammar,
    name: str,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    programs: int,
    timeout: int = 300,
    title: Optional[str] = None,
    seed: int = -1,
) -> Tuple[str, int, int, float, int, int, int, int]:
    n = 0
    non_terminals = len(pcfg.rules)
    derivation_rules = sum(len(pcfg.rules[S]) for S in pcfg.rules)
    used_time = 0

    pbar = tqdm.tqdm(total=programs, desc=title or name)
    enumerator = custom_enumerate(pcfg)
    gen = enumerator.generator()
    program = 1
    datum_each = 100000
    start = 0
    try:

        def fun():
            return next(gen)

        get_next = timeout_decorator.timeout(timeout, timeout_exception=StopIteration)(
            fun
        )
        start = time.perf_counter_ns()
        while program is not None:
            program = get_next()
            n += 1
            if n % datum_each == 0 or n >= programs:
                used_time = time.perf_counter_ns() - start
                bef = time.perf_counter_ns()
                if used_time >= timeout * 1e9:
                    break
                pbar.update(datum_each)
                if n >= programs or memory_exceeded(name, n):
                    break
                rem_time = timeout - used_time / 1e9
                get_next = timeout_decorator.timeout(
                    rem_time, timeout_exception=StopIteration
                )(fun)
                start -= time.perf_counter_ns() - bef
    except (StopIteration, RuntimeError):
        used_time = time.perf_counter_ns() - start
    pbar.close()
    datum_each = 1000000
    factor = datum_each / n
    return (
        name,
        non_terminals,
        derivation_rules,
        used_time * factor / 1e9,
        datum_each,
        int(enumerator.programs_in_queues() * factor),
        int(enumerator.programs_in_banks() * factor),
        seed,
    )


def enumerative_search(
    pcfg: ProbDetGrammar,
    name: str,
    custom_enumerate: Callable[
        [Union[ProbDetGrammar, ProbUGrammar]], ProgramEnumerator
    ],
    programs: int,
    timeout: int = 300,
    title: Optional[str] = None,
    seed: int = -1,
) -> Tuple[
    Tuple[str, int, int, float, int, int, int, int],
    List[Tuple[str, int, int, float, int, int, int, int]],
]:
    n = 0
    non_terminals = len(pcfg.rules)
    derivation_rules = sum(len(pcfg.rules[S]) for S in pcfg.rules)
    used_time = 0

    pbar = tqdm.tqdm(total=programs, desc=title or name, smoothing=0)
    enumerator = custom_enumerate(pcfg)
    gen = enumerator.generator()
    program = 1
    datum_each = 100000
    target_generation_speed = 1000000
    start = 0
    detailed = []
    try:

        def fun():
            return next(gen)

        get_next = timeout_decorator.timeout(timeout, timeout_exception=StopIteration)(
            fun
        )
        start = time.perf_counter_ns()
        while program is not None:
            program = get_next()
            n += 1
            if n % datum_each == 0 or n >= programs:
                used_time = time.perf_counter_ns() - start
                bef = time.perf_counter_ns()
                if used_time >= timeout * 1e9:
                    break
                pbar.update(datum_each)
                if n >= programs or memory_exceeded(name, n):
                    break
                detailed.append(
                    (
                        name,
                        non_terminals,
                        derivation_rules,
                        used_time / 1e9,
                        n,
                        enumerator.programs_in_queues(),
                        enumerator.programs_in_banks(),
                        seed,
                    )
                )
                rem_time = timeout - used_time / 1e9
                get_next = timeout_decorator.timeout(
                    rem_time, timeout_exception=StopIteration
                )(fun)
                start -= time.perf_counter_ns() - bef
    except (StopIteration, RuntimeError):
        used_time = time.perf_counter_ns() - start
    pbar.close()
    factor = target_generation_speed / n
    return (
        name,
        non_terminals,
        derivation_rules,
        used_time * factor / 1e9,
        target_generation_speed,
        int(enumerator.programs_in_queues() * factor),
        int(enumerator.programs_in_banks() * factor),
        seed,
    ), detailed


# Main ====================================================================


def gen_distance(non_terminals: int) -> CFG:
    syntax = {
        "1": "s1",
    }
    for i in range(2, non_terminals + 1):
        syntax[f"cast{i}"] = f"s{i} -> s1"
        syntax[f"s{i}"] = f"s{i}"
        syntax[f"+{i}"] = f"s1 -> s{i} -> s{i}"
        syntax[f"*{i}"] = f"s{i-1} -> s{i} -> s{i+1} -> s{i}"
    return CFG.infinite(DSL(auto_type(syntax)), auto_type("s1->s1"), n_gram=1)


def gen_nonterminals(non_terminals: int) -> CFG:
    syntax = {
        "1": "s1",
    }
    for i in range(2, non_terminals + 1):
        syntax[f"cast{i}"] = f"s{i} -> s1"
        syntax[f"s{i}"] = f"s{i}"
        syntax[f"+{i}"] = f"s1 -> s{i}"
    return CFG.infinite(DSL(auto_type(syntax)), auto_type("s1->s1"), n_gram=1)


def gen_derivations(non_terminals: int) -> CFG:
    syntax = {
        "1": "s1",
    }
    for i in range(2, non_terminals + 1):
        syntax[f"m{i}"] = f"s1 -> s1 -> s1"
        syntax[f"f{i}"] = f"s1 -> s1"
        syntax[f"s{i}"] = f"s1"
    return CFG.infinite(DSL(auto_type(syntax)), auto_type("s1->s1"), n_gram=1)


if __name__ == "__main__":
    # trace_rules = [
    #     (
    #         "search",
    #         "non_terminals",
    #         "derivation_rules",
    #         "time",
    #         "programs",
    #         "queue",
    #         "bank",
    #     )
    # ]
    # print("Working on derivation rules scaling")
    # for derivation_rules in range(2, max_rules + 1, 10):
    #     syntax = {
    #         "+": "s -> s",
    #         "1": "s",
    #     }
    #     for i in range(2, derivation_rules):
    #         syntax[f"{i}"] = "s"
    #     cfg = CFG.infinite(DSL(auto_type(syntax)), auto_type("s->s"), n_gram=1)
    #     pcfg = ProbDetGrammar.uniform(cfg)
    #     for name, enum in SEARCH_ALGOS.items():
    #         trace_rules.append(
    #             enumerative_search(pcfg, name, enum, programs, timeout=timeout, title=f"{name}-{derivation_rules}")  # type: ignore
    #         )
    #     save(trace_rules, file_name + "_rules.csv")
    # print("csv file was saved as:", file_name + "_rules.csv")
    gene = {
        "distance": gen_distance,
        "nonterminals": gen_nonterminals,
        "derivations": gen_derivations,
    }
    summary_trace = [
        (
            "search",
            "non_terminals",
            "derivation_rules",
            "time",
            "programs",
            "queue",
            "bank",
            "seed",
        )
    ]
    detailed_trace = [
        (
            "search",
            "non_terminals",
            "derivation_rules",
            "time",
            "programs",
            "queue",
            "bank",
            "seed",
        )
    ]
    print("Working on non terminals scaling")
    non_terminals_values = [4]
    while non_terminals_values[-1] < max_non_terminals:
        last = non_terminals_values[-1]
        if last <= 10:
            last += 2
        else:
            last += 10
        non_terminals_values.append(last)
    if parameters.sizes is not None:
        non_terminals_values = parameters.sizes
    first = True
    for non_terminals in non_terminals_values:
        cfg = gene[scaling](non_terminals)
        if seed < 0:
            pcfg = ProbDetGrammar.uniform(cfg)
        else:
            pcfg = ProbDetGrammar.random(cfg, seed=seed)
        for name, enum in SEARCH_ALGOS.items():
            if first:
                summary, detailed = enumerative_search(
                    pcfg,
                    name,
                    enum,
                    programs,
                    timeout=timeout,
                    title=f"{name}-{non_terminals}",
                    seed=seed,
                )  # type: ignore
                summary_trace.append(summary)
                detailed_trace += detailed
            else:
                summary_trace.append(
                    summary_enumerative_search(
                        pcfg,
                        name,
                        enum,
                        programs,
                        timeout=timeout,
                        title=f"{name}-{non_terminals}",
                        seed=seed,
                    )  # type: ignore
                )
        save(summary_trace, file_name + "_growth.csv")
        save(detailed_trace, file_name + "_detailed.csv")
        first = False
    print("growth csv file was saved as:", file_name + "_growth.csv")
    print("detailed csv file was saved as:", file_name + "_detailed.csv")
