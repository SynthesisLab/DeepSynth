"""
Enumeration on grammars with cycles between non-terminals,
compared with a brute force enumeration of all programs up to a cost bound.
"""

import heapq
import itertools
import math
import random
from typing import Callable, Dict, List

import pytest

from synth.syntax import CFG, DSL, auto_type, ProbDetGrammar, ProgramEnumerator
from synth.syntax.program import Function, Program
from synth.syntax.grammars.enumeration.heap_search import HeapSearch
from synth.syntax.grammars.enumeration.constant_delay import CDSearch, CDSearchEager
from synth.syntax.grammars.enumeration.constant_delay_queue import CDQueue, CostTuple
from synth.syntax.grammars.enumeration.bee_search import BeeSearch, BeeSearchBuckets
from synth.syntax.grammars.enumeration.beap_search import BeapSearch


def __make__(syntax: Dict[str, str], costs: Dict[str, int], start: str):
    cfg = CFG.infinite(DSL(auto_type(syntax)), auto_type(start), n_gram=1)
    int_costs = {S: {P: costs[str(P)] for P in cfg.rules[S]} for S in cfg.rules}
    return cfg, int_costs


def __nt__(cfg: CFG, S, P, i: int):
    Sp = cfg.rules[S][P][0][i]
    return (Sp[0], (Sp[1], None))


def __cost__(cfg: CFG, c, S, p: Program) -> int:
    if isinstance(p, Function):
        F = p.function
        return c[S][F] + sum(
            __cost__(cfg, c, __nt__(cfg, S, F, i), a) for i, a in enumerate(p.arguments)
        )
    return c[S][p]


def __brute_force__(cfg: CFG, c, bound: int) -> Dict[Program, int]:
    progs: Dict = {S: {} for S in cfg.rules}
    changed = True
    while changed:
        changed = False
        for S in cfg.rules:
            for P, (args, _) in cfg.rules[S].items():
                nts = [__nt__(cfg, S, P, i) for i in range(len(args))]
                for combo in itertools.product(*[list(progs[X].items()) for X in nts]):
                    cost = c[S][P] + sum(x[1] for x in combo)
                    prog = Function(P, [x[0] for x in combo]) if combo else P
                    if cost <= bound and prog not in progs[S]:
                        progs[S][prog] = cost
                        changed = True
    return progs[cfg.start]


def __probabilities__(c):
    return {S: {P: math.exp(-x) for P, x in v.items()} for S, v in c.items()}


ALGOS: Dict[str, Callable] = {
    "heap_search": lambda cfg, c: HeapSearch(ProbDetGrammar(cfg, __probabilities__(c))),
    "cd_search": lambda cfg, c: CDSearch(ProbDetGrammar(cfg, c), k=5),
    "cd_search_eager": lambda cfg, c: CDSearchEager(ProbDetGrammar(cfg, c), k=5),
    "bee_search": lambda cfg, c: BeeSearch(ProbDetGrammar(cfg, c)),
    "bee_search_buckets": lambda cfg, c: BeeSearchBuckets(ProbDetGrammar(cfg, c)),
    "beap_search": lambda cfg, c: BeapSearch(ProbDetGrammar(cfg, c)),
}


def __check__(enumerator: ProgramEnumerator, cfg: CFG, c, bound: int) -> None:
    truth = __brute_force__(cfg, c, bound)
    seen: List[Program] = []
    last = 0
    for program in enumerator.generator():
        cost = __cost__(cfg, c, cfg.start, program)
        assert cost >= last, f"{program} of cost {cost} after a program of cost {last}"
        last = cost
        if cost > bound or len(seen) == len(truth):
            break
        seen.append(program)
    assert len(seen) == len(set(seen))
    assert set(seen) == set(truth)


@pytest.mark.parametrize("algo", ALGOS.keys())
def test_counterexample(algo: str) -> None:
    # W -> k(U) cost 1, U -> u1 cost 4, U -> q(W) cost 0.5 (costs doubled)
    # the original order of Heap Search (pop, then expand the popped program)
    # only outputs k(u1)
    cfg, c = __make__(
        {"k": "U -> W", "u1": "U", "q": "W -> U"}, {"k": 2, "u1": 8, "q": 1}, "W"
    )
    __check__(ALGOS[algo](cfg, c), cfg, c, 40)


def __random_grammars__(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        types = [f"T{i}" for i in range(rng.choice([2, 3]))]
        syntax, costs = {}, {}
        for t in types:
            for _ in range(rng.randint(1, 3)):
                args = [rng.choice(types) for _ in range(rng.choice([0, 1, 1, 2, 2]))]
                name = f"f{len(syntax)}"
                syntax[name] = " -> ".join(args + [t])
                costs[name] = rng.randint(1, 5)
        try:
            cfg, c = __make__(syntax, costs, "T0")
        except KeyError:
            continue  # the start non-terminal does not derive any program
        if len(cfg.rules) < 2:
            continue
        truth = __brute_force__(cfg, c, 11)
        # all non-terminals must be productive, and there must be programs beyond the bound
        if not __all_productive__(cfg):
            continue
        if not 3 <= len(truth) <= 3000 or len(__brute_force__(cfg, c, 16)) == len(
            truth
        ):
            continue
        out.append((syntax, costs))
    return out


def __all_productive__(cfg: CFG) -> bool:
    productive = set()
    changed = True
    while changed:
        changed = False
        for S in cfg.rules:
            if S not in productive and any(
                all(__nt__(cfg, S, P, i) in productive for i in range(len(args)))
                for P, (args, _) in cfg.rules[S].items()
            ):
                productive.add(S)
                changed = True
    return len(productive) == len(cfg.rules)


RANDOM_GRAMMARS = __random_grammars__(25, 0)


@pytest.mark.parametrize("algo", ALGOS.keys())
@pytest.mark.parametrize("index", range(len(RANDOM_GRAMMARS)))
def test_random_cyclic_grammars(algo: str, index: int) -> None:
    syntax, costs = RANDOM_GRAMMARS[index]
    cfg, c = __make__(syntax, costs, "T0")
    __check__(ALGOS[algo](cfg, c), cfg, c, 11)


def test_heap_search_initialisation_with_cycles() -> None:
    # the best program of T1, f3(f2), is only known after that of T0:
    # the initial heap of T0 must contain f1(f3(f2)) and not f1(f5)
    syntax = {
        "f0": "T1 -> T1 -> T0",
        "f1": "T1 -> T0",
        "f2": "T0",
        "f3": "T0 -> T1",
        "f4": "T1",
        "f5": "T1",
    }
    costs = {"f0": 4, "f1": 3, "f2": 2, "f3": 1, "f4": 5, "f5": 4}
    cfg, c = __make__(syntax, costs, "T0")
    __check__(ALGOS["heap_search"](cfg, c), cfg, c, 10)


@pytest.mark.parametrize("seed", range(20))
def test_cd_queue_with_growing_window(seed: int) -> None:
    rng = random.Random(seed)
    queue = CDQueue(rng.choice([1, 3, 10]), rng.choice([2, 5, 20]))
    reference: List[int] = []
    for cost in sorted(3 * rng.randint(0, 5) for _ in range(rng.randint(1, 5))):
        queue.push(CostTuple(cost, [[cost]]))
        heapq.heappush(reference, cost)
    queue.update()
    for _ in range(200):
        if not reference:
            break
        popped = queue.pop()
        # equal costs are merged in a single cost tuple
        got = sorted(x[0] for x in popped.combinations)
        assert got == [heapq.heappop(reference) for _ in got]
        for _ in range(rng.randint(0, 3)):
            cost = got[0] + 3 * rng.randint(1, rng.choice([2, 10, 400]))
            queue.push(CostTuple(cost, [[cost]]))
            heapq.heappush(reference, cost)
        if not queue.is_empty():
            queue.update()
