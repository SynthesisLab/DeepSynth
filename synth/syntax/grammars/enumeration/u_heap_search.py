from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod

from synth.filter.filter import Filter
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.enumeration.heap_search import HeapElement, Bucket
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class StartHeapElement(Generic[U]):
    priority: Ordered
    program: Program = field(compare=False)
    start: Tuple[Type, U] = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.priority}, {self.program}, {self.start})"


def __wrap__(el: Union[U, List[U]]) -> Union[U, Tuple[U, ...]]:
    if isinstance(el, list):
        return tuple(el)
    return el


class UHSEnumerator(ProgramEnumerator[None], ABC, Generic[U, V, W]):
    def __init__(
        self,
        G: ProbUGrammar[U, V, W],
        threshold: Optional[Ordered] = None,
        filter: Optional[Filter[Program]] = None,
    ) -> None:
        super().__init__(filter)
        self.G = G
        symbols = [S for S in self.G.rules]
        self.threshold = threshold
        self.deleted: Set[Program] = set()

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[Tuple[Type, U], List[HeapElement]] = {S: [] for S in symbols}

        self._start_heap: List[StartHeapElement[U]] = []

        # the same program can be pushed in different heaps, with different probabilities
        # however, the same program cannot be pushed twice in the same heap

        # self.succ[S][P] is the successor of P from S
        self.succ: Dict[Tuple[Type, U], Dict[int, Program]] = {S: {} for S in symbols}
        # self.pred[S][P] is the hash of the predecessor of P from S
        self.pred: Dict[Tuple[Type, U], Dict[int, int]] = {S: {} for S in symbols}

        # self.hash_table_program[S] is the set of hashes of programs
        # ever added to the heap for S
        self.hash_table_program: Dict[Tuple[Type, U], Set[int]] = {
            S: set() for S in symbols
        }

        self._keys: Dict[Tuple[Type, U], Dict[Program, V]] = defaultdict(dict)

        self._init: Set[Tuple[Type, U]] = set()

        self.max_priority: Dict[
            Union[Tuple[Type, U], Tuple[Tuple[Type, U], Program, V]], Program
        ] = {}

    @classmethod
    def name(cls) -> str:
        return "u-heap-search"

    def programs_in_banks(self) -> int:
        return sum(len(val) for val in self.succ.values())

    def programs_in_queues(self) -> int:
        return sum(len(val) for val in self.heaps.values())

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        while True:
            program = self.start_query()
            if program is None:
                break
            if not self._should_keep_subprogram(program):
                self.deleted.add(program)
                continue
            yield program

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    def __init_helper__(
        self,
        Si: Tuple[Type, U],
        information: W,
        i: int,
        args: List[Program],
        nargs: int,
    ) -> Generator[List[Program], None, None]:
        # Try to init sub Tuple[Type, U] in case they were not initialised
        self.__init_non_terminal__(Si)
        args.append(self.max_priority[Si])
        if i + 1 >= nargs:
            yield args
            return
        for information, lst in self.G.derive_all(information, Si, args[-1]):
            Sip1 = lst[-1][0]
            for sol in self.__init_helper__(Sip1, information, i + 1, args[:], nargs):
                yield sol

    def __init_non_terminal__(self, S: Tuple[Type, U]) -> None:
        if S in self._init:
            return
        self._init.add(S)
        # 1) Compute max probablities
        best_program: Optional[Program] = None
        best_priority: Optional[Ordered] = None
        for P in self.G.rules[S]:
            nargs = self.G.arguments_length_for(S, P)
            if nargs > 0:
                for information, Si, v in self.G.derive(
                    self.G.start_information(), S, P
                ):
                    for arguments in self.__init_helper__(
                        Si, information, 0, [], nargs
                    ):
                        new_program = Function(
                            function=P,
                            arguments=arguments,
                        )
                        self._keys[S][new_program] = v
                        priority = self.compute_priority(S, new_program)
                        self.max_priority[(S, P, __wrap__(v))] = new_program  # type: ignore
                        if not best_priority or priority < best_priority:
                            best_program = new_program
                            best_priority = priority
            else:
                some_v = list(self.G.tags[S][P].keys())[0]
                self._keys[S][P] = some_v
                priority = self.compute_priority(S, P)
                self.max_priority[(S, P, some_v)] = P
                if not best_priority or priority < best_priority:
                    best_program = P
                    best_priority = priority
        assert best_program
        self.max_priority[S] = best_program

        # 2) add P(max(S1),max(S2), ...) to self.heaps[S]
        for P in self.G.rules[S]:
            for v in self.G.tags[S][P]:
                program = self.max_priority[(S, P, v)]
                hash_program = hash(program)
                # Remark: the program cannot already be in self.heaps[S]
                assert hash_program not in self.hash_table_program[S]
                self.hash_table_program[S].add(hash_program)
                # we assume that the programs from max_probability
                # are represented by the same object
                priority = self.compute_priority(S, program)
                assert program in self._keys[S]
                if not self.threshold or priority < self.threshold:
                    heappush(
                        self.heaps[S],
                        HeapElement(priority, program),
                    )
                    if S in self.G.starts:
                        heappush(
                            self._start_heap,
                            StartHeapElement(
                                self.adjust_priority_for_start(priority, S), program, S
                            ),
                        )

        # 3) Do the 1st query
        self.query(S, None)

    def start_query(self) -> Optional[Program]:
        if len(self._init) == 0:
            for start in self.G.starts:
                self.query(start, None)
        if len(self._start_heap) == 0:
            return None
        elem = heappop(self._start_heap)
        self.query(elem.start, elem.program)
        while elem.program in self.deleted:
            elem = heappop(self._start_heap)
            self.query(elem.start, elem.program)
        return elem.program

    def __add_successors_to_heap__(
        self,
        succ: Function,
        S: Tuple[Type, U],
        Si: Tuple[Type, U],
        info: W,
        i: int,
        v: V,
    ) -> bool:
        if i >= len(succ.arguments):
            return True
        # Si is non-terminal symbol used to derive the i-th argument
        program = succ.arguments[i]
        # Check if we are not using the correct derivation of S -> P
        # If it is not the right one (from which we can dervie back its arguments) then we can't generate successors
        for info, lst in self.G.derive_all(
            info, Si, succ.arguments[i], hints=self._keys
        ):
            Sip1 = lst[-1][0]
            if self.__add_successors_to_heap__(succ, S, Sip1, info, i + 1, v):
                succ_sub_program = self.query(Si, succ.arguments[i])
                if succ_sub_program is not None:
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program
                    new_program = Function(succ.function, new_arguments)
                    hash_new_program = hash(new_program)
                    if hash_new_program not in self.hash_table_program[S]:
                        self.hash_table_program[S].add(hash_new_program)
                        # try:
                        self._keys[S][new_program] = v
                        priority: Ordered = self.compute_priority(S, new_program)
                        if not self.threshold or priority < self.threshold:
                            heappush(self.heaps[S], HeapElement(priority, new_program))
                            if S in self.G.starts:
                                heappush(
                                    self._start_heap,
                                    StartHeapElement(
                                        self.adjust_priority_for_start(priority, S),
                                        new_program,
                                        S,
                                    ),
                                )
                return True
        return False

    def __add_successors__(self, succ: Program, S: Tuple[Type, U]) -> None:
        if isinstance(succ, Function):
            F = succ.function
            tgt_v = self._keys[S][succ]
            out = self.G.derive_specific(self.G.start_information(), S, F, tgt_v)  # type: ignore
            assert out is not None
            info, Si = out
            assert self.__add_successors_to_heap__(succ, S, Si, info, 0, tgt_v)

    def query(self, S: Tuple[Type, U], program: Optional[Program]) -> Optional[Program]:
        """
        computing the successor of program from S
        """
        if S not in self._init:
            self.__init_non_terminal__(S)
        if program is not None:
            hash_program = hash(program)
        else:
            hash_program = 123891

        # if we have already computed the successor of program from S, we return its stored value
        if hash_program in self.succ[S]:
            return self.succ[S][hash_program]

        # we first add all potential successors of program in heaps[S],
        # and only then pop: with cyclic grammars, the successors of the arguments
        # of program may only be known at this point
        if program is not None:
            self.__add_successors__(program, S)

        # the successor is the next element in the heap
        if len(self.heaps[S]) == 0:
            return None  # the heap is empty: there are no successors from S
        element = heappop(self.heaps[S])
        succ = element.program
        while succ in self.deleted:
            # a deleted program is never queried so it must be expanded here
            self.__add_successors__(succ, S)
            if len(self.heaps[S]) == 0:
                return None
            element = heappop(self.heaps[S])
            succ = element.program

        self.succ[S][hash_program] = succ  # we store the successor
        self.pred[S][hash(succ)] = hash_program  # we store the predecessor
        return succ

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        In other words, other will no longer be generated through heap search
        """
        our_hash = hash(other)
        self.deleted.add(other)
        for S in self.G.rules:
            if our_hash in self.pred[S] and our_hash in self.succ[S]:
                pred_hash = self.pred[S][our_hash]
                nxt = self.succ[S][our_hash]
                self.succ[S][pred_hash] = nxt
                self.pred[S][hash(nxt)] = pred_hash

    @abstractmethod
    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Ordered:
        pass

    @abstractmethod
    def adjust_priority_for_start(
        self, priority: Ordered, start: Tuple[Type, U]
    ) -> Ordered:
        pass

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "UHSEnumerator[U, V, W]":
        assert isinstance(G, ProbUGrammar)
        enum = self.__class__(G, self.threshold)
        enum.deleted = self.deleted.copy()
        return enum


class UHeapSearch(UHSEnumerator[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W], threshold: float = 0) -> None:
        super().__init__(G, -threshold)
        self.probabilities: Dict[Program, Dict[Tuple[Type, U], float]] = defaultdict(
            lambda: {}
        )

    def adjust_priority_for_start(
        self, priority: Ordered, start: Tuple[Type, U]
    ) -> Ordered:
        return priority * self.G.start_tags[start]  # type: ignore

    def __prob__(
        self, succ: Function, S: Tuple[Type, U], Si: Tuple[Type, U], info: W, i: int
    ) -> float:
        # Si is non-terminal symbol used to derive the i-th argument
        arg = succ.arguments[i]
        probability = self.probabilities[arg][Si]
        if i + 1 >= len(succ.arguments):
            return probability
        for info, lst in self.G.derive_all(info, Si, arg, hints=self._keys):
            Sip1 = lst[-1][0]
            prob = self.__prob__(succ, S, Sip1, info, i + 1)
            if prob >= 0:
                return prob * probability
        return -1

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> float:
        # Try to hit cached probability
        if new_program in self.probabilities and S in self.probabilities[new_program]:
            return -self.probabilities[new_program][S]
        if isinstance(new_program, Function):
            F = new_program.function
            v = self._keys[S][new_program]
            out = self.G.derive_specific(self.G.start_information(), S, F, v)  # type: ignore
            assert out is not None
            information, Si = out
            # We guarantee that F is a Primitive
            probability = self.G.probabilities[S][F][__wrap__(v)]  # type: ignore
            probability *= self.__prob__(new_program, S, Si, information, 0)
            assert (
                probability >= 0
            ), f"Could not find {new_program} in {S} [{self.G.__contains_rec__(new_program, S, self.G.start_information())[0]}]"
        else:
            possibles = self.G.derive_all(self.G.start_information(), S, new_program)
            assert len(possibles) == 1
            v = __wrap__(possibles[0][-1][-1][-1])  # type: ignore
            probability = self.G.probabilities[S][new_program][v]  # type: ignore
        self.probabilities[new_program][S] = probability
        return -probability


def enumerate_prob_u_grammar(
    G: ProbUGrammar[U, V, W], threshold: float = 0
) -> UHeapSearch[U, V, W]:
    return UHeapSearch(G, threshold)


class BucketSearch(UHSEnumerator[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W], bucket_size: int) -> None:
        super().__init__(G)
        self.bucket_tuples: Dict[Program, Dict[Tuple[Type, U], Bucket]] = defaultdict(
            lambda: {}
        )
        self.bucket_size = bucket_size

    def adjust_priority_for_start(
        self, priority: Ordered, start: Tuple[Type, U]
    ) -> Ordered:
        return priority.add_prob_uniform(self.G.start_tags[start])  # type: ignore

    def __prob__(
        self, succ: Function, S: Tuple[Type, U], Si: Tuple[Type, U], info: W, i: int
    ) -> Optional[Bucket]:
        # Si is non-terminal symbol used to derive the i-th argument
        arg = succ.arguments[i]
        if Si not in self.bucket_tuples[arg]:
            return None
        bucket = self.bucket_tuples[arg][Si]
        if i + 1 >= len(succ.arguments):
            return bucket
        for info, lst in self.G.derive_all(info, Si, arg):
            Sip1 = lst[-1][0]
            prob = self.__prob__(succ, S, Sip1, info, i + 1)
            if prob:
                return prob + bucket
        return None

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Bucket:
        new_bucket = Bucket(self.bucket_size)
        if isinstance(new_program, Function):
            F = new_program.function
            for information, lst in self.G.derive_all(self.G.start_information(), S, F):
                Si = lst[-1][0]
                v = __wrap__(lst[-1][-1])
                # We guarantee that F is a Primitive
                probability = self.G.probabilities[S][F][v]  # type: ignore
                prob = self.__prob__(new_program, S, Si, information, 0)
                if prob:
                    break
            new_bucket.add_prob_uniform(probability)
            assert prob is not None
            new_bucket += prob
        else:
            possibles = self.G.derive_all(self.G.start_information(), S, new_program)
            assert len(possibles) == 1
            v = __wrap__(possibles[0][-1][-1][-1])
            probability = self.G.probabilities[S][new_program][v]  # type: ignore
            new_bucket.add_prob_uniform(probability)
        self.bucket_tuples[new_program][S] = new_bucket
        return new_bucket


def enumerate_bucket_prob_u_grammar(
    G: ProbUGrammar[U, V, W], bucket_size: int
) -> BucketSearch[U, V, W]:
    return BucketSearch(G, bucket_size)
