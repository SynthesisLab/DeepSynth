from collections import defaultdict
from heapq import heappush, heappop
from typing import (
    Any,
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
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from synth.filter.filter import Filter
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.program import Program, Function
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class HeapElement:
    priority: Ordered
    program: Program = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.priority}, {self.program})"


class HSEnumerator(
    ProgramEnumerator[None],
    ABC,
    Generic[U, V, W],
):
    def __init__(
        self,
        G: ProbDetGrammar[U, V, W],
        threshold: Optional[Ordered] = None,
        filter: Optional[Filter[Program]] = None,
    ) -> None:
        super().__init__(filter)
        self.current: Optional[Program] = None
        self.threshold = threshold

        self.deleted: Set[Program] = set()

        self.G = G
        self.start = G.start
        self.rules = G.rules
        symbols = [S for S in self.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[Tuple[Type, U], List[HeapElement]] = {S: [] for S in symbols}

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

        self._init: Set[Tuple[Type, U]] = set()

        self.max_priority: Dict[
            Union[Tuple[Type, U], Tuple[Tuple[Type, U], Program]], Program
        ] = {}

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "heap-search"

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        self._reevaluate_()
        # Now we can init the heaps
        for S in self.G.rules:
            self.__init_heap__(S)
        # And now that ALL heaps have been init
        # Query(S, None) for all
        for S in self.G.rules:
            self.query(S, None)
        while True:
            program = self.query(self.start, self.current)
            if program is None:
                return
            self.current = program
            if not self._should_keep_subprogram(program):
                self.deleted.add(program)
                continue
            yield program

    def __compute_max_prio__(
        self,
        S: Tuple[Type, U],
        best_program: Optional[Program] = None,
        best_priority: Optional[Ordered] = None,
        init_arguments: bool = True,
    ) -> Tuple[Optional[Program], Optional[Ordered]]:
        for P in self.rules[S]:
            nargs = self.G.arguments_length_for(S, P)
            P_unique: Program = P
            if nargs > 0:
                arguments = []
                information, current = self.G.derive(self.G.start_information(), S, P)
                for _ in range(nargs):
                    if init_arguments:
                        self.__init_non_terminal__(current)
                    if current not in self.max_priority:
                        break
                    # Try to init sub Tuple[Type, U] in case they were not initialised
                    arguments.append(self.max_priority[current])
                    information, lst = self.G.derive_all(
                        information, current, arguments[-1]
                    )
                    current = lst[-1]
                if len(arguments) != nargs:
                    continue
                new_program = Function(
                    function=P_unique,
                    arguments=arguments,
                )
                P_unique = new_program
            priority = self.compute_priority(S, P_unique)
            self.max_priority[(S, P)] = P_unique
            if best_priority is None or priority < best_priority:
                best_program = P_unique
                best_priority = priority
        return best_program, best_priority

    def __init_non_terminal__(self, S: Tuple[Type, U]) -> None:
        if S in self._init:
            return
        if all((S, P) in self.max_priority for P in self.rules[S]):
            return
        self._init.add(S)
        # 1) Compute max probablities
        best_program, _ = self.__compute_max_prio__(S)

        self._init.remove(S)
        if best_program is not None:
            self.max_priority[S] = best_program

    def _reevaluate_(self) -> None:
        # compute the best program of each non-terminal as a fixpoint:
        # with cyclic grammars, the best program of a non-terminal can improve after
        # its first computation, and __init_non_terminal__ does not recompute it.
        # Arguments are not initialised recursively: with cycles, this recursion
        # recomputes the same non-terminals an exponential number of times
        changed = True
        while changed:
            changed = False
            for S in list(self.G.rules.keys()):
                old = [self.max_priority.get((S, P), None) for P in self.rules[S]]
                best_program, _ = self.__compute_max_prio__(S, init_arguments=False)
                if best_program is not None:
                    self.max_priority[S] = best_program
                if old != [self.max_priority.get((S, P), None) for P in self.rules[S]]:
                    changed = True

    def __init_heap__(self, S: Tuple[Type, U]) -> None:
        # 2) add P(max(S1),max(S2), ...) to self.heaps[S]
        for P in self.rules[S]:
            if (S, P) not in self.max_priority:
                continue  # P does not derive any program
            program = self.max_priority[(S, P)]
            hash_program = hash(program)
            # Remark: the program cannot already be in self.heaps[S]
            assert hash_program not in self.hash_table_program[S]
            # Init heap all others so that query will work
            self.hash_table_program[S].add(hash_program)
            # we assume that the programs from max_probability
            # are represented by the same object
            priority = self.compute_priority(S, program)
            if not self.threshold or priority < self.threshold:
                heappush(
                    self.heaps[S],
                    HeapElement(priority, program),
                )

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

    def __add_successors__(self, succ: Program, S: Tuple[Type, U]) -> None:
        if isinstance(succ, Function):
            F = succ.function
            information, lst = self.G.derive_all(self.G.start_information(), S, F)
            S2 = lst[-1]
            args_len = self.G.arguments_length_for(S, F)  # type: ignore
            for i in range(args_len):
                # S2 is non-terminal symbol used to derive the i-th argument
                succ_sub_program = self.query(S2, succ.arguments[i])
                if succ_sub_program is not None:
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program
                    new_program = Function(F, new_arguments)
                    hash_new_program = hash(new_program)
                    if (
                        hash_new_program not in self.hash_table_program[S]
                        and new_program not in self.deleted
                    ):
                        self.hash_table_program[S].add(hash_new_program)
                        try:
                            priority: Ordered = self.compute_priority(S, new_program)
                            if not self.threshold or priority < self.threshold:
                                heappush(
                                    self.heaps[S], HeapElement(priority, new_program)
                                )
                        except KeyError:
                            pass
                if i + 1 < args_len:
                    information, lst = self.G.derive_all(
                        information, S2, succ.arguments[i]
                    )
                    S2 = lst[-1]

    def query(self, S: Tuple[Type, U], program: Optional[Program]) -> Optional[Program]:
        """
        computing the successor of program from S
        """
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

    @abstractmethod
    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Ordered:
        pass

    def programs_in_banks(self) -> int:
        return sum(len(val) for val in self.succ.values())

    def programs_in_queues(self) -> int:
        return sum(len(val) for val in self.heaps.values())

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "HSEnumerator[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G, self.threshold)
        enum.deleted = self.deleted.copy()
        return enum


class HeapSearch(HSEnumerator[U, V, W]):
    def __init__(self, G: ProbDetGrammar[U, V, W], threshold: float = 0) -> None:
        super().__init__(G, -threshold)
        self.probabilities: Dict[Program, Dict[Tuple[Type, U], float]] = defaultdict(
            lambda: {}
        )

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> float:
        if new_program in self.probabilities and S in self.probabilities[new_program]:
            return -self.probabilities[new_program][S]
        if isinstance(new_program, Function):
            F = new_program.function
            # We guarantee that F is a Primitive
            new_arguments = new_program.arguments
            probability = self.G.probabilities[S][F]  # type: ignore
            information, lst = self.G.derive_all(self.G.start_information(), S, F)
            S2 = lst[-1]
            args_len = self.G.arguments_length_for(S, F)  # type: ignore
            for i in range(args_len):
                arg = new_arguments[i]
                probability *= self.probabilities[arg][S2]
                if i + 1 < args_len:
                    information, lst = self.G.derive_all(information, S2, arg)
                    S2 = lst[-1]
        else:
            probability = self.G.probabilities[S][new_program]  # type: ignore
        self.probabilities[new_program][S] = probability
        return -probability


def enumerate_prob_grammar(
    G: ProbDetGrammar[U, V, W], threshold: float = 0
) -> HeapSearch[U, V, W]:
    return HeapSearch(G, threshold)


class Bucket(Ordered):
    def __init__(self, size: int = 3):
        self.elems = [0 for _ in range(size)]
        self.size = size

    def __str__(self) -> str:
        s = "("
        for elem in self.elems:
            s += "{},".format(elem)
        s = s[:-1] + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: "Bucket") -> bool:
        if self.size == 0:
            return False
        for i in range(self.size):
            if self.elems[i] < other.elems[i]:
                return True
            elif self.elems[i] > other.elems[i]:
                return False
        return False

    def __gt__(self, other: "Bucket") -> bool:
        return other.__lt__(self)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Bucket) and all(
            self.elems[i] == other.elems[i] for i in range(self.size)
        )

    def __le__(self, other: "Bucket") -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: "Bucket") -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __iadd__(self, other: "Bucket") -> "Bucket":
        if self.size == other.size:
            for i in range(self.size):
                self.elems[i] += other.elems[i]
            return self
        else:
            raise RuntimeError(
                "size mismatch, Bucket{}: {}, Bucket{}: {}".format(
                    self, self.size, other, other.size
                )
            )

    def __add__(self, other: "Bucket") -> "Bucket":
        if self.size == other.size:
            dst = Bucket(self.size)
            dst += self
            dst += other
            return dst
        else:
            raise RuntimeError(
                "size mismatch, Bucket{}: {}, Bucket{}: {}".format(
                    self, self.size, other, other.size
                )
            )

    def add_prob_uniform(self, probability: float) -> "Bucket":
        """
        Given a probability add 1 in the relevant bucket assuming buckets are linearly distributed.
        """
        index = self.size - int(probability * self.size) - 1
        self.elems[index] += 1
        return self


class BucketSearch(HSEnumerator[U, V, W]):
    def __init__(self, G: ProbDetGrammar[U, V, W], bucket_size: int) -> None:
        super().__init__(G)
        self.bucket_tuples: Dict[Program, Dict[Tuple[Type, U], Bucket]] = defaultdict(
            lambda: {}
        )
        self.bucket_size = bucket_size

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Bucket:
        new_bucket = Bucket(self.bucket_size)
        if isinstance(new_program, Function):
            F = new_program.function
            new_arguments = new_program.arguments
            new_bucket.add_prob_uniform(self.G.probabilities[S][F])  # type: ignore

            information, lst = self.G.derive_all(self.G.start_information(), S, F)
            S2 = lst[-1]
            for i in range(self.G.arguments_length_for(S, F)):  # type: ignore
                arg = new_arguments[i]
                new_bucket += self.bucket_tuples[arg][S2]
                information, lst = self.G.derive_all(information, S2, arg)
                S2 = lst[-1]
        else:
            probability = self.G.probabilities[S][new_program]  # type: ignore
            new_bucket.add_prob_uniform(probability)
        self.bucket_tuples[new_program][S] = new_bucket
        return new_bucket


def enumerate_bucket_prob_grammar(
    G: ProbDetGrammar[U, V, W], bucket_size: int
) -> BucketSearch[U, V, W]:
    return BucketSearch(G, bucket_size)
