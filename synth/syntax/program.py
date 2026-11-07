from abc import ABC, abstractmethod
from typing import Dict, Generator, List as TList, Any, Optional, Set, Tuple
import itertools

from synth.syntax.type_system import (
    PrimitiveType,
    Type,
    UnknownType,
)
from synth.syntax.type_helper import FunctionType


class Program(ABC):
    """
    Object that represents a program: a lambda term with basic primitives.
    """

    def __init__(self, type: Type) -> None:
        self.type = type
        self.hash: int = 0

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self) -> str:
        return self.__str__()

    def uses_variables(self) -> bool:
        """
        Returns true if a variable is used.
        """
        return False

    def used_variables(self) -> Set[int]:
        """
        Returns the set of used variables numbers in this program.
        """
        s: Set[int] = set()
        self.__add_used_variables__(s)
        return s

    def __add_used_variables__(self, vars: Set[int]) -> None:
        pass

    def is_constant(self) -> bool:
        """
        Returns true if this program is an instance of a Constant.
        """
        return False

    def is_invariant(self, constant_types: Set[PrimitiveType]) -> bool:
        """SHOULD BE DELETED"""
        return True

    def count_constants(self) -> int:
        """
        Returns the number of constants that are present in this program.
        """
        return int(self.is_constant())

    def constants(self) -> Generator[Optional["Constant"], None, None]:
        """
        Iterates over all constants of this program, yields a None only if the program does NOT contain any constant.
        """
        yield None

    @abstractmethod
    def clone(self) -> "Program":
        """
        Produces an exact clone (deep copy) of this program.
        """
        pass

    def all_constants_instantiation(
        self, constants: Dict[Type, TList[Any]]
    ) -> Generator["Program", None, None]:
        yield self

    def size(self) -> int:
        """
        Returns the program's size.
        """
        return 1

    def depth(self) -> int:
        """
        Returns the program's depth seen as a tree.
        """
        return 1

    def depth_first_iter(self) -> Generator["Program", None, None]:
        """
        Depth first iteration over all objects that this program is built on.

        ``Function(P1, [P2, Function(P3, [P4])]).depth_first_iter()`` will yield
        P1, P2, P3, P4, Function(P3, [P4]), Function(P1, [P2, Function(P3, [P4])])
        """
        yield self

    def pretty_print(self) -> TList[str]:
        """
        Represents this program as a list of operations.
        """
        defined: Dict["Program", Tuple[int, str, str]] = {}
        self.__pretty_print__(defined, 0)
        data = sorted(defined.values())
        order = [x[2] for x in data if len(x[2]) > 0]
        return order

    def __pretty_print__(
        self,
        defined: Dict["Program", Tuple[int, str, str]],
        last: int,
    ) -> int:
        if self not in defined:
            defined[self] = (0, str(self), "")
        return last

    def __contains__(self, other: "Program") -> bool:
        return self == other

    @staticmethod
    @abstractmethod
    def __pickle__(o: "Program") -> Tuple:
        pass

    def type_checks(self) -> bool:
        return True


class Variable(Program):
    """
    Represents a variable (argument) in a program.

    Parameters:
    -----------
    - variable: the argument index
    - type: the variable's type
    """

    __hash__ = Program.__hash__

    def __init__(self, variable: int, type: Type = UnknownType()):
        super().__init__(type)
        self.variable: int = variable
        self.hash = hash((self.variable, self.type))

    def is_invariant(self, constant_types: Set[PrimitiveType]) -> bool:
        return False

    def uses_variables(self) -> bool:
        return True

    def clone(self) -> "Program":
        return Variable(self.variable, self.type)

    def __add_used_variables__(self, vars: Set[int]) -> None:
        vars.add(self.variable)

    def __str__(self) -> str:
        return "var" + format(self.variable)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Variable) and self.variable == other.variable

    def __pickle__(o: Program) -> Tuple:  # type: ignore[override]
        return Variable, (o.variable, o.type)  # type: ignore


class Constant(Program):
    """
    Represents a constant that may or may be assigned a value.

    Parameters:
    -----------
    - type: the constant's type
    - value: the constant's value
    - has_value: explicitly indicate that this constant has been assigned a value
    """

    __hash__ = Program.__hash__

    def __init__(self, type: Type, value: Any = None, has_value: Optional[bool] = None):
        super().__init__(type)
        self.value = value
        self._has_value = has_value or value is not None
        self.hash = hash((str(self.value), self._has_value, self.type))

    def has_value(self) -> bool:
        """
        Returns true if and only if this constant has been assigned a value.
        """
        return self._has_value

    def constants(self) -> Generator[Optional["Constant"], None, None]:
        yield self

    def clone(self) -> "Program":
        return Constant(self.type, self.value, self._has_value)

    def all_constants_instantiation(
        self, constants: Dict[Type, TList[Any]]
    ) -> Generator["Program", None, None]:
        for val in constants[self.type]:
            yield Constant(self.type, val)

    def __str__(self) -> str:
        if self.has_value():
            return format(self.value)
        return f"<{self.type}>"

    def is_constant(self) -> bool:
        return True

    def assign(self, value: Any) -> None:
        """
        Assign a value to this constant.
        """
        self._has_value = True
        self.value = value
        self.hash = hash((str(self.value), self._has_value, self.type))

    def reset(self) -> None:
        """
        Reset this constant as if no value was assigned to it.
        """
        self._has_value = False
        self.value = None
        self.hash = hash((str(self.value), self._has_value, self.type))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Constant)
            and self.type == other.type
            and self.value == other.value
        )

    def __pickle__(o: Program) -> Tuple:  # type: ignore[override]
        return Constant, (o.type, o.value, o._has_value)  # type: ignore


class Function(Program):
    """
    Represents a function call, it supports partial application and the type is guessed automatically.

    Parameters:
    -----------
    - function: the called function
    - arguments: the arguments to the function
    """

    __hash__ = Program.__hash__

    def __init__(self, function: Program, arguments: TList[Program]):
        # Build automatically the type of the function
        type = function.type
        args = type.arguments()[len(arguments) :]
        my_type = FunctionType(*args, type.returns())

        super().__init__(my_type)
        self.function = function
        self.arguments = arguments
        self.hash = hash((self.function, *self.arguments))

    def __pickle__(o: Program) -> Tuple:  # type: ignore[override]
        return Function, (o.function, o.arguments)  # type: ignore

    def __str__(self) -> str:
        if len(self.arguments) == 0:
            return format(self.function)
        else:
            s = "(" + format(self.function)
            for arg in self.arguments:
                s += " " + format(arg)
            return s + ")"

    def type_checks(self) -> bool:
        return all(
            arg.type.is_instance(f_arg_t)
            for arg, f_arg_t in zip(self.arguments, self.function.type.arguments())
        )

    def __pretty_print__(
        self,
        defined: Dict["Program", Tuple[int, str, str]],
        last: int,
    ) -> int:
        if self in defined:
            return last
        out = []
        for arg in self.arguments:
            last = arg.__pretty_print__(defined, last)
            out.append(defined[arg][1])
        var_name = f"x{last}"
        defined[self] = (
            last,
            var_name,
            f"{var_name}: {self.type} = {self.function}({', '.join(out)})",
        )
        return last + 1

    def clone(self) -> "Program":
        return Function(self.function.clone(), [x.clone() for x in self.arguments])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Function)
            and self.function == other.function
            and len(self.arguments) == len(other.arguments)
            and self.arguments == other.arguments
        )

    def is_constant(self) -> bool:
        return self.function.is_constant() and all(
            arg.is_constant() for arg in self.arguments
        )

    def uses_variables(self) -> bool:
        return self.function.uses_variables() or any(
            arg.uses_variables() for arg in self.arguments
        )

    def constants(self) -> Generator[Optional["Constant"], None, None]:
        g = [self.function.constants()] + [arg.constants() for arg in self.arguments]
        for gen in g:
            c = next(gen, None)
            while c is not None:
                yield c
                c = next(gen, None)

    def all_constants_instantiation(
        self, constants: Dict[Type, TList[Any]]
    ) -> Generator["Program", None, None]:
        for f in self.function.all_constants_instantiation(constants):
            possibles = [
                list(arg.all_constants_instantiation(constants))
                for arg in self.arguments
            ]
            for args in itertools.product(*possibles):
                yield Function(f, list(args))

    def is_invariant(self, constant_types: Set[PrimitiveType]) -> bool:
        return self.function.is_invariant(constant_types) and all(
            arg.is_invariant(constant_types) for arg in self.arguments
        )

    def count_constants(self) -> int:
        return self.function.count_constants() + sum(
            [arg.count_constants() for arg in self.arguments]
        )

    def size(self) -> int:
        return self.function.size() + sum([arg.size() for arg in self.arguments])

    def depth(self) -> int:
        return 1 + max(
            self.function.depth(), max(arg.depth() for arg in self.arguments)
        )

    def __add_used_variables__(self, vars: Set[int]) -> None:
        for el in [self.function] + self.arguments:
            el.__add_used_variables__(vars)

    def depth_first_iter(self) -> Generator["Program", None, None]:
        for sub in self.function.depth_first_iter():
            yield sub
        for arg in self.arguments:
            for sub in arg.depth_first_iter():
                yield sub
        yield self

    def __contains__(self, other: "Program") -> bool:
        return self == other or any(
            other in x for x in [self.function] + self.arguments
        )


class Lambda(Program):
    __hash__ = Program.__hash__

    def __init__(self, body: Program, type: Type = UnknownType()):
        super().__init__(type)
        self.body = body
        self.hash = hash(94135 + hash(self.body))

    def clone(self) -> "Program":
        return Lambda(self.body.clone())

    def __pickle__(o: Program) -> Tuple:  # type: ignore[override]
        return Lambda, (o.body, o.type)  # type: ignore

    def __str__(self) -> str:
        return "(lambda " + format(self.body) + ")"

    def constants(self) -> Generator[Optional["Constant"], None, None]:
        for x in self.body.constants():
            yield x

    def all_constants_instantiation(
        self, constants: Dict[Type, TList[Any]]
    ) -> Generator["Program", None, None]:
        for val in self.body.all_constants_instantiation(constants):
            yield Lambda(val)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Lambda) and self.body == other.body

    def uses_variables(self) -> bool:
        return self.body.uses_variables()

    def __add_used_variables__(self, vars: Set[int]) -> None:
        return self.body.__add_used_variables__(vars)

    def depth(self) -> int:
        return 1 + self.body.depth()

    def depth_first_iter(self) -> Generator["Program", None, None]:
        for sub in self.body.depth_first_iter():
            yield sub
        yield self

    def __contains__(self, other: "Program") -> bool:
        return self == other or other in self.body


class Primitive(Program):
    """
    Represents a DSL primitive.

    Parameters:
    -----------
    - primitive: the name of the primitive
    - type: the type of the primitive
    """

    __hash__ = Program.__hash__

    def __init__(self, primitive: str, type: Type = UnknownType()):
        super().__init__(type)
        self.primitive = primitive
        self.hash = hash((self.primitive, self.type))

    def clone(self) -> "Program":
        return Primitive(self.primitive, self.type)

    def __pickle__(o: Program) -> Tuple:  # type: ignore[override]
        return Primitive, (o.primitive, o.type)  # type: ignore

    def is_invariant(self, constant_types: Set[PrimitiveType]) -> bool:
        return not (self.type in constant_types)

    def __str__(self) -> str:
        """
        representation without type
        """
        return format(self.primitive)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Primitive)
            and self.primitive == other.primitive
            and self.type == other.type
        )


import copyreg

for cls in [Primitive, Constant, Lambda, Function, Variable]:
    copyreg.pickle(cls, cls.__pickle__)  # type: ignore
