from synth.semantic.evaluator import DSLEvaluator
from synth.syntax import DSL, auto_type

from examples.pbe.sygus.task_generator_string import reproduce_string_dataset

__syntax = auto_type(
    {
        "concat": "string -> string -> string",
        "replace": "string -> string -> string -> string",
        "substr": "string -> int -> int -> string",
        "ite": "bool -> 'a[string|int] -> 'a[string|int] -> 'a[string|int]",
        "int2str": "int -> string",
        "at": "string -> int -> string",
        "lower": "string -> string",
        "upper": "string -> string",
        "str2int": "string -> int",
        "+": "int -> int -> int",
        "-": "int -> int -> int",
        "len": "string -> int",
        "indexof": "string -> string -> int -> int",
        "firstindexof": "string -> string -> int",
        "*": "int -> int -> int",
        "%": "int -> int -> int",
        "=": "'a[string|int] -> 'a[string|int] -> bool",
        "contains": "string -> string -> bool",
        "prefixof": "string -> string -> bool",
        "suffixof": "string -> string -> bool",
        ">": "int -> int -> bool",
        "<": "int -> int -> bool",
    }
)

# SMT-LIB (SyGuS) semantics of the string theory.
# lower, upper, firstindexof, *, %, < and > are not part of SyGuS.


def __substr(x: str, i: int, n: int) -> str:
    # str.substr s i n: substring of length n starting at i
    if i < 0 or i >= len(x) or n <= 0:
        return ""
    return x[i : i + n]


def __at(x: str, i: int) -> str:
    if i < 0 or i >= len(x):
        return ""
    return x[i]


def __str2int(x: str) -> int:
    # str.to.int: -1 unless x is a non-empty string of digits
    if len(x) == 0 or not all("0" <= c <= "9" for c in x):
        return -1
    return int(x)


def __int2str(x: int) -> str:
    # int.to.str: "" for negative numbers
    if x < 0:
        return ""
    return str(x)


def __indexof(x: str, y: str, i: int) -> int:
    if i < 0 or i > len(x):
        return -1
    return x.find(y, i)


__semantics = {
    "concat": lambda x: lambda y: "" + x + y,
    "replace": lambda x: lambda y: lambda z: x.replace(y, z, 1),
    "substr": lambda x: lambda i: lambda n: __substr(x, i, n),
    "ite": lambda b: lambda x: lambda y: x if b else y,
    "int2str": __int2str,
    "at": lambda x: lambda pos: __at(x, pos),
    "lower": lambda x: x.lower(),
    "upper": lambda x: x.upper(),
    "str2int": __str2int,
    "+": lambda x: lambda y: x + y,
    "-": lambda x: lambda y: x - y,
    "len": lambda x: len(x),
    "indexof": lambda x: lambda y: lambda pos: __indexof(x, y, pos),
    "firstindexof": lambda x: lambda y: x.find(y),
    "*": lambda x: lambda y: x * y,
    "%": lambda x: lambda y: x % y,
    "=": lambda x: lambda y: x == y,
    "contains": lambda x: lambda y: y in x,
    # str.prefixof x y: x is a prefix of y
    "prefixof": lambda x: lambda y: y.startswith(x),
    "suffixof": lambda x: lambda y: y.endswith(x),
    ">": lambda x: lambda y: x > y,
    "<": lambda x: lambda y: x < y,
}

dsl = DSL(__syntax)
dsl.instantiate_polymorphic_types(1)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
evaluator.skip_exceptions.add(ZeroDivisionError)
evaluator.skip_exceptions.add(ValueError)
evaluator.skip_exceptions.add(TypeError)
lexicon = list([chr(i) for i in range(32, 126)])
constant_types = {auto_type("int"), auto_type("string")}
