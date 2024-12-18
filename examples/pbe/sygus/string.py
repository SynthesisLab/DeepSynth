from synth.semantic.evaluator import DSLEvaluator
from synth.syntax import DSL, auto_type

from examples.pbe.sygus.task_generator_string import reproduce_string_dataset

__syntax = auto_type(
    {
        "concat": "string -> string -> string",
        "replace": "string -> string -> string -> string",
        "substr": "string -> int -> int -> string",
        "ite": "bool -> 'a[string|int] -> 'a[string|int] -> 'a[string|int]",
        "int2str": "int -> str",
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

__semantics = {
    "concat": lambda x: lambda y: "" + x + y,
    "replace": lambda x: lambda y: lambda z: x.replace(y, z, 1),
    "substr": lambda x: lambda left: lambda right: x[left:right]
    if left < right and left > 0 and right < len(x)
    else "",
    "ite": lambda b: lambda x: lambda y: x if b else y,
    "int2str": str,
    "at": lambda x: lambda pos: x[pos] if pos > 0 and pos < len(x) else "",
    "lower": lambda x: x.lower(),
    "upper": lambda x: x.upper(),
    "str2int": int,
    "+": lambda x: lambda y: x + y,
    "-": lambda x: lambda y: x - y,
    "len": lambda x: len(x),
    "indexof": lambda x: lambda y: lambda pos: x.find(y, pos),
    "firstindexof": lambda x: lambda y: x.find(y),
    "*": lambda x: lambda y: x * y,
    "%": lambda x: lambda y: x % y,
    "=": lambda x: lambda y: x == y,
    "contains": lambda x: lambda y: y in x,
    "prefixof": lambda x: lambda y: x.startswith(y),
    "suffixof": lambda x: lambda y: x.endswith(y),
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
