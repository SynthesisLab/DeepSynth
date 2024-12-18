from synth.semantic.evaluator import DSLEvaluator
from synth.syntax import DSL, auto_type

from examples.pbe.sygus.task_generator_bitvector import reproduce_bitvector_dataset

__syntax = auto_type(
    {
        "bvnot": "bv -> bv",
        "bvxor": "bv -> bv -> bv",
        "bvand": "bv -> bv -> bv",
        "bvor": "bv -> bv -> bv",
        "bvneg": "bv -> bv",
        "bvadd": "bv -> bv -> bv",
        "bvmul": "bv -> bv -> bv",
        "bvudiv": "bv -> bv -> bv",
        "bvurem": "bv -> bv -> bv",
        "bvlshr": "bv -> bv -> bv",
        "bvashr": "bv -> bv -> bv",
        "bvshl": "bv -> bv -> bv",
        "bvsdiv": "bv -> bv -> bv",
        "bvsrem": "bv -> bv -> bv",
        "bvsub": "bv -> bv -> bv",
        "#x0000000000000000": "bv",
        "#x0000000000000001": "bv",
        "#xffffffffffffffff": "bv",
        "=": "bv -> bv -> bool",
        "bvsle": "bv -> bv -> bool",
        "bvule": "bv -> bv -> bool",
        "bvugt": "bv -> bv -> bool",
        "ite": "bool -> bv -> bv -> bv",
    }
)

MASK = (1 << 64) - 1

__semantics = {
    "bvnot": lambda x: ~x,
    "bvxor": lambda x: lambda y: x ^ y,
    "bvand": lambda x: lambda y: x & y,
    "bvor": lambda x: lambda y: x | y,
    "bvneg": lambda x: -x,
    "bvadd": lambda x: lambda y: x + y,
    "bvmul": lambda x: lambda y: x * y,
    "bvudiv": lambda x: lambda y: abs(x // y) if y != 0 else 0,
    "bvurem": lambda x: lambda y: abs(x % y) if y != 0 else 0,
    "bvlshr": lambda x: lambda y: (x >> y if x > 0 else -(-x >> y)) if y > 0 else x,
    "bvashr": lambda x: lambda y: x >> y if y > 0 else x,
    "bvshl": lambda x: lambda y: (x << y) & MASK if y > 0 and y < 63 else 0,
    "bvsdiv": lambda x: lambda y: x // y if y != 0 else 0,
    "bvsrem": lambda x: lambda y: x % y if y != 0 else 0,
    "bvsub": lambda x: lambda y: x - y,
    "#x0000000000000000": 0,
    "#x0000000000000001": 1,
    "#xffffffffffffffff": MASK,
    "=": lambda x: lambda y: x == y,
    "bvsle": lambda x: lambda y: x <= y,
    "bvule": lambda x: lambda y: abs(x) <= abs(y),
    "bvugt": lambda x: lambda y: abs(x) > abs(y),
    "ite": lambda b: lambda x: lambda y: x if b else y,
}
__forbidden = {
    ("bvnot", 0): {"bvnot", "bvneg"},
    ("bvneg", 0): {"bvnot", "bvneg"},
    ("bvxor", 0): {"bvxor", "#x0000000000000000", "#xffffffffffffffff"},
    ("bvxor", 1): {"#x0000000000000000", "#xffffffffffffffff"},
    ("bvor", 0): {"bvor", "#x0000000000000000", "#xffffffffffffffff"},
    ("bvxor", 1): {"#x0000000000000000", "#xffffffffffffffff"},
    ("bvand", 0): {"bvand", "#x0000000000000000", "#xffffffffffffffff"},
    ("bvand", 1): {"#x0000000000000000", "#xffffffffffffffff"},
    ("bvadd", 0): {"bvadd", "#x0000000000000000"},
    ("bvadd", 1): {"#x0000000000000000"},
    ("bvmul", 0): {"bvmul", "#x0000000000000000", "#x0000000000000001"},
    ("bvmul", 1): {"#x0000000000000000", "#x0000000000000001"},
}
dsl = DSL(__syntax, __forbidden)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
constant_types = {auto_type("bv")}
# evaluator.skip_exceptions.add(ZeroDivisionError)
# evaluator.skip_exceptions.add(ValueError)
# evaluator.skip_exceptions.add(TypeError)
# TODO: lexicon
lexicon = list([i for i in range(100000)])
