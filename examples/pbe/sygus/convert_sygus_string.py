"""
Convert the SyGuS string tasks of Bee Search (the 205 files of
https://github.com/lelis-research/BeeSearch/tree/main/sygus_string_tasks)
to a ProgSynth dataset.

Each task becomes a Task[PBEWithConstants]:
- the examples are the constraints (= (f in_1 ... in_n) out);
- the constants are the string and integer literals of the task's grammar, as in Bee Search;
- the solution is the define-fun of the file, when there is one, translated to the string DSL,
  and kept only if it is correct on all examples.

Run it as a module, since examples/pbe/sygus/string.py shadows the standard module string:
    python -m examples.pbe.sygus.convert_sygus_string <folder> -o sygus_strings.pickle
"""

import os
from typing import Any, Dict, List as TList, Optional, Tuple, Union

from synth import Dataset, Example, Task
from synth.specification import PBEWithConstants
from synth.syntax import (
    BOOL,
    INT,
    STRING,
    Constant,
    Function,
    FunctionType,
    Primitive,
    Program,
    Type,
    Variable,
)

from examples.pbe.sygus.string import dsl, evaluator

SExpr = Union[str, "StringLiteral", TList[Any]]


class StringLiteral(str):
    """A string literal, to distinguish "x" from the symbol x."""


SORTS: Dict[str, Type] = {"String": STRING, "Int": INT, "Bool": BOOL}

# SyGuS name -> DSL name
OPERATORS: Dict[str, str] = {
    "str.++": "concat",
    "str.replace": "replace",
    "str.at": "at",
    "int.to.str": "int2str",
    "str.from_int": "int2str",
    "str.to.int": "str2int",
    "str.to_int": "str2int",
    "ite": "ite",
    "str.substr": "substr",
    "+": "+",
    "-": "-",
    "str.len": "len",
    "str.indexof": "indexof",
    "=": "=",
    "str.prefixof": "prefixof",
    "str.suffixof": "suffixof",
    "str.contains": "contains",
}

PRIMITIVES: Dict[str, TList[Primitive]] = {}
for __p in dsl.list_primitives:
    PRIMITIVES.setdefault(__p.primitive, []).append(__p)


# =============================================================================
# S-expressions
# =============================================================================


def tokenize(text: str) -> TList[Union[str, StringLiteral]]:
    tokens: TList[Union[str, StringLiteral]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "()":
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        elif c == '"':
            # SMT-LIB: "" inside a string literal is an escaped quote
            j = i + 1
            value = []
            while True:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        value.append('"')
                        j += 2
                        continue
                    break
                value.append(text[j])
                j += 1
            tokens.append(StringLiteral("".join(value)))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '();"':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse(text: str) -> TList[SExpr]:
    tokens = tokenize(text)
    stack: TList[TList[Any]] = [[]]
    for token in tokens:
        if token == "(" and not isinstance(token, StringLiteral):
            stack.append([])
        elif token == ")" and not isinstance(token, StringLiteral):
            last = stack.pop()
            stack[-1].append(last)
        else:
            stack[-1].append(token)
    assert len(stack) == 1, "unbalanced parentheses"
    return stack[0]


def literal_value(expr: SExpr) -> Tuple[bool, Any, Optional[Type]]:
    """Returns (is_literal, value, type)."""
    if isinstance(expr, StringLiteral):
        return True, str(expr), STRING
    if isinstance(expr, list):
        # negative integers: (- 3)
        if len(expr) == 2 and expr[0] == "-":
            ok, v, t = literal_value(expr[1])
            if ok and t == INT:
                return True, -v, INT
        return False, None, None
    if expr in ("true", "false"):
        return True, expr == "true", BOOL
    try:
        return True, int(expr), INT
    except ValueError:
        return False, None, None


# =============================================================================
# Conversion
# =============================================================================


class UnsupportedSolution(Exception):
    pass


def to_program(expr: SExpr, variables: Dict[str, Tuple[int, Type]]) -> Program:
    ok, value, t = literal_value(expr)
    if ok:
        if t == BOOL:
            raise UnsupportedSolution("the DSL has no boolean constants")
        return Constant(t, value, True)  # type: ignore
    if isinstance(expr, str):
        if expr not in variables:
            raise UnsupportedSolution(f"unknown symbol {expr}")
        index, vtype = variables[expr]
        return Variable(index, vtype)
    head, args = expr[0], expr[1:]
    if head not in OPERATORS:
        raise UnsupportedSolution(f"unknown operator {head}")
    arguments = [to_program(arg, variables) for arg in args]
    arg_types = [arg.type for arg in arguments]
    for primitive in PRIMITIVES[OPERATORS[head]]:
        if primitive.type.arguments() == arg_types:
            return Function(primitive, arguments)
    raise UnsupportedSolution(f"no primitive {head} of argument types {arg_types}")


def grammar_literals(grammar: SExpr) -> Dict[Type, TList[Any]]:
    """The literals of the grammar of a synth-fun, by type, in order of appearance."""
    constants: Dict[Type, TList[Any]] = {STRING: [], INT: []}

    def visit(expr: SExpr) -> None:
        ok, value, t = literal_value(expr)
        if ok:
            if t in constants and value not in constants[t]:
                constants[t].append(value)
        elif isinstance(expr, list):
            for e in expr:
                visit(e)

    visit(grammar)
    return constants


def convert_file(path: str) -> Tuple[Task[PBEWithConstants], Optional[str]]:
    """Returns the task, and the reason why its solution was dropped, if any."""
    with open(path) as fd:
        commands = parse(fd.read())
    synth_fun = next(c for c in commands if isinstance(c, list) and c[0] == "synth-fun")
    fun_name = synth_fun[1]
    params = synth_fun[2]
    arg_types = [SORTS[sort] for _, sort in params]
    returns = SORTS[synth_fun[3]]
    type_request = FunctionType(*arg_types, returns)
    constants = grammar_literals(synth_fun[4])

    examples = []
    for c in commands:
        if not (isinstance(c, list) and c[0] == "constraint"):
            continue
        eq = c[1]
        assert eq[0] == "=", f"unsupported constraint in {path}"
        call, output = eq[1], eq[2]
        if not (isinstance(call, list) and call[0] == fun_name):
            call, output = output, call
        assert isinstance(call, list) and call[0] == fun_name, path
        inputs = []
        for arg in call[1:]:
            ok, v, _ = literal_value(arg)
            assert ok, f"non-literal input in {path}"
            inputs.append(v)
        ok, out, _ = literal_value(output)
        assert ok, f"non-literal output in {path}"
        examples.append(Example(inputs, out))

    solution: Optional[Program] = None
    dropped: Optional[str] = None
    definitions = [c for c in commands if isinstance(c, list) and c[0] == "define-fun"]
    if definitions:
        definition = definitions[0]
        variables = {name: (i, SORTS[sort]) for i, (name, sort) in enumerate(definition[2])}
        try:
            solution = to_program(definition[4], variables)
            for ex in examples:
                if evaluator.eval(solution, ex.inputs) != ex.output:
                    raise UnsupportedSolution("incorrect on its examples")
        except UnsupportedSolution as e:
            solution = None
            dropped = str(e)
    task = Task[PBEWithConstants](
        type_request,
        PBEWithConstants(examples, constants),
        solution,
        {"name": os.path.basename(path)},
    )
    return task, dropped


def convert_folder(folder: str, output_file: str, verbose: bool = False) -> Dataset:
    tasks = []
    dropped = 0
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".sl"):
            continue
        task, reason = convert_file(os.path.join(folder, file))
        if reason is not None:
            dropped += 1
            if verbose:
                print(f"{file}: solution dropped ({reason})")
        tasks.append(task)
    dataset = Dataset(tasks, metadata={"dataset": "sygus_strings", "source": folder})
    dataset.save(output_file)
    type_requests = dataset.type_requests()
    solved = sum(1 for t in tasks if t.solution is not None)
    print(
        f"Converted {len(tasks)} tasks, {len(type_requests)} type requests, "
        f"{solved} with a verified solution ({dropped} solutions dropped)"
    )
    if verbose:
        counts: Dict[Type, int] = {}
        for t in tasks:
            counts[t.type_request] = counts.get(t.type_request, 0) + 1
        for tr, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"\t{count}\t{tr}")
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert the SyGuS string tasks of Bee Search to ProgSynth format."
    )
    parser.add_argument(type=str, dest="folder", help="folder of .sl files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sygus_strings.pickle",
        help="output file (default: sygus_strings.pickle)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    params = parser.parse_args()
    convert_folder(params.folder, params.output, params.verbose)
