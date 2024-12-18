"""
Module to change to add your own DSL easily in all scripts.
Some constants may need to be changed directly in the script.
"""

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from synth.utils.import_utils import import_file_function


def __base_loader(
    name: str,
    keys: Iterable[Union[str, Tuple[str, str]]] = ["dsl", "evaluator", "lexicon"],
) -> Callable[[bool], Optional[SimpleNamespace]]:
    """
    Utility function that creates a simple loader for you if you only need
    > from name.name import X, Y, ...
    where "X, Y, ..." are elements of keys
    """

    return import_file_function(name, keys, ["pbe", "examples.pbe"])


# ======================================================================================
# ADD your line "my_dsl": my_loading_func
#   when the parameter to your loading_func is:
#       - False, we just want to check we CAN import
#       - True, we want to import everything and return it in a NameSpace
# The easiest way is to use the __base_loader(path without the .py to your python file defining the DSL, list of str)
# the second argument is the list of global variables that you want to make available;
# instead of a str you can also use a tuple to rename (name of the var in your file, name under which to make it available)
#
# /!\ Convention for names in your namespace:
#   dsl: DSL - the actual DSL
#   evaluator: Evaluator - the DSL's evaluator
#   lexicon: List - the DSL's lexicon
#   /!\ all of the following are optional:
#   constraints: List[str] - the list of constraints for sharpening
#   reproduce_dataset: Callable - synth.pbe.task_generator.reproduce_int_dataset like function
#   pretty_print_inputs: Callable[[List[Any]], str] - a function to change the default format to print the inputs to an example in a task
#   pretty_print_solution: Callable[[Any], str] - a function to change the default format to print the solution to a task
# =======================================================================================
__dsl_funcs: Dict[str, Callable[[bool], Optional[SimpleNamespace]]] = {
    "deepcoder": __base_loader("deepcoder.deepcoder"),
    "bitvectors": __base_loader(
        "sygus.bitvectors",
        [
            "dsl",
            "evaluator",
            "lexicon",
            ("reproduce_bitvector_dataset", "reproduce_dataset"),
        ],
    ),
    "string": __base_loader(
        "sygus.string",
        [
            "dsl",
            "evaluator",
            "lexicon",
            "constant_types",
            ("reproduce_string_dataset", "reproduce_dataset"),
        ],
    ),
    "deepcoder.raw": __base_loader(
        "deepcoder.deepcoder", [("dsl_raw", "dsl"), "evaluator", "lexicon"]
    ),
    "dreamcoder": __base_loader(
        "dreamcoder.dreamcoder", ["dsl", "evaluator", "lexicon", "constraints"]
    ),
    "regexp": __base_loader(
        "regexp.regexp",
        [
            "dsl",
            "evaluator",
            "lexicon",
            "pretty_print_inputs",
            "pretty_print_solution",
            ("reproduce_regexp_dataset", "reproduce_dataset"),
        ],
    ),
    "transduction": __base_loader(
        "transduction.transduction",
        [
            "dsl",
            "evaluator",
            "lexicon",
            "constant_types",
            "constraints",
            ("reproduce_transduction_dataset", "reproduce_dataset"),
        ],
    ),
    "calculator": __base_loader(
        "calculator.calculator",
        [
            "dsl",
            "evaluator",
            "lexicon",
            ("reproduce_calculator_dataset", "reproduce_dataset"),
        ],
    ),
    "karel": __base_loader(
        "karel.karel",
        [
            "dsl",
            "evaluator",
            "lexicon",
            "constraints",
            "pretty_print_inputs",
        ],
    ),
}
# =======================================================================================
# Nothing to change after this
# =======================================================================================

__buffer: Dict[str, Optional[SimpleNamespace]] = {}


def __wrapper_load(dsl_name: str, fully_load: bool = True) -> Optional[SimpleNamespace]:
    if dsl_name in __buffer:
        return __buffer[dsl_name]
    out = __dsl_funcs[dsl_name](fully_load)
    if fully_load:
        __buffer[dsl_name] = out
    return out


def available_DSL() -> List[str]:
    return [
        dsl_name
        for dsl_name in __dsl_funcs
        if __wrapper_load(dsl_name, False) is not None
    ]


def load_DSL(name: str) -> SimpleNamespace:
    if name not in __dsl_funcs:
        raise Exception(f"DSL {name} is not known!")
    return __wrapper_load(name, True)


def add_dsl_choice_arg(parser: ArgumentParser) -> None:
    dsls = available_DSL()
    parser.add_argument(
        "--dsl",
        type=str,
        default=dsls[0],
        choices=dsls,
        help=f"DSL choice (default: {dsls[0]})",
    )
