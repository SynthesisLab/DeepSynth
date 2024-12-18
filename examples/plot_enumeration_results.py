from collections import OrderedDict, defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import pltpublish as pub
import csv

from plot_helper import (
    plot_y_wrt_x,
    make_plot_wrapper,
)


__DATA__ = {
    "time": (0, "Time (in s)"),
    "programs": (1, "Programs Enumerated"),
    "queued": (2, "Queue Size"),
    "banked": (3, "Programs in Banks"),
    "non_terminals": (4, "Non Terminals in Grammar"),
    "rules": (5, "Derivation Rules in Grammar"),
}


def load_data(output_file: str, verbose: bool = False) -> Dict[str, Dict[int, List]]:
    # Dict[name, data]
    methods = {}

    # filename should end with a specific pattern
    name = output_file[:-4]
    if not (name.endswith("_detailed") or name.endswith("_growth")):
        if verbose:
            print(f"filename:{output_file} does not seem valid!")
        return {}
    trace = []
    with open(output_file, "r") as fd:
        reader = csv.reader(fd)
        trace = [tuple(row) for row in reader]
        # Pop columns names
        columns = {name: ind for ind, name in enumerate(trace.pop(0))}
        indices = [
            columns["search"],
            columns["time"],
            columns["programs"],
            columns["queue"],
            columns["bank"],
            columns["non_terminals"],
            columns["derivation_rules"],
            columns.get("seed", -1),
        ]
        data = [tuple(row[k] if k >= 0 else 0 for k in indices) for row in trace]
        if len(data) == 0:
            if verbose:
                print(f"filename:{output_file} is empty!")
            return {}
        agg = defaultdict(dict)
        for row in data:
            seed = int(row[-1])
            if seed not in agg[row[0]]:
                agg[row[0]][seed] = []
            agg[row[0]][seed].append(row[1:-1])
        for name, data in agg.items():
            if name == "beap_search":
                name = "eco_search_w/o_buckets"
            name = name.replace("_", " ")
            if "cd" in name:
                continue
            if name not in methods:
                methods[name] = {}
            # Save data for method
            for seed, vals in data.items():
                methods[name][seed] = [tuple(float(x) for x in row) for row in vals]
            # Backend support onl yseeded data so we register every data as seed 1
    return methods


# Generate all possible combinations
__PLOTS__ = {}
for ydata in list(__DATA__.keys()):
    for xdata in list(__DATA__.keys()):
        if xdata == ydata:
            continue
        __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_wrapper(
            plot_y_wrt_x,
            __DATA__[xdata],
            __DATA__[ydata],
            cumulative=False,
            logy=xdata == "non_terminals",
        )

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument(
        "file",
        type=str,
        help="data file to load",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose mode",
    )
    parser.add_argument("plots", nargs="+", choices=list(__PLOTS__.keys()))
    parameters = parser.parse_args()
    output_file: str = parameters.file
    verbose: bool = parameters.verbose
    plots: List[str] = parameters.plots

    # Load data
    pub.setup()
    methods = load_data(output_file, verbose)
    # Check we have at least one file
    if len(methods) == 0:
        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)
    # Order by name so that it is always the same color for the same methods if diff. DSL
    ordered_methods = OrderedDict()
    for met in sorted(methods.keys()):
        ordered_methods[met] = methods[met]
    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, ordered_methods)
    pub.save_fig("./current.png")
    plt.show()
