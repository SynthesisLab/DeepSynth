import re
import zlib

import numpy as np

import matplotlib.pyplot as plt

from typing import Any, List, Optional, Dict, Tuple

# Each algorithm has its own color, line style, marker and hatch, the same in all figures,
# so that the figures remain readable in grayscale.
# Colors: categorical palette in fixed order, a shade of the same blue for the bucket sizes of Eco Search.
__STYLES__: List[Tuple[List[str], Dict[str, Any]]] = [
    (
        ["eco search", "cd search"],
        dict(color="#2a78d6", linestyle="-", marker="o", hatch=""),
    ),
    (
        ["eco search w o buckets", "beap search"],
        dict(color="#eb6834", linestyle="--", marker="s", hatch="//"),
    ),
    (["heap search"], dict(color="#1baf7a", linestyle="-.", marker="^", hatch="\\\\")),
    (["bee search"], dict(color="#eda100", linestyle=":", marker="D", hatch="..")),
    (
        ["bee search buckets", "bee search with buckets"],
        dict(color="#e87ba4", linestyle=(0, (5, 1)), marker="v", hatch="xx"),
    ),
    (
        ["cd search eager", "eco search eager", "eco search with eager expansion"],
        dict(color="#008300", linestyle=(0, (3, 1, 1, 1)), marker="P", hatch="++"),
    ),
    (["a star"], dict(color="#6250d6", linestyle=(0, (1, 1)), marker="X", hatch="--")),
    (
        ["bucket search"],
        dict(color="#e34948", linestyle=(0, (5, 2, 1, 2)), marker="*", hatch="oo"),
    ),
    (
        ["cd4", "eco search bucket size 4"],
        dict(color="#86b6ef", linestyle=(0, (5, 1)), marker="h", hatch="||"),
    ),
    (
        ["cd16", "eco search bucket size 16"],
        dict(color="#256abf", linestyle="-", marker=">", hatch=""),
    ),
    (
        ["cd64", "eco search bucket size 64"],
        dict(
            color="#104281", linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="<", hatch="**"
        ),
    ),
]
__FALLBACK_STYLES__: List[Dict[str, Any]] = [
    dict(color="#6e6e6e", linestyle=ls, marker=m, hatch="")
    for ls in [(0, (2, 2)), (0, (4, 1, 1, 1, 1, 1)), (0, (8, 2))]
    for m in ["d", "p", "8", "H"]
]


def __normalize__(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", label.lower()).split())


def style_for(label: str) -> Dict[str, Any]:
    """
    Return the color, linestyle, marker and hatch of the algorithm with the given label.
    """
    key = __normalize__(label)
    for names, style in __STYLES__:
        if key in names:
            return dict(style)
    # deterministic, so that an unknown algorithm also gets the same style in all figures
    return dict(
        __FALLBACK_STYLES__[zlib.crc32(key.encode()) % len(__FALLBACK_STYLES__)]
    )


def plot_with_incertitude(
    ax: plt.Axes,
    x: List[np.ndarray],
    y: List[np.ndarray],
    label: str,
    std_factor: float = 1.96,
    miny: Optional[float] = None,
    maxy: Optional[float] = None,
    cumulative: bool = True,
    n_points: int = 50,
) -> None:
    max_len = max(len(xi) for xi in x)
    X = np.array([xi for xi in x if len(xi) == max_len])
    Y = np.array([yi for yi in y if len(yi) == max_len])

    x_min = np.min(X)
    x_max = np.max(X)
    if x_max == x_min:
        return
    if cumulative:
        x_mean = np.cumsum(np.mean(X, axis=0))
        x_min = np.min(x_mean)
        x_max = np.max(x_mean)
        step = (x_max - x_min) / n_points
        target_x = (
            np.arange(x_min, x_max + step / 2, step=step)
            if len(X.shape) > 1 or len(X) > n_points
            else X.reshape(-1)
        )

        y_mean = np.cumsum(np.mean(Y, axis=0))
        mean = np.interp(target_x, x_mean, y_mean)

        y_var = np.cumsum(np.var(Y, axis=0))
        y_var = np.interp(target_x, x_mean, y_var)
        std = std_factor * np.sqrt(y_var)
    else:
        step = (x_max - x_min) / n_points
        target_x = (
            np.arange(x_min, x_max + step / 2, step=step)
            if len(X.shape) > 1 or len(X) > n_points
            else X.reshape(-1)
        )
        data = []
        for xi, yi in zip(X, Y):
            nyi = np.interp(target_x, xi, yi)
            data.append(nyi)
        # Compute distribution
        Y = np.array(data)
        mean = np.mean(Y, axis=0)
        std = std_factor * np.std(Y, axis=0)

    style = style_for(label)
    style.pop("hatch")
    p = ax.plot(
        target_x,
        mean,
        label=label,
        linewidth=1.5,
        markersize=6,
        markevery=max(1, len(target_x) // 8),
        **style,
    )
    color = p[0].get_color()
    upper = mean + std
    if maxy is not None:
        upper = np.minimum(upper, maxy)
    lower = mean - std
    if miny is not None:
        lower = np.maximum(lower, miny)
    ax.fill_between(target_x, lower, upper, color=color, alpha=0.15, linewidth=0)


def make_plot_wrapper(func, *args, **kwargs) -> None:
    def f(ax: plt.Axes, methods: Dict[str, Dict[int, List]]) -> None:
        return func(ax, methods, *args, **kwargs)

    return f


def plot_y_wrt_x(
    ax: plt.Axes,
    methods: Dict[str, Dict[int, List]],
    x_data: Tuple[int, str],
    y_data: Tuple[int, str],
    cumulative: bool = True,
    logx: bool = False,
    logy: bool = False,
    xlim: Tuple[Optional[int], Optional[int]] = (0, None),
    ylim: Tuple[Optional[int], Optional[int]] = (0, None),
    hline_at_length: bool = False,
    vline_at_length: bool = False,
) -> None:
    # Plot data with incertitude
    a_index, a_name = y_data
    b_index, b_name = x_data
    max_a = 0
    max_b = 0
    data_length = 0
    for method, seeds_dico in methods.items():
        seeds = list(seeds_dico.keys())
        data = [
            [(elems[b_index], elems[a_index]) for elems in seeds_dico[seed]]
            for seed in seeds
        ]
        data_length = max(data_length, len(data[0]))

        xdata = [[x[0] for x in seed_data] for seed_data in data]
        ydata = [[x[1] for x in seed_data] for seed_data in data]
        plot_with_incertitude(
            ax,
            xdata,
            ydata,
            method.capitalize(),
            miny=0,
            maxy=data_length if hline_at_length else None,
            cumulative=cumulative,
        )
        max_a = max(max(np.max(yi) for yi in ydata), max_a)
        max_b = max(max(np.max(xi) for xi in xdata), max_b)
        if cumulative:
            max_a = max(max(np.sum(yi) for yi in ydata), max_a)
            max_b = max(max(np.sum(xi) for xi in xdata), max_b)
    ax.set_xlabel(f"{b_name} (log scale)" if logx else b_name)
    ax.set_ylabel(f"{a_name} (log scale)" if logy else a_name)
    if hline_at_length:
        ax.hlines(
            [data_length],
            xmin=0,
            xmax=(xlim[1] or max_b),
            label=f"All {a_name}",
            color="k",
            linestyles="dashed",
        )
    if vline_at_length:
        ax.vlines(
            [data_length],
            ymin=0,
            ymax=(xlim[1] or max_a),
            label=f"All {b_name}",
            color="k",
            linestyles="dashed",
        )
    if logx:
        ax.set_xscale("log")
    else:
        ax.set_xlim(xlim[0], xlim[1])
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid()
    ax.legend()


def get_rank_matrix(
    methods: Dict[str, Dict[int, List]], yindex: int, maximize: bool
) -> np.ndarray:
    method_names = list(methods.keys())
    task_len = len(list(list(methods.values())[0].values())[0])
    for val in methods.values():
        task_len = max(max(len(x) for x in val.values()), task_len)
    seeds = set(list(methods.values())[0].keys())
    for val in methods.values():
        local_seeds = set(x for x, y in val.items() if len(y) == task_len)
        seeds &= local_seeds
    rank_matrix = np.ndarray((len(methods), task_len, len(methods)), dtype=float)
    data = np.ndarray((len(methods), len(seeds)), dtype=float)
    rng = np.random.default_rng(1)
    for task_no in range(task_len):
        for i, method in enumerate(method_names):
            for j, seed in enumerate(seeds):
                data[i, j] = methods[method][seed][task_no][yindex]
        if maximize:
            data = -data
        rand_x = rng.random(size=data.shape)
        # This is done to randomly break ties.
        # Last key is the primary key,
        indices = np.lexsort((rand_x, data), axis=0)
        for i, method in enumerate(method_names):
            rank_matrix[i, task_no] = [
                np.sum(indices[i] == rank) / len(seeds) for rank in range(len(methods))
            ]
    return rank_matrix


def __ready_for_stacked_dist_plot__(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=True,
        right=False,
        labeltop=False,
        labelbottom=True,
        labelleft=True,
        labelright=False,
    )

    ax.legend(fancybox=True, fontsize="large")


def plot_rank_by(
    ax: plt.Axes,
    methods: Dict[str, Dict[int, List]],
    y_data: Tuple[int, str],
    maximize: bool = True,
) -> None:
    width = 1.0
    a_index, a_name = y_data
    rank_matrix = get_rank_matrix(methods, a_index, maximize)
    labels = list(range(1, len(methods) + 1))
    mean_ranks = np.mean(rank_matrix, axis=-2)
    bottom = np.zeros_like(mean_ranks[0])
    for i, key in enumerate(methods.keys()):
        label = key
        bars = ax.bar(
            labels,
            mean_ranks[i],
            width,
            label=label,
            bottom=bottom,
            alpha=0.9,
            linewidth=1,
            edgecolor="white",
            color=style_for(label)["color"],
            hatch=style_for(label)["hatch"],
        )
        ax.bar_label(bars, labels=[f"{x:.1%}" for x in mean_ranks[i]])
        bottom += mean_ranks[i]

    ax.set_ylabel("Fraction (in %)", size="large")
    yticks = np.array(range(0, 101, 20))
    ax.set_yticklabels(yticks)
    ax.set_yticks(yticks * 0.01)
    ax.set_xlabel("Ranking", size="large")
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    __ready_for_stacked_dist_plot__(ax)
    word = "Most" if maximize else "Least"
    ax.set_title(f"{word} {a_name}")


def plot_dist(
    ax: plt.Axes,
    methods: Dict[str, Dict[int, List]],
    y_data: Tuple[int, str],
    x_axis_name: str,
) -> None:
    width = 1.0
    data_length = 0
    a_index, a_name = y_data
    max_a = max(
        max(max([y[a_index] for y in x]) for x in seed_dico.values())
        for seed_dico in methods.values()
    )
    bottom = None
    nbins = 5
    bins = [max_a]
    while len(bins) <= nbins:
        bins.insert(0, np.sqrt(bins[0] + 1))
    for i in range(nbins):
        if bins[i + 1] < 2 * bins[i]:
            bins[i + 1] = 2 * bins[i]
    x_bar = list(range(1, nbins + 1))
    for method, seeds_dico in methods.items():
        hists = []
        for seed, raw_data in seeds_dico.items():
            data = [x[a_index] for x in raw_data]
            data_length = max(data_length, len(data))
            hist, edges = np.histogram(
                data, bins=bins, range=(1e-3, max_a), density=False
            )
            hists.append(hist)
        true_hist = np.mean(hists, axis=0) / data_length
        if bottom is None:
            bottom = np.zeros_like(true_hist)
        label = method
        bars = ax.bar(
            x_bar,
            true_hist,
            width,
            label=label,
            bottom=bottom,
            alpha=0.9,
            linewidth=1,
            edgecolor="white",
            color=style_for(label)["color"],
            hatch=style_for(label)["hatch"],
        )
        ax.bar_label(bars, labels=[f"{x:.1%}" for x in true_hist])
        bottom += true_hist
    __ready_for_stacked_dist_plot__(ax)
    ax.set_yticklabels([])
    ax.set_xlabel(a_name, size="large")
    ax.set_xticklabels(map(lambda x: f"<{x:.0f}", edges))
    ax.set_title(f"Distribution of {a_name} per {x_axis_name}")
