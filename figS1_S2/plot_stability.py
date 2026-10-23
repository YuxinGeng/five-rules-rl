import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.plotting import COLORS, save_figure, setup_style

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STRATEGY_COLORS = {"GRIM": COLORS[1], "WSLS": COLORS[0]}
EPSILON_LIMITS = (0.0001, 0.8)
RATIO_LIMITS = (1.001, 6.0)


def stability_bounds(ratio, epsilon):
    ratio, epsilon = np.broadcast_arrays(ratio, epsilon)
    grim_lower = 2 / (ratio * (1-epsilon) * (2-epsilon))
    grim_upper = np.minimum(2 / ((1-epsilon) * (ratio*epsilon+2)), 1.0)
    grim_valid = (grim_lower < grim_upper) & (grim_lower < 1)

    denominator = (1-epsilon) * (ratio*(1-epsilon)-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        wsls_lower = np.where(denominator > 0, 1 / denominator, np.nan)
    wsls_valid = np.isfinite(wsls_lower) & (wsls_lower > 0) & (wsls_lower < 1)
    return {
        "GRIM": (np.where(grim_valid, grim_lower, np.nan),
                 np.where(grim_valid, grim_upper, np.nan)),
        "WSLS": (np.where(wsls_valid, wsls_lower, np.nan),
                 np.where(wsls_valid, 1.0, np.nan)),
    }


def boundary_indices(valid):
    traces = []
    for transpose in (False, True):
        mask = valid.T if transpose else valid
        rows = np.flatnonzero(mask.any(axis=1))
        first = np.argmax(mask[rows], axis=1)
        last = mask.shape[1]-1-np.argmax(mask[rows, ::-1], axis=1)
        for columns in (first, last):
            traces.append((columns, rows) if transpose else (rows, columns))
    return traces


def outline_region(axis, ratios, epsilons, lower, upper, color):
    options = dict(color=color, linewidth=1.8, alpha=0.95)
    for surface in (lower, upper):
        for rows, columns in boundary_indices(np.isfinite(surface)):
            if len(rows) >= 2:
                axis.plot(ratios[rows, columns], epsilons[rows, columns],
                          surface[rows, columns], **options)
    corners = set()
    for rows, columns in boundary_indices(np.isfinite(lower) & np.isfinite(upper)):
        if len(rows):
            corners.update(((rows[0], columns[0]), (rows[-1], columns[-1])))
    for row, column in sorted(corners):
        if not np.isclose(lower[row, column], upper[row, column]):
            axis.plot([ratios[row, column]] * 2, [epsilons[row, column]] * 2,
                      [lower[row, column], upper[row, column]], **options)


def draw_box(axis):
    corners = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (1, 6)]
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
             (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))
    options = dict(color="0.75", linewidth=0.9, alpha=0.9, zorder=0)
    for start, end in edges:
        axis.plot(*zip(corners[start], corners[end]), **options)

    extension = 0.041
    axis.plot([6, 6], [1-extension, 1-extension/5], [1, 1], clip_on=False, **options)


def plot_surfaces(num_points=900):
    setup_style()
    ratios, epsilons = np.meshgrid(np.linspace(*RATIO_LIMITS, num_points),
                                 np.linspace(*EPSILON_LIMITS, num_points))
    bounds = stability_bounds(ratios, epsilons)
    figure = plt.figure(figsize=(16, 7.5))
    stride = max(1, num_points // 240)
    surface_options = dict(rstride=stride, cstride=stride, linewidth=0, antialiased=True)
    for index, (name, (lower, upper)) in enumerate(bounds.items(), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        rgb = np.array(to_rgb(STRATEGY_COLORS[name]))
        edge_color = tuple(rgb * 0.65)
        axis.plot_surface(ratios, epsilons, lower, color=tuple(rgb),
                          alpha=0.3, shade=True, **surface_options)
        axis.plot_surface(ratios, epsilons, upper, color=tuple(rgb+(1-rgb)*0.35),
                          alpha=0.3, shade=True, **surface_options)
        outline_region(axis, ratios, epsilons, lower, upper, edge_color)
        if name == "GRIM":
            epsilon_curve = np.linspace(*EPSILON_LIMITS, 2000)
            ratio_at_gamma_one = 2 / (1-epsilon_curve)
            valid = ((ratio_at_gamma_one >= RATIO_LIMITS[0])
                     & (ratio_at_gamma_one <= RATIO_LIMITS[1]))
            axis.plot(ratio_at_gamma_one[valid], epsilon_curve[valid],
                      np.ones(np.count_nonzero(valid)), color=edge_color,
                      linewidth=1.8, alpha=0.95)
        axis.set_xlabel(r"$b/c$", fontsize=14, labelpad=10)
        axis.set_ylabel(r"$\varepsilon$", fontsize=14, labelpad=10)
        axis.set_zlabel(r"$\gamma$", fontsize=14, labelpad=10)
        axis.set_title(name, fontsize=16, pad=15)
        axis.set(xlim=(1, 6), ylim=(0, 1), zlim=(0, 1))
        axis.set_xticks(np.arange(1, 7))
        axis.set_xticklabels([str(tick) for tick in range(1, 7)])
        ticks = np.linspace(0, 1, 6)
        axis.set_yticks(ticks)
        axis.set_zticks(ticks)
        axis.set_yticklabels([f"{tick:.1f}" for tick in ticks])
        axis.set_zticklabels([f"{tick:.1f}" for tick in ticks])
        axis.set_box_aspect((0.85*5, 1.8, 1.5))
        axis.tick_params(labelsize=12)
        axis.grid(False)
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.pane.set_alpha(0.08)
        draw_box(axis)
        axis.view_init(elev=24, azim=-56)
    figure.subplots_adjust(left=0.02, right=0.98, wspace=0.05)
    return figure


def plot_cross_sections(num_points=2000):
    setup_style()
    epsilons = np.linspace(*EPSILON_LIMITS, num_points)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True,
                               constrained_layout=True)
    for axis, ratio in zip(axes, (2.0, 4.0)):
        for name, (lower, upper) in stability_bounds(ratio, epsilons).items():
            color = STRATEGY_COLORS[name]
            edge_color = tuple(np.array(to_rgb(color)) * 0.7)
            axis.fill_between(epsilons, lower, upper, alpha=0.25, color=color, label=name)
            axis.plot(epsilons, lower, color=edge_color, linewidth=1.6)
            axis.plot(epsilons, upper, color=edge_color, linewidth=1.6)
        axis.set_xlabel(r"$\varepsilon$", fontsize=14)
        axis.set_title(rf"$b/c = {ratio:g}$", fontsize=16)
        axis.set_xlim(*EPSILON_LIMITS)
        axis.set_ylim(0, 1.05)
        axis.tick_params(labelsize=12)
        axis.legend(fontsize=11, loc="upper right")
        axis.grid(False)
    axes[0].set_ylabel(r"$\gamma$", fontsize=14)
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--surface-points", type=int, default=900,
                        help="Grid points per S1 axis (default: 900)")
    parser.add_argument("--slice-points", type=int, default=2000,
                        help="Exploration samples per S2 panel (default: 2000)")
    args = parser.parse_args()
    save_figure(plot_surfaces(args.surface_points), args.output_dir / "S1.png")
    save_figure(plot_cross_sections(args.slice_points), args.output_dir / "S2.png")


if __name__ == "__main__":
    main()
