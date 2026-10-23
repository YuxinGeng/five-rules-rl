from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import COLORS, setup_style


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BOUNDARY_LINE_COLOR = "#777777"
BOUNDARY_LINE_ALPHA = 0.5
REGION_FILL_ALPHA = 0.18
REGION_FILL_COLORS = (COLORS[0], COLORS[2], COLORS[3], COLORS[1])
FS = 32
FIG_SIZE = (7.2, 6.4)
LABEL_SIZE = FS - 2
TITLE_SIZE = FS
TICK_SIZE = FS - 4
X_LABEL_PAD = -3
REGION_LABEL_SIZE = 22
COMBINED_REGION_LABEL_SIZE = 20
BOUNDARY_LABEL_SIZE = 24
MAIN_AXES_RECT = (0.16, 0.135, 0.6667, 0.75)


def apply_plot_style() -> None:
    setup_style()
    rcParams["hatch.linewidth"] = 3


def delta_d_value(x_value, y_value, epsilon):
    return -y_value + epsilon * (y_value - x_value) / 2


def delta_c_value(x_value, y_value, epsilon):
    return -x_value - epsilon * (y_value - x_value) / 2


def clip_polygon_half_plane(polygon, value_function, keep_positive):
    clipped = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_value = value_function(*start)
        end_value = value_function(*end)
        start_inside = (
            start_value >= -1e-12 if keep_positive else start_value <= 1e-12
        )
        end_inside = (
            end_value >= -1e-12 if keep_positive else end_value <= 1e-12
        )
        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside != end_inside:
            denominator = start_value - end_value
            if abs(denominator) > 1e-12:
                fraction = start_value / denominator
                clipped.append(
                    (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    )
                )
            if end_inside:
                clipped.append(end)
    return clipped


def fill_equilibrium_regions(axis, epsilon):
    base_polygon = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    region_specs = (
        (REGION_FILL_COLORS[0], True, True),
        (REGION_FILL_COLORS[1], False, False),
        (REGION_FILL_COLORS[2], True, False),
        (REGION_FILL_COLORS[3], False, True),
    )
    for color, keep_d_positive, keep_c_positive in region_specs:
        polygon = clip_polygon_half_plane(
            base_polygon,
            lambda x_value, y_value: delta_d_value(
                x_value, y_value, epsilon
            ),
            keep_d_positive,
        )
        polygon = clip_polygon_half_plane(
            polygon,
            lambda x_value, y_value: delta_c_value(
                x_value, y_value, epsilon
            ),
            keep_c_positive,
        )
        if len(polygon) >= 3:
            axis.add_patch(
                patches.Polygon(
                    polygon,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=REGION_FILL_ALPHA,
                    zorder=0,
                )
            )


def plot_boundary_lines(axis, epsilon):
    beta = epsilon / (2 - epsilon)
    values = np.linspace(-1, 1, 400)
    axis.plot(
        values,
        -beta * values,
        color=BOUNDARY_LINE_COLOR,
        linewidth=2,
        alpha=BOUNDARY_LINE_ALPHA,
        zorder=4,
    )
    axis.plot(
        -beta * values,
        values,
        color=BOUNDARY_LINE_COLOR,
        linewidth=2,
        alpha=BOUNDARY_LINE_ALPHA,
        zorder=4,
    )


def add_region_labels(axis, epsilon):
    axis.text(
        -0.5,
        -0.5,
        r"$(\mathrm{C},\mathrm{C})$",
        fontsize=REGION_LABEL_SIZE,
        ha="center",
        va="center",
    )
    combined_x = 0.5 if np.isclose(epsilon, 0.01) else 0.575
    axis.text(
        -combined_x,
        0.5,
        r"$(\mathrm{D},\mathrm{D})$ & $(\mathrm{C},\mathrm{C})$",
        fontsize=COMBINED_REGION_LABEL_SIZE,
        ha="center",
        va="center",
    )
    axis.text(
        combined_x,
        -0.5,
        r"$(\mathrm{C},\mathrm{D})$ & $(\mathrm{D},\mathrm{C})$",
        fontsize=COMBINED_REGION_LABEL_SIZE,
        ha="center",
        va="center",
    )
    axis.text(
        0.5,
        0.5,
        r"$(\mathrm{D},\mathrm{D})$",
        fontsize=REGION_LABEL_SIZE,
        ha="center",
        va="center",
    )


def style_phase_axis(axis, epsilon):
    axis.set_xlabel(r"$T-R$", fontsize=LABEL_SIZE, labelpad=X_LABEL_PAD)
    axis.set_ylabel(r"$P-S$", fontsize=LABEL_SIZE)
    axis.set_title(rf"$\varepsilon={epsilon}$", fontsize=TITLE_SIZE, pad=10)
    axis.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    axis.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_xticks([-1, 0, 1])
    axis.set_yticks([-1, 0, 1])
    axis.set_aspect("equal")
    axis.tick_params(axis="both", labelsize=TICK_SIZE)
    axis.grid(False)


def plot_equilibrium_regions(epsilon, output_name):
    beta = epsilon / (2 - epsilon)
    fig = plt.figure(figsize=FIG_SIZE)
    axis = fig.add_axes(MAIN_AXES_RECT)
    fill_equilibrium_regions(axis, epsilon)
    plot_boundary_lines(axis, epsilon)

    upper_label = (0.5, 0.27)
    axis.add_patch(
        patches.FancyArrowPatch(
            (upper_label[0] + 0.15, upper_label[1] - 0.15),
            (0.7, -beta * 0.7),
            connectionstyle="arc3, rad=-0.3",
            arrowstyle="Simple, tail_width=0.5, head_width=8, head_length=8",
            color=BOUNDARY_LINE_COLOR,
            zorder=500,
        )
    )
    axis.text(
        *upper_label,
        r"$\dfrac{P-S}{T-R}=-\dfrac{\varepsilon}{2-\varepsilon}$",
        fontsize=BOUNDARY_LABEL_SIZE,
        color=BOUNDARY_LINE_COLOR,
        ha="center",
        va="center",
    )

    lower_label = (-0.5, -0.2)
    axis.add_patch(
        patches.FancyArrowPatch(
            (lower_label[0] + 0.14, lower_label[1] - 0.15),
            (beta * 0.7, -0.7),
            connectionstyle="arc3, rad=0.3",
            arrowstyle="Simple, tail_width=0.5, head_width=8, head_length=8",
            color=BOUNDARY_LINE_COLOR,
            zorder=500,
        )
    )
    axis.text(
        *lower_label,
        r"$\dfrac{P-S}{T-R}=-\dfrac{2-\varepsilon}{\varepsilon}$",
        fontsize=BOUNDARY_LABEL_SIZE,
        color=BOUNDARY_LINE_COLOR,
        ha="center",
        va="center",
    )

    add_region_labels(axis, epsilon)
    style_phase_axis(axis, epsilon)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=600)
    plt.close(fig)


def main():
    apply_plot_style()
    plot_equilibrium_regions(0.01, "stateless-eps-01.png")
    plot_equilibrium_regions(0.5, "stateless-eps-5.png")


if __name__ == "__main__":
    main()
