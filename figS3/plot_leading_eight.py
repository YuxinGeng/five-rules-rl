from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    COLORS,
    LINE_YLIM,
    draw_line_with_ci,
    save_figure,
    set_integer_time_ticks,
    setup_time_series_style,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
NORM_NAMES = tuple(f"L{index}" for index in range(1, 9))


def finish_line_axis(axis, frame, title):
    times = frame["time"].to_numpy()
    axis.set_xlabel("Time")
    axis.set_ylabel("Proportion")
    axis.set_xlim(times[0], times[-1])
    axis.set_ylim(*LINE_YLIM)
    axis.set_title(title)
    set_integer_time_ticks(axis)
    axis.legend(loc="best")


def plot_strategy_lines(
    frame, target, save_path, title, smooth_window=10
):
    setup_time_series_style()
    fig, axis = plt.subplots(constrained_layout=True)
    draw_line_with_ci(
        axis,
        frame,
        target,
        rf"$\mathrm{{{target}}}$",
        COLORS[0],
        smooth_window=smooth_window,
    )
    draw_line_with_ci(
        axis,
        frame,
        "DDDD",
        "ALLD",
        COLORS[2],
        smooth_window=smooth_window,
    )
    finish_line_axis(axis, frame, title)
    save_figure(fig, save_path)


def main(norms=None):
    selected = NORM_NAMES if norms is None else tuple(dict.fromkeys(norms))
    for norm_name in selected:
        csv_path = DATA_DIR / f"sim_{norm_name.lower()}.csv"
        frame = pd.read_csv(csv_path)
        target = str(frame["target_strategy"].iloc[0])
        title = f"Norm {norm_name}"
        plot_strategy_lines(
            frame,
            target,
            OUTPUT_DIR / f"sim_{norm_name.lower()}.png",
            title,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norms",
        nargs="+",
        choices=NORM_NAMES,
        default=list(NORM_NAMES),
    )
    arguments = parser.parse_args()
    main(norms=arguments.norms)
