import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    COLORS, LINE_YLIM, draw_line_with_ci, save_figure,
    set_integer_time_ticks, setup_time_series_style,
)

HERE = Path(__file__).resolve().parent


def plot_trajectories(frame, smooth_window=10, title=None):
    if frame.empty:
        raise ValueError("the trajectory CSV is empty")
    for key in ("n_agents", "update_interval", "time_steps", "record_every", "alpha", "gamma", "epsilon",
                "R", "S", "T", "P", "q_init_C", "q_init_D", "n_reps", "base_seed"):
        if frame[key].nunique() != 1:
            raise ValueError(f"a degree comparison must hold {key} fixed")
    if not (frame["statistic"] == "greedy_C_agents").all():
        raise ValueError("expected the proportion of agents with greedy action C")
    if frame.duplicated(["degree", "time"]).any():
        raise ValueError("duplicate degree/time rows")
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for index, (degree, group) in enumerate(frame.groupby("degree", sort=True)):
        draw_line_with_ci(
            axis, group.sort_values("time"), "greedy_c", rf"$d = {int(degree)}$",
            COLORS[index % len(COLORS)], smooth_window=smooth_window,
            band_alpha=0.18, band_linewidth=0,
        )
    axis.set_xlabel("Time")
    axis.set_ylabel("Proportion of cooperators")
    axis.set_xlim(frame["time"].min(), max(1, frame["time"].max()))
    axis.set_ylim(*LINE_YLIM)
    axis.yaxis.set_label_coords(-0.12, 0.45)
    if title:
        axis.set_title(title)
    set_integer_time_ticks(axis)
    axis.legend(loc="best")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "data/sim-regular-graph-SH.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    fig = plot_trajectories(pd.read_csv(args.input), args.smooth_window, args.title)
    save_figure(fig, args.output_dir / "sim-regular-graph-SH-lines.png")


if __name__ == "__main__":
    main()
