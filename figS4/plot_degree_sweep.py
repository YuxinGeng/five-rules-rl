from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import COLORS, draw_line_with_ci, save_figure, setup_time_series_style
from fig3.plot_fig3 import finish_line_axis


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def plot_trajectories(frame, smooth_window=10):
    if frame.empty or len(frame[["k", "B"]].drop_duplicates()) != 1:
        raise ValueError("expected one fixed group size and update interval")
    if frame.duplicated(["d", "time"]).any():
        raise ValueError("expected one trajectory per degree")
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for index, (degree, group) in enumerate(frame.groupby("d", sort=True)):
        draw_line_with_ci(
            axis,
            group.sort_values("time"),
            "CC",
            rf"$d = {int(degree)}$",
            COLORS[index % len(COLORS)],
            smooth_window=smooth_window,
            band_alpha=0.2,
            band_linewidth=0,
        )
    finish_line_axis(
        axis,
        frame,
        ylabel=r"Proportion of $(\mathrm{C},\mathrm{C})$",
        legend_kwargs={"fontsize": 32},
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot the network degree comparison.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "hypergraph_2pgg_degree_sweep.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    fig = plot_trajectories(pd.read_csv(args.input))
    save_figure(fig, args.output_dir / "S4_NR.png")


if __name__ == "__main__":
    main()
