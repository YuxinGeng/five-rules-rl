from __future__ import annotations

import sys
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


def finish_line_axis(
    axis, frame, ylabel="Proportion", title=None, legend_kwargs=None
):
    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    axis.set_xlim(frame["time"].min(), frame["time"].max())
    axis.set_ylim(*LINE_YLIM)
    if title:
        axis.set_title(title)
    set_integer_time_ticks(axis)
    axis.legend(**{"loc": "best", **(legend_kwargs or {})})


def plot_direct_reciprocity(frame):
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    draw_line_with_ci(axis, frame, "ALLD", "ALLD", COLORS[2])
    draw_line_with_ci(axis, frame, "WSLS", "WSLS", COLORS[0])
    finish_line_axis(axis, frame)
    save_figure(fig, OUTPUT_DIR / "fig3a_DR.png")


def plot_indirect_reciprocity(frame):
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    draw_line_with_ci(axis, frame, "CDCD", "Discriminator", COLORS[0])
    draw_line_with_ci(axis, frame, "DDDD", "ALLD", COLORS[2])
    finish_line_axis(axis, frame)
    save_figure(fig, OUTPUT_DIR / "fig3b_IR.png")


def plot_environmental_stochasticity(frame):
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    draw_line_with_ci(axis, frame, "CC", r"$(\mathrm{C},\mathrm{C})$", COLORS[0])
    draw_line_with_ci(axis, frame, "DD", r"$(\mathrm{D},\mathrm{D})$", COLORS[2])
    finish_line_axis(axis, frame, legend_kwargs={"fontsize": 32})
    save_figure(fig, OUTPUT_DIR / "fig3c_ES.png")


def plot_network_reciprocity(frame):
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    for index, (order, group) in enumerate(frame.groupby("k", sort=True)):
        group = group.sort_values("time")
        draw_line_with_ci(
            axis, group, "CC", rf"$k = {int(order)}$", COLORS[index],
            band_alpha=0.2, band_linewidth=0,
        )
    finish_line_axis(
        axis, frame, ylabel=r"Proportion of $(\mathrm{C},\mathrm{C})$",
        legend_kwargs={"fontsize": 32},
    )
    save_figure(fig, OUTPUT_DIR / "fig3d_NR.png")


def plot_demographic_stochasticity(frame):
    setup_time_series_style()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for index, (batch_size, group) in enumerate(
        frame.groupby("batch_size", sort=True)
    ):
        group = group.sort_values("time")
        draw_line_with_ci(
            axis, group, "greedy_c", rf"$B = {int(batch_size)}$", COLORS[index],
            band_alpha=0.18, band_linewidth=0,
        )
    finish_line_axis(axis, frame, ylabel="Proportion of Cooperator")
    axis.yaxis.set_label_coords(-0.12, 0.45)
    save_figure(fig, OUTPUT_DIR / "fig3e_DS.png")


def read_inputs():
    paths = {
        "dr": DATA_DIR / "sim-mem1.csv",
        "ir": DATA_DIR / "sim_image_scoring.csv",
        "environmental": DATA_DIR / "sim-2dg.csv",
        "network": DATA_DIR / "hypergraph_2pgg_sweep.csv",
        "demographic": DATA_DIR / "sim-SH-self-play.csv",
    }
    return {name: pd.read_csv(path) for name, path in paths.items()}


def main():
    frames = read_inputs()
    plot_direct_reciprocity(frames["dr"])
    plot_indirect_reciprocity(frames["ir"])
    plot_environmental_stochasticity(frames["environmental"])
    plot_network_reciprocity(frames["network"])
    plot_demographic_stochasticity(frames["demographic"])


if __name__ == "__main__":
    main()
