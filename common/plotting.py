from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, ticker

COLORS = ("#4285F4", "#34A853", "#EA4335", "#FBBC05", "#673AB7")
LINE_WIDTH = 6.0
LINE_YLIM = (-0.005, 1.005)
CI_Z = 1.96


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def setup_style() -> None:
    import seaborn as sns


    sns.set_theme(style="whitegrid")
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:style=italic",
            "mathtext.bf": "Arial:weight=bold",
            "mathtext.bfit": "Arial:style=italic:weight=bold",
            "axes.grid": False,
        }
    )


def setup_time_series_style(font_size=36):
    setup_style()
    rcParams.update(
        {
            "figure.figsize": (9, 6),
            "legend.title_fontsize": font_size - 8,
            "axes.labelsize": font_size - 2,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size - 8,
            "xtick.labelsize": font_size - 8,
            "ytick.labelsize": font_size - 8,
            "figure.facecolor": (242 / 255, 242 / 255, 242 / 255, 0.5),
            "savefig.facecolor": (242 / 255, 242 / 255, 242 / 255, 0.5),
            "axes.facecolor": "none",
            "legend.facecolor": "white",
        }
    )


def moving_average(values, window=10):
    values = np.asarray(values)
    if window <= 1 or len(values) <= window:
        return values
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def ci_halfwidth(std, n_repetitions):
    return np.asarray(std) * (CI_Z / np.sqrt(n_repetitions))


class FixedOrderFormatter(ticker.ScalarFormatter):

    def __init__(self, order):
        super().__init__(useMathText=True)
        self._fixed_order = order
        self.set_scientific(True)
        self.set_useOffset(False)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


def set_integer_time_ticks(axis, n_bins=6):
    lower, upper = axis.get_xlim()
    span = max(abs(lower), abs(upper))
    if not np.isfinite(span) or span == 0:
        return
    order = max(0, int(np.ceil(np.log10(span) - 0.05)) - 1)
    scale = 10.0**order
    locator = ticker.MaxNLocator(
        nbins=n_bins, steps=[1, 2, 5, 10], integer=True
    )
    ticks = locator.tick_values(lower / scale, upper / scale) * scale
    low_value, high_value = sorted((lower, upper))
    ticks = ticks[(ticks >= low_value - 1e-6) & (ticks <= high_value + 1e-6)]
    axis.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedOrderFormatter(order))


def draw_line_with_ci(
    axis,
    frame,
    key,
    label,
    color,
    smooth_window=10,
    *,
    linewidth=LINE_WIDTH,
    band_alpha=0.2,
    band_linewidth=None,
):
    times = frame["time"].to_numpy()
    n_repetitions = int(frame["n_reps"].iloc[0])
    mean = moving_average(frame[f"mean_{key}"].to_numpy(), smooth_window)
    std = moving_average(frame[f"std_{key}"].to_numpy(), smooth_window)
    half_width = ci_halfwidth(std, n_repetitions)
    axis.plot(
        times,
        mean,
        color=color,
        label=label,
        linewidth=linewidth,
    )
    band_options = {} if band_linewidth is None else {"linewidth": band_linewidth}
    axis.fill_between(
        times,
        np.clip(mean - half_width, 0, 1),
        np.clip(mean + half_width, 0, 1),
        color=color,
        alpha=band_alpha,
        **band_options,
    )
