from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import to_rgb
from scipy.integrate import solve_ivp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import COLORS, setup_style


OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "full_space"
)
FRAME_DIR = OUTPUT_DIR / "frames"
DATA_PATH = Path(__file__).resolve().parent / "data" / "full_space.npz"
MP4_NAME = "full_space.mp4"
BOUNDARY_STRATEGIES = np.array(
    [[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]],
    dtype=np.float64,
)

_VIEW_ELEV = 26
_VIEW_AZIM = -56
LABEL_SIZE = 55
TITLE_SIZE = 60
TICK_SIZE = 50
_CORNER_00 = np.array(to_rgb(COLORS[2]))
_CORNER_10 = np.array(to_rgb(COLORS[1]))
_CORNER_01 = np.array(to_rgb(COLORS[3]))
_CORNER_11 = np.array(to_rgb(COLORS[0]))
_CENTER = np.array([1.0, 1.0, 1.0])


setup_style()


def random_grid(n_random: int = 96, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(0.0, 1.0, size=(n_random, 2))
    return np.vstack([BOUNDARY_STRATEGIES, random_points]).astype(np.float64)


def _strategy_colors(strategies: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    p1 = np.clip(strategies[:, 0], 0.0, 1.0)[:, None]
    p2 = np.clip(strategies[:, 1], 0.0, 1.0)[:, None]
    corner = (
        (1.0 - p1) * (1.0 - p2) * _CORNER_00
        + p1 * (1.0 - p2) * _CORNER_10
        + (1.0 - p1) * p2 * _CORNER_01
        + p1 * p2 * _CORNER_11
    )
    distance = np.sqrt((p1 - 0.5) ** 2 + (p2 - 0.5) ** 2)
    weight = np.clip(distance / np.sqrt(0.5), 0.0, 1.0)
    rgb = (1.0 - weight) * _CENTER + weight * corner
    rgba = np.empty((strategies.shape[0], 4))
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, 3] = alpha
    return rgba


def reactive_payoff_matrix(
    strategies: np.ndarray,
    b: float,
    c: float,
) -> np.ndarray:
    p1 = strategies[:, 0:1]
    p2 = strategies[:, 1:2]
    q1 = strategies[:, 0][None, :]
    q2 = strategies[:, 1][None, :]
    delta_p = p1 - p2
    delta_q = q1 - q2
    denominator = 1.0 - delta_p * delta_q
    cooperation_x = (p2 + delta_p * q2) / denominator
    cooperation_y = (q2 + delta_q * p2) / denominator
    return b * cooperation_y - c * cooperation_x


def replicator_rhs(_time, distribution, payoff):
    rewards = payoff @ distribution
    average_reward = float(distribution @ rewards)
    return distribution * (rewards - average_reward)


def _style_axes_3d(axis) -> None:
    axis.view_init(elev=_VIEW_ELEV, azim=_VIEW_AZIM)
    axis.set_box_aspect((1.0, 1.0, 0.6))
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.set_pane_color((1.0, 1.0, 1.0, 0.0))
        dimension.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        dimension._axinfo["grid"].update(
            color=(0.6, 0.6, 0.6, 0.3), linewidth=0.5
        )
    axis.set_xticks([0.0, 1.0])
    axis.set_yticks([0.0, 1.0])
    axis.tick_params(colors=(0.25, 0.25, 0.25), labelsize=TICK_SIZE)
    axis.tick_params(axis="x", pad=12)
    axis.tick_params(axis="y", pad=12)
    axis.tick_params(axis="z", pad=24)


def _draw_distribution(
    axis,
    distribution,
    strategies,
    threshold,
    bar_width=0.03,
):
    mask = distribution > threshold
    p1 = strategies[mask, 0]
    p2 = strategies[mask, 1]
    heights = distribution[mask]
    top = 1.0
    if heights.size:
        order = np.argsort(p1 + p2)
        p1 = p1[order]
        p2 = p2[order]
        heights = heights[order]
        colors = _strategy_colors(np.column_stack([p1, p2]))
        axis.bar3d(
            p1 - bar_width / 2,
            p2 - bar_width / 2,
            np.zeros_like(heights),
            bar_width,
            bar_width,
            heights,
            color=colors,
            shade=True,
            linewidth=0.0,
        )
        top = float(heights.max())

    _style_axes_3d(axis)
    axis.set_xlabel(r"$p_1$", labelpad=12, fontsize=LABEL_SIZE)
    axis.set_ylabel(r"$p_2$", labelpad=12, fontsize=LABEL_SIZE)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, max(top * 1.05, 1e-3))
    axis.set_zticks([0.0, top])
    axis.set_zticklabels(["0", "1"])


def plot_snapshot(
    distribution,
    strategies,
    time,
    save_path,
    threshold=1e-10,
):
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_distribution(axis, distribution, strategies, threshold)
    fig.suptitle(rf"$t = {time:.0f}$", fontsize=TITLE_SIZE, y=0.9)
    fig.subplots_adjust(left=0.025, right=0.735, bottom=0.02, top=0.95)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _dominant_segments(distributions):
    dominant = np.argmax(distributions, axis=0)
    changes = np.flatnonzero(np.diff(dominant)) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(dominant)]])
    return dominant, starts, ends


def _population_averages(distributions, strategies, payoff):
    mean_p1 = strategies[:, 0] @ distributions
    mean_p2 = strategies[:, 1] @ distributions
    mean_payoff = np.einsum(
        "it,ij,jt->t", distributions, payoff, distributions
    )
    return mean_p1, mean_p2, mean_payoff


def _save_population_plot(fig, axes, times, save_path):
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].set_yticks([0.0, 1.0])
    axes[1].set_yticks([0.0, 1.0])
    axes[2].set_yticks([0.0, 4.0])
    axes[2].set_xlabel("Time", fontsize=20)
    axes[2].set_xlim(0.0, times[-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_population_averages(
    times,
    distributions,
    strategies,
    payoff,
    save_path,
):
    mean_p1, mean_p2, mean_payoff = _population_averages(
        distributions, strategies, payoff
    )
    dominant, starts, ends = _dominant_segments(distributions)
    dominant_rgb = _strategy_colors(
        strategies[dominant], alpha=1.0
    )[:, :3]

    def plot_colored(axis, values):
        for start, end in zip(starts, ends):
            stop = min(end + 1, len(times))
            axis.plot(
                times[start:stop],
                values[start:stop],
                color=dominant_rgb[start],
                linewidth=3.0,
            )

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Population average", fontsize=22)
    for axis, values, label in zip(
        axes,
        (mean_p1, mean_p2, mean_payoff),
        (r"$p_1$", r"$p_2$", "Payoff"),
    ):
        plot_colored(axis, values)
        axis.set_ylabel(label, fontsize=20)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=18)
    _save_population_plot(fig, axes, times, save_path)


def clear_frames(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def build_mp4(frame_paths, mp4_path, fps=30, max_width=1280):
    frame_duration = 1.0 / fps
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as list_file:
        list_path = Path(list_file.name)
        for frame_path in frame_paths:
            list_file.write(f"file '{frame_path.resolve().as_posix()}'\n")
            list_file.write(f"duration {frame_duration:.12f}\n")
        list_file.write(f"file '{frame_paths[-1].resolve().as_posix()}'\n")

    command = (
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        f"scale='min({max_width},iw)':-2",
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_path),
    )
    try:
        subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    setup_style()
    total_time = 5000.0
    n_snapshots = 5001
    b_value = 5.0
    c_value = 1.0
    rtol = 1e-12
    atol = 1e-14
    seed = 436
    strategies = random_grid(seed=seed)
    payoff = reactive_payoff_matrix(strategies, b_value, c_value)
    initial_distribution = np.full(
        strategies.shape[0], 1.0 / strategies.shape[0]
    )
    snapshot_times = np.linspace(0.0, total_time, n_snapshots)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_frames(FRAME_DIR)
    solution = solve_ivp(
        replicator_rhs,
        (0.0, total_time),
        initial_distribution,
        t_eval=snapshot_times,
        args=(payoff,),
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        times=solution.t,
        strategies=strategies,
        distributions=solution.y,
        b=b_value,
        c=c_value,
        seed=seed,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    dense_times = np.linspace(0.0, total_time, 2000)
    dense_distributions = np.maximum(solution.sol(dense_times), 0.0)
    dense_distributions /= dense_distributions.sum(axis=0, keepdims=True)
    plot_population_averages(
        dense_times,
        dense_distributions,
        strategies,
        payoff,
        OUTPUT_DIR / "population_averages.png",
    )
    frame_paths = []
    plot_arguments = []
    for index, time in enumerate(solution.t):
        distribution = np.maximum(solution.y[:, index], 0.0)
        distribution /= distribution.sum()
        save_path = FRAME_DIR / f"snapshot_{index:04d}_t{time:07.2f}.png"
        frame_paths.append(save_path)
        plot_arguments.append((distribution, strategies, time, save_path))

    Parallel(n_jobs=-1)(
        delayed(plot_snapshot)(*arguments) for arguments in plot_arguments
    )
    build_mp4(frame_paths, OUTPUT_DIR / MP4_NAME)


if __name__ == "__main__":
    main()
