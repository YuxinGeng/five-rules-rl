from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import setup_style

from simulation_full_space import (
    DATA_PATH,
    _strategy_colors,
    reactive_payoff_matrix,
    replicator_rhs,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "cycle"
)
SIMPLEX_VERTICES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
)


def load_cycle_initial_conditions(data_path, source_time):
    data_path = Path(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        times = data["times"]
        strategies = data["strategies"]
        distributions = data["distributions"]
    if (times.ndim != 1 or strategies.ndim != 2
            or strategies.shape[1] != 2 or len(strategies) < 3
            or distributions.shape != (len(strategies), len(times))):
        raise ValueError("Expected times (T,), strategies (N, 2), distributions (N, T), N >= 3.")
    matches = np.flatnonzero(times == source_time)
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one saved snapshot at t={source_time:g}; "
            f"found {matches.size}. No nearest-time substitution is used."
        )
    distribution = np.maximum(distributions[:, matches[0]], 0.0)
    if (not np.all(np.isfinite(distribution)) or distribution.sum() <= 0.0
            or not np.all(np.isfinite(strategies))):
        raise ValueError("Snapshot must have finite strategies and positive total mass.")
    distribution /= distribution.sum()

    selected = np.sort(np.argsort(-distribution, kind="stable")[:3])
    weights = distribution[selected]
    initial_distribution = weights / weights.sum()
    discarded = np.ones(len(distribution), dtype=bool)
    discarded[selected] = False
    discarded_mass = float(distribution[discarded].sum())
    return selected, strategies[selected], initial_distribution, discarded_mass


def simplex_grid(n_per_edge):
    points = []
    for first in range(n_per_edge + 1):
        for second in range(n_per_edge + 1 - first):
            third = n_per_edge - first - second
            points.append([first, second, third])
    return np.asarray(points, dtype=np.float64) / n_per_edge


def resample_arclength(trajectory, n_out):
    points = trajectory @ SIMPLEX_VERTICES
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(arc[-1])
    if total <= 0.0:
        return trajectory
    targets = np.linspace(0.0, total, n_out)
    output = np.empty((n_out, trajectory.shape[1]))
    for index in range(trajectory.shape[1]):
        output[:, index] = np.interp(targets, arc, trajectory[:, index])
    output = np.clip(output, 0.0, None)
    output /= output.sum(axis=1, keepdims=True)
    return output


def add_trajectory_arrowheads(
    axis,
    points,
    point_rgb,
    size,
    edge_fraction,
):
    n_points = len(points)
    for first, second in ((0, 1), (1, 2), (0, 2)):
        vertex_a = SIMPLEX_VERTICES[first]
        vertex_b = SIMPLEX_VERTICES[second]
        midpoint = 0.5 * (vertex_a + vertex_b)
        center_index = int(
            np.argmin(np.linalg.norm(points - midpoint, axis=1))
        )
        motion = (
            points[min(center_index + 1, n_points - 1)]
            - points[max(center_index - 1, 0)]
        )
        if np.hypot(motion[0], motion[1]) < 1e-12:
            continue
        if (vertex_a - points[center_index]) @ motion >= (
            vertex_b - points[center_index]
        ) @ motion:
            back, forward = vertex_b, vertex_a
        else:
            back, forward = vertex_a, vertex_b
        anchor = (1.0 - edge_fraction) * back + edge_fraction * forward
        index = int(np.argmin(np.linalg.norm(points - anchor, axis=1)))
        tangent = (
            points[min(index + 1, n_points - 1)]
            - points[max(index - 1, 0)]
        )
        if np.hypot(tangent[0], tangent[1]) < 1e-12:
            continue
        angle = np.degrees(np.arctan2(tangent[1], tangent[0])) - 90.0
        axis.plot(
            points[index, 0],
            points[index, 1],
            marker=(3, 0, angle),
            markersize=size,
            markerfacecolor=tuple(point_rgb[index]),
            markeredgecolor="none",
            linestyle="none",
            zorder=5,
        )


def plot_simplex_trajectories(
    trajectories,
    strategies,
    payoff,
    save_path,
    field_n_per_edge=14,
    arrow_length=0.055,
    trajectory_arrow_size=18.0,
    trajectory_arrow_fraction=2.0 / 3.0,
):
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.add_patch(
        plt.Polygon(
            SIMPLEX_VERTICES,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            zorder=1,
        )
    )

    vertex_rgb = _strategy_colors(strategies, alpha=1.0)[:, :3]
    for trajectory in trajectories:
        trajectory = resample_arclength(trajectory, n_out=2000)
        points = trajectory @ SIMPLEX_VERTICES
        point_rgb = np.clip(trajectory @ vertex_rgb, 0.0, 1.0)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segment_rgb = 0.5 * (point_rgb[:-1] + point_rgb[1:])
        axis.add_collection(
            LineCollection(
                segments,
                colors=segment_rgb,
                linewidth=4.0,
                capstyle="round",
                joinstyle="round",
                zorder=4,
            )
        )
        add_trajectory_arrowheads(
            axis,
            points,
            point_rgb,
            size=trajectory_arrow_size,
            edge_fraction=trajectory_arrow_fraction,
        )

    barycentric = simplex_grid(field_n_per_edge)
    interior = barycentric[np.sum(barycentric == 0.0, axis=1) == 0]
    positions = interior @ SIMPLEX_VERTICES
    velocities_barycentric = np.array(
        [replicator_rhs(0.0, state, payoff) for state in interior]
    )
    velocities = velocities_barycentric @ SIMPLEX_VERTICES
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    velocities = velocities / norms * arrow_length
    axis.quiver(
        positions[:, 0],
        positions[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="gray",
        alpha=0.6,
        width=0.005,
        zorder=3,
    )

    axis.set_xlim(-0.2, 1.2)
    axis.set_ylim(-0.15, 1.05)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, transparent=True)
    plt.close(fig)


def main() -> None:
    setup_style()
    total_time = 5000.0
    source_time = 2375.0
    b_value = 5.0
    c_value = 1.0
    selected, strategies, initial_distribution, discarded_mass = (
        load_cycle_initial_conditions(DATA_PATH, source_time)
    )
    print(f"Initial population from {DATA_PATH} at t={source_time:g}")
    print(f"Selected strategy indices (zero-based): {selected.tolist()}")
    print(f"Strategies:\n{np.array2string(strategies, precision=17)}")
    print(f"Normalized proportions: {np.array2string(initial_distribution, precision=17)}")
    print(f"Discarded population mass: {discarded_mass:.17g}")
    payoff = reactive_payoff_matrix(strategies, b_value, c_value)
    solution = solve_ivp(
        replicator_rhs,
        (0.0, total_time),
        initial_distribution,
        args=(payoff,),
        method="RK45",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")
    dense_times = np.linspace(0.0, total_time, 4000)
    dense = np.maximum(solution.sol(dense_times), 0.0)
    dense /= dense.sum(axis=0, keepdims=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_simplex_trajectories(
        [dense.T],
        strategies,
        payoff,
        OUTPUT_DIR / "cycle.png",
    )


if __name__ == "__main__":
    main()
