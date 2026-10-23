from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.colors import ListedColormap, LogNorm, to_rgb
from scipy.integrate import solve_ivp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import COLORS, setup_style










alpha, b, c, eps, gamma = sp.symbols(
    "alpha b c varepsilon gamma", real=True
)
qC, qD, q, z, tau = sp.symbols("q_C q_D q z tau", real=True)

qC_dot_regionC = alpha * (1 - eps / 2) * (
    (1 - eps / 2) * b - c + gamma * qC - qC
).factor()
qD_dot_regionC = alpha * (eps / 2) * (
    (1 - eps / 2) * b + gamma * qC - qD
).factor()
qC_dot_regionD = alpha * (eps / 2) * (
    (eps / 2) * b - c + gamma * qD - qC
).factor()
qD_dot_regionD = alpha * (1 - eps / 2) * (
    (eps / 2) * b + gamma * qD - qD
).factor()
qC_dot_regionE = tau * qC_dot_regionC + (1 - tau) * qC_dot_regionD
qD_dot_regionE = tau * qD_dot_regionC + (1 - tau) * qD_dot_regionD
qC_dot_regionE = qC_dot_regionE.subs({qC: q, qD: q}).factor()
qD_dot_regionE = qD_dot_regionE.subs({qC: q, qD: q}).factor()

solution_region_d = sp.solve(
    [sp.Eq(qC_dot_regionD, 0), sp.Eq(qD_dot_regionD, 0)],
    [qC, qD],
)
qC_sol_regionD = solution_region_d[qC].factor()
qD_sol_regionD = solution_region_d[qD].factor()

tau_solution = sp.solve(
    [sp.Eq(qC_dot_regionE, qD_dot_regionE)], [tau]
)
q_dot_regionE = qC_dot_regionE.subs(tau, tau_solution[tau]).factor()
q_solution = sp.solve([sp.Eq(q_dot_regionE, 0)], [q])

kappa = 1 - gamma
discriminant_z = (b - c) * (b * (1 - eps) ** 2 - c)
q_larger = ((b - c) + sp.sqrt(discriminant_z)) / (2 * kappa)

eta = eps / 2
lam = 1 - eps / 2
delta = 1 - eps
s_C_of_z = sp.expand(alpha * (delta * (lam * b - z) - lam * c))
s_D_of_z = sp.expand(alpha * (delta * (z - eta * b) - eta * c))
z_C_threshold = sp.solve(sp.Eq(s_C_of_z, 0), z)[0].factor()
z_D_threshold = sp.solve(sp.Eq(s_D_of_z, 0), z)[0].factor()


def filippov_segments(parameters):
    z_c = float(sp.N(z_C_threshold.subs(parameters)))
    z_d = float(sp.N(z_D_threshold.subs(parameters)))
    if z_d < z_c:
        return [
            ("escaping", -sp.oo, z_d),
            ("crossing", z_d, z_c),
            ("sliding", z_c, sp.oo),
        ]
    return [
        ("escaping", -sp.oo, z_c),
        ("crossing", z_c, z_d),
        ("sliding", z_d, sp.oo),
    ]


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FS = 32
LABEL_SIZE = FS - 2
TITLE_SIZE = FS
TICK_SIZE = FS - 4
X_LABEL_PAD = -3
FIG_SIZE = (7.2, 6.4)
MAIN_AXES_RECT = (0.16, 0.135, 0.6667, 0.75)
COLORBAR_RECT = (0.85, 0.135, 0.022, 0.75)
Q_D_LABEL = r"$\mathbf{q}_{\mathrm{D}}^{\ast}$"
Q_E_LABEL = r"$\mathbf{q}_{\mathrm{E}}^{\ast}$"
PARAMS = {alpha: 0.01, b: 2.0, c: 1.0, eps: 0.01, gamma: 0.5}
IVP_RTOL = 1e-8
IVP_ATOL = 1e-10


def truncate_colormap(colormap, minimum=0.0, maximum=1.0, n_colors=100):
    values = colormap(np.linspace(minimum, maximum, n_colors))
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"trunc({colormap.name},{minimum:.2f},{maximum:.2f})", values
    )


def lambdify_ode(expression):
    return sp.lambdify(
        (qC, qD), expression.subs(PARAMS), modules="numpy"
    )


def evaluate_scalar(expression):
    return float(sp.N(expression.subs(PARAMS)))


def integrate_filippov(
    initial,
    time_maximum,
    field_c,
    field_d,
    switching_c,
    switching_d,
    sample_dt=None,
    max_arcs=400,
):

    def rhs_c(_time, state):
        return field_c(state)

    def rhs_d(_time, state):
        return field_d(state)

    def rhs_slide(_time, state):
        s_c = switching_c(state)
        s_d = switching_d(state)
        tau = s_d / (s_d - s_c)
        field = tau * field_c(state) + (1.0 - tau) * field_d(state)
        midpoint = 0.5 * (field[0] + field[1])
        return np.array([midpoint, midpoint])

    def hit_from_c(_time, state):
        return state[0] - state[1]

    hit_from_c.terminal = True
    hit_from_c.direction = -1.0

    def hit_from_d(_time, state):
        return state[0] - state[1]

    hit_from_d.terminal = True
    hit_from_d.direction = 1.0

    def exit_to_c(_time, state):
        return switching_c(state)

    exit_to_c.terminal = True
    exit_to_c.direction = 1.0

    def exit_to_d(_time, state):
        return switching_d(state)

    exit_to_d.terminal = True
    exit_to_d.direction = -1.0

    modes = {
        "C": (rhs_c, (hit_from_c,)),
        "D": (rhs_d, (hit_from_d,)),
        "slide": (rhs_slide, (exit_to_c, exit_to_d)),
    }
    time = 0.0
    state = np.asarray(initial, dtype=float).copy()
    mode = "C" if state[0] >= state[1] else "D"
    points = [state.copy()]
    stalls = 0

    for _ in range(max_arcs):
        if time >= time_maximum or stalls > 3:
            break
        if mode == "slide":
            midpoint = 0.5 * (state[0] + state[1])
            state = np.array([midpoint, midpoint])
        rhs, events = modes[mode]
        solution = solve_ivp(
            rhs,
            (time, time_maximum),
            state,
            method="RK45",
            rtol=IVP_RTOL,
            atol=IVP_ATOL,
            events=events,
            dense_output=sample_dt is not None,
        )
        if sample_dt is not None and solution.t[-1] > solution.t[0]:
            samples = np.arange(
                solution.t[0], solution.t[-1], sample_dt
            )[1:]
            if samples.size:
                points.extend(solution.sol(samples).T)
        points.append(solution.y[:, -1].copy())
        stalls = stalls + 1 if solution.t[-1] <= time else 0
        time = solution.t[-1]
        state = solution.y[:, -1].copy()
        if solution.status != 1:
            break
        if mode == "slide":
            mode = "C" if solution.t_events[0].size else "D"
        else:
            s_c = switching_c(state)
            s_d = switching_d(state)
            if s_c < 0.0 < s_d:
                mode = "slide"
            else:
                mode = "D" if mode == "C" else "C"
    return np.asarray(points)


def classify_basin_point(
    x_value,
    y_value,
    equilibrium_d,
    equilibrium_diagonal,
    time_maximum,
    field_c,
    field_d,
    switching_c,
    switching_d,
):
    if equilibrium_diagonal is None:
        return 1.0
    trajectory = integrate_filippov(
        np.array([x_value, y_value]),
        time_maximum,
        field_c,
        field_d,
        switching_c,
        switching_d,
    )
    endpoint = trajectory[-1]
    distance_d = np.linalg.norm(endpoint - equilibrium_d)
    distance_e = np.linalg.norm(endpoint - equilibrium_diagonal)
    return 1.0 if distance_d < distance_e else 0.0


def compute_basin_grid(
    q_c_min,
    q_c_max,
    q_d_min,
    q_d_max,
    equilibrium_d,
    equilibrium_diagonal,
    field_c,
    field_d,
    switching_c,
    switching_d,
    coarse_n=50,
    fine_n=500,
    band_factor=1.5,
    time_maximum=40_000.0,
):
    if equilibrium_diagonal is None:
        x_values = np.linspace(q_c_min, q_c_max, coarse_n)
        y_values = np.linspace(q_d_min, q_d_max, coarse_n)
        return x_values, y_values, np.ones((coarse_n, coarse_n))

    x_coarse = np.linspace(q_c_min, q_c_max, coarse_n)
    y_coarse = np.linspace(q_d_min, q_d_max, coarse_n)
    labels_coarse = np.zeros((coarse_n, coarse_n))
    for row, y_value in enumerate(y_coarse):
        for column, x_value in enumerate(x_coarse):
            labels_coarse[row, column] = classify_basin_point(
                x_value,
                y_value,
                equilibrium_d,
                equilibrium_diagonal,
                time_maximum,
                field_c,
                field_d,
                switching_c,
                switching_d,
            )

    x_fine = np.linspace(q_c_min, q_c_max, fine_n)
    y_fine = np.linspace(q_d_min, q_d_max, fine_n)
    grid_x, grid_y = np.meshgrid(x_fine, y_fine)
    coarse_dx = max(
        (q_c_max - q_c_min) / max(coarse_n - 1, 1),
        (q_d_max - q_d_min) / max(coarse_n - 1, 1),
    )
    band_width = band_factor * coarse_dx
    boundary_x = equilibrium_d[1]
    diagonal_band = (
        (np.abs(grid_x - grid_y) <= band_width)
        & (grid_y <= boundary_x + band_width)
    )
    vertical_band = (
        (np.abs(grid_x - boundary_x) <= band_width)
        & (grid_y >= boundary_x - band_width)
    )
    in_band = diagonal_band | vertical_band

    column_indices = np.clip(
        np.round(
            (x_fine - q_c_min)
            / max(q_c_max - q_c_min, 1e-12)
            * (coarse_n - 1)
        ).astype(int),
        0,
        coarse_n - 1,
    )
    row_indices = np.clip(
        np.round(
            (y_fine - q_d_min)
            / max(q_d_max - q_d_min, 1e-12)
            * (coarse_n - 1)
        ).astype(int),
        0,
        coarse_n - 1,
    )
    labels_fine = labels_coarse[
        row_indices[:, None], column_indices[None, :]
    ].copy()
    for row, column in np.argwhere(in_band):
        labels_fine[row, column] = classify_basin_point(
            x_fine[column],
            y_fine[row],
            equilibrium_d,
            equilibrium_diagonal,
            time_maximum,
            field_c,
            field_d,
            switching_c,
            switching_d,
        )
    return x_fine, y_fine, labels_fine


def common_anchored_linspace(low, high, n_points, anchor_a, anchor_b):
    target_spacing = (high - low) / (n_points - 1)
    delta_anchor = abs(anchor_a - anchor_b)
    if delta_anchor < 1e-12:
        index = int(round((anchor_a - low) / target_spacing))
        index = max(0, min(n_points - 1, index))
        new_low = anchor_a - index * target_spacing
        return new_low + np.arange(n_points) * target_spacing
    n_intervals = max(1, int(round(delta_anchor / target_spacing)))
    spacing = delta_anchor / n_intervals
    midpoint = 0.5 * (low + high)
    index = int(
        round((anchor_a - midpoint) / spacing + (n_points - 1) / 2)
    )
    index = max(0, min(n_points - 1, index))
    new_low = anchor_a - index * spacing
    return new_low + np.arange(n_points) * spacing


def lighten(color, factor=0.2):
    red, green, blue = to_rgb(color)
    return (
        red + (1.0 - red) * factor,
        green + (1.0 - green) * factor,
        blue + (1.0 - blue) * factor,
    )


def _plot_filippov_segments(axis, segments, kappa, bounds):
    q_c_min, q_c_max, q_d_min, q_d_max = bounds
    segment_styles = {
        "escaping": dict(color=COLORS[3], linestyle=(0, (4, 3)), linewidth=4.0),
        "crossing": dict(color="#777777", linestyle=(0, (4, 3)), linewidth=4.0),
        "sliding": dict(color=COLORS[1], linestyle="-", linewidth=4.0),
    }
    diagonal_low = max(q_c_min, q_d_min)
    diagonal_high = min(q_c_max, q_d_max)
    for label, lower, upper in segments:
        q_lower = -np.inf if lower == -sp.oo else float(lower) / kappa
        q_upper = np.inf if upper == sp.oo else float(upper) / kappa
        segment_low = max(q_lower, diagonal_low)
        segment_high = min(q_upper, diagonal_high)
        if segment_high > segment_low:
            axis.plot(
                [segment_low, segment_high],
                [segment_low, segment_high],
                alpha=0.85,
                zorder=11,
                **segment_styles[label],
            )


def _plot_trajectories(
    axis, field_c, field_d, switching_c, switching_d,
    equilibrium_d, equilibrium_diagonal, time_scale,
):
    initial_conditions = ((0.9, 2.0), (2.7, 2.1), (0.0, -1.1), (-1.5, -1.0))
    for initial in initial_conditions:
        trajectory = integrate_filippov(
            np.array(initial),
            4000.0 * time_scale,
            field_c,
            field_d,
            switching_c,
            switching_d,
            sample_dt=2.0 * time_scale,
        )
        endpoint = trajectory[-1]
        distance_d = np.linalg.norm(endpoint - equilibrium_d)
        if equilibrium_diagonal is None:
            trajectory_color = lighten(COLORS[2])
        else:
            distance_e = np.linalg.norm(endpoint - equilibrium_diagonal)
            trajectory_color = lighten(
                COLORS[2] if distance_d < distance_e else COLORS[0]
            )
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=trajectory_color,
            linewidth=4.5,
            alpha=1.0,
            zorder=12,
            solid_capstyle="round",
        )
        axis.scatter(
            [initial[0]],
            [initial[1]],
            s=170,
            facecolor="white",
            edgecolor=trajectory_color,
            linewidth=2.4,
            zorder=13,
        )


def _plot_equilibria(axis, equilibrium_d, equilibrium_diagonal):
    axis.scatter(
        *equilibrium_d,
        s=280,
        color=COLORS[2],
        edgecolor="white",
        linewidth=2.2,
        zorder=14,
        label=Q_D_LABEL,
    )
    if equilibrium_diagonal is not None:
        axis.scatter(
            *equilibrium_diagonal,
            s=280,
            color=COLORS[0],
            edgecolor="white",
            linewidth=2.2,
            zorder=14,
            label=Q_E_LABEL,
        )


def _style_vector_axis(axis, bounds, epsilon):
    q_c_min, q_c_max, q_d_min, q_d_max = bounds
    integer_ticks = np.arange(
        np.ceil(min(q_c_min, q_d_min)),
        np.floor(max(q_c_max, q_d_max)) + 1,
    )
    axis.set_xlim(q_c_min, q_c_max)
    axis.set_ylim(q_d_min, q_d_max)
    axis.set_xticks(integer_ticks)
    axis.set_yticks(integer_ticks)
    axis.set_xlabel(r"$Q(\mathrm{C})$", fontsize=LABEL_SIZE, labelpad=X_LABEL_PAD)
    axis.set_ylabel(r"$Q(\mathrm{D})$", fontsize=LABEL_SIZE)
    axis.set_title(rf"$\varepsilon={epsilon}$", fontsize=TITLE_SIZE, pad=10)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=TICK_SIZE)
    axis.grid(False)


def plot_vector_field(
    q_c_min=-1.8,
    q_c_max=2.8,
    q_d_min=-1.8,
    q_d_max=2.8,
    grid_n=11,
    out_path=None,
):
    setup_style()
    bounds = (q_c_min, q_c_max, q_d_min, q_d_max)
    f_qc_c = lambdify_ode(qC_dot_regionC)
    f_qd_c = lambdify_ode(qD_dot_regionC)
    f_qc_d = lambdify_ode(qC_dot_regionD)
    f_qd_d = lambdify_ode(qD_dot_regionD)
    equilibrium_d = np.array(
        [evaluate_scalar(qC_sol_regionD), evaluate_scalar(qD_sol_regionD)]
    )

    b_value = PARAMS[b]
    c_value = PARAMS[c]
    epsilon = PARAMS[eps]
    threshold = 1.0 / (1.0 - epsilon) ** 2
    has_diagonal_equilibrium = b_value / c_value > threshold
    diagonal_value = (
        evaluate_scalar(q_larger) if has_diagonal_equilibrium else None
    )
    equilibrium_diagonal = (
        np.array([diagonal_value, diagonal_value])
        if diagonal_value is not None
        else None
    )

    def field_c(state):
        return np.array(
            [f_qc_c(state[0], state[1]), f_qd_c(state[0], state[1])]
        )

    def field_d(state):
        return np.array(
            [f_qc_d(state[0], state[1]), f_qd_d(state[0], state[1])]
        )

    eta = epsilon / 2.0
    lam = 1.0 - epsilon / 2.0
    delta = 1.0 - epsilon
    alpha_value = PARAMS[alpha]
    kappa = 1.0 - PARAMS[gamma]

    def switching_c(state):
        z_value = kappa * 0.5 * (state[0] + state[1])
        return alpha_value * (
            delta * (lam * b_value - z_value) - lam * c_value
        )

    def switching_d(state):
        z_value = kappa * 0.5 * (state[0] + state[1])
        return alpha_value * (
            delta * (z_value - eta * b_value) - eta * c_value
        )

    quiver_scale_factor = 1.5
    arrow_length = max(
        (q_c_max - q_c_min) / (grid_n * quiver_scale_factor),
        (q_d_max - q_d_min) / (grid_n * quiver_scale_factor),
    )
    shared_layout = np.isclose(epsilon, 0.01) or np.isclose(epsilon, 0.5)
    drawn_arrow_length = arrow_length * (1.25 if shared_layout else 1.0)

    grid_parameters = dict(PARAMS)
    grid_parameters[eps] = 0.5 if shared_layout else epsilon
    grid_equilibrium_d = np.array(
        [
            float(sp.N(qC_sol_regionD.subs(grid_parameters))),
            float(sp.N(qD_sol_regionD.subs(grid_parameters))),
        ]
    )
    grid_threshold = 1.0 / (1.0 - grid_parameters[eps]) ** 2
    if grid_parameters[b] / grid_parameters[c] > grid_threshold:
        grid_diagonal = float(sp.N(q_larger.subs(grid_parameters)))
        anchor_y = min(grid_equilibrium_d[1], grid_diagonal)
        anchor_x = max(grid_equilibrium_d[0], grid_diagonal)
    else:
        anchor_y = grid_equilibrium_d[1]
        anchor_x = grid_equilibrium_d[0]
    grid = common_anchored_linspace(
        min(q_c_min, q_d_min) + arrow_length,
        max(q_c_max, q_d_max) - arrow_length,
        grid_n,
        anchor_y,
        anchor_x,
    )
    q_c_grid = grid.copy()
    q_d_grid = grid.copy()
    if np.isclose(epsilon, 0.01):
        current_anchor_y = min(
            equilibrium_d[1],
            diagonal_value if diagonal_value is not None else equilibrium_d[1],
        )
        current_anchor_x = max(
            equilibrium_d[0],
            diagonal_value if diagonal_value is not None else equilibrium_d[0],
        )
        q_d_grid[np.argmin(np.abs(q_d_grid - current_anchor_y))] = current_anchor_y
        q_c_grid[np.argmin(np.abs(q_c_grid - current_anchor_x))] = current_anchor_x

    grid_qc, grid_qd = np.meshgrid(q_c_grid, q_d_grid)
    velocity_c = np.where(
        grid_qc >= grid_qd,
        f_qc_c(grid_qc, grid_qd),
        f_qc_d(grid_qc, grid_qd),
    )
    velocity_d = np.where(
        grid_qc >= grid_qd,
        f_qd_c(grid_qc, grid_qd),
        f_qd_d(grid_qc, grid_qd),
    )
    speed = np.hypot(velocity_c, velocity_d)
    norm = np.where(speed > 0, speed, 1.0)
    unit_c = velocity_c / norm
    unit_d = velocity_d / norm

    time_scale = 0.1 / alpha_value
    _, _, basin_labels = compute_basin_grid(
        q_c_min,
        q_c_max,
        q_d_min,
        q_d_max,
        equilibrium_d,
        equilibrium_diagonal,
        field_c,
        field_d,
        switching_c,
        switching_d,
        fine_n=500,
        time_maximum=4000.0 * time_scale,
    )

    fig = plt.figure(figsize=FIG_SIZE)
    axis = fig.add_axes(MAIN_AXES_RECT)
    axis.imshow(
        basin_labels,
        extent=(q_c_min, q_c_max, q_d_min, q_d_max),
        origin="lower",
        cmap=ListedColormap([COLORS[0], COLORS[2]]),
        vmin=0.0,
        vmax=1.0,
        alpha=0.15,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )

    positive_speed = speed[speed > 0]
    minimum = float(positive_speed.min()) if positive_speed.size else 1e-6
    maximum = float(speed.max()) if speed.max() > 0 else 1.0
    log_minimum = max(minimum, maximum * 1e-3)
    color_minimum = min(log_minimum, 1e-3) if np.isclose(epsilon, 0.5) else log_minimum
    log_speed = np.log10(np.clip(speed, log_minimum, maximum))
    factor = (log_speed - np.log10(log_minimum)) / max(
        np.log10(maximum) - np.log10(log_minimum), 1e-12
    )
    size = 0.5 + 0.5 * factor
    size = np.where(speed > 0, size, 0.0)
    arrows_c = unit_c * size
    arrows_d = unit_d * size



    tail_visible = (
        (grid_qc >= q_c_min)
        & (grid_qc <= q_c_max)
        & (grid_qd >= q_d_min)
        & (grid_qd <= q_d_max)
    )
    if shared_layout:
        arrow_visible = tail_visible
    else:
        head_c = grid_qc + drawn_arrow_length * arrows_c
        head_d = grid_qd + drawn_arrow_length * arrows_d
        arrow_visible = (
            tail_visible
            & (head_c >= q_c_min)
            & (head_c <= q_c_max)
            & (head_d >= q_d_min)
            & (head_d <= q_d_max)
        )
    arrows_c = np.where(arrow_visible, arrows_c, np.nan)
    arrows_d = np.where(arrow_visible, arrows_d, np.nan)

    segments = filippov_segments(PARAMS)
    sliding_range = None
    for label, lower, upper in segments:
        if label == "sliding":
            q_lower = -np.inf if lower == -sp.oo else float(lower) / kappa
            q_upper = np.inf if upper == sp.oo else float(upper) / kappa
            sliding_range = (q_lower, q_upper)
            break
    if sliding_range is not None:
        mask = (
            (np.abs(grid_qc - grid_qd) < 1e-9)
            & (grid_qc >= sliding_range[0])
            & (grid_qc <= sliding_range[1])
        )
        arrows_c = np.where(mask, np.nan, arrows_c)
        arrows_d = np.where(mask, np.nan, arrows_d)

    colormap = truncate_colormap(plt.get_cmap("viridis"), 0.18, 0.92)
    quiver = axis.quiver(
        grid_qc,
        grid_qd,
        arrows_c,
        arrows_d,
        speed,
        cmap=colormap,
        norm=LogNorm(vmin=color_minimum, vmax=maximum),
        angles="xy",
        scale_units="xy",
        scale=1.0 / drawn_arrow_length,
        pivot="tail",
        width=0.0050 if shared_layout else 0.0045,
        headwidth=4.5,
        headlength=5.0,
        headaxislength=4.5,
        minshaft=1.0,
    )
    colorbar_axis = fig.add_axes(COLORBAR_RECT)
    colorbar = fig.colorbar(quiver, cax=colorbar_axis)
    colorbar.outline.set_visible(False)
    if np.isclose(epsilon, 0.5):
        colorbar.set_ticks([1e-3, 1e-2])
        colorbar.set_ticklabels([r"$10^{-3}$", r"$10^{-2}$"])
    colorbar.ax.tick_params(length=0, which="both", labelsize=TICK_SIZE)

    _plot_filippov_segments(axis, segments, kappa, bounds)

    _plot_trajectories(
        axis, field_c, field_d, switching_c, switching_d,
        equilibrium_d, equilibrium_diagonal, time_scale,
    )

    _plot_equilibria(axis, equilibrium_d, equilibrium_diagonal)

    _style_vector_axis(axis, bounds, epsilon)

    if out_path is None:
        out_path = OUTPUT_DIR / "coupling_vector_field.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"saved -> {out_path}")
    return fig, axis


def main():
    for epsilon in (0.01, 0.5):
        PARAMS[eps] = epsilon
        plot_vector_field(
            out_path=OUTPUT_DIR / f"vector_field_eps{epsilon}.png"
        )
        plt.close()


if __name__ == "__main__":
    main()
