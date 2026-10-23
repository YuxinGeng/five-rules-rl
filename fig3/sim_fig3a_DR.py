from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    create_replay_buffer,
    epsilon_greedy,
    store_transition,
    flush_agent_buffer,
    epsilon_exp_decay,
)
from common.simulation import (
    aggregate_channels,
    run_repetitions,
    strategy_proportions,
)


DATA_DIR = Path(__file__).resolve().parent / "data"

C, D = 0, 1
NUM_OBS = 4
NUM_ACTIONS = 2
ALLD, GRIM, WSLS, OTHER = range(4)
STRATEGY_NAMES = ("ALLD", "GRIM", "WSLS", "Other")


@njit(cache=True)
def classify_strategy(q_player):
    c0 = q_player[0, C] > q_player[0, D]
    c1 = q_player[1, C] > q_player[1, D]
    c2 = q_player[2, C] > q_player[2, D]
    c3 = q_player[3, C] > q_player[3, D]

    if not (c0 or c1 or c2 or c3):
        return ALLD
    if c0 and not (c1 or c2 or c3):
        return GRIM
    if c0 and c3 and not c1 and not c2:
        return WSLS
    return OTHER


@njit(cache=True)
def _run_simulation_core(
    time_steps,
    batch_size,
    payoff,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
    decay_steps,
    initial_obs_0,
    initial_obs_1,
    q_tables,
    buffer,
    seed,
):
    np.random.seed(seed)
    n_updates = time_steps // batch_size
    strategies = np.empty((n_updates + 1, 2), dtype=np.int8)
    update_times = np.empty(n_updates + 1, dtype=np.int64)

    strategies[0, 0] = classify_strategy(q_tables[0])
    strategies[0, 1] = classify_strategy(q_tables[1])
    update_times[0] = 0
    snapshot_index = 1
    obs_0 = initial_obs_0
    obs_1 = initial_obs_1

    for time in range(1, time_steps + 1):
        epsilon = epsilon_exp_decay(
            time, epsilon_start, epsilon_end, decay_steps
        )
        action_0 = epsilon_greedy(q_tables[0], obs_0, epsilon)
        action_1 = epsilon_greedy(q_tables[1], obs_1, epsilon)

        next_obs_0 = action_0 * 2 + action_1
        next_obs_1 = action_1 * 2 + action_0
        store_transition(
            buffer,
            0,
            obs_0,
            action_0,
            payoff[action_0, action_1],
            next_obs_0,
        )
        store_transition(
            buffer,
            1,
            obs_1,
            action_1,
            payoff[action_1, action_0],
            next_obs_1,
        )

        if time % batch_size == 0:
            flush_agent_buffer(q_tables, 0, buffer, alpha, gamma)
            flush_agent_buffer(q_tables, 1, buffer, alpha, gamma)
            strategies[snapshot_index, 0] = classify_strategy(q_tables[0])
            strategies[snapshot_index, 1] = classify_strategy(q_tables[1])
            update_times[snapshot_index] = time
            snapshot_index += 1

        obs_0, obs_1 = next_obs_0, next_obs_1

    return strategies[:snapshot_index], update_times[:snapshot_index]


def run_simulation(
    time_steps=500_000,
    batch_size=100,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    decay_steps=100_000.0,
    initial_obs=(0, 0),
    seed=42,
    *,
    payoff_matrix,
    q_init,
):
    q_tables = np.asarray(q_init, dtype=np.float64).copy()
    return _run_simulation_core(
        time_steps,
        batch_size,
        np.asarray(payoff_matrix, dtype=np.float64),
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        decay_steps,
        initial_obs[0],
        initial_obs[1],
        q_tables,
        create_replay_buffer(2, batch_size),
        seed,
    )


def main():
    b_value = 5.0
    c_value = 1.0
    q0 = np.array(
        [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]],
        dtype=np.float64,
    )
    parameters = dict(
        time_steps=5_000_000,
        batch_size=64,
        payoff_matrix=np.array(
            [[b_value - c_value, -c_value], [b_value, 0.0]],
            dtype=np.float64,
        ),
        alpha=0.1,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=1_000_000.0,
        initial_obs=(3, 3),
        q_init=np.stack([q0, q0]),
    )

    print("Running multi-repetition direct-reciprocity simulation...")
    run_simulation(**{**parameters, "time_steps": 200, "batch_size": 100}, seed=9999)
    proportions, update_times = run_repetitions(
        run_simulation,
        lambda result: (strategy_proportions(result[0]),),
        n_reps=100,
        base_seed=0,
        parameters=parameters,
    )
    frame = aggregate_channels(
        proportions, update_times, STRATEGY_NAMES, save_every=1000,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "sim-mem1.csv"
    frame.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
