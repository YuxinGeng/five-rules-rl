from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    create_replay_buffer,
    argmax_random_tie,
    epsilon_greedy,
    flush_agent_buffer,
    store_transition,
)
from common.simulation import (
    run_repetitions,
    aggregate_channels,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
C = 0
OBS = 0


@njit(cache=True)
def greedy_c(q_table):
    return 1 if argmax_random_tie(q_table[OBS]) == C else 0


@njit(cache=True)
def _run_simulation_core(
    time_steps,
    batch_size,
    payoff_matrix,
    alpha,
    gamma,
    epsilon,
    q_table,
    buffer,
    seed,
):
    np.random.seed(seed)

    q_tables = q_table[np.newaxis, :, :]
    n_updates = time_steps // batch_size
    greedy_c_series = np.empty(n_updates + 1, dtype=np.int8)
    update_times = np.empty(n_updates + 1, dtype=np.int64)
    greedy_c_series[0] = greedy_c(q_table)
    update_times[0] = 0
    snapshot_index = 1

    for time in range(1, time_steps + 1):
        action_self = epsilon_greedy(q_table, OBS, epsilon)
        action_opponent = epsilon_greedy(q_table, OBS, epsilon)
        store_transition(
            buffer,
            0,
            OBS,
            action_self,
            payoff_matrix[action_self, action_opponent],
            OBS,
        )
        if time % batch_size == 0:
            flush_agent_buffer(q_tables, 0, buffer, alpha, gamma)
            greedy_c_series[snapshot_index] = greedy_c(q_table)
            update_times[snapshot_index] = time
            snapshot_index += 1

    return greedy_c_series[:snapshot_index], update_times[:snapshot_index]


def run_simulation(
    time_steps=500_000,
    batch_size=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.01,
    seed=42,
    *,
    payoff_matrix,
    q_init,
):
    q_table = np.asarray(q_init, dtype=np.float64).copy()
    return _run_simulation_core(
        time_steps,
        batch_size,
        np.asarray(payoff_matrix, dtype=np.float64),
        alpha,
        gamma,
        epsilon,
        q_table,
        create_replay_buffer(1, batch_size),
        seed,
    )


def main():
    q0 = np.array([[0.0, 10.0]], dtype=np.float64)
    time_steps = 1_000_000
    batch_sizes = (8, 32, 128)
    n_reps = 100
    common_parameters = dict(
        time_steps=time_steps,
        payoff_matrix=np.array([[4.0, 0.0], [2.0, 1.0]], dtype=np.float64),
        alpha=0.1,
        gamma=0.0, # No future reward in one-shot games
        epsilon=0.1,
        q_init=q0,
    )

    print("Running multi-batch self-play Stag-Hunt simulation...")
    run_simulation(
        **{**common_parameters, "time_steps": 200}, batch_size=10, seed=9999,
    )
    frames = []
    for batch_size in batch_sizes:
        print(f"\nRunning batch size {batch_size}...")
        series, update_times = run_repetitions(
            run_simulation,
            lambda result: (result[0].astype(np.float64),),
            n_reps=n_reps,
            base_seed=0,
            parameters=dict(batch_size=batch_size, **common_parameters),
        )
        frames.append(
            aggregate_channels(
                series[:, :, None], update_times, ("greedy_c",),
                save_every=20_000, batch_size=batch_size,
            )
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "sim-SH-self-play.csv"
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
