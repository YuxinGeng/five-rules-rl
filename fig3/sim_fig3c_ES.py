from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    create_replay_buffer,
    binary_strategy_index,
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

C = 0
STATE_1, STATE_2 = 0, 1
NUM_OBS = 2
NUM_ACTIONS = 2
STRATEGY_NAMES = ("CC", "CD", "DC", "DD")


def donation_payoff_matrices(b_values, c_value):
    return np.array(
        [
            [[b_values[0] - c_value, -c_value], [b_values[0], 0.0]],
            [[b_values[1] - c_value, -c_value], [b_values[1], 0.0]],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def next_state(action_0, action_1):
    return STATE_1 if action_0 == C and action_1 == C else STATE_2


@njit(cache=True)
def _run_simulation_core(
    time_steps,
    batch_size,
    payoff_matrices,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
    decay_steps,
    initial_state,
    q_tables,
    buffer,
    seed,
):
    np.random.seed(seed)
    n_updates = time_steps // batch_size
    strategies = np.empty((n_updates + 1, 2), dtype=np.int8)
    update_times = np.empty(n_updates + 1, dtype=np.int64)
    strategies[0, 0] = binary_strategy_index(q_tables[0])
    strategies[0, 1] = binary_strategy_index(q_tables[1])
    update_times[0] = 0
    snapshot_index = 1
    state = initial_state


    for time in range(1, time_steps + 1):
        epsilon = epsilon_exp_decay(
            time, epsilon_start, epsilon_end, decay_steps
        )
        action_0 = epsilon_greedy(q_tables[0], state, epsilon)
        action_1 = epsilon_greedy(q_tables[1], state, epsilon)
        next_state_value = next_state(action_0, action_1)
        store_transition(
            buffer,
            0,
            state,
            action_0,
            payoff_matrices[state, action_0, action_1],
            next_state_value,
        )
        store_transition(
            buffer,
            1,
            state,
            action_1,
            payoff_matrices[state, action_1, action_0],
            next_state_value,
        )
        if time % batch_size == 0:
            flush_agent_buffer(q_tables, 0, buffer, alpha, gamma)
            flush_agent_buffer(q_tables, 1, buffer, alpha, gamma)
            strategies[snapshot_index, 0] = binary_strategy_index(q_tables[0])
            strategies[snapshot_index, 1] = binary_strategy_index(q_tables[1])
            update_times[snapshot_index] = time
            snapshot_index += 1
        state = next_state_value

    return strategies[:snapshot_index], update_times[:snapshot_index]


def run_simulation(
    time_steps=500_000,
    batch_size=100,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    decay_steps=100_000.0,
    initial_state=STATE_1,
    seed=42,
    *,
    payoff_matrices,
    q_init,
):
    q_tables = np.asarray(q_init, dtype=np.float64).copy()
    return _run_simulation_core(
        time_steps,
        batch_size,
        np.asarray(payoff_matrices, dtype=np.float64),
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        decay_steps,
        initial_state,
        q_tables,
        create_replay_buffer(2, batch_size),
        seed,
    )


def main():
    b_values = (5.0, 2.0)
    c_value = 1.0
    q0 = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float64)
    parameters = dict(
        time_steps=2_000_000,
        batch_size=64,
        payoff_matrices=donation_payoff_matrices(b_values, c_value),
        alpha=0.1,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=1_000_000.0,
        initial_state=STATE_1,
        q_init=np.stack([q0, q0]),
    )
    print("Running environmental-stochasticity simulation...")
    run_simulation(**{**parameters, "time_steps": 200, "batch_size": 100}, seed=9999)
    proportions, update_times = run_repetitions(
        run_simulation,
        lambda result: (strategy_proportions(result[0]),),
        n_reps=100,
        base_seed=0,
        parameters=parameters,
    )
    frame = aggregate_channels(proportions, update_times, STRATEGY_NAMES)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "sim-2dg.csv"
    frame.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
