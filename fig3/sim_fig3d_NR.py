from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    create_replay_buffer,
    binary_strategy_index,
    epsilon_exp_decay,
    epsilon_greedy,
    store_transition,
    flush_agent_buffer,
)
from common.simulation import (
    aggregate_channels,
    run_repetitions,
    strategy_proportions,
)

from common.topology import (
    _find, _is_connected, _conflict_position, _repair_groups, regular_hypergraph,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
C = 0
STATE_1, STATE_2 = 0, 1
NUM_OBS = 2
NUM_ACTIONS = 2
STRATEGY_NAMES = ("CC", "CD", "DC", "DD")
SCHEDULE_DEFAULT = ((3, 4, 16), (4, 4, 16), (5, 4, 16))


@njit(cache=True)
def two_state_pgg_next_state(state, n_cooperators, group_size, q_c, q_r):
    if state == STATE_1:
        if np.random.random() < q_c * (group_size - n_cooperators) / group_size:
            return STATE_2
        return STATE_1
    if n_cooperators == group_size and np.random.random() < q_r:
        return STATE_1
    return STATE_2


def init_q_tables(q_init, n_agents):
    q_array = np.asarray(q_init, dtype=np.float64)
    return np.broadcast_to(q_array, (n_agents, NUM_OBS, NUM_ACTIONS)).copy()


@njit(cache=True)
def _run_simulation_core(
    n_agents,
    time_steps,
    update_interval,
    r_values,
    cost,
    q_c,
    q_r,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
    decay_steps,
    initial_state,
    q_tables,
    buffer,
    edges,
    seed,
):
    np.random.seed(seed)
    n_edges = edges.shape[0]
    group_size = edges.shape[1]
    edge_states = np.full(n_edges, initial_state, dtype=np.int64)
    n_updates = time_steps // update_interval
    strategies = np.empty((n_updates + 1, n_agents), dtype=np.int8)
    update_times = np.empty(n_updates + 1, dtype=np.int64)
    for agent in range(n_agents):
        strategies[0, agent] = binary_strategy_index(q_tables[agent])
    update_times[0] = 0
    snapshot_index = 1

    actions_per_state = np.empty((n_agents, NUM_OBS), dtype=np.int64)

    for time in range(1, time_steps + 1):
        epsilon = epsilon_exp_decay(
            time, epsilon_start, epsilon_end, decay_steps
        )
        for agent in range(n_agents):
            actions_per_state[agent, STATE_1] = epsilon_greedy(
                q_tables[agent], STATE_1, epsilon
            )
            actions_per_state[agent, STATE_2] = epsilon_greedy(
                q_tables[agent], STATE_2, epsilon
            )

        for edge_index in range(n_edges):
            state = edge_states[edge_index]
            n_cooperators = 0
            for member_index in range(group_size):
                agent = edges[edge_index, member_index]
                if actions_per_state[agent, state] == C:
                    n_cooperators += 1

            public_good = (
                r_values[state] * cost * n_cooperators / group_size
            )
            next_state = two_state_pgg_next_state(
                state, n_cooperators, group_size, q_c, q_r
            )
            for member_index in range(group_size):
                agent = edges[edge_index, member_index]
                action = actions_per_state[agent, state]
                action_cost = cost if action == C else 0.0
                store_transition(
                    buffer,
                    agent,
                    state,
                    action,
                    public_good - action_cost,
                    next_state,
                )
            edge_states[edge_index] = next_state

        if time % update_interval == 0:
            for agent in range(n_agents):
                flush_agent_buffer(q_tables, agent, buffer, alpha, gamma)
                strategies[snapshot_index, agent] = binary_strategy_index(
                    q_tables[agent]
                )
            update_times[snapshot_index] = time
            snapshot_index += 1

    return strategies[:snapshot_index], update_times[:snapshot_index]


def run_simulation(
    n_agents=30,
    order=5,
    degree=4,
    time_steps=500_000,
    update_interval=8,
    r_values=(4.0, 1.5),
    cost=1.0,
    q_c=0.2,
    q_r=0.05,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    decay_steps=100_000.0,
    initial_state=STATE_1,
    seed=42,
    *,
    q_init,
):
    rng = np.random.default_rng(seed)
    edges = regular_hypergraph(n_agents, order, degree, rng)
    q_tables = init_q_tables(q_init, n_agents)
    capacity = degree * update_interval
    return _run_simulation_core(
        n_agents,
        time_steps,
        update_interval,
        np.asarray(r_values, dtype=np.float64),
        float(cost),
        float(q_c),
        float(q_r),
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        decay_steps,
        initial_state,
        q_tables,
        create_replay_buffer(n_agents, capacity),
        edges,
        seed,
    )


def main():
    n_agents = 105
    r_values = (2.8, 1.2)
    cost = 1.0
    n_reps = 100
    time_steps = 300_000
    common_parameters = dict(
        time_steps=time_steps,
        r_values=r_values,
        cost=cost,
        q_c=1.0,
        q_r=1.0,
        alpha=0.1,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=100_000.0,
        initial_state=STATE_1,
        q_init=np.array([[0.0, 100.0], [0.0, 100.0]]),
    )

    print("JIT warmup...")
    run_simulation(
        **{**common_parameters, "time_steps": 200},
        n_agents=6,
        order=3,
        degree=2,
        update_interval=10,
        seed=9999,
    )

    frames = []
    for order, degree, update_interval in SCHEDULE_DEFAULT:
        print(
            f"\nSweep: k={order}, d={degree}, B={update_interval}, "
            f"batch={degree * update_interval}"
        )
        proportions, update_times = run_repetitions(
            run_simulation,
            lambda result: (strategy_proportions(result[0]),),
            n_reps=n_reps,
            base_seed=0,
            parameters=dict(
                n_agents=n_agents,
                order=order,
                degree=degree,
                update_interval=update_interval,
                **common_parameters,
            ),
        )
        frames.append(
            aggregate_channels(
                proportions,
                update_times,
                STRATEGY_NAMES,
                n_agents=n_agents,
                k=order,
                d=degree,
                n_edges=n_agents * degree // order,
                B=update_interval,
                hypergraph="regular",
                gamma=common_parameters["gamma"],
                c=cost,
                q_c=common_parameters["q_c"],
                q_r=common_parameters["q_r"],
                r1=r_values[0],
                r2=r_values[1],
            )
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "hypergraph_2pgg_sweep.csv"
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
