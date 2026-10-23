from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    binary_strategy_index,
    create_replay_buffer,
    epsilon_exp_decay,
    epsilon_greedy,
    flush_agent_buffer,
    store_transition,
)
from common.simulation import (
    aggregate_channels,
    run_repetitions,
    strategy_proportions,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
GOOD, BAD = 0, 1
NUM_OBS = 4
NUM_ACTIONS = 2
NUM_STRATEGIES = 16
STRATEGY_NAMES = tuple("".join(actions) for actions in product("CD", repeat=4))

IMAGE_SCORING_NORM = np.array(
    [
        [GOOD, GOOD, GOOD, GOOD],
        [BAD, BAD, BAD, BAD],
    ],
    dtype=np.int64,
)
IMAGE_SCORING_LEGAL_ACTIONS = np.array(
    [
        [True, True],
        [False, True],
        [True, True],
        [False, True],
    ],
    dtype=np.bool_,
)


def donation_payoff(benefit, cost):
    return np.array(
        [[benefit - cost, -cost], [benefit, 0.0]],
        dtype=np.float64,
    )


@njit(cache=True)
def pair_state_index(own_reputation, opponent_reputation):
    return own_reputation * 2 + opponent_reputation


def init_q_tables(q_init, n_agents):
    q_array = np.asarray(q_init, dtype=np.float64)
    return np.broadcast_to(q_array, (n_agents, NUM_OBS, NUM_ACTIONS)).copy()


@njit(cache=True)
def sample_pair(n_agents):
    first = np.random.randint(n_agents)
    while True:
        second = np.random.randint(n_agents)
        if second != first:
            return first, second


@njit(cache=True)
def observe_reputation(true_reputation, observation_accuracy):
    if np.random.random() < observation_accuracy:
        return true_reputation
    return GOOD


@njit(cache=True)
def record_snapshot(
    q_tables,
    reputations,
    legal_actions,
    strategies,
    good_fractions,
    index,
):
    for agent in range(q_tables.shape[0]):
        strategies[index, agent] = binary_strategy_index(
            q_tables[agent], mask=legal_actions
        )
    good_fractions[index] = np.mean(reputations == GOOD)


@njit(cache=True)
def close_previous_and_store_current(
    q_tables,
    agent,
    current_obs,
    current_action,
    current_reward,
    has_pending,
    pending_obs,
    pending_action,
    pending_reward,
    buffer,
    batch_size,
    alpha,
    gamma,
    legal_actions,
):
    if has_pending[agent]:
        store_transition(
            buffer,
            agent,
            pending_obs[agent],
            pending_action[agent],
            pending_reward[agent],
            current_obs,
        )
        if buffer.count[agent] >= batch_size:
            flush_agent_buffer(
                q_tables,
                agent,
                buffer,
                alpha,
                gamma,
                mask=legal_actions,
            )

    pending_obs[agent] = current_obs
    pending_action[agent] = current_action
    pending_reward[agent] = current_reward
    has_pending[agent] = True


@njit(cache=True)
def _run_simulation_core(
    time_steps,
    batch_size,
    n_snapshots,
    observation_accuracy,
    norm,
    payoff,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
    decay_steps,
    reputations,
    q_tables,
    legal_actions,
    buffer,
    seed,
):
    np.random.seed(seed)
    n_agents = q_tables.shape[0]
    has_pending = np.zeros(n_agents, dtype=np.bool_)
    pending_obs = np.empty(n_agents, dtype=np.int64)
    pending_action = np.empty(n_agents, dtype=np.int64)
    pending_reward = np.empty(n_agents, dtype=np.float64)

    strategies = np.empty((n_snapshots, n_agents), dtype=np.int8)
    good_fractions = np.empty(n_snapshots, dtype=np.float64)
    snapshot_times = np.empty(n_snapshots, dtype=np.int64)
    record_snapshot(
        q_tables,
        reputations,
        legal_actions,
        strategies,
        good_fractions,
        0,
    )
    snapshot_times[0] = 0
    snapshot_index = 1
    interval = (
        time_steps / (n_snapshots - 1)
        if n_snapshots > 1
        else time_steps + 1.0
    )

    for time in range(1, time_steps + 1):
        epsilon = epsilon_exp_decay(
            time, epsilon_start, epsilon_end, decay_steps
        )
        first, second = sample_pair(n_agents)
        first_reputation = reputations[first]
        second_reputation = reputations[second]
        first_obs = pair_state_index(
            first_reputation,
            observe_reputation(second_reputation, observation_accuracy),
        )
        second_obs = pair_state_index(
            second_reputation,
            observe_reputation(first_reputation, observation_accuracy),
        )


        first_action = epsilon_greedy(
            q_tables[first], first_obs, epsilon, mask=legal_actions,
        )
        second_action = epsilon_greedy(
            q_tables[second], second_obs, epsilon, mask=legal_actions,
        )
        first_reward = payoff[first_action, second_action]
        second_reward = payoff[second_action, first_action]

        close_previous_and_store_current(
            q_tables,
            first,
            first_obs,
            first_action,
            first_reward,
            has_pending,
            pending_obs,
            pending_action,
            pending_reward,
            buffer,
            batch_size,
            alpha,
            gamma,
            legal_actions,
        )
        close_previous_and_store_current(
            q_tables,
            second,
            second_obs,
            second_action,
            second_reward,
            has_pending,
            pending_obs,
            pending_action,
            pending_reward,
            buffer,
            batch_size,
            alpha,
            gamma,
            legal_actions,
        )

        reputations[first] = norm[
            first_action,
            pair_state_index(first_reputation, second_reputation),
        ]
        reputations[second] = norm[
            second_action,
            pair_state_index(second_reputation, first_reputation),
        ]

        if snapshot_index < n_snapshots and time >= snapshot_index * interval:
            record_snapshot(
                q_tables,
                reputations,
                legal_actions,
                strategies,
                good_fractions,
                snapshot_index,
            )
            snapshot_times[snapshot_index] = time
            snapshot_index += 1

    return strategies, good_fractions, snapshot_times


def run_simulation(
    time_steps=500_000,
    batch_size=100,
    n_agents=100,
    n_snapshots=1000,
    observation_accuracy=0.9,
    alpha=0.001,
    gamma=0.95,
    epsilon_start=0.05,
    epsilon_end=0.05,
    decay_steps=100_000.0,
    seed=42,
    *,
    norm_matrix,
    payoff_matrix,
    reputation_init,
    q_init,
    legal_actions,
):
    reputations = np.asarray(reputation_init, dtype=np.int64).copy()
    q_tables = init_q_tables(q_init, n_agents)
    return _run_simulation_core(
        time_steps,
        batch_size,
        n_snapshots,
        observation_accuracy,
        np.asarray(norm_matrix, dtype=np.int64),
        np.asarray(payoff_matrix, dtype=np.float64),
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        decay_steps,
        reputations,
        q_tables,
        np.asarray(legal_actions, dtype=np.bool_),
        create_replay_buffer(n_agents, batch_size),
        seed,
    )


def main():
    b_value = 5.0
    c_value = 1.0
    n_reps = 100
    base_seed = 0
    save_every = 1000
    n_agents = 100
    q_init = np.array(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    parameters = dict(
        time_steps=1_000_000,
        batch_size=64,
        n_agents=n_agents,
        n_snapshots=1001,
        observation_accuracy=0.9,
        norm_matrix=IMAGE_SCORING_NORM,
        payoff_matrix=donation_payoff(b_value, c_value),
        alpha=0.1,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=1_000_000.0,
        q_init=q_init,
        reputation_init=np.full(n_agents, GOOD, dtype=np.int64),
        legal_actions=IMAGE_SCORING_LEGAL_ACTIONS,
    )

    warmup = parameters.copy()
    warmup.update(time_steps=16, n_agents=2, batch_size=2, n_snapshots=2)
    warmup["reputation_init"] = parameters["reputation_init"][:2]
    run_simulation(seed=base_seed, **warmup)
    proportions, good_fractions, times = run_repetitions(
        run_simulation,
        lambda result: (
            strategy_proportions(result[0], n_strategies=NUM_STRATEGIES), result[1],
        ),
        n_reps=n_reps,
        base_seed=base_seed,
        parameters=parameters,
        n_channels=2,
    )
    channels = np.concatenate((proportions, good_fractions[:, :, None]), axis=2)
    frame = aggregate_channels(
        channels, times, (*STRATEGY_NAMES, "good_reputation"),
        save_every=save_every,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "sim_image_scoring.csv"
    frame.to_csv(output_path, index=False)

    final_counts = np.rint(proportions[:, -1] * parameters["n_agents"]).astype(int)
    runs = pd.DataFrame(
        {
            "seed": np.arange(base_seed, base_seed + n_reps),
            "time": int(times[-1]),
            **{
                f"count_{name}": final_counts[:, index]
                for index, name in enumerate(STRATEGY_NAMES)
            },
            "good_reputation_fraction": good_fractions[:, -1],
        }
    )
    runs.to_csv(DATA_DIR / "sim_image_scoring_runs.csv", index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
