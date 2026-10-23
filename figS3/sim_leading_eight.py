from __future__ import annotations

import sys
import argparse
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
COOPERATE, DEFECT = 0, 1
BB = 3
NUM_OBS = 4
NUM_ACTIONS = 2
NUM_STRATEGIES = 16
STRATEGY_NAMES = tuple("".join(actions) for actions in product("CD", repeat=4))

NORMS = {
    "L1": np.array([[GOOD, GOOD, GOOD, GOOD], [BAD, GOOD, BAD, BAD]]),
    "L2": np.array([[GOOD, BAD, GOOD, GOOD], [BAD, GOOD, BAD, BAD]]),
    "L3": np.array([[GOOD, GOOD, GOOD, GOOD], [BAD, GOOD, BAD, GOOD]]),
    "L4": np.array([[GOOD, GOOD, GOOD, BAD], [BAD, GOOD, BAD, GOOD]]),
    "L5": np.array([[GOOD, BAD, GOOD, GOOD], [BAD, GOOD, BAD, GOOD]]),
    "L6": np.array([[GOOD, BAD, GOOD, BAD], [BAD, GOOD, BAD, GOOD]]),
    "L7": np.array([[GOOD, GOOD, GOOD, BAD], [BAD, GOOD, BAD, BAD]]),
    "L8": np.array([[GOOD, BAD, GOOD, BAD], [BAD, GOOD, BAD, BAD]]),
}
NORMS = {name: norm.astype(np.int64) for name, norm in NORMS.items()}


def target_strategy_index(norm):
    bb_action = (
        COOPERATE
        if norm[COOPERATE, BB] == GOOD and norm[DEFECT, BB] == BAD
        else DEFECT
    )
    return 4 + bb_action


INITIAL_Q_BY_TARGET = {
    "CDCC": np.array(
        [[99.0, 100.0], [0.0, 100.0], [0.0, 1.0], [99.0, 100.0]]
    ),
    "CDCD": np.array(
        [[99.0, 100.0], [0.0, 100.0], [0.0, 1.0], [0.0, 100.0]]
    ),
}


def initial_q_for_norm(norm):
    target = STRATEGY_NAMES[target_strategy_index(norm)]
    return INITIAL_Q_BY_TARGET[target].copy()


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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--snapshots", type=int, default=1001)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--decay-steps", type=float, default=1_000.0)
    parser.add_argument("--b", type=float, default=5.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--norms",
        nargs="+",
        choices=tuple(NORMS),
        default=list(NORMS),
        help="Leading-Eight norms to simulate.",
    )
    return parser


def _append_metadata(frame, args, norm_name, target):
    frame.insert(0, "norm", norm_name)
    frame.insert(1, "target_strategy", target)
    frame.insert(2, "b", args.b)
    frame.insert(3, "c", args.c)
    frame.insert(4, "observation_accuracy", args.q)
    frame.insert(5, "gamma", args.gamma)
    frame.insert(6, "alpha", args.alpha)
    frame.insert(7, "batch_size", args.batch_size)
    frame.insert(8, "n_agents", args.agents)
    return frame


def run_norm(norm_name, args):
    norm = NORMS[norm_name]
    target = STRATEGY_NAMES[target_strategy_index(norm)]
    parameters = dict(
        time_steps=args.steps,
        batch_size=args.batch_size,
        n_agents=args.agents,
        n_snapshots=args.snapshots,
        observation_accuracy=args.q,
        norm_matrix=norm,
        payoff_matrix=donation_payoff(args.b, args.c),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        decay_steps=args.decay_steps,
        q_init=initial_q_for_norm(norm),
        reputation_init=np.full(args.agents, GOOD, dtype=np.int64),
        legal_actions=np.ones((NUM_OBS, NUM_ACTIONS), dtype=np.bool_),
    )

    warmup = parameters.copy()
    warmup.update(time_steps=16, n_agents=2, batch_size=2, n_snapshots=2)
    warmup["reputation_init"] = parameters["reputation_init"][:2]
    run_simulation(seed=args.seed, **warmup)

    print(
        f"{norm_name}: target {target}; "
        f"{args.steps:,} encounters x {args.reps} repetitions"
    )
    proportions, good_fractions, times = run_repetitions(
        run_simulation,
        lambda result: (
            strategy_proportions(result[0], n_strategies=NUM_STRATEGIES), result[1],
        ),
        n_reps=args.reps,
        base_seed=args.seed,
        parameters=parameters,
        n_channels=2,
    )
    channels = np.concatenate((proportions, good_fractions[:, :, None]), axis=2)
    frame = aggregate_channels(
        channels, times, (*STRATEGY_NAMES, "good_reputation"),
        save_every=args.save_every,
    )
    frame = _append_metadata(frame, args, norm_name, target)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"sim_{norm_name.lower()}.csv"
    frame.to_csv(output_path, index=False)

    final_counts = np.rint(proportions[:, -1] * args.agents).astype(int)
    runs = pd.DataFrame(
        {
            "norm": norm_name,
            "target_strategy": target,
            "seed": np.arange(args.seed, args.seed + args.reps),
            "time": int(times[-1]),
            "n_agents": args.agents,
            **{
                f"count_{name}": final_counts[:, index]
                for index, name in enumerate(STRATEGY_NAMES)
            },
            "good_reputation_fraction": good_fractions[:, -1],
        }
    )
    runs["target_count"] = runs[f"count_{target}"]
    runs["target_fraction"] = runs["target_count"] / args.agents
    runs["all_target"] = runs["target_count"] == args.agents
    runs.to_csv(
        DATA_DIR / f"sim_{norm_name.lower()}_runs.csv", index=False
    )
    print(f"wrote {output_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    for norm_name in dict.fromkeys(args.norms):
        run_norm(norm_name, args)


if __name__ == "__main__":
    main()
