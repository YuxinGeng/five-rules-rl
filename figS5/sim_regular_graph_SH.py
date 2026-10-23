from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.learning import (
    argmax_random_tie,
    create_replay_buffer,
    epsilon_greedy,
    flush_agent_buffer,
    store_transition,
)
from common.simulation import aggregate_channels, run_repetitions
from common.topology import regular_hypergraph

DATA_DIR = Path(__file__).resolve().parent / "data"
PAYOFF = np.array([[8.0, 0.0], [2.0, 1.0]])
DEGREES = (2, 4, 8)
DEFAULT_STEPS = 10_000
C = 0
OBS = 0

@njit(cache=True)
def greedy_c_fraction(q_tables):
    count = 0
    for agent in range(len(q_tables)):
        count += argmax_random_tie(q_tables[agent, OBS]) == C
    return count / len(q_tables)


@njit(cache=True)
def _run_simulation_core(
    time_steps, update_interval, payoff, alpha, gamma, epsilon,
    q_tables, buffer, edges, snapshot_times, seed,
):
    np.random.seed(seed)
    actions = np.empty(len(q_tables), dtype=np.int64)
    greedy_c = np.empty(len(snapshot_times), dtype=np.float64)
    greedy_c[0] = greedy_c_fraction(q_tables)
    snapshot = 1
    for time in range(1, time_steps + 1):
        for agent in range(len(q_tables)):
            actions[agent] = epsilon_greedy(q_tables[agent], OBS, epsilon)
        for u, v in edges:
            a_u, a_v = actions[u], actions[v]
            store_transition(buffer, u, OBS, a_u, payoff[a_u, a_v], OBS)
            store_transition(buffer, v, OBS, a_v, payoff[a_v, a_u], OBS)
        if time % update_interval == 0:
            for agent in range(len(q_tables)):
                flush_agent_buffer(q_tables, agent, buffer, alpha, gamma)

            fraction = greedy_c_fraction(q_tables)
            if snapshot < len(snapshot_times) and time == snapshot_times[snapshot]:
                greedy_c[snapshot] = fraction
                snapshot += 1
    return greedy_c


def snapshot_times(time_steps, update_interval, save_every):
    final_update = time_steps // update_interval * update_interval
    targets = np.arange(0, final_update + 1, save_every, dtype=np.int64)
    return np.unique(np.append(targets // update_interval * update_interval, final_update))


def run_simulation(
    n_agents=105, degree=4, time_steps=DEFAULT_STEPS, update_interval=4,
    alpha=0.1, gamma=0.9, epsilon=0.1,
    save_every=200, seed=0,
    *, q_init, payoff_matrix,
):
    payoff = np.asarray(payoff_matrix, dtype=np.float64)
    initial = np.asarray(q_init, dtype=np.float64)
    q_tables = np.broadcast_to(initial, (n_agents, 1, 2)).copy()
    edges = regular_hypergraph(n_agents, 2, degree, np.random.default_rng(seed))
    times = snapshot_times(time_steps, update_interval, save_every)
    greedy_c = _run_simulation_core(
        time_steps, update_interval, payoff, alpha, gamma, epsilon,
        q_tables, create_replay_buffer(n_agents, degree * update_interval),
        edges, times, seed,
    )
    return greedy_c, times


def _run_seed(seed, parameters):
    return run_simulation(seed=seed, **parameters)


def run_ensemble(parameters, n_reps=100, base_seed=0, workers=4):
    if workers == 1:
        return run_repetitions(run_simulation, lambda result: (result[0],),
                               n_reps, base_seed, parameters)
    trajectories = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(partial(_run_seed, parameters=parameters),
                           range(base_seed, base_seed + n_reps))
        for repetition, (values, times) in enumerate(results, 1):
            trajectories.append(values)
            print(f"rep {repetition}/{n_reps}: final greedy C={values[-1]:.3f}", flush=True)
    return np.asarray(trajectories), times


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if len(set(args.degrees)) != len(args.degrees):
        parser.error("degrees must be distinct")
    return args


def main(argv=None):
    args = parse_args(argv)
    parameters = dict(
        n_agents=args.agents, time_steps=args.steps, update_interval=args.update_interval,
        alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, save_every=args.save_every,
        payoff_matrix=PAYOFF, q_init=np.array([[0.0, 10.0]]),
    )

    for degree in args.degrees:
        run_simulation(**{**parameters, "time_steps": args.update_interval},
                       degree=degree, seed=args.seed)
    frames = []
    for degree in args.degrees:
        print(f"\nd={degree}, update interval={args.update_interval}, "
              f"batch size={degree * args.update_interval}, steps={args.steps}, reps={args.reps}")
        series, times = run_ensemble(dict(degree=degree, **parameters),
                                    args.reps, args.seed, args.workers)
        frames.append(aggregate_channels(
            series[:, :, None], times, ("greedy_c",), save_every=args.save_every,
            n_agents=args.agents, degree=degree, n_edges=args.agents * degree // 2,
            update_interval=args.update_interval, batch_size=degree * args.update_interval,
            time_steps=args.steps, record_every=args.save_every,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            R=PAYOFF[0, 0], S=PAYOFF[0, 1], T=PAYOFF[1, 0], P=PAYOFF[1, 1],
            q_init_C=0.0, q_init_D=10.0,
            base_seed=args.seed, graph="random_regular", statistic="greedy_C_agents",
        ))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "sim-regular-graph-SH.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
