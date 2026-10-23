from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.simulation import aggregate_channels, run_repetitions, strategy_proportions
from fig3.sim_fig3d_NR import STATE_1, STRATEGY_NAMES, run_simulation


DATA_DIR = Path(__file__).resolve().parent / "data"
DEGREES = (2, 4, 8)

PARAMETERS = dict(
    n_agents=105,
    order=3,
    update_interval=16,
    time_steps=300_000,
    r_values=(2.8, 1.2),
    cost=1.0,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Figure 3d network model at fixed group size and varying degree."
    )
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    parser.add_argument("--steps", type=int, default=PARAMETERS["time_steps"])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if len(set(args.degrees)) != len(args.degrees):
        parser.error("degrees must be distinct")
    if args.reps < 1 or args.save_every < 1 or args.steps < PARAMETERS["update_interval"]:
        parser.error("reps and save-every must be positive; steps must cover an update")
    return args


def main(argv=None):
    args = parse_args(argv)
    parameters = {**PARAMETERS, "time_steps": args.steps}
    print("JIT warmup...", flush=True)
    run_simulation(
        **{**parameters, "n_agents": 6, "time_steps": 200, "update_interval": 10},
        degree=2,
        seed=9999,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "hypergraph_2pgg_degree_sweep.csv"
    frames = []
    for degree in args.degrees:
        print(
            f"\nSweep: k={parameters['order']}, d={degree}, "
            f"update interval={parameters['update_interval']}, "
            f"batch size={degree * parameters['update_interval']}",
            flush=True,
        )
        proportions, update_times = run_repetitions(
            run_simulation,
            lambda result: (strategy_proportions(result[0]),),
            n_reps=args.reps,
            base_seed=args.seed,
            parameters=dict(degree=degree, **parameters),
        )
        frames.append(
            aggregate_channels(
                proportions,
                update_times,
                STRATEGY_NAMES,
                save_every=args.save_every,
                n_agents=parameters["n_agents"],
                k=parameters["order"],
                d=degree,
                n_edges=parameters["n_agents"] * degree // parameters["order"],
                B=parameters["update_interval"],
                batch_size=degree * parameters["update_interval"],
                time_steps=args.steps,
                record_every=args.save_every,
                base_seed=args.seed,
                hypergraph="regular",
                alpha=parameters["alpha"],
                gamma=parameters["gamma"],
                epsilon_start=parameters["epsilon_start"],
                epsilon_end=parameters["epsilon_end"],
                decay_steps=parameters["decay_steps"],
                initial_state=parameters["initial_state"],
                q_init_AC=parameters["q_init"][0, 0],
                q_init_AD=parameters["q_init"][0, 1],
                q_init_BC=parameters["q_init"][1, 0],
                q_init_BD=parameters["q_init"][1, 1],
                c=parameters["cost"],
                q_c=parameters["q_c"],
                q_r=parameters["q_r"],
                r1=parameters["r_values"][0],
                r2=parameters["r_values"][1],
            )
        )

        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
