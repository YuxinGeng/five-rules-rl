import numpy as np
import pandas as pd


def run_repetitions(run, transform, n_reps, base_seed, parameters, n_channels=1):
    channels = [[] for _ in range(n_channels)]
    times = None
    for repetition in range(n_reps):
        result = run(seed=base_seed + repetition, **parameters)
        if times is None:
            times = result[-1]
        values = transform(result)
        for channel, value in zip(channels, values):
            channel.append(value)
        print(f"\r  rep {repetition + 1}/{n_reps} done", end="", flush=True)
    print()
    return (*[np.asarray(channel) for channel in channels], times)


def strategy_proportions(strategies, n_strategies=4):
    proportions = np.zeros((strategies.shape[0], n_strategies), dtype=np.float64)
    for strategy in range(n_strategies):
        proportions[:, strategy] = np.mean(strategies == strategy, axis=1)
    return proportions


def snapshot_indices(times, save_every):
    targets = np.arange(times[0], times[-1] + 1, save_every)
    keep = np.searchsorted(times, targets, side="right") - 1
    keep = np.clip(keep, 0, len(times) - 1)
    return np.unique(np.append(keep, len(times) - 1))


def aggregate_channels(channels, times, names, /, save_every=1000, **identifiers):
    keep = snapshot_indices(times, save_every)
    mean = channels[:, keep].mean(axis=0)
    std = channels[:, keep].std(axis=0)
    data = {key: np.full(len(keep), value) for key, value in identifiers.items()}
    data["time"] = times[keep]
    for index, name in enumerate(names):
        data[f"mean_{name}"] = mean[:, index]
        data[f"std_{name}"] = std[:, index]
    data["n_reps"] = np.full(len(keep), channels.shape[0], dtype=int)
    return pd.DataFrame(data)
