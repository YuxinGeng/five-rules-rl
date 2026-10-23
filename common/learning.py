from collections import namedtuple

import numpy as np
from numba import njit

ReplayBuffer = namedtuple("ReplayBuffer", ("obs", "a", "r", "next_obs", "count"))


@njit(cache=True)
def epsilon_exp_decay(t, epsilon_start, epsilon_end, decay_steps):
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(
        -t / decay_steps
    )


@njit(cache=True)
def argmax_random_tie(q_values, mask=None):
    if mask is None:
        best, best_q, n_best, start = 0, q_values[0], 1, 1
    else:
        best, best_q, n_best, start = -1, 0.0, 0, 0
    for action in range(start, q_values.shape[0]):
        if mask is not None and not mask[action]:
            continue
        if n_best == 0 or q_values[action] > best_q:
            best = action
            best_q = q_values[action]
            n_best = 1
        elif q_values[action] == best_q:
            n_best += 1
            if np.random.randint(n_best) == 0:
                best = action
    return best


@njit(cache=True)
def epsilon_greedy(q_table, obs, epsilon, mask=None):
    if np.random.random() < epsilon:
        if mask is None:
            return np.random.randint(q_table.shape[1])
        n_legal = 0
        for action in range(q_table.shape[1]):
            if mask[obs, action]:
                n_legal += 1
        selected = np.random.randint(n_legal)
        for action in range(q_table.shape[1]):
            if mask[obs, action]:
                if selected == 0:
                    return action
                selected -= 1
    row_mask = None if mask is None else mask[obs]
    return argmax_random_tie(q_table[obs], row_mask)


@njit(cache=True)
def max_q_value(q_values, mask=None):
    if mask is None:
        best_q, found, start = q_values[0], True, 1
    else:
        best_q, found, start = 0.0, False, 0
    for action in range(start, q_values.shape[0]):
        if mask is not None and not mask[action]:
            continue
        if not found or q_values[action] > best_q:
            best_q = q_values[action]
            found = True
    return best_q


@njit(cache=True)
def binary_strategy_index(q_table, mask=None):
    index = 0
    for obs in range(q_table.shape[0]):
        row_mask = None if mask is None else mask[obs]
        index = index * 2 + argmax_random_tie(q_table[obs], row_mask)
    return index


@njit(cache=True)
def store_transition(buffer, agent, obs, action, reward, next_obs):
    index = buffer.count[agent]
    buffer.obs[agent, index] = obs
    buffer.a[agent, index] = action
    buffer.r[agent, index] = reward
    buffer.next_obs[agent, index] = next_obs
    buffer.count[agent] = index + 1


@njit(cache=True)
def flush_agent_buffer(q_tables, agent, buffer, alpha, gamma, mask=None):
    count = buffer.count[agent]
    if count == 0:
        return

    n_obs, n_actions = q_tables.shape[1:]
    td_sums = np.zeros((n_obs, n_actions), dtype=np.float64)
    td_counts = np.zeros((n_obs, n_actions), dtype=np.int64)
    q_player = q_tables[agent]

    for index in range(count):
        obs = buffer.obs[agent, index]
        action = buffer.a[agent, index]
        next_obs = buffer.next_obs[agent, index]
        next_mask = None if mask is None else mask[next_obs]
        td_error = (
            buffer.r[agent, index]
            + gamma * max_q_value(q_player[next_obs], next_mask)
            - q_player[obs, action]
        )
        td_sums[obs, action] += td_error
        td_counts[obs, action] += 1

    for obs in range(n_obs):
        for action in range(n_actions):
            if td_counts[obs, action] > 0:
                q_player[obs, action] += (
                    alpha
                    * td_sums[obs, action]
                    / td_counts[obs, action]
                )
    buffer.count[agent] = 0


def create_replay_buffer(n_agents, capacity):
    return ReplayBuffer(
        obs=np.empty((n_agents, capacity), dtype=np.int64),
        a=np.empty((n_agents, capacity), dtype=np.int64),
        r=np.empty((n_agents, capacity), dtype=np.float64),
        next_obs=np.empty((n_agents, capacity), dtype=np.int64),
        count=np.zeros(n_agents, dtype=np.int64),
    )
