import numpy as np


def _find(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _is_connected(edges, n_agents):
    if n_agents <= 1:
        return True
    parent = list(range(n_agents))
    for row in edges.tolist():
        root = _find(parent, row[0])
        for node in row[1:]:
            other = _find(parent, node)
            if other != root:
                parent[other] = root
    root = _find(parent, 0)
    return all(_find(parent, node) == root for node in range(1, n_agents))


def _conflict_position(groups):
    seen = set()
    for row_index, row in enumerate(np.sort(groups, axis=1).tolist()):
        for column in range(1, len(row)):
            if row[column] == row[column - 1]:
                columns = np.where(groups[row_index] == row[column])[0]
                return row_index, int(columns[0])
        key = tuple(row)
        if key in seen:
            return row_index, 0
        seen.add(key)
    return None


def _repair_groups(groups, rng, max_swaps):
    n_groups, order = groups.shape
    for _ in range(max_swaps):
        conflict = _conflict_position(groups)
        if conflict is None:
            return np.ascontiguousarray(
                np.sort(groups, axis=1), dtype=np.int32
            )
        row, column = conflict
        other_row = int(rng.integers(n_groups - 1))
        if other_row >= row:
            other_row += 1
        other_column = int(rng.integers(order))
        groups[row, column], groups[other_row, other_column] = (
            groups[other_row, other_column],
            groups[row, column],
        )
    return None


def regular_hypergraph(n_agents, order, degree, rng, max_tries=100):
    if (n_agents * degree) % order != 0:
        raise ValueError("n_agents * degree must be divisible by order")

    n_edges = n_agents * degree // order
    stubs = np.repeat(np.arange(n_agents, dtype=np.int32), degree)
    max_swaps = 50 * n_agents * degree
    for _ in range(max_tries):
        rng.shuffle(stubs)
        edges = _repair_groups(
            stubs.reshape(n_edges, order).copy(), rng, max_swaps
        )
        if edges is not None and _is_connected(edges, n_agents):
            return edges
    raise RuntimeError("could not construct a connected regular hypergraph")

