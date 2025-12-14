"""
Initial program for Betti number computation.

This program computes topological features of an undirected graph:
- beta_0: number of connected components
- beta_1: number of independent cycles

The LLM will evolve the EVOLVE-BLOCK to improve correctness.
"""


def transform(adjacency_matrix: list[list[int]]) -> list[int]:
    """
    Compute Betti numbers for an undirected graph.

    Args:
        adjacency_matrix: Symmetric matrix where A[i][j] = 1 means edge i-j

    Returns:
        [beta_0, beta_1] - number of components and cycles
    """
    # EVOLVE-BLOCK-START
    n = len(adjacency_matrix)
    if n == 0:
        return [0, 0]

    # Simple placeholder - counts components only
    visited = [False] * n
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for neighbor in range(n):
                if adjacency_matrix[node][neighbor] and not visited[neighbor]:
                    stack.append(neighbor)

    # TODO: compute cycles correctly
    cycles = 0

    return [components, cycles]
    # EVOLVE-BLOCK-END
