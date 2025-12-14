"""
Evaluator for ARC-style topology problems using brainstorm oracles.

This evaluator tests programs that implement graph topology algorithms:
- Betti number computation (connected components, cycles)
- Directed clique counting
- Integration score (min-cut)

The program must implement a `transform(input_grid)` function that returns
the expected topology metrics.
"""

import importlib.util
import random
import traceback
from typing import Any

from openevolve.evaluation_result import EvaluationResult

# =============================================================================
# Oracle implementations (from brainstorm)
# =============================================================================


def betti_numbers(adjacency_matrix: list[list[int]]) -> list[int]:
    """
    Compute Betti numbers for an undirected graph.

    beta_0 = number of connected components
    beta_1 = number of independent cycles (cyclomatic complexity)

    For a graph: beta_1 = E - V + beta_0 (Euler characteristic)
    """
    n = len(adjacency_matrix)
    if n == 0:
        return [0, 0]

    # Count edges (upper triangle only for undirected)
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                edges += 1

    # Find connected components using Union-Find
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                union(i, j)

    # Count components
    components = len(set(find(i) for i in range(n)))

    # Euler characteristic: beta_1 = E - V + beta_0
    cycles = edges - n + components

    return [components, cycles]


def directed_clique_count(adjacency_matrix: list[list[int]], max_k: int = 3) -> list[int]:
    """
    Count directed k-simplices (directed cliques) in a directed graph.
    """
    n = len(adjacency_matrix)

    # 0-simplices = nodes
    counts = [n]

    if max_k < 1:
        return counts

    # 1-simplices = directed edges
    edges = sum(adjacency_matrix[i][j] for i in range(n) for j in range(n))
    counts.append(edges)

    if max_k < 2:
        return counts

    # 2-simplices = directed triangles (i -> j, i -> k, j -> k)
    triangles = 0
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j]:
                for k in range(n):
                    if adjacency_matrix[i][k] and adjacency_matrix[j][k]:
                        triangles += 1
    counts.append(triangles)

    if max_k < 3:
        return counts

    # 3-simplices = directed tetrahedra
    tetrahedra = 0
    for i in range(n):
        for j in range(n):
            if not adjacency_matrix[i][j]:
                continue
            for k in range(n):
                if not (adjacency_matrix[i][k] and adjacency_matrix[j][k]):
                    continue
                for l in range(n):
                    if adjacency_matrix[i][l] and adjacency_matrix[j][l] and adjacency_matrix[k][l]:
                        tetrahedra += 1
    counts.append(tetrahedra)

    return counts


def integration_score(adjacency_matrix: list[list[int]]) -> list[int]:
    """
    Compute integration score (simplified Phi) = min-cut size.
    """
    n = len(adjacency_matrix)
    if n <= 1:
        return [0]

    # Check if connected first
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1

    while stack:
        node = stack.pop()
        for neighbor in range(n):
            if adjacency_matrix[node][neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                count += 1

    if count < n:
        return [0]  # Disconnected

    # For connected graphs, find min-cut by trying all partitions
    min_cut = float("inf")

    for mask in range(1, (1 << n) - 1):
        cut_size = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i][j]:
                    i_in_A = (mask >> i) & 1
                    j_in_A = (mask >> j) & 1
                    if i_in_A != j_in_A:
                        cut_size += 1
        min_cut = min(min_cut, cut_size)

    return [min_cut] if min_cut != float("inf") else [0]


# =============================================================================
# Test case generators
# =============================================================================


def generate_random_graph(
    n: int, edge_prob: float = 0.3, directed: bool = False
) -> list[list[int]]:
    """Generate a random adjacency matrix."""
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_prob:
                matrix[i][j] = 1
                if not directed or random.random() < 0.5:
                    matrix[j][i] = 1
    return matrix


def generate_test_cases(problem_type: str, count: int = 5) -> list[dict]:
    """Generate test cases for a given problem type."""
    cases = []

    for _ in range(count):
        n = random.randint(4, 8)

        if problem_type == "betti":
            matrix = generate_random_graph(n, edge_prob=0.4, directed=False)
            expected = betti_numbers(matrix)
            cases.append({"input": matrix, "expected": expected})

        elif problem_type == "clique":
            matrix = generate_random_graph(n, edge_prob=0.3, directed=True)
            expected = directed_clique_count(matrix)
            cases.append({"input": matrix, "expected": expected})

        elif problem_type == "integration":
            matrix = generate_random_graph(n, edge_prob=0.5, directed=False)
            expected = integration_score(matrix)
            cases.append({"input": matrix, "expected": expected})

    return cases


# =============================================================================
# Problem configuration
# =============================================================================

PROBLEM_TYPE = "betti"  # Change to "clique" or "integration" for other problems


# =============================================================================
# Evaluator functions
# =============================================================================


def evaluate(program_path: str) -> EvaluationResult:
    """
    Full evaluation of the program on multiple test cases.
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check for transform function
        if not hasattr(program, "transform"):
            return EvaluationResult(
                metrics={
                    "correctness": 0.0,
                    "combined_score": 0.0,
                    "error": "Missing transform function",
                },
                artifacts={"error_type": "MissingFunction"},
            )

        # Generate test cases
        test_cases = generate_test_cases(PROBLEM_TYPE, count=10)

        correct = 0
        total = len(test_cases)
        errors = []

        for i, case in enumerate(test_cases):
            try:
                result = program.transform(case["input"])

                # Compare results (allow partial matches for clique counting)
                expected = case["expected"]
                if isinstance(expected, list) and isinstance(result, list):
                    min_len = min(len(expected), len(result))
                    if expected[:min_len] == result[:min_len]:
                        correct += 1
                    else:
                        errors.append(f"Case {i}: expected {expected}, got {result}")
                elif result == expected:
                    correct += 1
                else:
                    errors.append(f"Case {i}: expected {expected}, got {result}")

            except Exception as e:
                errors.append(f"Case {i}: {type(e).__name__}: {e!s}")

        correctness = correct / total

        # Bonus for perfect score
        combined_score = correctness
        if correctness == 1.0:
            combined_score = 1.2  # 20% bonus for perfect

        return EvaluationResult(
            metrics={
                "correctness": correctness,
                "tests_passed": correct,
                "total_tests": total,
                "combined_score": combined_score,
            },
            artifacts={
                "errors": errors[:5],  # Limit error details
                "problem_type": PROBLEM_TYPE,
            },
        )

    except Exception as e:
        return EvaluationResult(
            metrics={
                "correctness": 0.0,
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts={
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Quick validation with fewer test cases."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "transform"):
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                artifacts={"error": "Missing transform function"},
            )

        # Quick test with 2 cases
        test_cases = generate_test_cases(PROBLEM_TYPE, count=2)

        for case in test_cases:
            try:
                result = program.transform(case["input"])
                if not isinstance(result, list):
                    return EvaluationResult(
                        metrics={"runs_successfully": 0.5, "combined_score": 0.1},
                        artifacts={"error": f"Expected list, got {type(result)}"},
                    )
            except Exception as e:
                return EvaluationResult(
                    metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                    artifacts={"error": str(e)},
                )

        return EvaluationResult(
            metrics={"runs_successfully": 1.0, "combined_score": 0.5},
            artifacts={"stage": "1", "status": "passed"},
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0}, artifacts={"error": str(e)}
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation."""
    return evaluate(program_path)
