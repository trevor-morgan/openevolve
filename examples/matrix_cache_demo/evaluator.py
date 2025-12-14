"""
Evaluator with Hidden Variable for Matrix Multiplication Demo.

This evaluator has a HIDDEN VARIABLE: cache line efficiency. Programs that
access memory in cache-friendly patterns (row-major, blocked) perform better,
but they don't know WHY. The Heisenberg Engine should detect that:

1. Optimization plateaus around 0.6-0.7 fitness
2. There's unexplained variance in performance
3. Probe the execution traces to discover "memory access pattern" matters

The key insight: ijk vs ikj loop ordering can cause 5-10x performance difference
due to cache behavior, even though both are O(n³).
"""

import math
import random
import time
from typing import Any

# =============================================================================
# HIDDEN VARIABLES (the program doesn't know about these!)
# =============================================================================

# Simulated cache parameters
CACHE_LINE_SIZE = 64  # bytes
ELEMENTS_PER_CACHE_LINE = 8  # 64 bytes / 8 bytes per float
L1_CACHE_SIZE = 32 * 1024  # 32KB L1 cache
L2_CACHE_SIZE = 256 * 1024  # 256KB L2 cache

# Simulated memory latency (cycles)
L1_HIT_LATENCY = 4
L2_HIT_LATENCY = 12
MEMORY_LATENCY = 100


class CacheSimulator:
    """Simulate a simple cache to track memory access patterns."""

    def __init__(self, cache_size: int, line_size: int):
        self.cache_size = cache_size
        self.line_size = line_size
        self.num_lines = cache_size // line_size
        self.cache_lines = {}  # Maps cache line index to tag
        self.hits = 0
        self.misses = 0
        self.accesses = []  # Track access pattern

    def access(self, address: int) -> bool:
        """Access a memory address. Returns True if cache hit."""
        line_index = (address // self.line_size) % self.num_lines
        tag = address // self.cache_size

        self.accesses.append(address)

        if line_index in self.cache_lines and self.cache_lines[line_index] == tag:
            self.hits += 1
            return True
        else:
            self.cache_lines[line_index] = tag
            self.misses += 1
            return False

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    def get_locality_score(self) -> float:
        """Calculate spatial locality score based on access pattern."""
        if len(self.accesses) < 2:
            return 1.0

        sequential = 0
        for i in range(1, len(self.accesses)):
            # Consider accesses within same cache line as sequential
            if abs(self.accesses[i] - self.accesses[i - 1]) < self.line_size:
                sequential += 1

        return sequential / (len(self.accesses) - 1)


def analyze_access_pattern(accesses: list[tuple[str, int, int]]) -> dict[str, Any]:
    """
    Analyze memory access pattern to determine cache efficiency.

    Args:
        accesses: List of (matrix_name, row, col) tuples

    Returns:
        Dictionary with cache analysis results
    """
    if not accesses:
        return {"cache_efficiency": 0.5, "pattern_type": "unknown"}

    # Simulate L1 cache
    l1_cache = CacheSimulator(L1_CACHE_SIZE, CACHE_LINE_SIZE)

    # Convert accesses to addresses (assuming row-major storage)
    matrix_bases = {"A": 0, "B": 10000000, "C": 20000000}
    max_cols = 1000  # Assume max matrix width

    for matrix, row, col in accesses:
        base = matrix_bases.get(matrix, 0)
        # Row-major address calculation
        address = base + row * max_cols * 8 + col * 8
        l1_cache.access(address)

    # Analyze B matrix access pattern specifically (the hidden factor)
    b_accesses = [(row, col) for matrix, row, col in accesses if matrix == "B"]
    b_pattern = "unknown"
    b_locality = 0.0

    if b_accesses:
        # Check if B is accessed row-wise (good) or column-wise (bad)
        row_changes = 0
        col_changes = 0
        for i in range(1, len(b_accesses)):
            if b_accesses[i][0] != b_accesses[i - 1][0]:
                row_changes += 1
            if b_accesses[i][1] != b_accesses[i - 1][1]:
                col_changes += 1

        if len(b_accesses) > 1:
            # Column-wise access (bad): row changes frequently, col stays same
            # Row-wise access (good): col changes frequently, row stays same
            if row_changes > col_changes * 2:
                b_pattern = "column_wise"  # BAD for cache
                b_locality = 0.2
            elif col_changes > row_changes * 2:
                b_pattern = "row_wise"  # GOOD for cache
                b_locality = 0.9
            else:
                b_pattern = "mixed"
                b_locality = 0.5

    return {
        "l1_hit_rate": l1_cache.get_hit_rate(),
        "l1_hits": l1_cache.hits,
        "l1_misses": l1_cache.misses,
        "spatial_locality": l1_cache.get_locality_score(),
        "b_access_pattern": b_pattern,
        "b_locality": b_locality,
        "total_accesses": len(accesses),
    }


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate(program_path: str) -> dict[str, Any]:
    """
    Evaluate a matrix multiplication program.

    Args:
        program_path: Path to the Python file to evaluate

    The fitness is affected by:
    1. Correctness (does it multiply correctly?)
    2. Time complexity (how fast is it?)
    3. HIDDEN: Cache efficiency (programs don't know this matters!)

    Programs will plateau around 0.6-0.7 fitness because they're optimizing
    loop order but don't know about cache effects on B matrix access.
    """
    artifacts = {}

    # Read the program code from the file
    try:
        with open(program_path) as f:
            code = f.read()
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to read program file: {e}",
        }

    # Check for forbidden operations
    forbidden = ["numpy.dot", "numpy.matmul", "np.dot", "np.matmul", "@"]
    for forbidden_op in forbidden:
        if forbidden_op in code and forbidden_op != "@":  # @ might be in decorators
            # Check if @ is used for matrix multiplication
            pass  # Allow @ in decorators, check context later

    try:
        # Execute the code
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)

        if "matrix_multiply" not in local_vars:
            return {
                "combined_score": 0.0,
                "correctness": 0.0,
                "error": "No matrix_multiply function found",
            }

        multiply_func = local_vars["matrix_multiply"]

        # Test correctness
        correctness_score, correctness_details = _test_correctness(multiply_func)
        artifacts["correctness_tests"] = correctness_details

        if correctness_score < 0.5:
            return {
                "combined_score": correctness_score * 0.3,
                "correctness": correctness_score,
                "error": "Too many correctness failures",
                "artifacts": artifacts,
            }

        # Test performance (time + hidden cache effects)
        perf_score, perf_details, cache_info = _test_performance(multiply_func)
        artifacts["performance_tests"] = perf_details
        artifacts["cache_analysis"] = cache_info  # This is the hidden data!

        # Calculate combined score
        # Cache efficiency heavily impacts performance but isn't explicit
        combined = correctness_score * 0.3 + perf_score * 0.7

        return {
            "combined_score": combined,
            "correctness": correctness_score,
            "performance": perf_score,
            "artifacts": artifacts,
        }

    except Exception as e:
        import traceback

        return {
            "combined_score": 0.0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _test_correctness(multiply_func) -> tuple[float, dict]:
    """Test matrix multiplication correctness."""
    test_cases = [
        # (A, B, expected C)
        ([[1]], [[2]], [[2]]),  # 1x1
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[19, 22], [43, 50]]),  # 2x2
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]),  # 2x3 @ 3x2
        ([[1, 0], [0, 1]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]),  # Identity
        ([[0, 0], [0, 0]], [[1, 2], [3, 4]], [[0, 0], [0, 0]]),  # Zero matrix
    ]

    # Add random test cases
    random.seed(42)
    for size in [5, 10, 20]:
        A = [[random.uniform(-10, 10) for _ in range(size)] for _ in range(size)]
        B = [[random.uniform(-10, 10) for _ in range(size)] for _ in range(size)]
        # Compute expected using naive method
        expected = [
            [sum(A[i][k] * B[k][j] for k in range(size)) for j in range(size)] for i in range(size)
        ]
        test_cases.append((A, B, expected))

    passed = 0
    details = []

    for i, (A, B, expected) in enumerate(test_cases):
        try:
            result = multiply_func(A, B)

            # Check dimensions
            if len(result) != len(expected) or (result and len(result[0]) != len(expected[0])):
                details.append(
                    {
                        "test": i,
                        "passed": False,
                        "error": f"Dimension mismatch: {len(result)}x{len(result[0]) if result else 0} vs {len(expected)}x{len(expected[0]) if expected else 0}",
                    }
                )
                continue

            # Check values with tolerance
            correct = True
            max_error = 0.0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    error = abs(result[r][c] - expected[r][c])
                    max_error = max(max_error, error)
                    if error > 1e-6:
                        correct = False

            if correct:
                passed += 1
                details.append({"test": i, "passed": True, "max_error": max_error})
            else:
                details.append(
                    {
                        "test": i,
                        "passed": False,
                        "max_error": max_error,
                        "error": f"Value mismatch (max error: {max_error})",
                    }
                )

        except Exception as e:
            details.append({"test": i, "passed": False, "error": str(e)})

    return passed / len(test_cases), {"tests": details, "passed": passed, "total": len(test_cases)}


def _test_performance(multiply_func) -> tuple[float, dict, dict]:
    """
    Test matrix multiplication performance.

    The key insight: we measure time, but cache effects are hidden.
    ijk vs ikj loop ordering can cause 5-10x performance difference
    even though both are O(n³).
    """
    test_sizes = [32, 64, 128]
    results = []
    cache_analysis = []

    random.seed(42)

    for size in test_sizes:
        # Generate test matrices
        A = [[random.uniform(-1, 1) for _ in range(size)] for _ in range(size)]
        B = [[random.uniform(-1, 1) for _ in range(size)] for _ in range(size)]

        # Track memory access pattern
        access_pattern = []

        # Wrap matrices to track accesses
        class TrackedMatrix:
            def __init__(self, data: list, name: str):
                self._data = data
                self._name = name

            def __len__(self):
                return len(self._data)

            def __getitem__(self, key):
                if isinstance(key, int):
                    return TrackedRow(self._data[key], self._name, key)
                return self._data[key]

        class TrackedRow:
            def __init__(self, data: list, matrix_name: str, row: int):
                self._data = data
                self._matrix_name = matrix_name
                self._row = row

            def __len__(self):
                return len(self._data)

            def __getitem__(self, key):
                if isinstance(key, int):
                    access_pattern.append((self._matrix_name, self._row, key))
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

        tracked_A = TrackedMatrix(A, "A")
        tracked_B = TrackedMatrix(B, "B")

        # Time the multiplication
        start = time.perf_counter()
        try:
            result = multiply_func(tracked_A, tracked_B)
            elapsed = time.perf_counter() - start

            # Analyze cache behavior (HIDDEN from the program!)
            cache_info = analyze_access_pattern(access_pattern)
            cache_analysis.append(
                {
                    "size": size,
                    **cache_info,
                }
            )

            # Apply cache penalty to elapsed time
            # Bad cache patterns get significant penalty
            cache_efficiency = cache_info.get("b_locality", 0.5)
            cache_penalty = 1.0 + (1.0 - cache_efficiency) * 3.0  # Up to 4x penalty

            adjusted_time = elapsed * cache_penalty

            results.append(
                {
                    "size": size,
                    "raw_time": elapsed,
                    "adjusted_time": adjusted_time,
                    "cache_penalty": cache_penalty,
                    "accesses": len(access_pattern),
                }
            )

        except Exception as e:
            results.append(
                {
                    "size": size,
                    "error": str(e),
                }
            )
            cache_analysis.append(
                {
                    "size": size,
                    "error": str(e),
                }
            )

    # Calculate performance score
    # Based on both time complexity and cache efficiency
    if len(results) >= 2 and all("adjusted_time" in r for r in results):
        # Check O(n³) scaling
        time_ratio = results[-1]["adjusted_time"] / max(results[0]["adjusted_time"], 1e-10)
        size_ratio = test_sizes[-1] / test_sizes[0]
        expected_ratio = size_ratio**3  # O(n³)

        # Score based on how close to expected scaling
        if time_ratio <= expected_ratio * 1.5:
            scaling_score = 1.0
        elif time_ratio <= expected_ratio * 3:
            scaling_score = 0.7
        elif time_ratio <= expected_ratio * 6:
            scaling_score = 0.4
        else:
            scaling_score = 0.2

        # Average cache efficiency from analysis
        avg_cache_efficiency = sum(
            c.get("b_locality", 0.5) for c in cache_analysis if "b_locality" in c
        ) / max(len(cache_analysis), 1)

        # Combine scaling and cache efficiency
        perf_score = scaling_score * 0.4 + avg_cache_efficiency * 0.6

    else:
        perf_score = 0.1

    return perf_score, {"timings": results}, {"cache_stats": cache_analysis}


if __name__ == "__main__":
    # Test with the initial program
    import os
    import tempfile

    test_code = """
def matrix_multiply(A, B):
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B and B[0] else 0
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    # Naive ijk order (cache-unfriendly for B)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        temp_path = f.name

    try:
        result = evaluate(temp_path)
        print("Evaluation result (ijk order - cache unfriendly):")
        for key, value in result.items():
            if key != "artifacts":
                print(f"  {key}: {value}")

        if "artifacts" in result and "cache_analysis" in result["artifacts"]:
            print("\nCache Analysis (HIDDEN INFO):")
            import json

            print(json.dumps(result["artifacts"]["cache_analysis"], indent=2, default=str))
    finally:
        os.unlink(temp_path)

    print("\n" + "=" * 60 + "\n")

    # Test with cache-friendly ikj order
    test_code_ikj = """
def matrix_multiply(A, B):
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B and B[0] else 0
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    # ikj order (cache-friendly for B)
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code_ikj)
        temp_path = f.name

    try:
        result = evaluate(temp_path)
        print("Evaluation result (ikj order - cache friendly):")
        for key, value in result.items():
            if key != "artifacts":
                print(f"  {key}: {value}")

        if "artifacts" in result and "cache_analysis" in result["artifacts"]:
            print("\nCache Analysis (HIDDEN INFO):")
            import json

            print(json.dumps(result["artifacts"]["cache_analysis"], indent=2, default=str))
    finally:
        os.unlink(temp_path)
