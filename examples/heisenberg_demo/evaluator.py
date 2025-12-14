"""
Evaluator with Hidden Variable for Heisenberg Engine Demo.

This evaluator has a HIDDEN VARIABLE: cache_line_size. Programs that happen
to access memory in cache-friendly patterns will perform better, but they
don't know WHY. The Heisenberg Engine should detect that:

1. Optimization is plateauing (programs can't get past ~0.7 fitness)
2. There's unexplained variance in performance
3. Probe the execution traces to discover "memory access pattern" matters

This demonstrates the core insight: sometimes you can't optimize further
because you're missing a variable in your model of the problem.
"""

import random
import time
from typing import Any

# =============================================================================
# HIDDEN VARIABLES (the program doesn't know about these!)
# =============================================================================

# Simulated cache line size - affects performance based on access patterns
CACHE_LINE_SIZE = 64  # bytes, typical CPU cache line

# Simulated memory latency (nanoseconds)
CACHE_HIT_LATENCY = 1
CACHE_MISS_LATENCY = 100


def simulate_memory_access(access_pattern: list[int], data_size: int) -> dict[str, Any]:
    """
    Simulate memory access with cache effects.

    Programs that access memory sequentially will have better cache behavior.
    Programs with random access patterns will suffer cache misses.

    This is the HIDDEN VARIABLE that affects performance!
    """
    # Calculate cache line utilization
    # Sequential accesses within a cache line are "free"
    elements_per_cache_line = CACHE_LINE_SIZE // 8  # Assuming 8-byte integers

    cache_hits = 0
    cache_misses = 0
    last_cache_line = -1

    for addr in access_pattern:
        current_cache_line = addr // elements_per_cache_line

        if current_cache_line == last_cache_line:
            cache_hits += 1
        else:
            cache_misses += 1
            last_cache_line = current_cache_line

    total_accesses = len(access_pattern)
    hit_rate = cache_hits / max(total_accesses, 1)

    # Simulated time based on cache behavior
    simulated_time = (
        cache_hits * CACHE_HIT_LATENCY + cache_misses * CACHE_MISS_LATENCY
    ) / 1000  # Convert to microseconds

    return {
        "cache_hit_rate": hit_rate,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "simulated_time_us": simulated_time,
        "access_locality": _calculate_locality(access_pattern),
    }


def _calculate_locality(access_pattern: list[int]) -> float:
    """Calculate a locality score (0-1) based on access pattern."""
    if len(access_pattern) < 2:
        return 1.0

    sequential_count = 0
    for i in range(1, len(access_pattern)):
        if abs(access_pattern[i] - access_pattern[i - 1]) <= 1:
            sequential_count += 1

    return sequential_count / (len(access_pattern) - 1)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate(program_path: str) -> dict[str, Any]:
    """
    Evaluate a sorting program.

    Args:
        program_path: Path to the Python file to evaluate

    The fitness is affected by:
    1. Correctness (does it sort correctly?)
    2. Time complexity (how fast is it?)
    3. HIDDEN: Cache efficiency (programs don't know this matters!)

    Programs will plateau around 0.6-0.7 fitness because they're optimizing
    time complexity but don't know about cache effects.
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

    try:
        # Execute the code
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)

        if "sort_data" not in local_vars:
            return {
                "combined_score": 0.0,
                "correctness": 0.0,
                "error": "No sort_data function found",
            }

        sort_func = local_vars["sort_data"]

        # Test correctness on multiple inputs
        correctness_score, correctness_details = _test_correctness(sort_func)
        artifacts["correctness_tests"] = correctness_details

        if correctness_score < 0.5:
            return {
                "combined_score": correctness_score * 0.3,
                "correctness": correctness_score,
                "error": "Too many correctness failures",
                "artifacts": artifacts,
            }

        # Test performance (time + hidden cache effects)
        perf_score, perf_details, cache_info = _test_performance(sort_func)
        artifacts["performance_tests"] = perf_details
        artifacts["cache_analysis"] = cache_info  # This is the hidden data!

        # Calculate combined score
        # Note: cache effects are baked into perf_score but not explicitly
        combined = correctness_score * 0.4 + perf_score * 0.6

        return {
            "combined_score": combined,
            "correctness": correctness_score,
            "performance": perf_score,
            # Hidden info in artifacts - probes can discover this
            "artifacts": artifacts,
        }

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": str(e),
        }


def _test_correctness(sort_func) -> tuple[float, dict]:
    """Test sorting correctness."""
    test_cases = [
        [],
        [1],
        [2, 1],
        [3, 1, 2],
        [5, 4, 3, 2, 1],
        list(range(100)),
        list(range(100, 0, -1)),
        [random.randint(0, 1000) for _ in range(100)],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1, 2],
    ]

    passed = 0
    details = []

    for i, test in enumerate(test_cases):
        try:
            result = sort_func(test.copy())
            expected = sorted(test)

            if result == expected:
                passed += 1
                details.append({"test": i, "passed": True})
            else:
                details.append(
                    {
                        "test": i,
                        "passed": False,
                        "expected_len": len(expected),
                        "got_len": len(result),
                    }
                )
        except Exception as e:
            details.append({"test": i, "passed": False, "error": str(e)})

    return passed / len(test_cases), {"tests": details, "passed": passed, "total": len(test_cases)}


def _test_performance(sort_func) -> tuple[float, dict, dict]:
    """
    Test sorting performance.

    The key insight: we measure time, but cache effects are hidden.
    Two algorithms with the same O(n log n) complexity can have
    very different real-world performance due to cache behavior.
    """
    test_sizes = [100, 500, 1000]
    results = []
    cache_analysis = []

    for size in test_sizes:
        # Generate test data
        test_data = [random.randint(0, 10000) for _ in range(size)]

        # Track memory access pattern during sort
        access_pattern = []

        # Wrap the data to track accesses
        class TrackedList(list):
            def __getitem__(self, key):
                if isinstance(key, int):
                    access_pattern.append(key)
                return super().__getitem__(key)

            def __setitem__(self, key, value):
                if isinstance(key, int):
                    access_pattern.append(key)
                return super().__setitem__(key, value)

        tracked_data = TrackedList(test_data)

        # Time the sort
        start = time.perf_counter()
        try:
            result = sort_func(tracked_data)
            elapsed = time.perf_counter() - start

            # Analyze cache behavior (HIDDEN from the program!)
            cache_info = simulate_memory_access(access_pattern, size)
            cache_analysis.append(
                {
                    "size": size,
                    **cache_info,
                }
            )

            # Adjust elapsed time based on cache behavior
            # This simulates real hardware behavior where cache misses are expensive
            cache_penalty = 1.0 + (1.0 - cache_info["cache_hit_rate"]) * 2.0
            adjusted_time = elapsed * cache_penalty

            results.append(
                {
                    "size": size,
                    "raw_time": elapsed,
                    "adjusted_time": adjusted_time,
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

    # Calculate performance score
    # Penalize based on time complexity growth
    if len(results) >= 2 and "adjusted_time" in results[0] and "adjusted_time" in results[-1]:
        time_ratio = results[-1]["adjusted_time"] / max(results[0]["adjusted_time"], 0.0001)
        size_ratio = test_sizes[-1] / test_sizes[0]

        # O(n log n) would have time_ratio ~ size_ratio * log(size_ratio)
        expected_ratio = size_ratio * (1 + 0.5 * (size_ratio**0.5))

        if time_ratio <= expected_ratio:
            perf_score = 1.0
        elif time_ratio <= expected_ratio * 2:
            perf_score = 0.7
        elif time_ratio <= expected_ratio * 5:
            perf_score = 0.4
        else:
            perf_score = 0.2
    else:
        perf_score = 0.1

    return perf_score, {"timings": results}, {"cache_stats": cache_analysis}


if __name__ == "__main__":
    # Test with the initial program
    test_code = """
def sort_data(data):
    result = data.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
"""

    result = evaluate(test_code)
    print("Evaluation result:")
    for key, value in result.items():
        if key != "artifacts":
            print(f"  {key}: {value}")

    if "artifacts" in result:
        print("\nArtifacts (contains hidden cache info!):")
        import json

        print(json.dumps(result["artifacts"], indent=2, default=str))
