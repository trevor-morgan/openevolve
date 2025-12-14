"""
Evaluator for Sorting Discovery Example

This evaluator tests sorting implementations for:
1. Correctness - Does it actually sort?
2. Efficiency - How fast is it?
3. Robustness - Does it handle edge cases?
4. Memory - How much extra space does it use?

The Discovery Engine will use these metrics to:
- Evaluate solutions
- Track phenotypes (behavior characteristics)
- Guide problem evolution
"""

import importlib.util
import random
import sys
import time
from typing import Any


def load_program(program_path: str):
    """Load the program module"""
    spec = importlib.util.spec_from_file_location("solution", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)
    return module


def generate_test_cases() -> list[dict[str, Any]]:
    """Generate test cases of varying difficulty"""
    test_cases = [
        # Basic cases
        {"input": [], "expected": [], "name": "empty"},
        {"input": [1], "expected": [1], "name": "single"},
        {"input": [2, 1], "expected": [1, 2], "name": "two_elements"},
        {"input": [3, 2, 1], "expected": [1, 2, 3], "name": "reversed"},
        {"input": [1, 2, 3], "expected": [1, 2, 3], "name": "already_sorted"},
        # Edge cases
        {"input": [1, 1, 1], "expected": [1, 1, 1], "name": "all_same"},
        {"input": [-1, -2, -3], "expected": [-3, -2, -1], "name": "negative"},
        {"input": [0, -1, 1], "expected": [-1, 0, 1], "name": "with_zero"},
        # Larger cases
        {
            "input": list(range(100, 0, -1)),
            "expected": list(range(1, 101)),
            "name": "hundred_reversed",
        },
    ]

    # Random cases
    for i in range(5):
        size = random.randint(10, 50)
        data = [random.randint(-1000, 1000) for _ in range(size)]
        test_cases.append({"input": data, "expected": sorted(data), "name": f"random_{i}"})

    return test_cases


def evaluate_stage1(program_path: str) -> dict[str, float]:
    """
    Stage 1: Quick validation - does it sort at all?

    This is the first gate. Programs that can't handle basic cases
    are rejected early to save computational resources.
    """
    try:
        module = load_program(program_path)

        if not hasattr(module, "solve"):
            return {"correctness": 0.0, "error": 1.0, "stage1_passed": 0.0}

        # Test basic cases only
        basic_cases = [
            ([], []),
            ([1], [1]),
            ([2, 1], [1, 2]),
            ([3, 1, 2], [1, 2, 3]),
        ]

        passed = 0
        for input_data, expected in basic_cases:
            try:
                result = module.solve(input_data.copy())
                if result == expected:
                    passed += 1
            except Exception:
                pass

        correctness = passed / len(basic_cases)

        return {
            "correctness": correctness,
            "stage1_passed": 1.0 if correctness >= 0.5 else 0.0,
        }

    except Exception as e:
        return {
            "correctness": 0.0,
            "error": 1.0,
            "stage1_passed": 0.0,
            "error_message": str(e),
        }


def evaluate_stage2(program_path: str) -> dict[str, float]:
    """
    Stage 2: Comprehensive testing - correctness and edge cases
    """
    try:
        module = load_program(program_path)
        test_cases = generate_test_cases()

        passed = 0
        edge_case_passed = 0
        edge_case_total = 0

        for test in test_cases:
            try:
                result = module.solve(test["input"].copy())

                if result == test["expected"]:
                    passed += 1

                    # Track edge cases separately
                    if test["name"] in ["empty", "single", "all_same", "negative", "with_zero"]:
                        edge_case_passed += 1
                        edge_case_total += 1
                else:
                    if test["name"] in ["empty", "single", "all_same", "negative", "with_zero"]:
                        edge_case_total += 1

            except Exception:
                if test["name"] in ["empty", "single", "all_same", "negative", "with_zero"]:
                    edge_case_total += 1

        correctness = passed / len(test_cases)
        robustness = edge_case_passed / max(edge_case_total, 1)

        return {
            "correctness": correctness,
            "robustness": robustness,
            "stage2_passed": 1.0 if correctness >= 0.7 else 0.0,
        }

    except Exception:
        return {
            "correctness": 0.0,
            "robustness": 0.0,
            "stage2_passed": 0.0,
        }


def evaluate_stage3(program_path: str) -> dict[str, float]:
    """
    Stage 3: Performance evaluation - efficiency metrics
    """
    try:
        module = load_program(program_path)

        # Benchmark on different sizes
        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            data = [random.randint(-10000, 10000) for _ in range(size)]

            start = time.perf_counter()
            for _ in range(3):  # Multiple runs for stability
                module.solve(data.copy())
            elapsed = (time.perf_counter() - start) / 3

            times.append(elapsed)

        # Calculate efficiency score (lower time = higher score)
        # Normalize against expected O(n log n) time
        avg_time = sum(times) / len(times)
        baseline_time = 0.01  # Expected time for good implementation

        if avg_time < baseline_time:
            efficiency = 1.0
        elif avg_time < baseline_time * 10:
            efficiency = 1.0 - (avg_time - baseline_time) / (baseline_time * 9)
        else:
            efficiency = 0.1

        efficiency = max(0.0, min(1.0, efficiency))

        # Memory estimation (crude - based on code analysis)
        with open(program_path) as f:
            code = f.read()

        # Heuristic: more list/dict literals and comprehensions = more memory
        memory_indicators = code.count("[") + code.count("{") + code.count("append")
        memory_score = max(0.0, 1.0 - memory_indicators * 0.05)

        return {
            "efficiency": efficiency,
            "memory_efficiency": memory_score,
            "avg_time_ms": avg_time * 1000,
            "stage3_passed": 1.0,
        }

    except Exception:
        return {
            "efficiency": 0.0,
            "memory_efficiency": 0.0,
            "stage3_passed": 0.0,
        }


def evaluate(program_path: str) -> dict[str, float]:
    """
    Main evaluation function - runs all stages and combines scores

    Returns metrics that the Discovery Engine will use for:
    - Fitness calculation (combined_score)
    - Phenotype extraction (efficiency, robustness, etc.)
    - MAP-Elites grid placement
    """
    # Run all stages
    stage1 = evaluate_stage1(program_path)
    stage2 = evaluate_stage2(program_path)
    stage3 = evaluate_stage3(program_path)

    # Combine results
    metrics = {
        **stage1,
        **stage2,
        **stage3,
    }

    # Calculate combined score
    # Weight: correctness (50%), efficiency (30%), robustness (20%)
    correctness = metrics.get("correctness", 0.0)
    efficiency = metrics.get("efficiency", 0.0)
    robustness = metrics.get("robustness", 0.0)

    combined_score = correctness * 0.5 + efficiency * 0.3 + robustness * 0.2

    metrics["combined_score"] = combined_score

    return metrics


if __name__ == "__main__":
    # Test with initial program
    import sys

    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print("Evaluation Results:")
        for key, value in sorted(result.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Usage: python evaluator.py <program_path>")
