"""
Initial Program: Basic Sorting

This is the starting point for the Discovery Mode example.
The Discovery Engine will:
1. Evolve solutions to this problem
2. Test them adversarially
3. Once solutions are found, EVOLVE THE PROBLEM ITSELF
   (e.g., "sort without comparisons", "sort in O(n) space")
"""


def solve(data):
    """
    Sort a list of numbers.

    Args:
        data: List of numbers to sort

    Returns:
        Sorted list in ascending order
    """
    # EVOLVE-BLOCK-START
    # Simple bubble sort - a clear target for optimization
    result = list(data)
    n = len(result)

    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]

    return result
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Quick test
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Input:  {test_data}")
    print(f"Output: {solve(test_data)}")
