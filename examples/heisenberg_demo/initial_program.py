"""
Initial sorting program for Heisenberg Engine demo.

This program will be evolved to optimize cache efficiency.
The evaluator has a HIDDEN VARIABLE (cache line size) that affects performance,
but the program doesn't know about it. The Heisenberg Engine should discover
that memory access patterns matter.
"""


# EVOLVE-BLOCK-START
def sort_data(data: list) -> list:
    """
    Sort the input data.

    Args:
        data: List of integers to sort

    Returns:
        Sorted list
    """
    # Simple bubble sort - inefficient but works
    result = data.copy()
    n = len(result)

    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]

    return result
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the sorting function
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_data}")
    print(f"Sorted: {sort_data(test_data)}")
