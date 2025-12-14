"""
Matrix Multiplication for Heisenberg Engine Demo.

This program will be evolved to optimize matrix multiplication.
The evaluator has a HIDDEN VARIABLE (cache line efficiency) that affects performance,
but the program doesn't know about it. The Heisenberg Engine should discover
that memory access patterns matter.

CONSTRAINT: You must implement the multiplication yourself.
            Using numpy.dot, numpy.matmul, or @ operator is NOT ALLOWED.
"""


# EVOLVE-BLOCK-START
def matrix_multiply(A: list, B: list) -> list:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix as list of lists (m x n)
        B: Second matrix as list of lists (n x p)

    Returns:
        Result matrix C = A @ B as list of lists (m x p)

    CONSTRAINT: Must implement multiplication manually.
                Cannot use numpy.dot, numpy.matmul, or @ operator.
    """
    # Get dimensions
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B and B[0] else 0

    # Initialize result matrix with zeros
    C = [[0.0 for _ in range(p)] for _ in range(m)]

    # Naive triple-loop multiplication (ijk order)
    # This is cache-unfriendly because B is accessed column-wise
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the matrix multiplication
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    C = matrix_multiply(A, B)
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = A @ B = {C}")
    # Expected: [[58, 64], [139, 154]]
