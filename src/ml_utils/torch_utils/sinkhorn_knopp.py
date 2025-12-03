import torch as th
import jaxtyping as jt


def sinkhorn_knopp(
    matrix: jt.Float[th.Tensor, " #batch_size seq_len seq_len #features"],
    dim1: int = -2,
    dim2: int = -1,
    num_iters: int = 3,
):
    """Applies the Sinkhorn-Knopp algorithm to a given non-negative matrix to
    convert it into a doubly stochastic matrix.

    Args:
        matrix: A non-negative matrix of shape (..., seq_len, seq_len).
        dim1: The dimension corresponding to the rows of the matrix.
        dim2: The dimension corresponding to the columns of the matrix.
        num_iters: The number of iterations to perform.

    Returns:
        A doubly stochastic matrix of the same shape as the input matrix.
    """
    # Ensure the input matrix is non-negative
    assert th.all(matrix >= 0), "Input matrix must be non-negative."

    # Initialize the output matrix
    output = matrix

    for _ in range(num_iters):
        # Normalize rows
        row_sums = output.sum(dim=dim2, keepdim=True)
        output = output / (row_sums + 1e-8)

        # Normalize columns
        col_sums = output.sum(dim=dim1, keepdim=True)
        output = output / (col_sums + 1e-8)

    return output
