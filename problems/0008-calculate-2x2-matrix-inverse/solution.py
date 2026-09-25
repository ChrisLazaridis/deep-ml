import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    mat = np.asarray(matrix)
    det = mat.flat[0]* mat.flat[3] - mat.flat[1]*mat.flat[2]
    if det == 0:
        return None
    inv = np.zeros(mat.shape)
    factor = 1 / det
    inv.flat[0] = mat.flat[3]
    inv.flat[1] = - mat.flat[1]
    inv.flat[2] = - mat.flat[2]
    inv.flat[3] = mat.flat[0]

    return (inv * factor).tolist()