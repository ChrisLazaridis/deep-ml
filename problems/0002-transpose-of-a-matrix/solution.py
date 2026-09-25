def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    rows,cols = len(a), len(a[0])
    res = []
    for i in range(cols):
        row = []
        for j in range(rows):
            row.append(a[j][i])
        res.append(row)
    return res