import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    a = np.asarray(a)
    rows, cols = a.shape
    new_rows, new_cols = new_shape
    total = rows * cols
    
    if total != (new_rows * new_cols):
        return []
        
    res = np.zeros(new_shape)
    # Construct the original indexing grid.
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]

    # Convert original (row, col) into flat indices.
    k = i * cols + j

    # Convert flat indices into new (row, col).
    nr = k // new_cols
    nc = k % new_cols

    # Transfer the elements.
    res[nr, nc] = a[i, j]

    return res.tolist()