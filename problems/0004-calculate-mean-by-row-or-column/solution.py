import torch


def mean_rows(mat) -> torch.Tensor:
    rows = mat.shape[0]
    cols = mat.shape[1]
    res = torch.empty(rows, dtype=torch.float)
    for i in range (rows):
        tmp = mat[i, :]
        res[i] = torch.mean(tmp)
    return res

def mean_cols(mat) -> torch.Tensor:
    rows = mat.shape[0]
    cols = mat.shape[1]
    res = torch.empty(cols, dtype=torch.float)
    for i in range (cols):
        tmp = mat[:, i]
        res[i] = torch.mean(tmp)
    return res


def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == 'row':
        return mean_rows(a_t)
    elif mode == 'column':
        return mean_cols(a_t)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'row' or 'column'.")
