import torch


def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    check_inversible = lambda mat: mat.shape[0] == mat.shape[1] and torch.linalg.det(mat) != 0
    if not check_inversible(T_t) or not check_inversible(S_t):
        return -1
    T_inv = torch.linalg.inv(T_t)
    return torch.round(T_inv @ A_t @ S_t, decimals = 3)
