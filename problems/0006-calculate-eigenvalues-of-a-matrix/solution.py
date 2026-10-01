import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    trace = torch.trace(matrix)
    det = torch.linalg.det(matrix)
    lambdas = torch.zeros(matrix.shape[1])
    discr = trace**2 - 4*det
    if discr <=0:
        return "eigenvaues do not exist"
    #lambdas[1] = -trace + torch.sqrt(discr)/ 2
    #lambdas[2] = -trace - torch.sqrt(discr) /2
    lambdas[0]= (trace + torch.sqrt(discr))/ 2
    lambdas[1] = (trace - torch.sqrt(discr)) /2
    return lambdas
