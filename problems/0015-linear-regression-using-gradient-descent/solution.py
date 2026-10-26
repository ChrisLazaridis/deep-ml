import torch
 
def mse(pred, y) -> torch.tensor :
    return torch.sum((pred-y)**2)

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32) #[m,n]
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1) # [m,1]
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True) # [n,1]
    for i in range(iterations):
        pred = X_t @ theta #[m,1]
        loss = torch.div(mse(pred, y_t), 2*m)
        loss.backward()
        with torch.no_grad():
            theta -= alpha*theta.grad
            theta.grad.zero_()
    return theta.detach().squeeze()
    