import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Calculate coefficients using the normal equation: 
	# (X^T * X)^{-1} * X^T * y
    theta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ y_arr
    
    return [round(float(val), 4) for val in theta.flatten()]