import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	mat = np.asarray(a)
	vec = np.asarray(b)
	if mat.shape[1] == vec.shape[0]:
		return (mat @ vec).tolist()
	else:
		return -1 