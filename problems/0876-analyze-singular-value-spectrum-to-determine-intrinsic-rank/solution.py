import numpy as np
from scipy.linalg import svd


def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	if np.all(delta_W == 0):
		return 0
	_, s, _ = svd(delta_W, full_matrices = False)
	s = s**2
	total_energy: float = np.sum(s)
	energies = np.cumsum(s) / total_energy
	found = False
	k = 0
	while not found or not k <= energies.shape[0]:
		if energies[k] >= energy_threshold:
			found = True
		k += 1
	return k
		
		

	