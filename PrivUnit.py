import math
import numpy as np
import random
import matplotlib.pyplot as plt


def privUnit(X, eps):

	(d, n) = X.shape
	B = (math.exp(eps)+1)/(math.exp(eps)-1)*np.sqrt(math.pi*d/2)
	pi_a = math.exp(eps)/(1+math.exp(eps))
	X_perturb = X.copy()
	
	for i in range(n):
		# only handle when X[:, i] is a unit vector
		v = np.random.normal(0, 1, size = d)
		v = v/np.linalg.norm(v, 2) # v uniform over l_2 unit ball
		if np.sum(v * X[:, i]) < 0:
			v = -v

		T = 2*np.random.binomial(1, pi_a)-1
		
		X_perturb[:, i] = T*v
		
	return [X_perturb, B]

