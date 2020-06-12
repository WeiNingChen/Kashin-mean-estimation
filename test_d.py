import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from PrivUnit import privUnit
from Kashin_sampling import *
import scipy.io as io

if __name__ == "__main__":

	# set parameters
	d = 20
	n = 10000
	# Comm. and privacy constraints
	k = 5
	eps = 5
	k_equiv = min(math.ceil(0.8*eps), k)
	print("(eps, k, k_equiv) = (%f, %f, %f)" %(eps, k, k_equiv))

	num_itr = 5
	init_size = 20
	step_size = 20
	step_num = 5
	indices = [init_size + i*step_size for i in range(step_num)]


	kashin_mse_list = np.zeros(step_num)
	privUnit_mse_list = np.zeros(step_num)

	for n in [8000]: 
		for step in range(step_num):
			d = init_size + step_size*step
			print("d = %d" %d)
			for itr in range(num_itr):

				# Generate a random tight frame satisfying UP
				N = 2**int(math.ceil(math.log(d, 2)))
				U = (1/np.sqrt(N)*hadamard(N)@np.diag(2*np.random.binomial(1, 1/2*np.ones(N))-1))[:, 0:d]

				# Generate data matrix
				X = np.zeros((d, n))
				for j in range(int(n/2)):
					v_1 = np.concatenate([np.random.normal(10,1,int(d))])
					v_2 = np.concatenate([np.random.normal(1,1,int(d))])
					X[:,j] = v_1/np.linalg.norm(v_1)
					X[:,j+int(n/2)] = v_2/np.linalg.norm(v_2)

				# Kashin
				print('k_equiv = %d' %k_equiv)
				[q_quantize, q_sampling, q_perturb] = Kashin_encode(U, X, k_equiv, eps)
				X_hat = Kashin_decode(U, k, eps, q_perturb)
				mse = np.linalg.norm(np.mean(X, axis = 1).reshape(-1,1) - X_hat)**2
				kashin_mse_list[step] = kashin_mse_list[step] + mse*1/num_itr

				# privUnit
				[a, q_quantize, q_sampling, B] = privUnit_encode(U, X, k, eps)
				X_hat = privUnit_decode(U, k, eps, q_sampling, B)
				mse = np.linalg.norm(np.mean(X, axis = 1).reshape(-1,1) - X_hat)**2
				privUnit_mse_list[step] = privUnit_mse_list[step] + mse*1/num_itr

		print(kashin_mse_list)
		print(privUnit_mse_list)
		print(indices)

		plt.plot(np.array(indices), kashin_mse_list)
		plt.plot(np.array(indices), privUnit_mse_list)
		plt.show()

		data = {
			'n' : n,
			'eps': eps,
			'k' : k,
			'k_equiv' : k_equiv,

			'kashin_mse' : kashin_mse_list,
			'privUnit_mse' : privUnit_mse_list,
			'indices' : indices
		}
		para = 'n_{}_eps_{}_'.format(n, eps)
		filename = 'Data/' + para + '.mat'
		io.savemat(filename, data)


