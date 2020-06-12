import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from PrivUnit import privUnit

def Kashin_representation(x, U, eta = 0.5, delta = 0.7):
	# compute kashin representation of x with respect to the frame U at level K
	# U = [u_1, u_2, ..., u_N]^T
	(N, d) = U.shape
	a = np.zeros((N, 1))
	K = 1/((1-eta)*np.sqrt(delta))
	M = eta/np.sqrt(delta*N)
	y = x
	itr = int(np.log(N))
	for i in range(itr):
		b = U @ y
		b_hat = np.clip(b, -M, M)
		y = y - U.T @ b_hat
		a = a + b_hat
		M = eta*M

	b = U @ y
	#b = np.clip(b, -M, M)
	Ty = U.T @ b
	y = y - Ty
	a = a + b
	return [a, K/np.sqrt(N)]

def rand_quantize(a, a_bdd):
	return (np.random.binomial(1, (np.clip(a, -a_bdd, a_bdd)+a_bdd)/(2*a_bdd))-1/2)*2*a_bdd

def rand_sampling(q, k):
	# each column of q represents a quantized observation with Kashin representation
	# output k sampling matrices and an aggregation of q*sampling_mat
	(N, n) = q.shape
	sampling_mat_sum = np.zeros((n, N))
	sampling_mat_list = []
	for i in range(k):
		spl = np.eye(N)[np.random.choice(N, n)]
		sampling_mat_sum = sampling_mat_sum + spl 
		sampling_mat_list.append(spl.T)

	return [sampling_mat_list, sampling_mat_sum.T, q * sampling_mat_sum.T/k]

def kRR(k, eps, q_sampling, sampling_mat_list, a_bdd):
	# perturb each row of q, as a k-bit string, via k-RR mechanism 
	q_perturb = q_sampling.copy()
	(N, n) = q_sampling.shape
	for j in range(n):
		if (np.random.uniform(0,1) > (math.exp(eps)-1)/(math.exp(eps)+2**k-1)):
			noise = np.zeros(N)
			for i in range(k):
				# create a random {-1, +1}^N vector, and filter it by sampling matrix
				noise = noise + (2*np.random.binomial(1, 1/2*np.ones(N))-1)*sampling_mat_list[i][:, j].reshape(-1,)/k
			q_perturb[:, j] = noise*a_bdd
	return q_perturb

def estimate(k, eps, q_perturb):
	return	(math.exp(eps)+2**k-1)/(math.exp(eps)-1)*q_perturb 

def kRR_string(d, num, eps):
	if (np.random.uniform(0,1) < math.exp(eps)/(math.exp(eps)+d-1)):
		return num
	else:
		return np.radom.choice(d)

def Kashin_encode(U, X, k, eps):
	[a, a_bdd] = Kashin_representation(X, U)
	q = rand_quantize(a, a_bdd)
	[sampling_mat_list, sampling_mat_sum, q_sampling] =  rand_sampling(q, k)
	q_perturb = kRR(k, eps, q_sampling, sampling_mat_list, a_bdd)
	return [q, q_sampling, q_perturb]

def Kashin_decode(U, k, eps, q_perturb):
	(N, d) = U.shape
	q_unbiased = estimate(k, eps, q_perturb)
	return U.T @ (np.mean(q_unbiased*N, axis = 1)).reshape(-1, 1)
	#return U.T @ (np.mean(q_perturb*N, axis = 1)).reshape(-1,1)

def privUnit_encode(U, X, k, eps):
	[X_perturb, B] = privUnit(X, eps)
	[a, a_bdd] = Kashin_representation(X_perturb, U)
	q = rand_quantize(a, a_bdd)
	[sampling_mat_list, sampling_mat_sum, q_sampling] =  rand_sampling(q, k)
	return [a, q, q_sampling, B]

def privUnit_decode(U, k, eps, q_sampling, B):
	(N, d) = U.shape
	return B*U.T @ (np.mean(q_sampling*N, axis = 1)).reshape(-1,1)



if __name__ == "__main__":
	# generate X
	d = 40
	n = 10000
	k = 10 # sampling rate
	eps = 1
	N = 2**int(math.ceil(math.log(d, 2)))

	# Generate a (random) tight frame based on Hadamard matrix
	U = (1/np.sqrt(N)*hadamard(N)@np.diag(2*np.random.binomial(1, 1/2*np.ones(N))-1))[:, 0:d]


	# Generate data
	X = np.zeros((d, n))
	for j in range(int(n/2)):
		v_1 = np.concatenate([np.random.normal(10,1,int(d))])
		v_2 = np.concatenate([np.random.normal(1,1,int(d))])
		X[:,j] = v_1/np.linalg.norm(v_1)
		X[:,j+int(n/2)] = v_2/np.linalg.norm(v_2)

	# Kashin
	k_equiv = min(math.ceil(0.8*eps), k)
	print('k_equiv = %d' %k_equiv)
	[q_quantize, q_sampling, q_perturb] = Kashin_encode(U, X, k_equiv, eps)
	X_hat = Kashin_decode(U, k_equiv, eps, q_perturb)
	print(np.linalg.norm(np.mean(X, axis = 1).reshape(-1,1) - X_hat)**2)

	# PrivUnit
	[a, q_quantize, q_sampling, B] = privUnit_encode(U, X, k, eps)
	X_hat = privUnit_decode(U, k, eps, q_sampling, B)
	print(np.linalg.norm(np.mean(X, axis = 1).reshape(-1,1) - X_hat)**2)

	