import scipy
from scipy.linalg import hadamard
import numpy as np
import random
from numpy import linalg

from vamp import vamp

#1- List of priors

def prior_gb(A, B, prmts):
    rho, mu, sig=prmts

    m=(B*sig+mu)/(1.+A*sig)
    v=sig/(1+A*sig)
    p_s=rho/(rho+(1-rho)*np.sqrt(1.+A*sig)*
        np.exp(-.5*m**2/v+.5*mu**2/sig))

    a=p_s*m
    c=p_s*v+p_s*(1.-p_s)*m**2
    return a, np.mean(c);

def prior_pm1(A, B, prmts):
	rho = prmts[0]
	a = np.tanh(.5 * np.log(rho / (1. - rho)) + B)
	c = np.maximum(1e-11, 1. - a ** 2)
	return a, np.mean(c);

#2- List of matrices and useful functions

def reduce_matrix(alpha,full_matrix):
	n_cols=full_matrix.shape[0]
	n_lines=int(round(n_cols*alpha))
	lines=random.sample(range(n_cols),n_lines)
	reduced_matrix=np.zeros((n_lines, n_cols))
	for i in range(n_lines):
		for k in range(n_cols):
			reduced_matrix[i,k]=full_matrix[lines[i],k]
	return reduced_matrix;

def iid(alpha, n):
	m=int(round(alpha*n))
	X=np.random.randn(m, n) / np.sqrt(n)
	return X;

def ReLU_zero_mean(x):
	mean = 1 / np.sqrt(2*np.pi)
	return (np.where(x < 0, 0, x) - mean);

def RF(f, alpha, beta, n):
	#A of size n0*n1, B of size n1*n
	n0=int(round(alpha*n))
	n1=int(round(beta*n))
	A=iid(alpha/beta,n1)
	B=iid(beta,n)
	g=np.vectorize(f)
	C=g(A.dot(B))
	return C;

def DCT_matrix(n):
    A=np.zeros((n,n))
    epsilon=np.ones(n)
    epsilon[0]=1/np.sqrt(2)
    for j in range(n):
        for k in range(n):
            A[j,k]=np.sqrt(2/n)*epsilon[k]*np.cos(np.pi*(2*j+1)*k/(2*n))
    return A;

def DCT(alpha,n):
	w=DCT_matrix(n)
	A=reduce_matrix(alpha,w)
	return A;

def Hadamard(alpha,n):
	w=hadamard(n)
	A=reduce_matrix(alpha,w)
	return A;


#3- Main function: returns table of MSE for a given matrix type and rho fixed

def MSEtable_generator(matrix_type=ReLU_zero_mean, RF_matrix=True, prior=prior_gb, rho=0.5, beta=1, n=500, iterations=50, step=0.01, var_noise=1e-8):
	#matrix_type is a matrix generator if RF_matrix=false, or the function f of the RF matrix if RF_matrix=True

	size=int(1/step)
	alpha_table=np.linspace(step, size*step, num=size)
	MSE_table=np.zeros(size)
	
	for i in range(size):
		#Generate matrix for a alpha[i]
		if RF_matrix is True:
			A=RF(matrix_type, alpha_table[i], beta, N)
		else:
			A=matrix_type(alpha_table[i],N)

		#Iterate execution of VAMP 50 times
		MSE_iterations=np.zeros(iterations)
		for k in range(iterations):

			#Generate signal with chosen prior
			if prior is prior_pm1:
				w = np.zeros(N)
				w=np.sign(w)
			else :
				w = np.zeros(N)
				n_nonzeros=int(np.ceil(rho*N))
				supp = np.random.choice(N, n_nonzeros, replace=False)
				w[supp] = np.random.randn(n_nonzeros)

			y = A.dot(w) + np.sqrt(var_noise)*np.random.randn(int(round(alpha_table[i]*N)))

			#Apply VAMP and compute MSE
			w_hat_rho = vamp(A, y, var_noise, prior, prior_prmts=(rho, 0, 1), true_coef=w, max_iter=250, verbose=0)						
			MSE_iterations[k]=np.mean((w - w_hat_rho)**2)

		#Average MSE for given alpha on 50 executions	
		MSE_table[i]=np.mean(MSE_iterations)

	return MSE_table;

#4- Examples for rho=0.5 and Gauss-Bernoulli prior

#MSE transition Bayes-optimal with DCT, rho=0.5
DCT_gb_2000_rho5=MSEtable_generator(matrix_type=DCT, RF_matrix=False, prior=prior_gb, rho=0.5, n=2000, iterations=20, step=0.02, var_noise=1e-8)
np.savetxt("DCT_gb_2000_rho5.txt", DCT_gb_2000_rho5)

#MSE transition Bayes-optimal with Hadamard, rho 0.5
Hadamard_gb_2000_rho5=MSEtable_generator(matrix_type=Hadamard, RF_matrix=False, prior=prior_gb, rho=0.5, n=2048, iterations=20, step=10/512, var_noise=1e-8)
np.savetxt("Hadamard_gb_2000_rho5.txt", Hadamard_gb_2000_rho5)

#Mse transition Bayes-optimal with RF atrix f=tanh, rho=0.5
tanhRF_gb_2000_rho25=MSEtable_generator(f=np.tanh, RF_matrix=True, prior=prior_gb, rho=0.5, n=2000, iterations=20, step=0.02, var_noise=1e-8)
np.savetxt("tanhRF_gb_2000_rho25.txt", tanhRF_gb_2000_rho25)