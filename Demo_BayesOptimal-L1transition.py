import scipy
from scipy.linalg import hadamard
import numpy as np
from numpy import linalg

from vamp import vamp
from vamp import vamp_modified

#1- List of priors

def prior_gb(A, B, prmts):
	rho, mu, sig = prmts

    m = (B * sig + mu) / (1. + A * sig)
    v = sig / (1 + A * sig)
    p_s = rho / ( rho + (1 - rho) * np.sqrt(1. + A * sig) *
        np.exp(-.5 * m ** 2 / v + .5 * mu ** 2 / sig) )

    a = p_s * m
    c = p_s * v + p_s * (1. - p_s) * m ** 2
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

#3- Bayes-optimal phase diagram : returns a table of alpha's between 0 and 1, and a table of MSEs for rho between 0 and 1

def MSEgrid_generator(matrix_type, n=1000, iterations=50, step=0.05, var_noise=1e-8, prior=prior_gb):
	size=int(1/step)
	alpha_table=np.linspace(step, size*step, num=size)
	rho_table=np.linspace(step, size*step, num=size)
	MSE_table=np.zeros((size,size))

	for i in range(size):
		#Generate matrix for a alpha[i]
		A=matrix_type(alpha_table[i], n)
		
		#Apply VAMP for every rho, for fixed alpha[i]
		for j in range(size):	

			#Iterate execution of VAMP 50 times for (rho[j], alpha[i])
			MSE_iterations=np.zeros(iterations)
			for k in range(iterations):
				#Generate signal with Gauss-Bernoulli prior
				w = np.zeros(n)
				n_nonzeros=int(np.ceil(rho_table[j]*n))
				supp = np.random.choice(n, n_nonzeros, replace=False)
				w[supp] = np.random.randn(n_nonzeros)
				y = A.dot(w) + np.sqrt(var_noise)
				#Apply VAMP and compute MSE
				w_hat_rho = vamp(A, y, var_noise, prior, prior_prmts=(rho_table[j], 0, 1), true_coef=w, max_iter=250, verbose=0) 
				MSE_iterations[k]=np.mean((w - w_hat_rho)**2)

			#Average MSE for (rho[j], alpha[i]) on 50 iterations
			MSE_table[i,j]=np.mean(MSE_iterations)

	return MSE_table;


#4- L1 transition line generator : returns a table of alpha's, and a table of corresponding rho's for the l1 phase transition line

def L1transition_finder(matrix_type, RF_matrix=False, n=2000, iterations=10, step=0.02, alpha_start=0.02, alpha_end=1, var_noise=1e-8):
	#matrix_type is a matrix generator if RF_matrix=false, or the function f of the RF matrix if RF_matrix=True

	size=int((alpha_end-alpha_start)/step)
	alpha_table=np.linspace(alpha_start, alpha_end, num=size+1)
	rho_transition_table=np.zeros(size+1)

	for i in range(size+1):
		#Generate matrix for alpha[i]
		if RF_matrix is True:
			A=RF(matrix_type, alpha_table[i], 1, n)
		elif RF_matrix is False:
			A=matrix_type(alpha_table[i],n)
		
		#Iterate search for the transition rho 50 times
		rho_iterations=np.zeros(iterations)
		print("alpha :", alpha_table[i], ", iterations : ", end=" ", flush=True)
		for l in range(iterations):		
			print(l, end=" , ", flush=True)

			#Try rho between 0 and 1 to find the L1 transition				
			new_start=np.zeros(3)
			#Span [0,1] to look for transition with step 0.1, then 0.01, then 0.001
			for k in range(3):    
				for j in range(10):
					rho=new_start[0]/10 + new_start[1]/100 + new_start[2]/1000 + (j+1)/(10**(k+1))
					w = np.zeros(n)
					n_nonzeros=int(np.ceil(rho*n))
					supp = np.random.choice(n, n_nonzeros, replace=False)
					w[supp] = np.random.randn(n_nonzeros)
					y = A.dot(w) + np.sqrt(var_noise)
					#Apply VAMP with soft-thresholding prior to find MSE
					w_hat_rho = vamp_modified(A, y, var_noise=1e-8, prior=soft_thres, tol=5e-7, prior_prmts=None, true_coef=w, max_iter=250, verbose=0) 
					MSE=np.mean((w - w_hat_rho)**2)
					if MSE>1e-3 :
						new_start[k]=j
						break

			rho_iterations[l] = new_start[0]/10 + new_start[1]/100 + new_start[2]/1000

		#Average on 50 executions to obtain rho_transition[i] for alpha[i]
		rho_transition_table[i]=np.mean(rho_iterations)
	return alpha_table, rho_transition_table;


#5- Examples

#Bayes-optimal MSE grid for a DCT matrix (size 2000)
DCT_BayesOptimal_2000_table=MSEgrid_generator(DCT, n=2000, iterations=20, step=0.02, var_noise=1e-8)
np.savetxt("DCT_BayesOptimal_2000_table.txt", DCT_BayesOptimal_2000_table)

#L1 phase transition line for a RF matrix with f=tanh (size 2000)
L1transition_RF_tanh_2000=L1transition_finder(np.tanh, RF_matrix=True, n=2000, iterations=20, step=0.02, var_noise=1e-8)

#L1 phase transition line for a Hadamard matrix (size 2048)
L1transition_Hadamard_2048=L1transition_finder(Hadamard, RF_matrix=False, n=2048, iterations=10, step=10/512, alpha_start=10/512, alpha_end=510/512, var_noise=1e-8)