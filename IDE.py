
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

def domain_size(X, U, threshold):
	support_index = [ i for i in range(len(U)) if U[i] > threshold ]
	if len(support) == 0:
		return 0
	else:
		support = [ X[i] for i in support_index ]
		return max(M) - min(M)




def iterate(U,X,g,k):
	dx = X[1]-X[0]
	U_next = dx * scipy.signal.fftconvolve(k(X), g(U), mode='same')
	return U_next
	

def plot(U_seq):
	T = len(U_seq)
	for t in range(T):
		U = U_seq[t]
		if t % 2 == 0:
			v = np.array((1,0,0))
		else:
			v = np.array((0,1,0))
		plt.plot(U, color=v,alpha=t/T)
	plt.show()

def generateInitialData(self,type, u):
	if type == 'bounded':
		U0 = u * np.heaviside(1 - np.abs(X), 1)
	elif type == 'wave':
		U0 = u * np.heaviside(X, 1)
	elif type == 'cdf':
		U0 = u * K
	return U0
	
def getBoundary(U): return np.array((U[0],U[-1]))

def pad(A,count,type):

	if type == 'density':
		(a_min, a_max) = A[0], A[-1]
		
		left_tail = np.repeat(a_min, count)
		right_tail = np.repeat(a_max, count)
		
		A_padded = np.concatenate((left_tail, A, right_tail))
		
		return A_padded
	
	elif type == 'space':
	
		(a_min, a_max) = A[0], A[-1]
		step = A[1]-A[0]
		
		left_tail = np.linspace(a_min-step*count, a_min-step, count)
		right_tail = np.linspace(a_max+step, a_max+step*count, count)
		
		A_padded = np.concatenate((left_tail, A, right_tail))
		
		return A_padded




def model(g,k,u0,T, dx):

	xmin = -10
	xmax = 10
	# X0 is the real interval from xmax to xmin
	X0 = np.linspace(xmin,xmax,1+int((xmax-xmin)/dx))

	# U0 is an array of the initial distribution
	U0 = u0(X0)


	U_seq = [U0]

	X = X0
	U = U0

	iterated_boundary = getBoundary(U0)

	for t in range(T):

		u1, u2 = iterated_boundary
		iterated_boundary = g(u1), g(u2)
		
		U_next = iterate(U,X,g,k)
		
		while np.linalg.norm(getBoundary(U_next) - iterated_boundary) >= 0.01:
			
			count = int(1/dx)
			
			X = pad(X,count,'space')
			U_seq = [pad(V,count,'density') for V in U_seq]
			
			U = U_seq[-1]
			U_next = iterate(U,X,g,k)
			
		#U_seq.append(U)
		#U_seq = [pad(V,count,'density') for V in U_seq[:-1]]
			
		U_seq.append(U_next)
		U = U_next
	return U_seq
	
	

