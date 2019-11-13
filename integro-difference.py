
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class integro_difference:

	domain = np.array(0)
	
	density_sequence = []
	
	growth_function = None
	dispersal_kernel = None
	
	
	
	def __init__(self, growth_function, dispersal_kernel):
		self.growth_function = growth_function
		self.dispersal_kernel = dispersal_kernel
	
	
	def set_domain(self,xmin,xmax,dx):
	
		N = 1+int((xmax-xmin)/dx)
		
		self.domain = np.linspace(xmin,xmax,N)
		
		return
	
	def set_initial_density(self,u0):
		X = self.domain
		self.density_sequence = [u0(X)]
		return

	def iterate(self,U):
		X = self.domain
		dx = X[1] - X[0]
		
		g = self.growth_function
		k = self.dispersal_kernel
		
		U = dx * signal.fftconvolve(k(X), g(U), mode='same')
		return U
	
	def run(self,steps):
		
		U_sequence = self.density_sequence
		
		U = U_sequence[0]
		
		for t in range(steps):
			U = self.iterate(U)
			U_sequence.append(U)
			
		self.density_sequence = U_sequence
	
	def plot(self):
		X = self.domain
		U_seq = self.density_sequence
		T = len(U_seq)
		
		for t in range(T):
			U = U_seq[t]
			plt.plot(X, U, color=(t/T,1-t/T,0))
		plt.show()