

import numpy as np
import matplotlib.pyplot as plt

class ide:

	domain = np.array(0)
	
	density_sequence = []
	
	growth_function = None
	dispersal_kernel = None
	pdf_array = np.array([])
	cdf_array_left = np.array([])
	cdf_array_right = np.array([])
	
	
	
	
	def __init__(self, growth_function, dispersal_kernel):
		self.growth_function = growth_function
		self.dispersal_kernel = dispersal_kernel
	
	
	def set_domain(self,xmin,xmax,step_size):
	
		N = int((xmax-xmin)/step_size/2)
		
		self.domain = np.linspace(xmin,xmax,2*N+1)
		
		domain_large = np.linspace(2*xmin,2*xmax,4*N+1)
		
		k = self.dispersal_kernel
		pdf_array_large = k(domain_large)
		self.pdf_array = pdf_array_large[N:(3*N+1)]
		
		cdf_array_left = [0]
		for i in range(2*N):
			cdf_array_left.append(pdf_array_large[i]+cdf_array_left[i])
		
		cdf_array_right = [0]
		for i in range(2*N):
			cdf_array_right.append(pdf_array_large[4*N-i]+cdf_array_right[i])
		cdf_array_right = cdf_array_right[::-1]
		
		self.cdf_array_left = step_size*np.array(cdf_array_left)
		self.cdf_array_right = step_size*np.array(cdf_array_right)
			
		return
	
	def set_initial_density(self,u0):
		X = self.domain
		self.density_sequence = [u0(X)]
		return

	def iterate(self,U):
		X = self.domain
		step_size = X[1] - X[0]
		
		g = self.growth_function
		
		N = int((len(X)-1)/2)
		
		pdf = self.pdf_array
		cdf_left = self.cdf_array_left
		cdf_right = self.cdf_array_right
		
		U_center = step_size * np.convolve(pdf, g(U), mode='same')
		U_left = g(U[-1]) * cdf_left
		U_right = g(U[0]) * cdf_right
		
		return U_center + U_left + U_right
	
	def run(self,time_steps):
		
		U_sequence = self.density_sequence
		
		U = U_sequence[0]
		
		for t in range(time_steps):
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
