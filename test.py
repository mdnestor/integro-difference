
import IDE
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

u0 = lambda x : np.heaviside(1 - np.abs(x), 1)
g = lambda u : np.heaviside(u-0.1,1) * u * np.exp(2.5*(1-u))
from scipy.stats import norm
k = norm.pdf
dx = 0.01
T = 100

U_seq = IDE.model(g,k,u0,T,dx)




IDE.plot(U_seq)

#D = [IDE.domain_size(U,0.01) for U in U_seq]
#S = [D[i+1]-D[i] for i in range(len(D)-1)]
#plt.plot(S)
#plt.show()
#N = [i for i in range(len(D))]


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(N,D)
print(slope)
print(p_value)