


from pyide import *
from scipy.stats import norm


H = lambda x : np.heaviside(x, 1)

alpha=0.33
mu=0.5
beta=0.66

g = lambda u: H(u-alpha) + (mu-1) * H(u-beta)

U = np.arange(0, 1, 0.01)

import matplotlib.pyplot as plt

#plt.plot(U, g(U))

# horizontal lines
plt.plot([0,alpha],[0,0], color='blue')
plt.plot([alpha,beta],[1,1], color='blue')
plt.plot([beta,1],[mu,mu], color='blue')
# vertical lines
plt.plot([alpha,alpha],[0,1], color='blue' , linestyle='dotted', label='g(u)')
plt.plot([beta,beta],[1,mu], color='blue', linestyle='dotted')

plt.plot([0,1],[0,1], color='black', linestyle='dashed', label='Identitfy function')

plt.legend()

plt.savefig('fig1.png')
