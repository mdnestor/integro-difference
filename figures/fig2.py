


from pyide import *
from scipy.stats import norm


H = lambda x : np.heaviside(x, 1)

K = lambda x : H(x) + 1/2 * (1-2*H(x)) * np.exp((2*H(x)-1)*(-x))

w1 = lambda x : K(-x)
X = np.arange(-5, 5, 0.01)

import matplotlib.pyplot as plt

plt.plot(X, w1(X), label='y=w1(x)')
plt.ylim(0, 1.1)
plt.legend()

plt.savefig('fig2A.png')


plt.clf()
alpha=0.3
beta = 0.8
mu = 0.5

# Find x_alpha and x_beta
K_inv = lambda p : np.log(2*p)*H(1/2-p) - np.log(2-2*p)*H(p-1/2)
P = np.arange(0, 1, 0.01)
plt.plot(P, K_inv(P) ,label='y=K^(-1)(x)')

plt.savefig('K.png')

x_alpha = K_inv(1-alpha)
x_beta =  K_inv(1-beta)
print(x_alpha,x_beta)
w2 = lambda x : 1 - K(x-x_alpha) - (1-mu) * (1-K(x-x_beta))

plt.clf()
plt.plot(X, w2(X), label='y = w2(x)')
plt.ylim(0, 1.1)
plt.legend()

plt.savefig('fig2B.png')



