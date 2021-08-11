

import  pyide
import importlib
importlib.reload(pyide)
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from imp import reload # for debugging
alpha=0.2
beta=0.8
mu=0.3

H = lambda x : np.heaviside(x, 1)

g = lambda x : H(x-alpha) - (1-mu)*H(x-beta)

import matplotlib.pyplot as plt

U = np.arange(0, 1, 0.01)
k = lambda x: 1/2 * np.exp(-np.abs(x))

model = pyide.IDEModel(growthFunction=pyide.GrowthFunction(g),
                  dispersalKernel=pyide.DisperalKernel(k))
                  
                
sim = pyide.IDESimulation(model)
sim.setDomain(xmin=-10, xmax=10, dx=0.005)
r=0.58
sim.setInitialCondition(lambda x : H(x+r)-H(x-r))
sim.setBoundaryCondition('static')

n = 15
sim.run(n)
sim.plot(times=np.arange(1, n), file='fig_accelerating_wave1.png')
plt1 = sim.plot_wavespeed(times=np.arange(1, n), 
predictedcstar = 0.8568,
predictedc1 = 2*0.8568 + np.log(2/5),
predictedc2 = -np.log(2/5),
level=alpha,
                         file='fig_accelerating_wave2.png')
