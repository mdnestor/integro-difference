

import pyide
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
sim.setDomain(xmin=-10, xmax=10, dx=0.01)
sim.setInitialCondition(lambda x : H(-x))
sim.setBoundaryCondition('static')

n = 11
sim.run(n)
sim.plot(times=np.arange(1, n), file='fig3.png')

