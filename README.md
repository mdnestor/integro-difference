# integro-difference
Integro-difference equations (IDEs) define discrete time integral recursions. They are used in population biology to measure the spread of an invasive species with discrete reproduction periods. This repository contains code to simulate integro-difference equations and calculate their spreading speed and travelling wave speed.

# Installation
The module requires numpy and matplotlib. The primary class can be imported as follows:
```
from integro_difference import ide
```

# Usage
IDEs are defined in terms of two parameters, a growth function and a dispersal kernel. Common choices in population biology for the growth function are the Ricker map g(u)=u\*exp(r(1-u)) and the logistic map $g(u)=ru(1-u)$, where $r > 0$ is a free parameter.

The dispersal kernel is a probability density function with mean zero. Common choices are the Gaussian kernel and the Laplace kernel.

```
ricker_map = lambda u : u * np.exp(r*(1-u))
std_normal = lambda x : 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
```

An IDE model can be created as follows:

```
model = pyide.IDEModel(growthFunction=pyide.GrowthFunction(g),
                  dispersalKernel=pyide.DisperalKernel(k))
```

We then create a simulation object:
```
sim = pyide.IDESimulation(model)
```

```
sim.setDomain(xmin=-10, xmax=10, dx=0.01)
sim.setInitialCondition(lambda x : H(-x))
sim.setBoundaryCondition('static')
```

Run the model:
```
sim.run(n = 11)

sim.plot()
```

![Alt text](figures/fig3.png?raw=true "Title")



