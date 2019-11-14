# integro-difference
Integro-difference equations (IDEs) define discrete time integral recursions. They are used in population biology to measure the spread of an invasive species with discrete reproduction periods. This module contains a numerical algorithm to simulate integro-difference equations, and to calculate their spreading speed and travelling wave speed.

# Installation
The module requires numpy and matplotlib. The primary class can be imported as follows:
```
from integro_difference import ide
```

# Usage
IDEs are defined in terms of two parameters, a growth function and a dispersal kernel. Common choices in population biology for the growth function are the Ricker map g(u)=u\*exp(r(1-u)) and the logistic map g(u)=r*\u\*(1-u), where r > 0 is a free parameter.

The dispersal kernel is a probability density function with mean zero. Common choices are the Gaussian kernel and the Laplace kernel.

```
ricker_map = lambda u : u * np.exp(r*(1-u))
std_normal = lambda x : 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
```

An IDE model can be created with two functions as parameters

```
model = ide(growth_function = ricker,
            dispersal_kernel = std_normal)
```

Determine the spatial domain to run the model on. I chose the interval [-20,20] with a step size of 0.01:

```
model.set_domain(xmin=-20, xmax=20, step_size=0.01)
```

The initial data is defined by a function. Assuming the growth function has an equilibrium at zero, a common choice is the unit plateau or the unit heaviside step function.

```
# u0 is the unit plateau
u0 = lambda x : np.heaviside(1 - np.abs(x), 1)
model.set_initial_density(u0)
```

Run the model for 10 time steps:

```
model.run(time_steps=10)
```

Plot the results using matplotlib:

```
model.plot()
```




