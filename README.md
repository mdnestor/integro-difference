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
# g is the Ricker map
g = lambda u : u * np.exp(r*(1-u))
# k is the Gaussian kernel
k = lambda x : 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
```


