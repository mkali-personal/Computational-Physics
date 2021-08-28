# %% Import modules:
import numpy as np


# %%
def derivative2p(f,x,dx):
    return (f(x + dx) - f(x)) / dx

def derivative3p(f,x,dx):
    return (f(x+dx) - f(x-dx)) / (2*dx)

def derivative5p(f,x,dx):
    return 1/(12*dx) * (f(x - 2*dx) -8*f(x - dx)  + 8*f(x + dx) - f(x + 2*dx))

# %% Check Results
def func(x):
    return np.sin(x)

def true_deriv(x):
    return np.cos(x)

np.random.seed(12345)
x = np.random.rand(10)
dx = 0.002
print('%12s %12s %12s %12s %12s ' % ('x','dfdx','err2p','err3p','err5p'))
for a in x:
    der2p = derivative2p(func,a,dx)
    der3p = derivative3p(func,a,dx)
    der5p = derivative5p(func,a,dx)
    der = true_deriv(a)
    print('%12.4e %12.4e %12.4e %12.4e %12.4e ' % (a,der,der2p/der-1,der3p/der-1,der5p/der-1))

# %%
a = 0.7
der = true_deriv(a)
print('%s %12.4e %s %12.4e ' % ('x=', a, 'dfdx=', der))
print('%12s %12s %12s %12s ' % ('dx', 'err2p', 'err3p', 'err5p'))
for dx in np.logspace(-1, -10, 10):
    der2p = derivative2p(func, a, dx)
    der3p = derivative3p(func, a, dx)
    der5p = derivative5p(func, a, dx)
    print('%12.4e %12.4e %12.4e %12.4e ' % (dx, der2p / der - 1, der3p / der - 1, der5p / der - 1))
