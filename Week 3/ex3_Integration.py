# %% Import modules:
import numpy as np
from warnings import warn
from matplotlib import pyplot as plt
from pandas import DataFrame

# %%Define main functions:
def simpson_method(f, a, b):
    # Estimates area under graph of a function for a single segment using Simpson's method:
    h = (b-a) / 2
    return (h/3) * (f(a) + 4 * f((a+b) / 2) + f(b))


def gauss_method(f, a, b):
    # Estimates area under graph of a function for a single segment using Gauss's method
    h = (b-a) / 2
    x_0 = (a+b) / 2
    return (h/9) * (  5*f(x_0 - 0.6**0.5 * h) + 8*f(x_0) + 5*f(x_0 + 0.6**0.5 * h))


def fix_segments_integration(f, a, b, integration_method, num_of_segments=100):
    # Direct integral calculation - dividing the domain into equi-width segments,
    # estimating area for each one separately using the chosen method and summing.
    I = 0
    d = (b - a) / num_of_segments
    for i in range(num_of_segments):
        I += integration_method(f, a + i*d, a + (i+1)*d)
    return I


def recursive_integration(f, a, b, eps, integration_method, max_depth=50, min_depth=2, current_depth=1):
    # Recursive funciton for integration
    # Calculating segment's Integral, without deviding:
    I = integration_method(f, a, b)
    # Stop here if max depth reached:
    if current_depth == max_depth:
        # warn('Reached max recursion depth, for better precision inrease it.')
        return I
    # Calculate with the same function the integral value for each of the two halves of the domain:
    I_1 = integration_method(f, a, (a + b) / 2)
    I_2 = integration_method(f, (a + b) / 2, b)
    # Check to see if the higher resolution resulted in better precision:
    # If yes, calculate each half's value again, by deviding it too:
    if abs(I - (I_1 + I_2)) > eps or current_depth < min_depth:
        I_1 = recursive_integration(f, a, (a + b) / 2, eps / 2,
                                    integration_method,
                                    max_depth,
                                    min_depth,
                                    current_depth+1)
        I_2 = recursive_integration(f, (a + b) / 2, b, eps / 2,
                                    integration_method,
                                    max_depth,
                                    min_depth,
                                    current_depth+1)
    return I_1 + I_2


def f_integral(f, a, b, eps, global_type, integral_type):
    # Integrates a function f, in region [a,b], using precision eps, dividing method global_type, and
    # and evaluation method integral_type:
    # Convert method text to the corresponding python function:
    if integral_type == 'simpson':
        integration_method = simpson_method
    elif integral_type == 'gauss':
        integration_method = gauss_method
    else:
        warn('Unknown integral type. using simpson as default')
        integration_method = simpson_method
    # Convert segmentaion method text to the corresponding python function:
    if global_type == 'fix_segments':
        return fix_segments_integration(f, a, b, integration_method)
    if global_type == 'recursive':
        return recursive_integration(f, a, b, eps, integration_method)


def bisection_root(f, a, b, epsx, epsf):
    h = (b-a) / 2
    x_mid = a + h #(=0.5*(a+b))
    f_mid = f(x_mid)
    sign_f_mid = np.sign(f_mid)

    if h < epsx or abs(f_mid) < epsf:    # Break recursion if desired approximation achieved:
        return x_mid
    elif sign_f_mid == 1:
        return bisection_root(f, a , x_mid, epsx, epsf)
    elif sign_f_mid == -1:
        return bisection_root(f, x_mid, b, epsx, epsf)
    else:
        raise Exception('Debbug me')


def secant_root(f, a, b, epsx, epsf, x_n=None, x_n_1=None, f_x_n=None, f_x_n_1=None):
    # Here x_n_1 denotes x_{n-1}
    # If previous-previous point does not exist, take a as x_{n-1} and the first quarter of [a,b] as x_n:
    # (This should happen only in the first iteration of the function)
    if x_n_1 is None:
        x_n_1 = a
        f_x_n_1 = f(x_n_1)
    if x_n is None:
        x_n = a + (b-a) / 4
        f_x_n = f(x_n)

    # If the desired precision achieved, return the previous point:
    if abs(x_n - x_n_1) < epsx or abs(f_x_n) < epsf:
        return x_n

    # Estimate root's location based on NR method:
    x_next = x_n - f_x_n * (x_n - x_n_1) / (f_x_n - f_x_n_1)

    # If the prediction for the next x is out of range, use the mid of [a,b] instead:
    if abs(x_next - x_n) > 0.9*(b-a) or x_next > b or x_next < a: # NOTE ARBITRARY VALUE
        x_next = (a+b) / 2

    f_x_next = f(x_next)
    sign_f_x_next = np.sign(f_x_next)

    if sign_f_x_next == 1:
        return secant_root(f, a, x_next, epsx, epsf, x_next, x_n, f_x_next, f_x_n)
    elif sign_f_x_next == -1:
        return secant_root(f, x_next, b, epsx, epsf, x_next, x_n, f_x_next, f_x_n)
    elif sign_f_x_next == 0:
        return x_next


def x_root(f, a, b, epsx, epsf, type):
    f_a = f(a)
    f_b = f(b)
    sign_f_a = np.sign(f_a)
    sign_f_b = np.sign(f_b)
    if sign_f_a == 1 and sign_f_b == -1:
        f = lambda x: -f(x)
    elif sign_f_a == 0:
        return a
    elif sign_f_b == 0:
        return b
    elif sign_f_a == sign_f_b:
        warn('f(a) must have the opposite sign of f(b), returning None')
        return None
    if type == 'bisection':
        return bisection_root(f, a, b, epsx, epsf)
    elif type == 'secant':
        return secant_root(f, a, b, epsx, epsf)
    else:
        warn('Unknown method type, using bisection as default')
        return bisection_root(f, a, b, epsx, epsf)


def momentum(x, potential, e_n):
    return (e_n - potential(x))**(1/2)
# %% Question 1


def x_squared(x):
    global x_squared_mone
    x_squared_mone += 1
    return x ** 2

x_squared_mone = 0
a_r_g = f_integral(x_squared, 0, 8, 0.01, 'recursive', 'gauss')
a_r_g_mone = x_squared_mone
x_squared_mone = 0
a_r_s = f_integral(x_squared, 0, 8, 0.01, 'recursive', 'simpson')
a_r_s_mone = x_squared_mone
x_squared_mone = 0
a_f_g = f_integral(x_squared, 0, 8, 0.01, 'fix_segments', 'gauss')
a_f_g_mone = x_squared_mone
x_squared_mone = 0
a_f_s = f_integral(x_squared, 0, 8, 0.01, 'fix_segments', 'simpson')
a_f_s_mone = x_squared_mone
df_1 = DataFrame({'Global Type': ['recursive', 'recursive', 'fix_segments', 'fix_segments'],
                  'Integral Type': ['gauss', 'simpson', 'gauss', 'simpson'],
                  'Num of Iterations': [a_r_g_mone, a_r_s_mone, a_f_g_mone, a_f_s_mone],
                  'Integral Result': [a_r_g, a_r_s, a_f_g, a_f_s]})
df_1['Deviation'] = df_1['Integral Result'] / (8**3/3) - 1


df_1

# %% Question 2


# def poly_3(x):
#     global mone_poly_3
#     mone_poly_3 +=1
#     return x ** 3 - 17*x**2+76*x-60
#
# mone_poly_3 = 0
# r_bisection = x_root(poly_3, -5, 5, 0.001, 0.001, 'bisection')
# r_bisection_mone = mone_poly_3
# mone_poly_3 = 0
# r_secant = x_root(poly_3, -5, 5, 0.001, 0.001, 'secant')
# r_secantn_mone = mone_poly_3
#
# df_2 = DataFrame({'Method Type': ['bisection', 'secant'],
#                   'Integral Result': [r_bisection, r_secant],
#                   'Num of iterations': [r_bisection_mone, r_secantn_mone],
#                   'Deviation': [r_bisection - 1, r_secant - 1]})
# df_2

# %% Question 3


# def v_sq(x):
#     global mone_v
#     mone_v += 1 # counts how many times v was calculated for efficiency check
#     return x*x
#
#
# gamma = 1
#
#
# def s(energy) :
#     global mone_s
#     mone_s += 1 # counts how many times s was calculated for efficiency check
#     x_1 = -energy**(1/2)
#     x_2 = energy**(1/2)
#     ff = lambda x: momentum(x, v_sq, energy) # Make sure the factor 2 here is necessary
#     return gamma * f_integral(ff,x_1,x_2,1e-6,'recursive','gauss')
#
#
# for n in range(5):
#     f = lambda energy: s(energy) - (n+1/2) * np.pi
#     mone_v, mone_s = 0, 0
#     e_n = x_root(f,0,1000,1e-3,1e-3,'secant')
#     print('%3d %20.16f %6d %4d %12.4e %12.4e' %(n,e_n,mone_v,mone_s,f(e_n),e_n/(2*n+1)-1))


# %% Question 4

def x_in_out_hr(e_n):
    # Function to calculate x_in and x_out for a given energy e_n

    if e_n >= 0:
        raise Exception('Only negative values for e_n are allowed!')

    x_minus = -(((2 ** (1 / 3)) / 36) * (e_n + 1)) ** (1 / 2) + 2 ** (1 / 6)
    x_plus = (((2 ** (1 / 3)) / 36) * (e_n + 1)) ** (1 / 2) + 2 ** (1 / 6)

    return min(x_minus, x_plus), max(x_minus, x_plus)


def v_hr(x):
    global mone_v
    mone_v += 1  # counts how many times v was calculated for efficiency check
    return -1 + (36 / 2 ** (1 / 3)) * (x - 2 ** (1 / 6)) ** 2


gamma = 150


def s(energy):
    global mone_s
    mone_s += 1  # counts how many times s was calculated for efficiency check
    x_1, x_2 = x_in_out_hr(energy)
    ff = lambda x: momentum(x, v_hr, energy)
    return gamma * f_integral(ff, x_1, x_2, 1e-6, 'recursive', 'gauss')


en_hr = np.zeros(15)
m_v = np.zeros(15)
m_s = np.zeros(15)
f_values = np.zeros(15)

for n in range(15):
    f = lambda energy: s(energy) - (n + 1 / 2) * np.pi
    mone_v, mone_s = 0, 0
    en_hr[n] = x_root(f, -.999, -1e-7, 1e-3, 1e-3, 'secant')
    m_v[n], m_s[n] = mone_v, mone_s
    f_values[n] = f(en_hr[n])

df_4 = DataFrame({'HR Energy levels': en_hr[0:-1],
                  'Mone v': m_v[0:-1],
                  's Counter': m_s[0:-1],
                  'Function Values': f_values[0:-1],
                  'Energy Steps': en_hr[0:-1] - en_hr[1:]
                  })
df_4.index.name = 'n'
df_4.style.format({'LN Energy levels': '{:20.16f}',
                   'Function Values': '{:20.16f}',
                   'Energy Steps': '{:20.16f}'})


# for n in range (14):
#     f = lambda energy: s(energy) - (n + 1/2) * np.pi
#     print('%3d %20.16f %6d %4d %12.4e %20.16f' %(n,en_hr[n],m_v[n],m_s[n],f(en_hr[n]),en_hr[n]-en_hr[n+1]))

# %% Question 5


def x_in_out(e_n):
    # Function to calculate x_in and x_out for a given energy e_n

    if e_n >= 0:
        raise Exception('Only negative values for e_n are allowed!')
    x_minus = (2 * (-1 - (1+e_n)**(1/2)) / e_n )**(1/6)
    x_plus  = (2 * (-1 + (1+e_n)**(1/2)) / e_n )**(1/6)

    return min(x_minus, x_plus), max(x_minus, x_plus)


def v_lj(x):
    global mone_v
    mone_v += 1 # counts how many times v was calculated for efficiency check
    return 4 * (1/(x**12) - 1/(x**6))


gamma = 150


def s_lj(energy, eps) :
    global mone_s
    mone_s += 1 # counts how many times s was calculated for efficiency check
    x_1, x_2 = x_in_out(energy)
    ff = lambda x: momentum(x, v_lj, energy)
    return gamma * f_integral(ff,x_1,x_2, eps,'recursive','gauss')


n_max = 39

n_values = np.arange(n_max)
en_lj = np.zeros(n_max)
m_v = np.zeros(n_max)
m_s = np.zeros(n_max)
f_values = np.zeros(n_max)

for n in range (n_max):
    f = lambda energy: s_lj(energy, 1e-6) - (n + 1 / 2) * np.pi
    mone_v, mone_s = 0, 0
    en_lj[n] = x_root(f, -.999, -1e-7, 1e-6, 1e-6, 'secant')
    m_v[n], m_s[n] = mone_v, mone_s
    f_values[n] = f(en_lj[n])

df_5 = DataFrame({'LN Energy levels': en_lj[0:-1],
                  'Mone v': m_v[0:-1],
                  's Counter': m_s[0:-1],
                  'Function Values': f_values[0:-1],
                  'Energy Steps': en_lj[0:-1] - en_lj[1:]
                  })
df_5.index.name = 'n'
df_5.style.format({'LN Energy levels': '{:20.16f}',
                   'Function Values': '{:20.16f}',
                   'Energy Steps': '{:20.16f}'})

# for n in range (n_max-1):
#     f = lambda energy: s_lj(energy, 1e-6) - (n + 1 / 2) * np.pi
#     print('%3d %20.16f %6d %4d %12.4e %20.16f' %(n,en_lj[n],m_v[n],m_s[n],f(en_lj[n]),en_lj[n]-en_lj[n+1]))


# %% Question 6


# plt.figure(figsize=(6,5))
# n1 = np.arange(15)
# plt.plot(n1[:],en_hr[0:15],'o',label='harmonic-approx')
# n2 = np.arange(39)
# plt.plot(n2[:],en_lj[0:39],'x',label='lenard-Jones')
# plt.grid()
# plt.legend()
# plt.show()

# %% Question 7

n = 20
ep = 1e-3
print('bisection - recursive - gauss')
print(' %3s %20s %12s %6s %4s %12s' % ('n',' en_p ',' ep ','it_v','it_s',' err '))
for ip in range(10):
    f = lambda energy: s_lj(energy, 1e4*ep) - (20 + 1 / 2) * np.pi
    mone_v, mone_s = 0, 0
    en_p = x_root(f,-.999,-1e-7,ep,ep,'secant')
    print(' %3d %20.16f %12.4e %6d %4d %12.4e' % (n,en_p,ep,mone_v,mone_s,f(en_p)))
    ep = ep/10


# %% Question 8
from scipy import optimize
from scipy import integrate

gamma = 150


def s(energy):
    global mone_s
    mone_s += 1  # counts how many times s was calculated for efficiency check
    x_1, x_2 = x_in_out(energy)
    ff = lambda x: momentum(x, v_lj, energy)
    return gamma * integrate.quad(ff, x_1, x_2)[0]


n_max = 39

en_lj = np.zeros(n_max)
m_v = np.zeros(n_max)
m_s = np.zeros(n_max)
f_values = np.zeros(n_max)

for n in range(39):
    f = lambda energy: s(energy) - (n + 1 / 2) * np.pi
    mone_v, mone_s = 0, 0
    en_lj[n] = optimize.brentq(f, -.999, -1e-7)
    m_v[n], m_s[n] = mone_v, mone_s
    f_values[n] = f(en_lj[n])
# print('%20.16f' % (2 * (-1 - en_lj[0])))


df_8 = DataFrame({'LN Energy levels': en_lj[0:-1],
                  'Mone v': m_v[0:-1],
                  's Counter': m_s[0:-1],
                  'Function Values': f_values[0:-1],
                  'Energy Steps': en_lj[0:-1] - en_lj[1:]
                  })
df_8.index.name = 'n'
df_5_8 = df_8.join(df_5, lsuffix='_scipy')

df_5_8.style.format({'Mone v_scipy': '{:20.0f}',
                     'LN Energy levels': '{:20.6f}',
                     'Function Values': '{:20.6f}',
                     'Energy Steps': '{:20.6f}'})