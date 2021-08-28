import matplotlib.pyplot as plt
import numpy as np


# %% Function decelerations:
def find_position(x, x1):
    # Returns the position index of the last element that is smaller than or equals x1.
    # Assumes x is ordered array
    return sum(x <= x1) - 1


def linear_inter(x1, x, y):
    # Return y1(x1) based on linear interpolation.
    # Assumes x[0] is the point before x1 and x[1] is the point after. same for y[0], y[1]
    return ((x1 - x[0]) * y[1] + (x[1] - x1) * y[0]) / (x[1] - x[0])


def lagrange4_inter(x1, x, y):
    # Return y1(x1) based on Lagrange 4-points interpolation.
    # Evaluate the 4 polynoms f_1..f_4 evaluated at the point x_1
    f_i_x = np.empty(len(x))
    for i in range(len(x)):
        f_i_x[i] = np.prod(np.delete(x1 - x, i)) / np.prod(np.delete(x[i] - x, i))

    # Evaluate the polynom that is the sum of each f_i multiplied by y_iat the point x_1:
    P_x = np.matmul(f_i_x, y)

    return P_x


def derivative3p(x, y):
    # Return y1(x1) based on 3-points discrete interpolation:
    # Assumes x = [x_{i-1}, x, x_{i}], y = [y_{i-1}, ~, y_{i}]
    k = x[1] - x[0]
    h = x[2] - x[1]
    common_denominator = ((k+h) * k * h)
    return (k**2 * y[2] - h**2 * y[0] - (k**2 - h**2) * y[1]) / common_denominator


def hermit4_inter(x1, x, y):
    # Return y1(x1) based on Lagrange 4-points interpolation.
    # Assumes:
    # x1 = float, x = [x_{i-2}, x_{i-1}, x_{i}, x_{i+1}], y = [y_{i-2}, y_{i-1}, y_{i}, y_{i+1}]

    # Start by extracting derivatives:
    y_der = np.array([derivative3p(x[:-1], y[:-1]), derivative3p(x[1:], y[1:])])

    # Calculate desired functions f_{1}, g_{i}:
    common_denominator = (x[1] - x[2]) ** 2
    evaluated_functions = np.empty(4)  # [g_{1}(x), g_{2}(x), f_{1}(x), f_{2}(x)]
    evaluated_functions[0] = ((x1 - x[1]) * (x1 - x[2]) ** 2) / common_denominator # g_{1}(x)
    evaluated_functions[1] = ((x1 - x[1]) ** 2 * (x1 - x[2])) / common_denominator # g_{2}(x)
    evaluated_functions[2] = (x1 - x[2]) ** 2 / common_denominator - 2 * evaluated_functions[0] / (x[1] - x[2]) # f_{1}(x)
    evaluated_functions[3] = (x1 - x[1]) ** 2 / common_denominator - 2 * evaluated_functions[1] / (x[2] - x[1]) # f_{2}(x)

    # Multiply and sum:
    P_x = np.matmul(evaluated_functions, np.append(y_der, y[[1, 2]]))

    return P_x


def interp(x, y, x1, itype):
    # Interpolate functions value at the point x1, given x and y arrays, and method itype
    # Assumes: x = nX1 array, y = nX1 array, x1 = float, itype = 'Linear'/'Lagrange4'/'Hermit4'

    # Find the position index of the last element that is smaller than or equals x1:
    idx = find_position(x, x1)

    if itype == 'Linear':
        # Extract previous and next points values:
        x_temp, y_temp = x[[idx, idx + 1]], y[[idx, idx + 1]]
        # Interpolate
        return linear_inter(x1, x_temp, y_temp)

    elif itype == 'Lagrange4':
        # Extract previous two and next two points values:
        x_temp, y_temp = x[idx - 1:idx + 3], y[idx - 1:idx + 3]
        # Interpolate
        return lagrange4_inter(x1, x_temp, y_temp)

    elif itype == 'Hermit4':
        # Extract previous two and next two points values:
        x_temp, y_temp = x[idx - 1:idx + 3], y[idx - 1:idx + 3]
        # Interpolate
        return hermit4_inter(x1, x_temp, y_temp)

    else:
        print("Unknown method, please use one of the following: 'Linear', 'Lagrange4', 'Hermit4'")


def generate_data():
    # Function to generate data. sparsity_factor is a characteristic distance between
    # adjacent points and xmax is the maximum x of the data
    npoints = 101
    xmax = 10
    np.random.seed(12345)
    x = np.arange(npoints, dtype="float64")
    r = np.random.rand(npoints - 2)
    x[1:npoints - 1] = (x[1:npoints - 1] + (r[0:npoints - 2] - 0.5) / 1.5) * xmax / (npoints - 1)
    x[-1] = x[-1] * xmax / (npoints - 1)
    y = np.sin(x)
    return x, y, npoints, xmax


def print_out(xv, yv):
    # Print out results for the interpolation on the dataset:
    print('%18s %12.4e' % ('max. error was:', np.amax(np.abs(yv - np.sin(xv)))))
    print('%12s %12s %12s' % ('test_x', 'y_interp', 'error'))
    for i in range(len(xv)):
        print('%12.4e %12.4e %12.4e' % (xv[i], yv[i], yv[i] - np.sin(xv[i])))
    return


def compare_methods(x, y, x_test):
    # Save interpolations results on one array and print them all out
    # Also returns the interpolations array
    y_interp = np.zeros((len(x_test), 3))
    for i in range(len(x_test)):
        y_interp[i] = [interp(x, y, x_test[i], 'Linear'),
                       interp(x, y, x_test[i], 'Lagrange4'),
                       interp(x, y, x_test[i], 'Hermit4')]

    print_out(x_test, y_interp[:, 0])
    print_out(x_test, y_interp[:, 1])
    print_out(x_test, y_interp[:, 2])
    return y_interp


# %% Generate Data:

ntests = 10
x, y, npoints, xmax = generate_data()
x_test = np.random.rand(ntests) * xmax

print('Comparing with sparsity_factor = 1')
y_interp = compare_methods(x, y, x_test)

print('Comparing with sparsity_factor = 2')
x_sparse = x[0:npoints:2]
y_sparse = y[0:npoints:2]
compare_methods(x_sparse, y_sparse, x_test)

print('Comparing with sparsity_factor = 0.25')
x_sparser = x[0:npoints:4]
y_sparser = y[0:npoints:4]
compare_methods(x_sparser, y_sparser, x_test)

# %%

plt.plot(x, y)
plt.scatter(x_test, y_interp[:, 0], color='red')
plt.show()
