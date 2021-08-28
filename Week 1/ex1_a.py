import numpy as np
import matplotlib.pyplot as plt

# %%
golden_ratio_global = (np.sqrt(5) - 1) / 2


# Function to calculate power of phi using the backwards method:
def backward_recursive_power(n, initial_guess, df_type):
    # Initial empty list:
    powers = np.empty(n + 2, dtype=df_type)
    powers[-2] = initial_guess
    powers[-1] = initial_guess * golden_ratio_global

    # Iterate backwards using formula:
    for i in range(np.size(powers) - 3, -1, -1):
        powers[i] = powers[i + 1] + powers[i + 2]

    # Normalize and return:
    return powers[0:n] / powers[0]

def direct_powers_calculation(n):
    # Function to calculate power of phi directly from the definition of power
    powers = [0] * n
    powers[0] = 1
    for i in range(1, n):
        powers[i] = powers[i-1] * golden_ratio_global
    return powers


# Function to generate the requested table:
def golden_powers_down(n, df_type):
    # Initialize initial variables:
    data = np.empty((n, 3), dtype=df_type)
    direct_powers = direct_powers_calculation(n)
    data[:, 0] = direct_powers
    data[:, 1] = backward_recursive_power(n, 1, df_type)
    data[:, 2] = (data[:, 1] - direct_powers) / direct_powers

    return data


# %%
goldown = golden_powers_down(31, 'float32')
plt.plot(range(2, 25), np.log10(abs(goldown[2:25, 2])))
plt.title('stab_down', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('log-abs-error', fontsize=14)
plt.show()

plt.plot(range(2, 19), goldown[2:19, 2])
plt.title('stab_down', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('error', fontsize=14)
plt.show()
