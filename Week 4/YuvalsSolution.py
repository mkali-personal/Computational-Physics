import numpy as np
from matplotlib import pyplot as plt
# %%

def find_zero(F, dF, X, epsF, epsX, no_flipping=False):
    """
    Find the root of the multivariable vector function f.
    Args:
        f (`function`): The function for which to find a root. Accepts `x` (a vector) and returns a vector.
        df (`function`): The differential function of `f`. Accepts `x` and returns the Jacobian matrix at `x`.
        x (`np.array`): The initial guess for a root.
        eps (`float`): .
    Returns:
        x (`np.array`): The x value of the root of the function in the range [a,b].
    Raises:
        ValueError:
    """
    if np.linalg.norm(F(X)) < epsF:
        # Initial guess is close enough
        return X

    X = run_step(F, dF, X, no_flipping)

    while np.linalg.norm(F(X)) >= epsF:
        X = run_step(F, dF, X, no_flipping)

    return X


def run_step(F, dF, X_guess, no_flipping=False):
    delta = newton_step(F, dF, X_guess)
    print('Guess:', X_guess, 'Delta:', delta)

    if no_flipping:
        alpha = scale_to_prevent_flipping(X_guess, delta)
        delta = alpha * delta

    delta = line_search(F, X_guess, delta)
    return X_guess + delta


def scale_to_prevent_flipping(X, delta):
    Xd = X + delta - np.roll(X + delta, 1)
    Xd[0] = X[0] + delta[0]
    if any(Xd < 0):
        X_1d = X - np.roll(X, 1)
        X_1d[0] = X[0]

        deltad = delta - np.roll(delta, 1)
        deltad[0] = delta[0]  # Set boundary (delta1 - delta0 = delta1)

        alpha_n = - X_1d / (deltad)
        return 0.9 * min(alpha_n[Xd < 0])
    return 1


def is_delta():
    pass


def newton_step(F, dF, X):
    J = dF(X)
    F_value = F(X)
    delta = - np.linalg.solve(J, F_value)

    return delta


def fmin(F, X):
    return 0.5 * sum(F(X) ** 2)


def line_search(F, X_old, delta):
    plt.figure(figsize=(12, 5))
    x = np.geomspace(1e-10, 1, num=100)
    y = np.zeros(x.size)
    for i in range(y.size):
        y[i] = fmin(F, X_old + x[i] * delta)
    plt.plot(x, y, '.')
    plt.grid()
    plt.show()

    #     scale_factor = 1e16
    #     if sum(delta**2) > scale_factor:
    #         delta *= scale_factor / sum(delta**2)

    # Based majorly on the C implementation found in "Numerical Recipes in C, 2nd Edition" by W. H. Press et. al., p.385-386
    g_0 = fmin(F, X_old)
    dg_0 = - 2 * g_0

    l_1 = 1

    while True:
        X = X_old + l_1 * delta
        g = fmin(F, X)
        print('\t\tl = %12.4e\tg(l) = %20.16f' % (l_1, g))

        if (g <= g_0 + 1e-4 * l_1 * dg_0):
            # Sufficient function decrease
            return l_1 * delta

        # Find suitable lambda.
        if l_1 == 1:
            # First iteration, approximate g as a quadratic polynomial
            # import pdb;
            # pdb.set_trace()
            l = - (dg_0) / (2 * (g - g_0 - dg_0))

        else:
            a = (1 / (l_1 - l_2)) * (
                        (1 / (l_1 ** 2)) * (g - dg_0 * l_1 - g_0) + (-1 / (l_2 ** 2)) * (g_2 - dg_0 * l_2 - g_0))
            b = (1 / (l_1 - l_2)) * (
                        (-l_2 / (l_1 ** 2)) * (g - dg_0 * l_1 - g_0) + (l_1 / (l_2 ** 2)) * (g_2 - dg_0 * l_2 - g_0))

            if a == 0:
                l = - dg_0 / (2 * b)
            else:
                d = b ** 2 - 3 * a * dg_0  # the discriminant
                if d < 0:
                    l *= 0.5
                elif b <= 0:
                    l = (-b + np.sqrt(d)) / (3 * a)
                else:
                    l = - dg_0 / (b + np.sqrt(d))
            l = min(l, 0.5 * l_1)

        l_2, l_1 = l_1, max(l, 0.1 * l_1)

        g_2 = g

# %%
UNITS = 'CGS'

# Gravitational Constant
G_CGS = 6.674 * 1e-8 # cm^3 * g^-1 * s^-2
G_SUN = 6.674 * 1e-8 * (7 * 1e10)**-3 * 2 * 1e33 # R^3 * M^-1 * s^-2

# kappa for gamma = 5 / 3
GAMMA_NOT_RELATIVISTIC = 5 / 3
KAPPA_NOT_RELATIVISTIC_CGS = 3.15 * 10**12 # cm^4 * g^-(2/3) * s^-2
KAPPA_NOT_RELATIVISTIC_SUN = 3.15 * 10**12 * (7 * 1e10)**-4 * (2 * 1e33)**(2/3) # R^4 * M^-(2/3) * s^-2

# kappa for gamma = 4 / 3
GAMMA_RELATIVISTIC = 4 / 3
KAPPA_RELATIVISTIC_CGS = 4.9 * 10**14 # cm * g^-(1/3) * s^-2
KAPPA_RELATIVISTIC_SUN = 4.9 * 10**14 * (7 * 1e10)**-1 * (2 * 1e33)**(1/3) # R * M^-(1/3) * s^-2

if UNITS.upper() == 'CGS':
    G = G_CGS
    KAPPA_NOT_RELATIVISTIC = KAPPA_NOT_RELATIVISTIC_CGS
    KAPPA_RELATIVISTIC = KAPPA_RELATIVISTIC_CGS
elif UNITS.upper() == 'SUN':
    G = G_SUN
    KAPPA_NOT_RELATIVISTIC = KAPPA_NOT_RELATIVISTIC_SUN
    KAPPA_RELATIVISTIC = KAPPA_RELATIVISTIC_SUN
else:
    raise Exception()

# %%
# Debugging implementation of generate_F_and_dF

def generate_F_and_dF(n, m, r, gamma, kappa):
    star_constant = (2 * kappa * 3 ** gamma) / ((4 * np.pi) ** (gamma - 1) * G)

    def F(r):
        F = np.zeros(n)
        for i in range(n):
            if i == 0:
                F[i] = star_constant * (r[i] ** 4 / (m[i] * (m[i + 1] - 0))) * (
                            ((m[i + 1] - m[i]) / (r[i + 1] ** 3 - r[i] ** 3)) ** gamma - (
                                (m[i] - 0) / (r[i] ** 3 - 0 ** 3)) ** gamma) + 1
            elif i == n - 1:
                F[i] = star_constant * (r[i] ** 4 / (m[i] * (m[-1] - m[i - 1]))) * (
                    - ((m[i] - m[i - 1]) / (r[i] ** 3 - r[i - 1] ** 3)) ** gamma) + 1
            else:
                F[i] = star_constant * (r[i] ** 4 / (m[i] * (m[i + 1] - m[i - 1]))) * (
                            ((m[i + 1] - m[i]) / (r[i + 1] ** 3 - r[i] ** 3)) ** gamma - (
                                (m[i] - m[i - 1]) / (r[i] ** 3 - r[i - 1] ** 3)) ** gamma) + 1

        #         print('r:', r)
        #         print('F:', F)
        return F

    def dF(r):
        dF = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if j == i:
                    if i == 0:
                        dF[i][j] = star_constant * (r[i] ** 3 / (m[i] * (m[i + 1] - 0))) * (4 * (
                                    ((m[i + 1] - m[i]) / (r[i + 1] ** 3 - r[i] ** 3)) ** gamma - (
                                        (m[i] - 0) / (r[i] ** 3 - 0 ** 3)) ** gamma) + 3 * r[i] ** 3 * gamma * (((m[
                                                                                                                      i + 1] -
                                                                                                                  m[
                                                                                                                      i]) ** gamma / (
                                                                                                                             r[
                                                                                                                                 i + 1] ** 3 -
                                                                                                                             r[
                                                                                                                                 i] ** 3) ** (
                                                                                                                             gamma + 1)) + (
                                                                                                                            (
                                                                                                                                        m[
                                                                                                                                            i] - 0) ** gamma / (
                                                                                                                                        r[
                                                                                                                                            i] ** 3 - 0 ** 3) ** (
                                                                                                                                        gamma + 1))))
                    elif i == n - 1:
                        dF[i][j] = star_constant * (r[i] ** 3 / (m[i] * (m[-1] - m[i - 1]))) * (
                                    4 * (- ((m[i] - m[i - 1]) / (r[i] ** 3 - r[i - 1] ** 3)) ** gamma) + 3 * r[
                                i] ** 3 * gamma * (
                                    ((m[i] - m[i - 1]) ** gamma / (r[i] ** 3 - r[i - 1] ** 3) ** (gamma + 1))))
                    else:
                        dF[i][j] = star_constant * (r[i] ** 3 / (m[i] * (m[i + 1] - m[i - 1]))) * (4 * (
                                    ((m[i + 1] - m[i]) / (r[i + 1] ** 3 - r[i] ** 3)) ** gamma - (
                                        (m[i] - m[i - 1]) / (r[i] ** 3 - r[i - 1] ** 3)) ** gamma) + 3 * r[
                                                                                                       i] ** 3 * gamma * (
                                                                                                               ((m[
                                                                                                                     i + 1] -
                                                                                                                 m[
                                                                                                                     i]) ** gamma / (
                                                                                                                            r[
                                                                                                                                i + 1] ** 3 -
                                                                                                                            r[
                                                                                                                                i] ** 3) ** (
                                                                                                                            gamma + 1)) + (
                                                                                                                           (
                                                                                                                                       m[
                                                                                                                                           i] -
                                                                                                                                       m[
                                                                                                                                           i - 1]) ** gamma / (
                                                                                                                                       r[
                                                                                                                                           i] ** 3 -
                                                                                                                                       r[
                                                                                                                                           i - 1] ** 3) ** (
                                                                                                                                       gamma + 1))))
                elif j == i + 1:
                    if i == 0:
                        dF[i][j] = - star_constant * (
                                    (3 * gamma * r[i] ** 4 * r[i + 1] ** 2) / (m[i] * (m[i + 1] - 0))) * (
                                               (m[i + 1] - m[i]) ** gamma / (r[i + 1] ** 3 - r[i] ** 3) ** (gamma + 1))
                    elif i == n - 1:
                        # Out of bounds
                        pass
                    else:
                        dF[i][j] = - star_constant * (
                                    (3 * gamma * r[i] ** 4 * r[i + 1] ** 2) / (m[i] * (m[i + 1] - m[i - 1]))) * (
                                               (m[i + 1] - m[i]) ** gamma / (r[i + 1] ** 3 - r[i] ** 3) ** (gamma + 1))
                elif j == i - 1:
                    if i == 0:
                        # Out of bounds
                        pass
                    elif i == n - 1:
                        dF[i][j] = - star_constant * (
                                    (3 * gamma * r[i] ** 4 * r[i - 1] ** 2) / (m[i] * (m[-1] - m[i - 1]))) * (
                                               (m[i] - m[i - 1]) ** gamma / (r[i] ** 3 - r[i - 1] ** 3) ** (gamma + 1))
                    else:
                        dF[i][j] = - star_constant * (
                                    (3 * gamma * r[i] ** 4 * r[i - 1] ** 2) / (m[i] * (m[i + 1] - m[i - 1]))) * (
                                               (m[i] - m[i - 1]) ** gamma / (r[i] ** 3 - r[i - 1] ** 3) ** (gamma + 1))

        #         print('r:', r)
        #         print('dF:', dF)
        return dF

    return F, dF


# %%
def find_star_structure(n, M, R, gamma, kappa):
    """
    Calculate the structure of a white dwarf
    Args:
        n (`int`): Number of layers of the dwarf star.
        M (`int` or `float`): The mass of the star (in sun units (i.e. the mass is M * 2 * 10^33 g).
        R (`int` or `float`): Initial guess of the outer radius of the star in sun units (i.e. the radius is R * 7 * 10^10 cm).
        gamma ('float'):
    Returns:
        x (`float`): The x value of the root of the function in the range [a,b].
    Raises:
        ValueError:
    """
    #     if UNITS.upper() == 'CGS':
    #         # Convert to CGS
    #         M = M * 2 * 1e33
    #         R = R * 7 * 1e10

    m = (M / n) * np.arange(1, n + 1, dtype='float64')
    r = (R / n) * np.arange(1, n + 1, dtype='float64')

    #     if gamma == 5 / 3:
    #         kappa = KAPPA_NOT_RELATIVISTIC
    #     elif gamma == 4 / 3:
    #         kappa = KAPPA_RELATIVISTIC
    #     else:
    #         raise Exception('What should kappa be?')

    # kappa = KAPPA_NOT_RELATIVISTIC_SUN
    # G = G_SUN
    F, dF = generate_F_and_dF(n, m, r, gamma, kappa)

    return find_zero(F, dF, r, 1e-4, 1e-4, no_flipping=True)


def calculate_layer_densities(m, r):
    layer_mass = np.zeros(m.size)
    for i in range(m.size):
        if i == 0:
            layer_mass[i] = m[i]
        else:
            layer_mass[i] = m[i] - m[i - 1]

    layer_volume = np.zeros(r.size)
    for i in range(r.size):
        if i == 0:
            layer_volume[i] = (4 * np.pi / 3) * r[i] ** 3
        else:
            layer_volume[i] = (4 * np.pi / 3) * (r[i] ** 3 - r[i - 1] ** 3)

    return layer_mass / layer_volume


def plot_star_density(m, r):
    layer_density = calculate_layer_densities(m, r)

    plt.figure(figsize=(12, 5))
    plt.plot(r, layer_density, '.', label='V')
    plt.title('V')
    plt.grid()

# %%
#find_star_structure(100, 1, 1, (5/3))
M = 2e33
R = 7e10
# kappa = KAPPA_NOT_RELATIVISTIC_SUN
n = 100
# G = G_SUN
gamma = GAMMA_NOT_RELATIVISTIC
kappa = KAPPA_NOT_RELATIVISTIC
rs = find_star_structure(n, M, R, gamma, kappa)
print(rs)
#plot_star_density()
