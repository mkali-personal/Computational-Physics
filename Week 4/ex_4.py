import numpy as np
from warnings import warn
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
import matplotlib.cm as cmx
from pandas import DataFrame

sun_mass = 2e33
sun_radius = 7e10
kappa_regular = 3.15e12
kappa_rel = 4.9e14
gamma_regular = 5/3
gamma_rel = 1.335
G = 6.674e-8
my_cmap = plt.get_cmap('viridis_r')
# %% Define Functions:


def gauss_p(sa,sb):
    ni, nj = sa.shape
    x = np.zeros(ni)
    if ni != nj:
        print('not rectangular matrix')
        return x
    else:
        n = ni
    if n != sb.shape[0]:
        print('no vector b')
        return x
    a = np.zeros([n,n])
    a[:, :] = sa[:]
    b = np.zeros(n)
    b[:] = sb[:]

    eps = np.finfo(np.float).eps * np.amax(np.abs(np.diag(a)))
    for j in range(0, n - 1):
        if abs(a[j, j]) < eps * 1e5:
            if np.amax(np.abs(a[j + 1:n, j])) < eps:
                print('zero pivot')
                return x
            loc = np.argmax(np.abs(a[j + 1:n, j])) + 1
            if j == 0:
                a[j:j + 1 + loc:loc, j:] = a[j + loc::-loc, j:]
                b[j:j + 1 + loc:loc] = b[j + loc::-loc]
            else:
                a[j:j + 1 + loc:loc, j:] = a[j + loc:j - 1:-loc, j:]
                b[j:j + 1 + loc:loc] = b[j + loc:j - 1:-loc]
        for i in range(j + 1, n):
            p = a[i, j] / a[j, j]
            a[i, j + 1:n] -= p * a[j, j + 1:n]
            b[i] -= p * b[j]
    for i in range(n - 1, -1, -1):
        a_i_i = a[i,i]
        b_i_mone = (b[i] - np.sum(a[i, i + 1:n] * x[i + 1:n]))
        if abs(a_i_i) < eps:
            if abs(b_i_mone) < eps:
                x[i] = 0
                warn('Infinite possible values for x[%s], setting it to 0' % i)
            if abs(b_i_mone) >= eps:
                warn('Unsolvable system of equation - No solution. Returning None')
                return None
        else:
            x[i] = (b[i] - np.sum(a[i, i + 1:n] * x[i + 1:n])) / a[i, i]
    return x


def all_plot(data, type, *args):
    # Plots progress of the root function
    global fig, ax
    # For question 2 - Plot the two points with the style specified in *args:
    if type == '2d function':
        for a in ax:
            a.plot(data[0], data[1] , *args)
    # if the tyep is star, plot the star density graph and the outer radius progression:
    elif type == 'star':
        rs = data['r']
        outer_radius = rs[-1]
        rs = rs / np.max(rs)
        ax[0].clear()
        ax[0].set_xlim((-1.2, 1.2))
        ax[0].set_ylim((-1.2, 1.2))
        chosen_layers = np.concatenate((rs[::5], [rs[-1]]))
        cNorm = colors.Normalize(vmin=np.min(chosen_layers), vmax=np.max(chosen_layers))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap)
        for r in reversed(chosen_layers):
            colorVal = scalarMap.to_rgba(r)
            circle = plt.Circle((0, 0), r, color=colorVal)
            ax[0].add_artist(circle)
        global line1
        current_data = line1.get_data()
        line1.set_data(np.append(current_data[0], current_data[0][-1]+1), np.append(current_data[1], outer_radius))
        ax[1].set_xlim((1, current_data[0][-1]+1.5))
        ax[1].set_ylim((0, max(current_data[1])*1.1))


    fig.show()
#     fig.canvas.draw()


def f(F_value):
    return (1/2) * np.dot(F_value, F_value)


def cube_guess(g, g_0, g_0_der, l_1, g_l_1, l_2, g_l_2):
    mat = np.array([
                    [1/l_1**2, -1/l_2**2],
                    [-l_2/l_1**2, l_1/l_2**2]
                  ])
    vec = np.array([g_l_1-g_0_der*l_1 - g_0,
                    g_l_2-g_0_der*l_2 - g_0
                  ])
    coeffs = ( 1/(l_1 - l_2) ) * ( np.dot(mat, vec) )
    a = coeffs[0]
    b = coeffs[1]

    l_new = (-b+np.sqrt(b**2 - 3*a*g_0_der)) / (3*a)
    l_new = max(l_new, 1/2)
    l_new = min(l_new, 0.1)
    g_l_new = g(l_new)

    if g_l_new <= g_0 + 1e-4 * g_0_der * l_new:
        return l_new, g_l_new
    else:
        return cube_guess(g, g_0, g_0_der, l_new, g_l_new, l_1, g_l_1)


def find_1d_min(g, g_0, g_0_der, g_1):
    # x = scipy.optimize.fminbound(g, 0, 1)
    # return x, g(x)
    guess = - g_0_der / (2 * (g_1 - g_0 - g_0_der))
    guess = max(guess, 0.1)
    g_guess = g(guess)
    if g_guess <= g_0 + 1e-4 * g_0_der * guess:
        return guess, g_guess
    else:
        return cube_guess(g, g_0, g_0_der, guess, g_guess, 1, g_1)


def fix_ordered_coeff(x, step, boundary):
    x = np.insert(x, 0, 0)
    step = np.insert(step, 0, 0)
    x_new = x + step
    if min(x_new[2:] - x_new[1:-1]) >= boundary:
        return x_new[1:], step[1:]
    else:
        # This is a boolean vector of the elements which passed (upwards) their next levels
        filter = np.concatenate(([False], x_new[2:] - x_new[1:-1] < boundary, [False]))
        # This is a boolean vector of the elements which passed (upwards) their next levels
        filter_p1 = np.insert(filter[:-1], False, 0)
        c = min((x[filter_p1] - x[filter] - boundary) / (step[filter] - step[filter_p1]))*0.9
        return fix_ordered_coeff(x[1:], c*step[1:], boundary)


def find_root_nd(F, dF, x, eps, boundary=None, visualization_type=None, level=1, max_iter=1000):
    F_current = F(x)
    f_current = f(F_current)
    # print(level, '   ', f_current, '   ', x[-1])
    if visualization_type == 'star':
        all_plot({'f': f_current, 'F': F_current, 'r': x}, visualization_type, 'r+')
    J = dF(x)
    step = -gauss_p(J, F_current)
    x_new = x + step
    if boundary is not None:
        x_new, step = fix_ordered_coeff(x, step, boundary)
    if visualization_type == '2d function':
        all_plot(x_new, visualization_type, 'r+')
    F_new = F(x_new)
    f_new = f(F_new)
    if f_new > f_current:
        g = lambda t: f(F(x + t*step))
        g_0_der = np.dot(np.dot(F_current, J), step)
        t, f_new = find_1d_min(g, f_current, g_0_der, f_new)
        x_new = x + t * step
        if visualization_type == '2d function':
            all_plot(x_new, visualization_type, 'mo')
    if (f_new < eps) or (level == max_iter):
        return x_new
    else:
        level += 1
        return find_root_nd(F, dF, x_new, eps, boundary, visualization_type, level=level, max_iter=max_iter)


def gigantic_coeff(m, G, gamma, kappa, n):
    indices = np.linspace(1, n, n)
    C = 3**gamma * (4*np.pi)**(1-gamma) * kappa * m**(gamma-2)/ (G * indices)
    C[-1] = 2 * C[-1]
    return C


def create_initial_rs(R, n):
    return np.linspace(R/n, R, n)


def initialize_star_problem(n, M, R, gamma, kappa):
    M = M * sun_mass
    R = R * sun_radius
    m = M / n
    C = gigantic_coeff(m, G, gamma, kappa, n)


    def F(r):
        r_i_p1 = np.append(r[1:], r[-1] * 2)  # The last value is a dummy value and will not have effect on the results.
        r_i_m1 = np.insert(r[:-1], 0, 0)
        special_nth_filter = np.append(np.ones(n - 1), 0)
        F_value = 1 + C * r**4 * (
                                 (r_i_p1**3 - r**3)**(-gamma) * special_nth_filter - (r**3 - r_i_m1**3)**(-gamma)
                                )
        return F_value


    def dF(r):
        # Calculate derivatives of F_{n} separately as they have different fourmula::
        dFn_drn = C[-1] * r[-1]**3 * (r[-1]**3 - r[-2]**3)**(-gamma) * (3*gamma*r[-1]**3 / (r[-1]**3 - r[-2]**3) - 4)
        dFn_drn_m1 = -3*gamma * C[-1] * r[-2]**2 * r[-1] **4 * (r[-1]**3 - r[-2]**3)**(-gamma-1)
        # Calculate elements where i <!=> n:
        # lead of the vector: r_{i+1}, i\in{2..n} (1..n-1 in python)
        r_i_p1 = r[1:]
        # lag of the vector: r_{i-1}, i\in{0..n-2} (-1..n-3 in python)
        r_i_m1 = np.insert(r[:-2], 0, 0)
        # For the next calculations as we calculated already dFn_dr, we dont want r_{n} in the vector:
        r_i = r[:-1]
        C_i = C[:-1]
        # r_{i+1}^{3}-r_{i}^{3} and r_{i}^{3}-r_{i-1}^{3} are common and therefore calculated once here:
        delta_cubed_p1 = r_i_p1**3 - r_i**3
        delta_cubed_m1 = r_i**3 - r_i_m1**3


        dFi_dri = C_i * r_i**3 * (
                          4*(delta_cubed_p1**(-gamma) - delta_cubed_m1**(-gamma)) +
                          3*gamma * r_i**3 * (
                                              delta_cubed_p1 ** (-gamma - 1) + delta_cubed_m1 ** (-gamma - 1)
                                             )
                                  )
        dFi_dri_m1 = -3*C_i * gamma * r_i_m1**2 * r_i**4 * delta_cubed_m1**(-gamma - 1)
        # Ignore dF_1 / dr_0:
        dFi_dri_m1 = dFi_dri_m1[1:]
        dFi_dri_p1 = -3*C_i * gamma * r_i_p1**2 * r_i**4 * delta_cubed_p1**(-gamma - 1)

        J = np.pad(np.diag(dFi_dri_m1), ((1, 1), (0, 2))) + \
            np.pad(np.diag(dFi_dri_p1), ((0, 1), (1, 0))) + \
            np.pad(np.diag(dFi_dri), (0, 1))
        J[-1, -1] = dFn_drn
        J[-1, -2] = dFn_drn_m1

        return J

    r_initial = create_initial_rs(R, n)
    return F, dF, r_initial

def solve_star_problem(n, M, R, gamma, kappa, visualization_type=None, max_iter=1000):
    F_star, dF_star, r_initial = initialize_star_problem(n, M, R, gamma, kappa)
    if visualization_type is not None:
        global fig, ax, line1
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_title('Density Visualization')
        ax[0].set_title('Density Visualization')
        ax[1].set_title('Outer Radius')
        ax[1].set_xlabel(r'Iteration', usetex=True)
        ax[1].set_ylabel(r'Radius [cm]', usetex=True)
        line1, = ax[1].plot(0, R * sun_radius, 'b-')
        cNorm = colors.Normalize(vmin=0, vmax=1)
        star_colorbar = fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=my_cmap), ax=ax[0])
        star_colorbar.set_label('Radius Normed')
    rs = find_root_nd(F_star, dF_star, r_initial, eps, boundary, visualization_type=visualization_type, max_iter=max_iter)
    return rs


# %% Question 2:


def F(x):
    return np.array([x[0]**2 +2*x[1] - 6, -x[0]**2+5*x[1]**3-1])


def dF(x):
    return np.array([[2*x[0], 2], [-2*x[0], 15*x[1]**2]])


xlist = np.linspace(-5, 5, 200)
ylist = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(xlist, ylist)
F0 = X**2 + 2*Y - 6
F1 = -X**2 + 5*Y**3 - 1

# fig, ax = plt.subplots(1,2, subplot_kw={'projection': '3d'})
# ax[0].plot_surface(X, Y, F0,cmap='viridis', edgecolor='none')
# ax[0].set_title('F0')
# ax[1].plot_surface(X, Y, F1,cmap='viridis', edgecolor='none')
# ax[1].set_title('F1')
# x_0 = [0.4, 0.3]
# F_0 = F(x_0)
# ax[0].plot([x_0[0]], [x_0[1]], [F_0[0]], 'ro')
# ax[1].plot([x_0[0]], [x_0[1]], [F_0[1]], 'ro')
# ax[0].view_init(elev=40., azim=120)
# ax[1].view_init(elev=40., azim=120)
# plt.show()

fig, ax = plt.subplots(1,2, subplot_kw={'projection': '3d'})
fig, ax = plt.subplots(1,2)
ax[0].contourf(X, Y, F0, cmap='viridis', levels = 20)
ax[0].set_title('F0')
ax[1].contourf(X, Y, F1, cmap='viridis', levels = 20)
ax[1].set_title('F1')
ax[0].plot(2, 1, 'bo')
ax[1].plot(2, 1, 'bo')
plt.show()

eps = 0.001
x_initial = np.array([0.4, 0.3])
all_plot(x_initial, '2d function', 'y+')

s = find_root_nd(F, dF, x_initial, eps, visualization_type='2d function')

d = scipy.optimize.fsolve(F, x_initial)
print(s, '\n', d)

# %% Question 5:

M = 1
gamma = gamma_regular
n = 100
R = 1
boundary = 1e3
eps = 0.001
kappa = kappa_regular

rs = solve_star_problem(n, M, R, gamma, kappa, visualization_type='star')
rs = np.insert(rs, 0, 0)
m = M/n * sun_mass
ro = m / (4*np.pi / 3 * (rs[1:]**3 - rs[:-1]**3))
fig_5, ax5 = plt.subplots()
ax5.plot(rs[:-1], ro)
ax5.set_xlabel(r'$r\left[cm^{3}\right]$', usetex=True)
ax5.set_ylabel(r'$\rho\left[\frac{gr}{cm^{3}}\right]$', usetex=True)
ax5.set_title(r'density as a function of the radius', usetex=True)
fig_5.show()

df_5 = DataFrame({'Radius': rs[:-1],
                  'Density': ro})
df_5


# %% Question 6:
k = 8
M = 1
gamma = gamma_regular
n = 100
R = 0.01
kappa = kappa_regular
boundary = 1e5
eps = 0.001

R_s = np.empty(k)
Ms = np.linspace(0.5, 5, k)

for i, M in enumerate(Ms):
    rs = solve_star_problem(n, M, R, gamma, kappa, visualization_type=None)
    R_s[i] = rs[-1]

fig_6, ax6 = plt.subplots()
ax6.set_xlabel(r'$M$', usetex=True)
ax6.set_ylabel(r'$RM^{\frac{1}{3}}$', usetex=True)
ax6.set_title(r'$RM^{\frac{1}{3}}$ trend', usetex=True)
y_values = R_s * Ms**(1/3) / 7e10
ax6.set_ylim((0, max(y_values) * 1.1))
ax6.plot(Ms, y_values)
fig_6.show()

df_6 = DataFrame({'Samples Average': [np.mean(y_values)],
                  'Samples Standard Deviation': [np.std(y_values)],
                  'Approximated Value': [0.006],
                  })
df_6

# %% Question 7:
k=7
gamma = gamma_rel
n = 100
R = 100
boundary = 1e3
eps = 1e-17
R_s = np.empty(k)
Ms = np.linspace(1.40, 1.46, k)
kappa = kappa_rel

for i, M in enumerate(Ms):
    # Plot only first example:
    if i == 0:
        rs = solve_star_problem(n, M, R, gamma, kappa, visualization_type='star', max_iter=40)
    else:
        rs = solve_star_problem(n, M, R, gamma, kappa, visualization_type=None, max_iter=40)
    R_s[i] = rs[-1] / sun_radius


fig_7, ax7 = plt.subplots()
ax7.set_xlabel(r'$M$', usetex=True)
ax7.set_ylabel(r'$R$', usetex=True)
ax7.set_title(r'$R$ as a function of $M$ - Relativistic', usetex=True)
ax7.set_ylim((0, max(R_s) * 1.1))
ax7.plot(Ms, R_s)
fig_7.show()

df_7 = DataFrame({'Radius': Ms,
                  'Density': R_s})
df_7


# %% Question 8:
n = 100
R = 0.006
boundary = 1e1
eps = 0.000001
M_rel = 1.42
r_indices = np.arange(n)+1

F_star_reg, dF_star_reg, r_initial_reg = initialize_star_problem(n, M=1, R=R, gamma=gamma_regular, kappa=kappa_regular)
r_mine = find_root_nd(F_star_reg, dF_star_reg, r_initial_reg, eps, boundary, visualization_type=None)
r_scipy = scipy.optimize.fsolve(F_star_reg, r_initial_reg, fprime=dF_star_reg)

F_star_rel, dF_star_rel, r_initial_rel = initialize_star_problem(n, M=M_rel, R=100, gamma=gamma_rel, kappa=kappa_rel)
r_rel_mine = find_root_nd(F_star_rel, dF_star_rel, r_initial_rel, eps, boundary, visualization_type=None)
r_rel_scipy = scipy.optimize.fsolve(F_star_rel, r_rel_mine, fprime=dF_star_rel)


fig8, ax8 = plt.subplots(1,2, figsize=(10,5))
ax8[0].set_xlabel('Accumulating mass [gr]')
ax8[0].set_ylabel('Radius [cm]')
ax8[0].set_title('Basic State')
ax8[0].plot(r_indices[::4] * 1 / max(r_indices), r_mine[::4] / sun_radius, linewidth=2, label='My function')
ax8[0].plot(r_indices[::4] * 1 / max(r_indices), r_scipy[::4] / sun_radius, '.', label='Scipy function')
ax8[0].legend()

ax8[1].set_xlabel('Accumulating mass [solar units]')
ax8[1].set_ylabel('Radius [solar units]')
ax8[1].set_title('Relativistic State')
ax8[1].plot(r_indices[::4] * 1.42 / max(r_indices), r_rel_mine[::4] / sun_radius, linewidth=2, label='My function')
ax8[1].plot(r_indices[::4] * 1.42 / max(r_indices), r_rel_scipy[::4] / sun_radius, '.', label='Scipy function')
ax8[1].legend()
fig8.show()

df_8 = DataFrame({'Scipy Radius': [r_scipy[-1], r_rel_scipy[-1]],
                  'My Radius': [r_mine[-1], r_rel_mine[-1]],
                  'State': ['Basic State', 'Relativistic State']})
df_8.set_index('State', inplace=True)
df_8

