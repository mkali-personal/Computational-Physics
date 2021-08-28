# Import Modules:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pandas import DataFrame
#%matplotlib inline

# Set global variables:
nice_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
N = 1e5
I_0 = 1
R_0 = 3


# Copy-Paste functions from Exercise 3 for function's root finding:
def bisection_root(f, a, b, epsx, epsf):
    h = (b - a) / 2
    x_mid = a + h  # (=0.5*(a+b))
    f_mid = f(x_mid)
    sign_f_mid = np.sign(f_mid)

    if h < epsx or abs(f_mid) < epsf:  # Break recursion if desired approximation achieved:
        return x_mid
    elif sign_f_mid == 1:
        return bisection_root(f, a, x_mid, epsx, epsf)
    elif sign_f_mid == -1:
        return bisection_root(f, x_mid, b, epsx, epsf)
    else:
        raise Exception('Debbug me')


def secant_root(f, a, b, epsx, epsf, x_n=None, x_n_1=None, f_x_n=None, f_x_n_1=None):
    # Here x_n_1 denotes x_{n-1}
    # If previous-previous point does not exist, take a as x_{n-1} and the first quarter of [a,b] as x_n:
    # (This should happen only in the first iteration of the function)
    print(a, b)
    if x_n_1 is None:
        x_n_1 = a
        f_x_n_1 = f(x_n_1)
    if x_n is None:
        x_n = a + (b - a) / 4
        f_x_n = f(x_n)

    # If the desired precision achieved, return the previous point:
    if abs(x_n - x_n_1) < epsx or abs(f_x_n) < epsf:
        return x_n

    # Estimate root's location based on NR method:
    x_next = x_n - f_x_n * (x_n - x_n_1) / (f_x_n - f_x_n_1)

    # If the prediction for the next x is out of range, use the mid of [a,b] instead:
    if abs(x_next - x_n) > 0.9 * (b - a) or x_next > b or x_next < a:  # NOTE ARBITRARY VALUE
        x_next = (a + b) / 2

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
        print('f(a) must have the opposite sign of f(b), returning None')
        return None
    if type == 'bisection':
        return bisection_root(f, a, b, epsx, epsf)
    elif type == 'secant':
        return secant_root(f, a, b, epsx, epsf)
    else:
        return bisection_root(f, a, b, epsx, epsf)


# %% Define funtions:


def rk_step_3(f, x, y, h):
    k_1 = h * f(x, y)
    k_2 = h * f(x + (1 / 2) * h, y + (1 / 2) * k_1)
    k_3 = h * f(x + h, y + 2 * k_2 - k_1)
    return y + (1 / 6) * (k_1 + 4 * k_2 + k_3)


def rk_step_4(f, x, y, h):
    k_1 = h * f(x, y)
    k_2 = h * f(x + (1 / 2) * h, y + (1 / 2) * k_1)
    k_3 = h * f(x + (1 / 2) * h, y + (1 / 2) * k_2)
    k_4 = h * f(x + h, y + k_3)
    return y + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def test_h(rk_step_func, f, x, y, h):
    # Tries one step with size h, two steps with size h/2, and compare.
    # Return their maximum difference and weighted average:

    # One step:
    y_1step = rk_step_func(f, x, y, h)

    # Two steps:
    y_2steps_mid = rk_step_func(f, x, y, h / 2)
    y_2steps = rk_step_func(f, x + h / 2, y_2steps_mid, h / 2)

    # Their difference:
    diff = max(abs(y_1step - y_2steps))

    # wighted average according to the order:
    if rk_step_func == rk_step_4:
        y_step = (16 / 15) * y_2steps - (1 / 15) * y_1step
    elif rk_step_func == rk_step_3:
        y_step = (8 / 7) * y_2steps - (1 / 7) * y_1step

    # Return their difference and (x+h, y(x+h)):
    return diff, x + h, y_step


def adjust_step_size(rk_step_func, f, x, y, h, eps, x_end):
    # Tries out several step sizes h and returns ideal h and its resulted (x+h,y(x+h))

    if rk_step_func == rk_step_3:
        # Since we are using the double step to increase the degree of the precision by one,
        # the rk_step_3 is treated as fourth degree precision and the rk_step_4 is treated as fifth degree precision
        # The values are set so that after increasing or decreasing h size the error will grow\shrink by +-30%
        decrease_coeff, increase_coeff = 0.7**(1/5), 1.3**(1/5)
    elif rk_step_func == rk_step_4:
        decrease_coeff, increase_coeff = 0.7**(1/6), 1.3**(1/6)

    # Shorten step if it's about to go beyond x_end
    if x + h > x_end:
        h = x_end - x

    # Values to compare the step's error with:
    delta = eps * h / (x_end - x)
    mu, eta = delta / 2, delta / 4
    diff, x_step, y_step = test_h(rk_step_func, f, x, y, h)
    found = 0

    while found == 0:
        # Make h smaller and smaller until the desired precision is achieved:
        if diff > delta:
            h = h / 2
            delta = eps * h / (x_end - x)
            diff, x_step, y_step = test_h(rk_step_func, f, x, y, h)
        # When h is small enough, stop iterating, adjust h size, and shrink epsilon
        else:
            found = 1
            eps = eps - diff
            if mu < diff <= delta:
                h = decrease_coeff * h
            # elif eta < diff <= mu: # In this case nothing should happen and it's here only for readability
            #     h = h
            elif diff <= eta:
                h = increase_coeff * h

    return x_step, y_step, h, eps


def runge_kutta(f, x_0, y_0, x_end, eps, h=None, save_graph=1, order=4):  # ,
    # Calculate ODE using runge kutta method

    # Adjust the rungae kutta step function according to the desired order:
    if order == 4:
        rk_step_func = rk_step_4
    elif order == 3:
        rk_step_func = rk_step_3

    # If there is no h as an input, use 0.01 of the interval as first guess:
    if h is None:
        h = (x_end - x_0) / 100

    if save_graph:
        graph = np.insert(y_0, 0, x_0)

    x_n = x_0
    y_n = y_0
    reached_x_end = 0

    # As long as the x_end was not reached iterate again and again until while continuously adjust epsilon and h:
    while not reached_x_end:
        x_n, y_n, h, eps = adjust_step_size(rk_step_func, f, x_n, y_n, h, eps, x_end)
        if abs(x_end - x_n) < 1e-30:  # Float comparison
            reached_x_end = 1
        if save_graph:
            graph = np.vstack((graph, np.insert(y_n, 0, x_n)))

    if save_graph:
        return y_n, graph
    else:
        return y_n


def run_uniform_epidemic_simulation(beta, gamma, N, t_0, t_end, I_0, recovered=0, eps=0.001):
    # Initiate inital condition using the initial infected and recovered population
    y_0 = [N - recovered - I_0, I_0]

    # function to calulate ODE derivatives:
    def f_uniform(t, y):
        # format is y = [S, I]
        dS_dt = - beta * y[0] * y[1] / N
        dI_dt = beta * y[0] * y[1] / N - gamma * y[1]
        return np.array([dS_dt, dI_dt])

    # Calculate pandemic evolution using runge_kutta function:
    _, graph = runge_kutta(f_uniform, t_0, y_0, t_end, eps)
    # Add recovered column:
    graph = np.c_[graph, N - graph[:, 1:3].sum(axis=1)]

    return graph


def run_diverse_epidemic_sumulation(gamma, N, t_0, t_end, d, C_V, R_0, I_0_vec,
                                    recovered_vec=np.array([0, 0, 0]), eps=0.001):
    # Calculate population properties:
    alpha = 3 * C_V ** 2 / (2 * d ** 2)
    x = np.array([1 - d, 1, 1 + alpha * d])
    s_m = 2 * alpha / (3 * (1 + alpha))
    s_p = 2 / (3 * (1 + alpha))
    n = np.array([N * s_m, (1 / 3) * N, N * s_p])
    y_0 = np.concatenate((n - recovered_vec - I_0_vec, I_0_vec))
    beta = gamma * R_0 * 1 / (1 / N * sum(n * x ** 2))

    # function to calulate ODE derivatives:
    def f_diverse(t, y):
        # format is y = [S1, S2, S3, I1, I2, I3]
        xI_sum = sum(x * y[3:])
        dS_dt = - beta * (y[:3] * x) * xI_sum / N
        dI_dt = - dS_dt - gamma * y[3:]
        return np.concatenate((dS_dt, dI_dt))

    # Calculate pandemic evolution using runge_kutta function:
    _, graph = runge_kutta(f_diverse, t_0, y_0, t_end, eps)

    # Add total recovered column:
    graph = np.c_[graph, N - graph[:, 1:].sum(axis=1)]

    # return the graph results with the relative population sizes:
    return graph, n


def herd_immunity_uni(beta, gamma, N, t_0, t_end, outbreak_threshold=0.2, I_0=1, precision=0.001, type='bisection'):
    # Searches for initial conditions such that no more than [outbreak_threshold] of the population is infected

    # Function to calculate how much of the population is infected given [initial_recovered] recovered people
    # and 1 infected
    def total_infected_portion(initial_recovered):
        graph = run_uniform_epidemic_simulation(beta=beta,
                                                gamma=gamma,
                                                N=N,
                                                t_0=t_0,
                                                t_end=t_end,
                                                I_0=I_0,
                                                recovered=initial_recovered)
        return outbreak_threshold - (graph[0, 1] - graph[-1, 1]) / N

    # Find value with which the total_infected_portion is exactly outbreak_threshold:
    a = x_root(f=total_infected_portion, a=0, b=N * (1 - outbreak_threshold) - 1, epsx=precision, epsf=precision,
               type=type)
    return a


def herd_immunity_div(gamma, N, t_0, t_end, worst_day_portions, d, C_V, R_0,
                      I_0_vec=np.array([1, 1, 1]), outbreak_threshold=0.2, precision=1, type='bisection'):
    # Searches for initial conditions such that no more than [outbreak_threshold] of the population is infected

    # Function to calculate how much of the population is infected given [initial_recovered] recovered people
    # and 1 infected
    def total_infected_portion(initial_recovered):
        # Initiate recovered people vector using the amount of initialy recovered and the portions between the groups:
        recovered_vec = initial_recovered * worst_day_portions
        graph, _ = run_diverse_epidemic_sumulation(gamma=gamma,
                                                   N=N,
                                                   t_0=t_0,
                                                   t_end=t_end,
                                                   d=d,
                                                   C_V=C_V,
                                                   R_0=R_0,
                                                   I_0_vec=I_0_vec,
                                                   recovered_vec=recovered_vec)
        return outbreak_threshold - (graph[0, 1:4].sum() - graph[-1, 1:4].sum()) / N

    # Find value with which the total_infected_portion is exactly outbreak_threshold:
    return x_root(f=total_infected_portion, a=0, b=N * (1 - outbreak_threshold) - 1, epsx=precision, epsf=precision,
                  type=type)


def plot_uniform_epidemic(ax, graph):
    ax.plot(graph[:, 0], graph[:, 1], color=nice_colors[1], label="Susceptible")
    ax.plot(graph[:, 0], graph[:, 2], color=nice_colors[0], label="Infectious")
    ax.plot(graph[:, 0], graph[:, 3], color=nice_colors[2], label="Recovered")
    ax.legend()
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of People')


def plot_diverse_epidemic(ax, graph):
    names = ['S_1', 'S_2', 'S_3', 'I_1', 'I_2', 'I_3']
    styles = ['-', '-', '-', '-.', '-.', '-.']
    colors = [nice_colors[0], nice_colors[1], nice_colors[2], nice_colors[0], nice_colors[1], nice_colors[2]]
    # line_widths = [1, 1, 1, 0.5, 0.5, 0.5]

    for i in np.arange(6):
        ax.plot(graph[:, 0], graph[:, i + 1], styles[i], color=colors[i],
                label=names[i])  # , linewidth=line_widths[i]

    ax.plot(graph[:, 0], graph[:, 1:4].sum(axis=1), '-', color=(0.9, 0.9, 0.9), label='Total Susceptible')
    ax.plot(graph[:, 0], graph[:, 4:7].sum(axis=1), '-.', linewidth=0.8, color=(0.9, 0.9, 0.9),
            label='Total Infectious')
    ax.plot(graph[:, 0], graph[:, 7], '--', color=(0.9, 0.9, 0.9), label='Total Recovered')

    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(prop=fontP)
    ax.set_label('Days')
    ax.set_ylabel('Number of People')


# %% Qeustion 2


eps = 0.001
beta = 0.75
gamma = 0.25
t_0 = 0
t_end = 50
y_0 = [N - I_0, I_0]


def f(t, y):
    # format is y = [S, I]
    dS_dt = - beta * y[0] * y[1] / N + gamma * y[1]
    dI_dt = - dS_dt
    return np.array([dS_dt, dI_dt])


def analytic_solution(t):
    mone = N * (1 - gamma / beta)
    mechane = 1 + (N / I_0 * (1 - gamma / beta) - 1) * np.exp((gamma - beta) * t)
    return mone / mechane


y_n, graph = runge_kutta(f, t_0, y_0, t_end, eps)

fig_2, ax_2 = plt.subplots(figsize=(10,6))

ax_2.plot(graph[:, 0], graph[:, 1], color=nice_colors[1], label="Numerical Susceptible")
ax_2.plot(graph[:, 0], graph[:, 2], color=nice_colors[0], label="Numerical Infectious")
t_dummy = np.linspace(t_0, t_end, 40)
ax_2.plot(t_dummy, analytic_solution(t_dummy), '.', color=nice_colors[3], label="Analytical Infectious")
ax_2.legend()
ax_2.set_xlabel('Days')
ax_2.set_ylabel('Number of People')
ax_2.set_title('Numerical Vs. Analytical Solution')
fig_2.show()
print('relative error: ', analytic_solution(graph[-1, 0]) / graph[-1, 2] - 1)

# %% Question 3

epss = [1e-3, 1e-5, 1e-7, 1e-9]
orders = [3, 4]

a = 0.1
b = [-5, -1, 1, 4, 7]
t_0 = -10
t_end = 10
y_0 = 0


def f(x, y):
    f_x = [1 / (a * np.sqrt(np.pi)) * np.exp(-(x - b[i]) ** 2 / a ** 2) for i in range(len(b))]
    global f_call_counter
    f_call_counter += 1
    return np.array([sum(f_x)])


df_3 = DataFrame(columns=['RK order', 'Requested accuracy', 'No. of steps', 'No, of function calls', 'Relative error'])

for order in orders:
    print('Now calculating with order = ', order)
    for eps in epss:
        f_call_counter = 0
        # steps_counter = 0
        y_n, graph = runge_kutta(f, t_0, y_0, t_end, order=order, eps=eps)
        df_3.loc[len(df_3)] = [order, eps, len(graph) - 1, f_call_counter, y_n / 5 - 1]

df_3

# %% Question 4

beta = 0.3
gamma = 0.1
t_0 = 0
t_end = 150

graph_4 = run_uniform_epidemic_simulation(beta, gamma, N, t_0, t_end, I_0)

fig_4, ax_4 = plt.subplots(figsize=(10,6))
plot_uniform_epidemic(ax_4, graph_4)
ax_4.set_title('Uniform Population Epidemic Evolution')
fig_4.show()

print('Recovered people Percentage:', (N - graph_4[-1, 1:3].sum()) / N)

# %% Question 5


gamma = 0.1
t_0 = 0
t_end = 150
d = 0.8
C_V = 1
I_0_vec = [0, 0, 1]

graph_5, n_5 = run_diverse_epidemic_sumulation(gamma, N, t_0, t_end, d, C_V, R_0, I_0_vec)

fig_5a, ax_5a = plt.subplots(figsize=(10,6))
plot_diverse_epidemic(ax_5a, graph_5)
ax_5a.set_title('Diverse Population Epidemic Evolution')
ax_5a.axvline(47.5, linewidth=0.8, color=(0.7, 0.7, 0.7))
fig_5a.show()

print('Recovered people Percentage:', graph_5[-1, 7] / N)

fig_5b, ax_5b = plt.subplots(figsize=(10,6))
ax_5b.plot(graph_4[:, 0], graph_4[:, 1], '-', color=nice_colors[0], label="Susceptible Uniform")
ax_5b.plot(graph_4[:, 0], graph_4[:, 2], '-.', linewidth=0.8, color=nice_colors[0], label="Infectious Uniform")
ax_5b.plot(graph_4[:, 0], graph_4[:, 3], '--', color=nice_colors[0], label="Recovered Uniform")
ax_5b.plot(graph_5[:, 0], graph_5[:, 1:4].sum(axis=1), '-', color=nice_colors[3], label='Susceptible Diverse')
ax_5b.plot(graph_5[:, 0], graph_5[:, 4:7].sum(axis=1), '-.', linewidth=0.8, color=nice_colors[3], label='Infectious Diverse')
ax_5b.plot(graph_5[:, 0], graph_5[:, 7], '--', color=nice_colors[3], label='Recovered Diverse')
fontP = FontProperties()
fontP.set_size('small')
ax_5b.legend(prop=fontP)
ax_5b.legend()
ax_5b.axvline(47.5, linewidth=0.8, color=(0.7, 0.7, 0.7))
ax_5b.set_label('Days')
ax_5b.set_ylabel('Number of People')
ax_5b.set_title('Diverse Popul. Epidemic Vs. Uniform Popul. Epidemic')
fig_5b.show()

# %% Question 6

fig_6, ax_6 = plt.subplots(figsize=(10,6))
ax_6.plot(graph_4[5:, 0], graph_4[5:, 3], '--', color=nice_colors[0], label="Recovered Uniform")
ax_6.plot(graph_5[5:, 0], graph_5[5:, 7], '--', color=nice_colors[3], label='Recovered Diverse')
ax_6.legend()
ax_6.set_label('Days')
ax_6.set_ylabel('Number of People')
ax_6.set_title('Recovered - Uniform Vs. Diverse')
ax_6.set_yscale('log')
fig_6.show()


max_days = 40

filter_4 = (graph_4[:, 0] < max_days) & (2 <= graph_4[:, 3])  # np.ones(len((graph_4[:, 0])), dtype=bool)
filter_5 = (graph_5[:, 0] < max_days) & (2 <= graph_5[:, 7])  # np.ones(len((graph_5[:, 0])), dtype=bool)
poly_4 = np.polyfit(graph_4[filter_4, 0], np.log(graph_4[filter_4, 3]), 1)
poly_5 = np.polyfit(graph_5[filter_5, 0], np.log(graph_5[filter_5, 7]), 1)
t_dummy = np.linspace(min(graph_4[filter_4, 0]), max_days, 100)
# fig_test, ax_test = plt.subplots()
# ax_test.plot(graph_4[filter_4, 0], np.log(graph_4[filter_4, 3]))
# ax_test.plot(graph_5[filter_5, 0], np.log(graph_5[filter_5, 7]))
# ax_test.plot(t_dummy, poly_4[1] + t_dummy * poly_4[0])
# ax_test.plot(t_dummy, poly_5[1] + t_dummy * poly_5[0])
# fig_test.show()

print('The typical doubling time for the uniform population (during the first stage of the outbreak) is {:.2f} while the typical doubling time for the diverse '
      'population is {:.2f}.'.format(np.log(2) / poly_4[0], np.log(2) / poly_5[0]))

# %% Question 7


beta = 0.3
gamma = 0.1
t_0 = 0
t_end = 1250
outbreak_threshold = 0.2

worst_day_4 = np.argmax(graph_4[:, 2])
worst_day_non_sensitive_4 = graph_4[worst_day_4, 1:3].sum()

threshold_7 = herd_immunity_uni(beta, gamma, N, t_0, t_end, I_0=1, precision=0.001,
                                outbreak_threshold=outbreak_threshold)

graph_7 = run_uniform_epidemic_simulation(beta=beta,
                                          gamma=gamma,
                                          N=N,
                                          t_0=t_0,
                                          t_end=500,
                                          I_0=I_0,
                                          recovered=threshold_7)
fig_7, ax_7 = plt.subplots(figsize=(10,6))
plot_uniform_epidemic(ax_7, graph_7)
ax_7.set_title('Epidemic Evolution with Minimal Herd Immunity')
fig_7.show()

print('With {:.1f}% Initialy non-sensitive (recovered) population, the total infected people'
      ' is {:.1f}%. \nThis is the herd-immunity threshold.'.format(threshold_7 / N * 100, outbreak_threshold * 100))

print('With 0 Initially non-sensitive people (as in question 4), the day with the '
      'most infected people has already {:.1%} which\nis {:.1f} times the number required for '
      'herd immunity'.format(worst_day_non_sensitive_4 / N, worst_day_non_sensitive_4 / threshold_7))

# %% Question 8:

C_Vs = [0.5, 1, 1.5, 2]
eps = 0.001
gamma = 0.1
t_0 = 0
t_end = 1250
d = 0.8
outbreak_threshold = 0.2

# Extract aggregated values from the question 5 results:
# Day with most infected people:
graph_5_aggr = {'worst_day_5': np.argmax(graph_5[:, 4:7].sum(axis=1), axis=0)}
# Number of non-sensitive  in the worst day (e.g. not Susceptible):
graph_5_aggr['worst_day_non_sensitive'] = np.array(n_5 - graph_5[graph_5_aggr['worst_day_5'], 1:4])
# The relative portions of the infected groups:
graph_5_aggr['worst_day_portions'] = graph_5_aggr['worst_day_non_sensitive'] / sum(
                                                graph_5_aggr['worst_day_non_sensitive'])
graph_5_aggr['worst_day_non_sensitive_percentage'] = sum(graph_5_aggr['worst_day_non_sensitive']) / N

thresholds_8 = np.zeros([len(C_Vs)])

print('Calculating C_V = ', end='')
for i, C_V in enumerate(C_Vs):
    thresholds_8[i] = herd_immunity_div(gamma=gamma, N=N, t_0=t_0, t_end=t_end,
                                        worst_day_portions=graph_5_aggr['worst_day_portions'], d=d, C_V=C_V, R_0=R_0,
                                        I_0_vec=np.array([0, 0, 1]), outbreak_threshold=outbreak_threshold,
                                        precision=eps, type='bisection')
    print(C_V, ', ', end='')

print('\nWith 0 Initially non-sensitive people (as in question 5), the day with the most infected people'
      'has already {:.1%} of the\npopulation infected which is {:.1f} times the number required for '
      'herd immunity'.format(graph_5_aggr['worst_day_non_sensitive_percentage'],
                             (graph_5_aggr['worst_day_non_sensitive_percentage']) / (thresholds_8[1] / N)))

df_8 = DataFrame({'C_Vs': C_Vs, 'Herd Immunity Threshold': thresholds_8})

df_8