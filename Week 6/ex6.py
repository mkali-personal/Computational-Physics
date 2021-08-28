# Import Modules:
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib; matplotlib.use("TkAgg")

# from matplotlib.font_manager import FontProperties
from pandas import DataFrame
#%matplotlib inline

# Set global variables:
nice_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# %%


class State:
    def __init__(self, x_0, u_0, p_0, ro_0, m, sigma, gamma=1.4, t=0):

        # NX1 points arrays:
        self.x = x_0
        self.u = u_0
        self.a = np.pad(-2 * (p_0[1:] - p_0[:-1]) / (m[1:] + m[:-1]), pad_width=1, constant_values=0)
        self.t = t

        # (N-1)X1 segments array:
        self.m = m
        self.p = p_0
        self.ro = ro_0
        self.q = sigma * self.ro * np.maximum(self.u[1:] - self.u[:-1], 0) ** 2
        self.e = p_0 / ((gamma - 1) * ro_0)


    def extract_energy(self):
        E_pot = sum(self.e * self.m)
        E_kin_1 = sum((1/8) * self.m * (self.u[1:] + self.u[:-1])**2)
        # m_padded = np.pad(self.m, pad_width=1, constant_values=0)
        # E_kin_2 = sum(0.5 * self.u**2 * m_padded[1:] - m_padded[:-1])
        # E_kin_3 = sum((1/6) * self.m * (self.u[1:]**2 + self.u[1:]*self.u[:-1] + self.u[:-1]**2))

        E_1 = E_pot + E_kin_1
        # E_2 = E_pot + E_kin_2
        # E_3 = E_pot + E_kin_3
        return E_1#, E_2, E_3


    def extract_interesting_points(self):

        d = dict()

        d['contact_plane_x'] = self.x[1:-1]

        d['contact_plane_y'] = abs(self.ro[1:] - self.ro[:-1] - (self.p[1:] - self.p[:-1]))

        if len(self.x) < 50:
            w = 2
        else:
            w = 4

        min_change = np.minimum(abs(self.p[1:] - self.p[:-1]), abs(self.ro[1:] - self.ro[:-1]))

        d['shock_wave_x'] = self.x[w+1:-w]

        d['shock_wave_y'] = min_change[w:-w+1] - np.maximum(min_change[:-2*w+1], min_change[2*w-1:])

        x9_idx = np.argmax(d['shock_wave_x'] > 9)

        d['shock_wave_pos'] = d['shock_wave_x'][np.argmax(d['shock_wave_y'][x9_idx:])+x9_idx]

        d['contact_plane_pos'] = d['contact_plane_x'][np.argmax(d['contact_plane_y'])]

        d['fan_pos'] = [self.x[np.argmax(abs(self.p[1:] - self.p[:-1]) > 0.003)],
                        self.x[np.argmax((self.p[2:] - self.p[1:-1] > 0) &
                                    (self.p[2:] - 2 * self.p[1:-1] + self.p[:-2] > 0))]]


        return d


    def plot_state(self, plot_interesting_points, ax):

        ax.plot((self.x[1:] + self.x[:-1])/2, self.p, label='p')
        ax.plot((self.x[1:] + self.x[:-1]) / 2, self.ro, label='ro')

        if plot_interesting_points:
            d = self.extract_interesting_points()
            ax.axvline(d['shock_wave_pos'], color=nice_colors[2], linewidth=0.3, label='shock wave')
            ax.axvline(d['contact_plane_pos'], color=nice_colors[3], linewidth=0.3, label='contact plane')
            ax.axvline(d['fan_pos'][0], color=nice_colors[4], linewidth=0.3, label='fan beggining')
            ax.axvline(d['fan_pos'][1], color=nice_colors[4], linewidth=0.3, label='fan end')

        ax.set_title('N = {}'.format(len(self.x)))
        ax.set_ylabel('ro, p [cgs]')
        ax.set_xlabel('x [cm]')
        ax.legend()


    def plot_interesting_points_graph(self):
        fig, ax = plt.subplots(3, 1)
        d = self.extract_interesting_points()

        ax[0].plot(self.x[1:-1], abs(self.ro[1:] - self.ro[:-1]), color=nice_colors[2], label='ro')
        ax[0].plot(self.x[1:-1], abs(self.p[1:] - self.p[:-1]), color=nice_colors[1], label='p')
        ax[0].legend()
        ax[0].axvline(d['fan_pos'][0], linewidth=0.5)
        ax[0].axvline(d['fan_pos'][1], linewidth=0.5)

        ax[1].plot(d['contact_plane_x'], d['contact_plane_y'], color=nice_colors[2], label='contact wave filter')
        ax[1].legend()
        ax[0].axvline(d['contact_plane_pos'], linewidth=0.5)
        ax[1].axvline(d['contact_plane_pos'], linewidth=0.5)

        ax[2].plot(d['shock_wave_x'], d['shock_wave_y'], color=nice_colors[1], label='shock wave filter')
        ax[2].legend()
        ax[0].axvline(d['shock_wave_pos'], linewidth=0.5)
        ax[2].axvline(d['shock_wave_pos'], linewidth=0.5)

        plt.show()
        return fig, ax


class Simulation:

    def __init__(self, x_0, u_0, p_0, ro_0, m, sigma, gamma=1.4):

        # global scalars:
        self.gamma = gamma
        self.sigma = sigma

        self.state = State(x_0, u_0, p_0, ro_0, m, sigma, gamma, 0)

        self.states_list = [copy.copy(self.state)]


    def preform_step(self, dt):

        # estimate mid velocities:
        mid_u = self.state.u + (dt / 2) * self.state.a

        # update positions:
        previous_V = self.state.x[1:] - self.state.x[:-1]
        self.state.x = self.state.x + dt * mid_u
        new_V = self.state.x[1:] - self.state.x[:-1]

        # update densities:
        self.state.ro = self.state.m / new_V

        # calculate viscosity:
        new_q = self.sigma * self.state.ro * np.maximum(mid_u[:-1] - mid_u[1:], 0) ** 2

        # Update energy
        self.state.e = (self.state.e - 1 / (2 * self.state.m) * (self.state.p + new_q + self.state.q) *
                        (new_V - previous_V)) / (1 + 1 / (2 * self.state.m) * (self.gamma - 1) * self.state.ro * (new_V - previous_V))

        # update viscosity
        self.state.q = new_q

        # update pressure
        self.state.p = (self.gamma - 1) * self.state.ro * self.state.e

        # update acceleration:
        a_unpadded = -2 * (self.state.p[1:] - self.state.p[:-1] + self.state.q[1:] - self.state.q[:-1]) / (self.state.m[1:] + self.state.m[:-1])
        self.state.a = np.pad(a_unpadded, pad_width=1, constant_values=0)

        # update velocities
        self.state.u = mid_u + (dt/2) * self.state.a

        self.state.t = self.state.t + dt


    def extract_courant_value(self):
        return 0.2 * min((self.state.x[1:] - self.state.x[:-1]) / (np.sqrt(self.gamma * self.state.p / self.state.ro)))


    def run_simulation(self, final_t):

        dt = self.extract_courant_value()
        t = 0
        while t <= final_t:
            dt_too_big = 1
            while dt_too_big:
                self.preform_step(dt)
                max_difference = max(abs(self.state.p / self.states_list[-1].p - 1))
                if max_difference > 0.1:
                    self.state = copy.copy(self.states_list[-1])
                    dt = 0.5*dt
                else:
                    dt_too_big = 0
            if t >= final_t:
                break
            t = t+dt

            max_dt = self.extract_courant_value()
            if  0.07 < max_difference < 0.1:
                dt = min([max_dt, 0.8*dt, final_t - t])
            elif 0.03 < max_difference < 0.07:
                dt = min([max_dt, dt, final_t - t])
            elif max_difference < 0.03:
                dt_temp = min([max_dt, 1.2*dt, final_t - t])
                dt = dt_temp

            self.states_list.append(copy.copy(self.state))


    def extract_waves_velocities(self, time_interval):
        t_values = np.array([s.t for s in self.states_list])
        first_interval_time = np.argmax(t_values >= t_values[-1] - time_interval)

        d1 = self.states_list[first_interval_time].extract_interesting_points()
        d2 = self.state.extract_interesting_points()

        shock_wave_v = abs((d2['shock_wave_pos'] - d1['shock_wave_pos']) / time_interval)
        contact_plane_v = abs((d2['contact_plane_pos'] - d1['contact_plane_pos']) / time_interval)
        fan_v = abs((d2['fan_pos'][0] - d1['fan_pos'][0]) / time_interval)

        return shock_wave_v, contact_plane_v, fan_v


    def animate(self, i):
        global line1, line2, line3, line4, line5, line6, line7, line8
        x_mids = (self.states_list[i].x[1:] + self.states_list[i].x[:-1]) / 2
        line1.set_xdata(self.states_list[i].x)
        line2.set_xdata(x_mids)
        line2.set_ydata(self.states_list[i].p)
        line3.set_xdata(x_mids)
        line3.set_ydata(self.states_list[i].ro * self.states_list[i].m)
        line4.set_xdata(x_mids)
        line4.set_ydata(self.states_list[i].e * self.states_list[i].m)

        if self.states_list[i].t > 2:
            d = self.states_list[i].extract_interesting_points()

            line5.set_xdata(d['fan_pos'][0])
            line6.set_xdata(d['fan_pos'][1])
            line7.set_xdata(d['contact_plane_pos'])
            line8.set_xdata(d['shock_wave_pos'])
        else:
            line5.set_xdata(-1)
            line6.set_xdata(-1)
            line7.set_xdata(-1)
            line8.set_xdata(-1)
        return line1, line2, line3, line4, line5, line6, line7, line8


    def plot_simulation(self, interval=10):
        fig, ax = plt.subplots()
        global line1, line2, line3, line4, line5, line6, line7, line8
        x_mids = (self.states_list[0].x[1:] + self.states_list[0].x[:-1]) / 2
        line1, = ax.plot(self.states_list[0].x, np.zeros(self.states_list[0].x.size), '.', label='x')
        line2, = ax.plot(x_mids, self.states_list[0].p, label='p')
        line3, = ax.plot(x_mids, self.states_list[0].ro, label='ro')
        line4, = ax.plot(x_mids, self.states_list[0].e, label='e')
        line5 = ax.axvline(-1)
        line6 = ax.axvline(-1)
        line7 = ax.axvline(-1)
        line8 = ax.axvline(-1)
        ax.set_ylim([0, 1.3 * max(np.concatenate(
            [self.states_list[0].p, self.states_list[0].ro * self.states_list[0].m, self.states_list[0].e * self.states_list[0].m]))])
        ax.set_xlim([0, 16])
        ax.legend()


        ani = animation.FuncAnimation(fig, self.animate, np.arange(1, len(self.states_list)),  # init_func=init,
                                      interval=interval, blit=True)
        plt.show()




# %%

Ns = [34, 100, 334, 1000]

df = DataFrame({'N': [],
                'Energy Fraction': [],
                'Shock Wave veloc.': [],
                'Contact Plane veloc.': [],
                'Fan Width veloc.': []})

fig, ax = plt.subplots(4,1, figsize=(8,22))

simulations = []

for i, N in enumerate(Ns):
    x_0 = np.linspace(0, 16, N)
    u_0 = np.zeros(N)
    p_0 = np.concatenate([np.ones(int((N-1)/2)) * 1, np.ones((N-1)-int((N-1)/2)+1)[1:] * 0.1])
    ro_0 = np.concatenate([np.ones(int((N-1)/2)) * 1, np.ones((N-1)-int((N-1)/2)+1)[1:] * 0.125])
    m = (x_0[1:] - x_0[:-1]) * ro_0

    gamma = 1.4
    sigma = 3
    s = Simulation(x_0, u_0, p_0, ro_0, m, sigma, gamma)
    s.run_simulation(2.5)

    shock_wave_v, contact_plane_v, fan_v = s.extract_waves_velocities(0.5)

    energies_fraction = s.states_list[1].extract_energy() / s.states_list[0].extract_energy() - 1

    df.loc[len(df)] = [N, energies_fraction, shock_wave_v, contact_plane_v, fan_v]

    s.state.plot_state(1, ax[i])
    simulations.append(s)

fig.show()
print(df)

simulations[1].plot_simulation(interval=1)



