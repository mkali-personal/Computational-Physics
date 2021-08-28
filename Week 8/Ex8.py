import numpy as np
import matplotlib.pyplot as plt
import copy
# %%


class Barrier:
    # This is the Barrier class.
    # It contains all the system properties such as size and the fields psi, zeta.
    # The methods of this class contain:
    # find_critical_Re: A function to find the critical Re value for a circulation after the barrier
    # Re_to_post_barrier_direction: A simple method to calculate the direction of flow after the
    # barrier for a given value of Re;
    # dynamic_relax_system and simple_relax_system: The Relaxation methods;
    # gauss_seidel_round: a Gauss-Seidel function round for psi and zeta (that are used by the Relaxation methods).
    # plot_flow: A visualization method to plot the flow lines of the fluid (the contour of psi);
    # classify_sector: A function to classify a coordinate as part of an edge that is used by the Gauss seidel functions


    def __init__(self, R=None, Re=None, h=1, V_0=1, x_dim=120, y_dim=40, E=36, W=8, C=8, w=0.1):
        # Given Values:
        self.V_0 = V_0
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.w = w
        self.W = W
        self.h = h
        self.E = E
        self.W = W
        self.EplusC = E + C
        # Define R and Re based on whichever input was givenL
        if R is None and Re is None:
            self.R = 0.01
            self.Re = 2 * self.W * self.R / self.h
        elif R is not None and Re is not None:
            if abs(R - Re * self.h / (2*self.W)) > 1e-9:
                raise Exception('R and Re do not match. Please insert only one of them')
        elif R is not None:
            self.R = R
            self.Re = 2 * self.W * R / self.h
        elif Re is not None:
            self.Re = Re
            self.R = Re * self.h / (2 * self.W)
        # Initiate psi and zeta fields with zeros on the barrier and bottom row:
        self.psi = np.ones((self.x_dim, self.y_dim)) * np.arange(self.y_dim) * self.V_0 * self.h
        self.psi[:, 0] = 0
        self.psi[self.E-1:self.EplusC+1, 0:self.W+1] = 0
        self.zeta = np.zeros((self.x_dim, self.y_dim))


    def find_critical_Re(self,
                         Re_initial=1,
                         y_tolerance = 0.1,
                         x_tolerance=0.1,
                         barrier_params={},
                         initial_w=0.1,
                         relative_tolerance=0.3,
                         max_iter=200,
                         print_progress=0):
        params = {'barrier_params': barrier_params,
                  'initial_w': initial_w,
                  'relative_tolerance': relative_tolerance,
                  'max_iter': max_iter,
                  'print_progress': print_progress
                 }
        # use the find_mono_functions_root function to find the root of post-barrier-velocity(Re):
        Re_critic_h1, vertical_velocity = find_mono_functions_root(func=self.Re_to_post_barrier_velocity,
                                                                   initial_guess=Re_initial,
                                                                   y_tolerance=y_tolerance,
                                                                   x_tolerance=x_tolerance,
                                                                   params=params)
        return Re_critic_h1, vertical_velocity # Convert from R to Re


    def Re_to_post_barrier_velocity(self, Re, barrier_params, initial_w, relative_tolerance, max_iter, print_progress):
        # returns the velocity after the barrier given an Re value and varrier parameters (such as h, dimwnsions, etc)
        barrier_params['Re'] = Re
        self.dynamic_relax_system(initial_w,
                                  relative_tolerance,
                                  max_iter,
                                  barrier_params,
                                  print_progress=print_progress)
        return self.post_barrier_velocity()


    def post_barrier_velocity(self):
        # Returns the velocity in the middle of the barrier (the B edge) given a relaxed psi field:
        j = int(self.W / 2)
        i = self.EplusC
        direction = -(self.psi[i + 1, j] - self.psi[i, j])
        return direction


    def dynamic_relax_system(self,
                             initial_w,
                             relative_tolerance,
                             max_iter,
                             barrier_params={},
                             print_progress=0,
                             plot_progress=0):
        # This is the main function of the excercise. It solves psi and zeta starting with a given w value and:
        # 1) converge to a solution.
        # 2) reaching max_iter parameter and giving up.
        # 3) diveging, cutting w by half and trying again.
        w_ratio = 2
        divergence_ratio = 2
        w = initial_w
        self.__init__(w=initial_w, **barrier_params)
        if plot_progress:
            self.plot_flow()
        history = [copy.copy(self.psi), copy.copy(self.zeta)]
        self.psi_zeta_update_round()
        initial_differences = np.append(self.psi, self.zeta) - np.append(history[0], history[1])
        initial_difference_size = np.linalg.norm(initial_differences)
        tolerance = initial_difference_size * relative_tolerance
        history = [copy.copy(self.psi), copy.copy(self.zeta)]
        k = 1
        while True:
            k += 1
            self.psi_zeta_update_round()
            differences = np.append(self.psi, self.zeta) - np.append(history[0], history[1])
            difference_size = np.linalg.norm(differences)
            if print_progress:
                print(f'{difference_size:.5f} / {tolerance:.5f}\r', end="")
            if plot_progress and not(k%10):
                self.plot_flow()

            if difference_size > divergence_ratio * initial_difference_size:
                print(f'\nDiverged with w = {w}, Now trying with w = {w / w_ratio}')
                return self.dynamic_relax_system(w / w_ratio,
                                                 relative_tolerance,
                                                 max_iter,
                                                 barrier_params,
                                                 print_progress=print_progress)
            elif k == max_iter:
                print(f'\nReached max_iter = {max_iter} iterations - Stopping.')
                return k, w, 'Stuck' #, self.dynamic_relax_system(w * (w_ratio**(1/2)), relative_tolerance, max_iter, params)
            elif difference_size < tolerance:
                print(f'\nCoverged after {k} iterations with w = {w}')
                return k, w, 'Converged'
            else:
                history = [copy.copy(self.psi), copy.copy(self.zeta)]


    def simple_relax_system(self, tolerance):
        # Same as previous funtion but only converging, without playing with the value of w.
        history = [copy.copy(self.psi), copy.copy(self.zeta)]
        k = 0
        while True:
            k += 1
            self.psi_zeta_update_round()
            differences = np.append(self.psi, self.zeta) - np.append(history[0], history[1])
            difference_size = np.linalg.norm(differences)
            print(f'{difference_size:.5f} / {tolerance:.5f}\r', end="")
            if difference_size < tolerance:
                return k
            else:
                history = [copy.copy(self.psi), copy.copy(self.zeta)]


    def psi_zeta_update_round(self):
        # Executing one psi gauss seidel round and one zeta gaus seidel round
        self.gauss_seidel_round(self.psi_update_element)
        self.gauss_seidel_round(self.zeta_update_element)


    def gauss_seidel_round(self, func):
        for j in range(self.y_dim):
            for i in range(self.x_dim):
                func(i, j)


    def psi_update_element(self, i, j):
        # updates a single element of the psi field
        sector = self.classify_sector(i, j)
        # Those edges require no updating:
        if sector in ('A', 'B', 'C', 'D', 'E', 'In Barrier'):
            return
        else:
            # The other values will be updated according to the laplace equation and boundaries condition:
            new_value = self.evaluate_psi_element(i, j, sector)
            self.psi[i, j] = self.interpolation(self.psi[i, j], new_value)


    def zeta_update_element(self, i, j):
        # updates a single element of the psi field
        sector = self.classify_sector(i, j)
        # Those edges require no updating:
        if sector in ('A', 'E', 'F', 'FG', 'G', 'HG', 'In Barrier'):
            return
        else:
            # The other values will be updated according to the laplace equation and boundaries condition:
            new_value = self.evaluate_zeta_element(i, j, sector)
            self.zeta[i, j] = self.interpolation(self.zeta[i, j], new_value)


    def evaluate_psi_element(self, i, j, sector):
        if sector in ('Mid', 'BC', 'DC'):
            return (1/4) * (self.psi[i+1, j] + self.psi[i-1, j] + self.psi[i, j+1] + self.psi[i, j-1] - self.zeta[i, j])
        elif sector == 'F':
            return (1/3) * (self.psi[i+1, j] + self.psi[i, j+1] + self.psi[i, j-1] - self.zeta[i, j])
        elif sector == 'FG':
            return (1/2) * (self.psi[i+1, j] + self.psi[i, j-1] + self.V_0 - self.zeta[i, j])
            # Maybe add * self.h to the V_0 argument
        elif sector == 'G':
            return (1/3) * (self.psi[i+1, j] + self.psi[i-1, j] + self.psi[i, j-1] + self.V_0 - self.zeta[i, j])
            # Maybe add * self.h to the V_0 argument
        elif sector == 'HG':
            return (1/2) * (self.psi[i-1, j] + self.psi[i, j-1] + self.V_0 - self.zeta[i, j])
            # Maybe add * self.h to the V_0 argument
        elif sector == 'H':
            return (1/3) * (self.psi[i-1, j] + self.psi[i, j+1] + self.psi[i, j-1] - self.zeta[i, j])
        else:
            raise Exception('sector not treated: ' + sector + str(i) + str(j))

    def evaluate_zeta_element(self, i, j, sector):
        if sector in ('Mid', 'BC', 'DC'): # NOTE THAT BC AND DC MIGHT REQUIRE SPECIAL TREATMENT!!
            # new_value = (self.zeta[i+1,j] + self.zeta[i-1,j] + self.zeta[i,j+1] + self.zeta[i,j-1] +
            #              self.R/4 * (self.psi[i,j+1] * self.zeta[i+1,j] - self.psi[i,j] * self.zeta[i+1,j]-
            #                          self.psi[i+1,j] * self.zeta[i,j+1] - self.psi[i,j] * self.zeta[i,j+1]))\
            #             / (self.R/4 * (-self.psi[i,j+1] + self.psi[i+1,j]) + 4)
            new_value = (1/4) * (self.zeta[i+1,j] + self.zeta[i-1,j] + self.zeta[i,j+1] + self.zeta[i,j-1] +
                            (self.R / 4) * (
                                         (self.psi[i+1,j] - self.psi[i-1,j]) * (self.zeta[i,j+1] - self.zeta[i,j-1]) -
                                         (self.psi[i,j+1] - self.psi[i,j-1]) * (self.zeta[i+1,j] - self.zeta[i-1,j])
                                           )
            )
        elif sector == 'H':
            new_value = (1/3) * (self.zeta[i-1,j] + self.zeta[i,j+1] + self.zeta[i,j-1])
        elif sector == 'C':
            new_value = 2*self.psi[i,j+1]# / self.h**2
        elif sector == 'B':
            new_value = 2*self.psi[i+1,j]# / self.h**2
        elif sector == 'D':
            new_value = 2*self.psi[i-1,j]# / self.h**2
        else:
            raise Exception('sector not treated: ' + sector + str(i) + str(j))
        return new_value


    def classify_sector(self, i, j):
        # Classify an element to the edge\domain it belongs to:
        if self.E > i >= 0 == j:
            sector = 'E'
        elif self.EplusC <= i < self.x_dim  and j == 0:
            sector = 'A'
        elif i == self.E - 1 and 0 <= j < self.W:
            sector = 'D'
        elif self.E <= i < self.EplusC and j == self.W:
            sector = 'C'
        elif i == self.EplusC and 0 <= j < self.W:
            sector = 'B'
        elif i == self.E - 1 and j == self.W:
            sector = 'DC'
        elif i == self.EplusC and j == self.W:
            sector = 'BC'
        elif i == 0 and j == self.y_dim - 1:
            sector = 'FG'
        elif i == self.x_dim - 1 and j == self.y_dim - 1:
            sector = 'HG'
        elif i == 0:
            sector = 'F'
        elif i == self.x_dim - 1:
            sector = 'H'
        elif j == self.y_dim - 1:
            sector = 'G'
        elif self.E <= i < self.EplusC and 0 <= j < self.W:
            sector = 'In Barrier'
        else:
            sector = 'Mid'

        return sector


    def interpolation(self, older, new):
        # Simple interpolation
        return self.w * new + (1-self.w) * older


    def plot_flow(self, resolution=50, title=None):
        # Plots psi contour plot which is the flow lines. adds the barrier and the vertical velocity after it
        fig, ax = plt.subplots(figsize=(22,8))
        ax.contour(self.psi.T, resolution)
        ax.fill([self.E, self.E, self.EplusC-1, self.EplusC-1], [0, self.W-1, self.W-1, 0])
        velocity = self.post_barrier_velocity()
        if np.sign(velocity) == 1:
            tip = min(self.y_dim / 2, self.W / 2 + velocity*100)
        else:
            tip = max(0, self.W / 2 + velocity*100)
        ax.annotate(s = '',
                    xy=(self.EplusC, tip),
                    xytext=(self.EplusC, self.W / 2),
                    arrowprops={'arrowstyle':'->', 'lw': 2})
        if title is not None:
            ax.set_title(title)
        fig.show()
        return fig, ax


    @ staticmethod
    def visualize_matrix(matrix):
        # This function is for printing the matrix with the same orientation as in the painting:
        return np.flip(matrix.T, 0)


def find_mono_functions_root(func, initial_guess, y_tolerance, x_tolerance, params={}):
    # This function finds a monotonic functions root, assuming f(0) < 0 and f(\infty) > 0
    x = initial_guess
    simplified_func = lambda x: func(x, **params)
    y = simplified_func(x)
    print(f'Re = {x:.5f} yields post barrier vertical velocity of {y:.5f}')
    first_sign = np.sign(y)
    changed_sign_once = 0
    if first_sign == 1:
        a, y_a, b, y_b = 0, -1, x, y
    else:
        a, y_a, b, y_b = x, y, np.Inf, 1
    while not(abs(y) < y_tolerance or b - a < x_tolerance) or changed_sign_once == 0:
        a, y_a, b, y_b, x, y = update_root_boundaries(simplified_func, a, y_a, b, y_b)
        if np.sign(y) != first_sign:
            changed_sign_once = 1
        print(f'Re = {x:.5f} yields post barrier vertical velocity of {y:.5f}')
    if abs(y) > abs(y_a):
        return (a - x) / (y - y_a) * y_a + a, y_a
    elif abs(y) > abs(y_b):
        return (b - x) / (y - y_b) * y_b + b, y_b
    else:
        return x, y


def update_root_boundaries(simplified_func, a, y_a, b, y_b):
    # Update the boundaries of the search using the function
    x_new, y_new = update_x_value(simplified_func, a, y_a, b, y_b)
    if np.sign(y_new) == 1:
        return a, y_a, x_new, y_new, x_new, y_new
    else:
        return x_new, y_new, b, y_b, x_new, y_new


def update_x_value(simplified_func, a, y_a, b, y_b):
    if a == 0:
        x_new = b / 2
    elif b == np.Inf:
        x_new = a * 2
    else:
        x_new = (a - b) / (y_b - y_a) * y_a + a
    return x_new, simplified_func(x_new)


# %% Question 1
B_1 = Barrier()
k_1, _, _ = B_1.dynamic_relax_system(initial_w=0.3,
                                 relative_tolerance=0.025,
                                 max_iter=300,
                                 barrier_params={'R': 0.01, 'h': 1},
                                 print_progress=1)
fig_1, ax_1 = B_1.plot_flow(title=f'System state after {k_1} gauss-seidel iterations')
fig_1.savefig('fig_1.png')

# %% Question 2
B_2 = Barrier()
k_2, _, _ = B_2.dynamic_relax_system(initial_w=0.3,
                                 relative_tolerance=0.025,
                                 max_iter=300,
                                 barrier_params={'R': 4, 'h': 1},
                                 print_progress=1,
                                 plot_progress=0)
fig_2, ax_2 = B_2.plot_flow(title=f'System state after {k_2} gauss-seidel iterations')
fig_2.savefig('fig_2.png')

# %% Question 3
# Here for better accuracy the relative tolerance used is higher than in previous cells, as it strongly affect
# the result. actually, it is simply stopped after about 250 iterations.
B_3 = Barrier()
Re_critic_h_1, _ = B_3.find_critical_Re(Re_initial=1,
                            initial_w=0.3,
                            relative_tolerance=0.001,
                            y_tolerance = 0.01,
                            x_tolerance=0.01,
                            max_iter=250,
                            barrier_params={'h': 1}
                            )
print(f'\n Critical Re value with h = 1 is {Re_critic_h_1}')
fig_3, ax_3 = B_3.plot_flow(title='System state with critical Re value for post-barrier vortex, h=1')
fig_3.savefig('fig_3.png')

# %% Question 4
B_4 = Barrier()
Re_critic_h_half, _ = B_4.find_critical_Re(Re_initial=1,
                            initial_w=0.3,
                            relative_tolerance=0.001,
                            y_tolerance = 0.01,
                            x_tolerance=0.01,
                            max_iter=250,
                            barrier_params={'h': 0.5}
                            )
print(f'\nCritical Re value with h = 0.5 is {Re_critic_h_half}')
fig_4, ax_4 = B_4.plot_flow(title='System state with critical Re value for post-barrier vortex, h=0.5')
fig_4.savefig('fig_4.png')


# %% Question 5
# B_5 = Barrier()
r = 1.5
# k_5, _, _ = B_5.dynamic_relax_system(initial_w=0.5,
#                                      relative_tolerance=0.01,
#                                      max_iter=1000,
#                                      barrier_params={'Re': Re_critic_h_1 * r, 'h': 1},
#                                      print_progress=1
#                                      )
B_5 = Barrier(Re=Re_critic_h_1 * r, h=1, w = 0.5)
k_5 = B_5.simple_relax_system(0.35)
fig_5, ax_5 = B_5.plot_flow(title=f'System state which is {(r-1)*100}% greater the critical Re from (3) and h=1.\nConverged after {k_5} iterations')
fig_5.savefig('fig_5.png')

# %% Question 6
# B_6 = Barrier()
# k_6, _, _ = B_6.dynamic_relax_system(initial_w=0.5,
#                                      relative_tolerance=0.01,
#                                      max_iter=1000,
#                                      barrier_params={'Re': Re_critic_h_1 * r, 'h': 0.5},
#                                      print_progress=1
#                                      )
B_6 = Barrier(Re=Re_critic_h_1 * r, h=0.5, w = 0.5)
k_6 = B_6.simple_relax_system(0.35)
fig_6, ax_6 = B_6.plot_flow(title=f'System state which is {(r-1)*100}% greater the critical Re from (3) and h=0.5.\nConverged after {k_6} iterations')
fig_6.savefig('fig_6.png')
