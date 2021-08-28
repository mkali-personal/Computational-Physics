import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
import pickle
nice_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
with open(r'C:\Users\mkali\Google Drive\University\Current Semester\Computational Physics\Week 7\random_data.pickle', 'rb') as f:
    random_data = pickle.load(f)
outcome_counter = 0
direction_counter = 0
distance_counter = 0
# %%

class Ball:
    def __init__(self, ro, p_0, p_1, R=None, sigma=1e-23, A=300):
        self.ro = ro
        self.sigma = sigma
        self.A = A
        self.ro_0 = 0.6e24 * ro / A
        self.l = 1 / (self.ro_0 * sigma)
        self.p_0 = p_0
        self.p_1 = p_1
        if R is not None:
            self.R = R
        else:
            self.R = self.l
        self.stable_R = None

    @staticmethod
    # Generate random movement direction:
    def initiate_direction():
        global direction_counter
        direction_counter += 1
        return random_data.loc[direction_counter, 'direction']

    def calculate_weight(self, Radius=None):
        # Calculate mass of the ball based on it's radius, or any other input radius:
        if Radius is not None:
            return 4 / 3 * np.pi * Radius ** 3 * self.ro
        elif self.stable_R is None:
            return None
        else:
            return 4 / 3 * np.pi * self.stable_R ** 3 * self.ro

    def predict_final_position(self, studention_initial_conditions):
        # roll final position based on the step distribution:
        global distance_counter
        distance_counter += 1
        distance = random_data.loc[distance_counter, 'distance']
        step = distance * studention_initial_conditions[1]
        return studention_initial_conditions[0] + step

    # Roll a random collision result (swallowed, deflected, or fussion)
    def roll_collision_result(self):
        global outcome_counter
        outcome_counter += 1
        return random_data.loc[outcome_counter, 'outcome']

    def students_life(self, studention):
        # Given a student - rolls direction, final position, and final outcome:
        final_destination = self.predict_final_position(studention)
        if np.linalg.norm(final_destination) > self.R:
            studention[[2, 3]] = [final_destination, 3]
        else:
            collision_result = self.roll_collision_result()
            studention[[2, 3]] = final_destination, collision_result
        return studention

    def reset_studention(self, studention):
        # Reset direction, outcome, and final position:
        return np.array((studention[2], self.initiate_direction(), (None, None, None), -1))

    def run_simulation(self):
        n = 10000
        iterations_counter = 0
        # Outcomes are {-1: not yet calculated, 0: swallowed, 1: deflected, 2: splitted}
        # each recorder element is {0: initial_position, 1: direction, 2: final_position, 3: outcome}
        recorder = [np.array(((0, 0, 0), self.initiate_direction(), (None, None, None), -1))]
        focused_stuention = 0
        students_created = 1
        while focused_stuention < len(recorder) and students_created < n:
            recorder[focused_stuention] = self.students_life(recorder[focused_stuention])
            if recorder[focused_stuention][3] == 1:
                recorder[focused_stuention] = self.reset_studention(recorder[focused_stuention])
                students_created += 1
            elif recorder[focused_stuention][3] == 2:
                recorder[focused_stuention] = self.reset_studention(recorder[focused_stuention])
                recorder.append(self.reset_studention(recorder[focused_stuention]))
                students_created += 2
            else:
                focused_stuention += 1

        # print(len(recorder), self.R)
        if students_created >= n:
            return 1
        else:
            return 0

    def determine_stability(self, num_of_iterations=100):
        for i in range(num_of_iterations):
            simulation_result = self.run_simulation()
            if simulation_result == 1:
                return 1
        return 0

    def find_equilibrium(self, print_progress=0, num_of_iterations=100):
        stabilities = DataFrame({'Radius': [self.R], 'Is Stable': [self.determine_stability(num_of_iterations)]})
        while True:
            if print_progress:
                print("{0}".format(self.R), end="\r")
            if stabilities.loc[len(stabilities) - 1, 'Is Stable'] == 0:
                self.R = self.R * 1.1
            else:
                self.R = self.R * 0.9
            stabilities.loc[len(stabilities)] = (self.R, self.determine_stability(num_of_iterations))
            if stabilities.loc[len(stabilities) - 1, 'Is Stable'] != stabilities.loc[len(stabilities) - 2, 'Is Stable']:
                self.stable_R = (stabilities.loc[len(stabilities) - 1, 'Radius'] +
                                 stabilities.loc[len(stabilities) - 2, 'Radius']) / 2
                return stabilities


def iterate_over_ps(ps, num_of_iterations=100):
    df = DataFrame({'p_0': [],
                    'p_1': [],
                    'p_2': [],
                    'Delta p': [],
                    'Radius': [],
                    'Mass': [],
                    'Mass Prediction': []})

    for i in range(len(ps)):
        B = Ball(30, ps[i][0], ps[i][1])
        B.find_equilibrium(num_of_iterations)
        print(ps[i][0], ps[i][1], B.stable_R, end='\r')
        df.loc[len(df)] = [ps[i][0],
                             ps[i][1],
                             1 - ps[i][0] - ps[i][1],
                             1 - 2 * ps[i][0] - ps[i][1],
                             B.stable_R,
                             B.calculate_weight(),
                             B.calculate_weight(
                                 -D * np.log((1 - 2 * ps[i][0] - ps[i][1]) / (2 - 2 * ps[i][0] - ps[i][1])))]
    return df


# %% Question 1
B = Ball(30, 0.2, 0.5, R=9)
stabilities = B.find_equilibrium(print_progress=1)
print('Stable R: ', B.stable_R)
print('Weight: ', B.calculate_weight())
stabilities

# %% Question 2

# ros = (np.arange(10) + 1) * 10
# initial_guesses = [10, 6, 4, 3, 2.5, 2, 1.75, 1.5, 1.3, 1]
# df_2 = DataFrame({'ro': [], 'Radius': [], 'Mass': []})
# for i in range(len(ros)):
#     print('density: ', ros[i], end='\r')
#     B = Ball(ros[i], 0.2, 0.5, R=initial_guesses[i])
#     B.find_equilibrium()
#     df_2.loc[len(df_2)] = [ros[i], B.stable_R, B.calculate_weight()]
#
# df_2

# %% Question 2 Plot:
# A = 300
# p_0 = 0.2
# p_1 = 0.5
# p_2 = 1-p_0 - p_1
# N = 0.6e24
# sigma = 1e-23
#
# df_2.plot('ro', 'Radius')
# C = -A / (N * sigma) * np.log((p_2 - p_0) / (1+p_2 - p_0))
# x = np.linspace(8, 100, 100)
# y = C / x
# plt.plot(x, y, label='Prediction')
# plt.legend()
# plt.xlabel('$\rho$')
# plt.xlabel('Mass');
# %% Question 3

A = 300
N = 0.6e24
sigma = 1e-23
ro = 30
D = A / (N * sigma * ro)

ps = [(0.1, 0.75),
      (0.1, 0.7),
      (0.1, 0.65),
      (0.2, 0.57),
      (0.2, 0.55),
      (0.2, 0.53),
      (0.2, 0.51),
      (0.2, 0.5),
      (0.2, 0.49),
      (0.2, 0.47),
      (0.2, 0.45),
      (0.3, 0.35),
      (0.3, 0.3),
      (0.3, 0.25),
     ]

df_3 = iterate_over_ps(ps)
# %% Question 3 Plot:
ax = df_3.plot.scatter(x='Delta p', y='Mass', color=nice_colors[0], label='Mass')
df_3.plot.scatter('Delta p', 'Mass Prediction', color=nice_colors[1], label='Predicted Mass', ax=ax)
ax.set_ylim(0, max(df_3['Mass'])*1.1)
ax.set_ylabel('Mass');