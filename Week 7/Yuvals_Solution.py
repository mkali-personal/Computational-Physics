import numpy as np
import random

TARGILIUM_ATOMIC_MASS = 300
AVOGADRO_NUMBER = 0.6 * 1e24
CROSS_SECTIONAL_AREA_FOR_COLLISION = 10 * 1e-24  # cm^2

MAX_TALMIDONS = 1e4

ANNIHILATION = 0
SCATTER = 1
FISSION = 2

SUPER_CRITICAL = 0
SUB_CRITICAL = 1


def find_critical_radius(density, p_0, p_1, p_2, radius=None, steps=None):
    """
    Find the critical mass of a ball of "Targilium"
    Args:
        density (`float`): The density of the "Targilium" in the ball.
        p_0, p_1, p_2 (`float`): The probabilities for annihilation, scattering and fission.
        radius (`float`): Initial guess for the radius of the "Targilium" ball. If not supplied, will start with mean free path.
    Returns:
        r (`float`): The critical radius of the "Targilium" ball in which the number of "Talmidonim" stays constant.
    Raises:
        ValueError:
    """
    action_probabilities = (p_0, p_1, p_2)
    if abs(sum(action_probabilities) - 1) > 1e-3:
        raise ValueError('Sum of probabilities must equal 1')

    mean_free_path = calculate_mean_free_path(density, TARGILIUM_ATOMIC_MASS, CROSS_SECTIONAL_AREA_FOR_COLLISION)
    if radius is None:
        radius = mean_free_path

    prev_result = check_radius(radius, mean_free_path, action_probabilities)
    if isinstance(steps, list):
        steps.append((radius, prev_result))
    prev_radius = radius

    if prev_result is SUB_CRITICAL:
        radius *= 1.1
    else:
        radius *= 0.9

    while True:
        result = check_radius(radius, mean_free_path, action_probabilities)
        if isinstance(steps, list):
            steps.append((radius, result))
        if result is not prev_result:
            return (prev_radius + radius) / 2

        prev_radius = radius
        prev_result = result

        if result is SUB_CRITICAL:
            radius *= 1.1
        else:
            radius *= 0.9


def calculate_atomic_density(mass_density, atomic_mass):
    return AVOGADRO_NUMBER * mass_density / atomic_mass


def calculate_mean_free_path(mass_density, atomic_mass, cross_sectional_area_for_collision):
    return 1 / (calculate_atomic_density(mass_density, atomic_mass) * cross_sectional_area_for_collision)


def check_radius(radius, mean_free_path, action_probabilities):
    for i in range(100):
        result = run_radius_experiment(radius, mean_free_path, action_probabilities)
        if result is SUPER_CRITICAL:
            return SUPER_CRITICAL

    return SUB_CRITICAL


def run_radius_experiment(radius, mean_free_path, action_probabilities):
    talmidons = [np.asarray([0, 0, 0])]
    total_talmidons_count = 1

    while True:
        new_talmidons = follow_talmidon(talmidons.pop(0), mean_free_path, radius, action_probabilities)
        talmidons.extend(new_talmidons)
        total_talmidons_count += len(new_talmidons)

        if total_talmidons_count > MAX_TALMIDONS:
            return SUPER_CRITICAL

        if len(talmidons) == 0:
            return SUB_CRITICAL


def follow_talmidon(starting_point, mean_free_path, radius, action_probabilities):
    new_talmidons = []
    talmidon = advance_talmidon(starting_point, mean_free_path)

    while True:
        if np.sqrt(np.sum(talmidon ** 2)) > radius:
            # Talmidon escaped Targilium
            return new_talmidons

        interaction = generate_random_interaction(*action_probabilities)
        if interaction is ANNIHILATION:
            return new_talmidons
        if interaction is FISSION:
            # Talmidon was annihilated and two new Talmidons appeared.
            new_talmidons.append(talmidon)
            new_talmidons.append(talmidon)
            return new_talmidons
        # Assuming interaction is SCATTER and continuing
        talmidon = advance_talmidon(talmidon, mean_free_path)


def advance_talmidon(starting_point, mean_free_path):
    distance = generate_random_distance(mean_free_path)
    direction = generate_random_direction()
    return starting_point + distance * direction


def generate_random_distance(mean_free_path):
    y = random.random()
    return - mean_free_path * np.log(1 - y)


def generate_random_interaction(p_0, p_1, p_2):
    random_value = random.random()
    if random_value < p_0:
        return ANNIHILATION
    if random_value < (p_0 + p_1):
        return SCATTER
    return FISSION


def generate_random_direction():
    cos_theta = -1 + 2 * random.random()
    phi = 2 * np.pi * random.random()

    return np.asarray([np.sin(np.arccos(cos_theta)) * np.cos(phi),
                       np.sin(np.arccos(cos_theta)) * np.sin(phi),
                       cos_theta])
# %%
from IPython.display import display
import pandas as pd

density = 30 # g / cm^3
p_0 = 0.2
p_1 = 0.5
p_2 = 0.3

steps = []
radius = find_critical_radius(density, p_0, p_1, p_2, steps=steps)

radii = np.asarray([step[0] for step in steps])
results = (['SUPER CRITICAL' if step[1] is SUPER_CRITICAL else 'SUB CRITICAL' for step in steps])

df = pd.DataFrame({'Radius': radii,
                   'Mass': (4 * np.pi / 3) * radii**3 * density,
                   'Result': results})

display(df)

print('The critical radius is:', radius, 'and the critical mass is', (4 * np.pi / 3) * radius**3 * density)