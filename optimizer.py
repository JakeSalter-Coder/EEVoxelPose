import json
import os
import numpy as np
from scipy.optimize import linprog

# Define variables
profiles_2d = json.load(open('./Profiles/2D_profile.json', 'r'))
profiles_3d = json.load(open('./Profiles/3D_profile.json', 'r'))

profiles_2d_client = profiles_2d['2D_profile_client']
profiles_2d_server = profiles_2d['2D_profile_server']

intermediate_output_sizes = json.load(open('./Profiles/intermediate_output.json', 'r')) # bytes
faux_transmission_rate = 10000000 # megabytes/sec

faux_mobile_power = 2.5 # joules/sec
faux_transmission_power = 3 # joules/sed

alpha = 0.01 #scale energy (joules) to match time (sec)

# Scales all the time values by a given scalar
def scales_profile(profile, scalar):
    return dict((partition, time * scalar) for partition, time in profile.items())

def get_cost(scalar):
    # Find Transmission_time
    transmission_time = [size/faux_transmission_rate for size in intermediate_output_sizes.values()]

    # Find mobile device computation
    local_time = [time for time in scales_profile(profiles_2d_client, scalar).values()]

    # Find server computation
    server_time = [time for time in scales_profile(profiles_2d_server, 1/8).values()]

    # Find power
    energy_local = [time * faux_mobile_power for time in local_time]
    energy_upload = [time * faux_mobile_power for time in transmission_time]

    # Define totals
    total_time = [transmission + local + server for transmission, local, server in zip(transmission_time, local_time, server_time)]
    total_energy = [local + upload for local, upload in zip(energy_local, energy_upload)]

    cost = [time + alpha* energy for time, energy in zip(total_time, total_energy)]

    return cost

def Min(c, num_devices):
    # define a 2-d matrix to allow the evaluation of one device at a time
    A_eq = []
    for i in range(num_devices):
        intermediate = []
        for n in range(num_devices):
            intermediate += np.multiply(np.ones(21), (n==i)).astype(int).tolist()
        A_eq.append(intermediate)

    return linprog(c, A_eq=A_eq, b_eq=[1 for n in range(num_devices)], bounds=(0,1))

def Optimize():
    # get cost for the five devices with a theoretical computational scalar of some value
    c = get_cost(1)
    c += get_cost(2)
    c += get_cost(2)
    c += get_cost(0.5)
    c += get_cost(0.5)

    # find solution
    solution = Min(c, 5)

    # Creat dictionary for easy ready of the results
    solution_dict = dict((f'Solution for device {n//21}', solution.x.tolist()[n:n+21]) for n in range(0, len(solution.x.tolist()), 21))

    # save to logs
    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    with open('logs/solution.json', 'w') as file:
        json.dump(solution_dict, file, indent=4)

if __name__ == '__main__':
    Optimize()