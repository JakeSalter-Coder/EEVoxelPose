import json

# Define variables
profiles_2d = json.load(open('./Profiles/2D_profile.json', 'r'))
profiles_3d = json.load(open('./Profiles/3D_profile.json', 'r'))

profiles_2d_client = profiles_2d['2D_profile_client']
profiles_2d_server = profiles_2d['2D_profile_server']

intermediate_output_sizes = json.load(open('./Profiles/intermediate_output.json', 'r')) # bytes
faux_transmission_rate = 1000000 # megabytes/sec

faux_mobile_power = 2.5 # joules/sec
faux_transmission_power = 3 # joules/sed

# Scales all the time values by a given scalar
def scales_profile(profile, scalar):
    return dict((partition, time * scalar) for partition, time in profile.items())

# Find Transmission_time
transmission_time = [size/faux_transmission_rate for size in intermediate_output_sizes.values()]

# Find mobile device computation
local_time = [time for time in scales_profile(profiles_2d_client, 2).values()]

# Find server computation
server_time = [time for time in scales_profile(profiles_2d_server, 1/8).values()]

# Find power
energy_local = [time * faux_mobile_power for time in local_time]
energy_upload = [time * faux_mobile_power for time in transmission_time]

# Define totals
total_time = [transmission + local + server for transmission, local, server in zip(transmission_time, local_time, server_time)]
total_energy = [local + upload for local, upload in zip(energy_local, energy_upload)]

alpha = 0.5
cost = [time + alpha* energy for time, energy in zip(total_time, total_energy)]

print(cost)


