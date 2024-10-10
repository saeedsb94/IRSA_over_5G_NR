# -*- coding: utf-8 -*-
#---------------------------------------------------------------------------
# Author: Cedric Adjih - Inria - 2017-2021
#---------------------------------------------------------------------------
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
# Distributions for degree selection: the degree always starts from 0 

def get_ideal_soliton(d_max):
    """Truncated soliton distribution, as in wikipedia"""
    res = [0, 1/d_max]
    for i in range(2,d_max+1):
        res.append(1/(i*(i-1)))
    return res

def get_soliton_np(L, a):
    """The distribution from Narayanan&Pfister (NP) article"""
    res = [0, 0, (1-a)/2]
    for i in range(3,L+2):
        res.append(1/(i*(i-1)))
    res = np.array(res)
    res = res / res.sum()
    return res
# Some examples of distributions Lambda  
l3_dist = [0,0,0.5,0.28,0,0,0,0,0.22] # this is \Lambda_3 from [Liva2011]
truc_sol_dist = get_ideal_soliton(30)    
np_sol_dist = get_soliton_np(30, 0.02)  
d3_dist = [0,0,0,1] # degree always 3
d2_dist = [0,0,1] # degree always 2

def generate_slots_of_users(N,M,lambda_dist):
    """N nodes select a number of slots of the frame of size M:
    return a list of N sub-lists which indicate the slots selected by each user """
    d_array = np.random.choice(np.arange(len(lambda_dist)), size=N, p=lambda_dist)
    slot_of_frame_choice_matrix = np.random.uniform(size=(N,M)).argsort()
    frame_choice_array = ((np.arange(0,N) // N)*M ).reshape(-1,1)
    slot_choice_matrix = slot_of_frame_choice_matrix+frame_choice_array
    slot_choice_list = [ list(slot_choice_matrix[i,0:d_array[i]]) for i in range(N)]
    return slot_choice_list

#%%

def decode_irsa(slots_of_users, M):
    """ Given a frame of size `M`, 
    and a list `slots_of_users` that gives for each user, the list of the slots where it transmits
    -> perform the IRSA decoding.
    """
    users_of_slots = [ set() for i in range(M) ]
    decoded_set = set()
    decoded_iteration = {}
    
    for user_idx,slot_list in enumerate(slots_of_users):
        for i in slot_list:
            users_of_slots[i].add(user_idx)
        
    users_of_slots = set([tuple(sorted(slot_list)) 
                          for slot_list in users_of_slots
                          if len(slot_list) > 0])
    
    nb_iter = 0
    while len(users_of_slots) > 0:
        new_decoded_set = decoded_set.copy()
        for user_tuple in users_of_slots.copy():
            user_set = set(user_tuple)
            unknown_set = user_set.difference(decoded_set)
            #print nbIter,userSet, unknownSet
            if len(unknown_set) >= 2:
                continue
            elif len(unknown_set) == 1:
                new_decoded_user = unknown_set.pop()
                new_decoded_set.add(new_decoded_user)
                assert new_decoded_user not in decoded_set
                #assert newU not in decodedIteration
                new_decoded_set.add(new_decoded_user)
                if new_decoded_user not in decoded_iteration:
                    decoded_iteration[new_decoded_user] = nb_iter
            users_of_slots.remove(user_tuple)
        if len(new_decoded_set) == len(decoded_set):
            break
        decoded_set = new_decoded_set
        nb_iter += 1

    return decoded_set #, decoded_iteration #, users_of_slots
#%%
np.random.seed(1)
# 7 users transmitting in 10 slots with degree 2
N = 7
M = 10
lambda_dist = [0,0,1]
slots_of_users = generate_slots_of_users(N, M, lambda_dist)
# perform decoding on the frame of size 10
decoded_set = decode_irsa(slots_of_users, M)
print("N={N} M={M} decoded={decoded_set}".format(N=N, M=M, decoded_set=decoded_set))

#%%

def simul_nb_decoded(N, M, L, lambda_dist):
    """Run `L` simulations with `N` users and a frame of `M` slots
    Return the average number of decoded users"""
    nb_decoded_list = []
    for simul_idx in range(L):
        slots_of_users = generate_slots_of_users(N, M, lambda_dist)
        decoded_users = decode_irsa(slots_of_users, M)
        nb_decoded_list.append(len(decoded_users))
    return np.array(nb_decoded_list).mean()

avg_decoded = simul_nb_decoded(90, 100, 10, [0,0,1])
print("avg decoded=%s" %avg_decoded, " for 10 trials with N=90 M=100 degree=2"  )

#%%
L = 1000 # number of simulations
M = 15 # number of slots
# lambda_dist = np_sol_dist # [0,0,1]
lambda_dist=[0, 0.3, 0.15, 0.55]
np.random.seed(1)

xl = []
yl = []
for N in range(1,M+1,1): # number of users
    avg_decoded = simul_nb_decoded(N, M, L, lambda_dist)
    xl.append(N / M)  # Normalize N by M to get load
    yl.append(avg_decoded / M)  # Normalize avg_decoded by N to get throughput
xarray = np.array(xl)
yarray = np.array(yl)

# Create a new figure
plt.figure()

# Plot the new curve
plt.plot(xarray, yarray, ".-", label="Perfect SIC")

# Load the data from irsa_performance.csv
data = pd.read_csv('irsa_performance.csv')

# Extract the number of users and the average number of decoded users
num_users = data['num_users'].values
avg_decoded_imperfect_sic = data['avg_decoded_imperfect_sic'].values

# Normalize the data for load and throughput
load = num_users / M
throughput = avg_decoded_imperfect_sic / M

# Plot the new curve on the same figure
plt.plot(load, throughput, ".-", label="imperfect SIC")

# Add labels, legend, grid, and show the plot
plt.xlabel("Load")
plt.ylabel("Throughput")
plt.legend(loc="best")
plt.grid()

# Save the figure
plt.savefig('irsa_performance_plot.png')

plt.show()

# %%
# Save the figure# %%

# %%
