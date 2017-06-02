import numpy as np
import sys
import matplotlib.pyplot as plt
from RRM_lib import Radio_Resource_Manager

"""
Simulation Parameters
"""

downlink_bw = 20e6          # Hz
time_slot_duration = 1.0    # msec
N_MS = 8                  # number of mobile stations

N_array = [4]   # number of users
N_array = range(1, 1000)
T = 5         # number of simulations slots

labels = []

for N in N_array:

    RRM = Radio_Resource_Manager(RRM_id=1, N=N, N_MS=N_MS, downlink_bw=downlink_bw)
    RRM.assign_class_to_users()
    RRM.assign_SE_to_MS()

    pkt = RRM.get_new_packet()
    pkt["src"] = 1
    pkt["dest"] = 4
    pkt["size"] = 1600
    RRM.transmit_queue.append(pkt)

    #for slt in range(T):

        # update user state
        #RRM.update_user_state(current_slt=slt)

        # check if a user has a packet to add to scheduler queue
        #RRM.update_user_queues(slt)

        # call on the scheduler to
        #RRM.priority_based_scheduler()

        # call in transmitter portion of radio

        #RRM.call_transmitter(current_slt=slt)

    #print RRM.transmissions_per_user
    #print RRM.delays_per_user