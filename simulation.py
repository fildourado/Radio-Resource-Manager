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
#N_array = range(1, 10)
T = 5         # number of simulations slots

labels = []

for N in N_array:

    RRM = Radio_Resource_Manager(RRM_id=1, N=N, N_MS=N_MS, downlink_bw=downlink_bw)
    RRM.assign_class_to_users()
    RRM.assign_SE_to_MS()
    RRM.randomize_usr_start_time(0)

    pkt = RRM.get_new_packet(1, 4, 1600, -1)
    RRM.transmit_queue.append(pkt)

    RRM.push_packet(1, pkt)
    print RRM.user_queues
    new_pkt = RRM.pop_packet(1)
    print new_pkt
    print RRM.user_queues

    for slt in range(T):

        # update user state
        RRM.update_user_state(current_slt=slt)

        # check if a user has a packet to add to scheduler queue
        #RRM.update_user_queues(slt)

        # call on the scheduler to
        #RRM.priority_based_scheduler()

        # call in transmitter portion of radio

        #RRM.call_transmitter(current_slt=slt)

    #print RRM.transmissions_per_user
    #print RRM.delays_per_user