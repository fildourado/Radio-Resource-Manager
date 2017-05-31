import numpy as np
import sys
import matplotlib.pyplot as plt
from RRM_lib import Radio_Resource_Manager

"""
Simulation Parameters
"""

downlink_bw = 20e6          # Hz
time_slot_duration = 1.0    # msec
Num_MS = 8                  # number of mobile stations

# for 40% of MS
SE_1 = 0.2                  # bps/Hz
# for 30% of MS
SE_2 = 1.0                  # bps/Hz
# for 30% of MS
SE_3 = 2.0                  # bps/Hz

N_array = [5]   # number of users
T = 100         # number of simulations slots




labels = []

for N in N_array:

    RRM = Radio_Resource_Manager(1, N, N_MS)

    RRM.user_queues[0].appendleft(0)
    RRM.user_queues[1].appendleft(1)
    RRM.user_queues[2].appendleft(2)
    RRM.user_queues[3].appendleft(3)
    RRM.user_queues[4].appendleft(4)

    print RRM.user_queues

    packet = RRM.assign_class_to_users()

    #RRM.assign_QOS_to_Mobile_Station()

    for slt in range(T):

        labels.append(RRM.get_class_label())

        # check if a user has a packet to add to scheduler queue
        #RRM.update_user_queues()

        # call on the scheduler to
        #RRM.call_scheduler()



    RRM.user_queues[0].appendleft(1)
    RRM.user_queues[1].appendleft(2)
    RRM.user_queues[2].appendleft(3)
    RRM.user_queues[3].appendleft(4)
    RRM.user_queues[4].appendleft(5)
    print RRM.user_queues

    RRM.user_queues[1].pop()
    print RRM.user_queues



labels = np.array(labels)

total_num_labels = len(labels)
idx_1 = np.where(labels == 1)
idx_2 = np.where(labels == 2)
idx_3 = np.where(labels == 3)

print 1.0*len(idx_1[0])/total_num_labels
print 1.0*len(idx_2[0])/total_num_labels
print 1.0*len(idx_3[0])/total_num_labels

