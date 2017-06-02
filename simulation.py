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
N_array = range(1, 100)
T = 10000         # number of simulations slots

class_1_throughput = []
class_2_throughput = []
class_3_throughput = []

for N in N_array:

    RRM = Radio_Resource_Manager(RRM_id=1, N=N, N_MS=N_MS, downlink_bw=downlink_bw)
    RRM.assign_class_to_users()
    RRM.assign_SE_to_MS()
    RRM.randomize_usr_start_time(0)

    #pkt = RRM.get_new_packet(1, 4, 1600, -1)
    #RRM.transmit_queue.append(pkt)
    #RRM.push_packet(1, pkt)
    #print RRM.user_queues
    #new_pkt = RRM.pop_packet(1)
    #print new_pkt
    #print RRM.user_queues

    for slt in range(T):
        #print "\nSLOT: %d" % (slt)
        # update user state
        RRM.update_user_state(current_slt=slt)

        # check if a user has a packet to add to scheduler queue
        RRM.update_user_queues(current_slt=slt)

        # call on the scheduler to schedule packets for transmission
        RRM.priority_based_scheduler()

        # call in transmitter portion of radio to actually TX the packets
        RRM.call_transmitter(current_slt=slt)



    num_tx = RRM.transmissions_per_user
    c1_idx = RRM.class_n_usrs[0]
    c2_idx = RRM.class_n_usrs[1]
    c3_idx = RRM.class_n_usrs[2]

    class_1_tx = num_tx[c1_idx]
    class_1_tx = np.sum(class_1_tx)

    class_2_tx = num_tx[c2_idx]
    class_2_tx = np.sum(class_2_tx)

    class_3_tx = num_tx[c3_idx]
    class_3_tx = np.sum(class_3_tx)

    bps_1 = ((1.0*class_1_tx / (T*1e-3)) * RRM.class_lookup[0].get("packet_size")) / len(c1_idx)
    bps_2 = ((1.0*class_2_tx / (T*1e-3)) * RRM.class_lookup[1].get("packet_size")) / len(c2_idx)
    bps_3 = ((1.0*class_3_tx / (T*1e-3)) * RRM.class_lookup[2].get("packet_size")) / len(c3_idx)

    class_1_throughput.append( bps_1/1e3 ) # kbps
    class_2_throughput.append( bps_2/1e3 ) # kbps
    class_3_throughput.append( bps_3/1e3 ) # kbps

    #print RRM.transmissions_per_user
    #print RRM.delays_per_user



#print class_1_throughput
#print N_array
plt.plot(N_array, class_1_throughput, label='Class 1')
plt.plot(N_array, class_2_throughput, label='Class 2')
plt.plot(N_array, class_3_throughput, label='Class 3')
plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Throughput (Kbps)")
plt.title("Number of Users vs")
plt.show()