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


scheduler = 1
T = 20000         # number of simulations slots

if scheduler == 0:
    N_usrs = 129

elif scheduler == 1:
    N_usrs = 50
    Tf = 10  # 1 msec slots
    W = np.array([0.15, 0.45, 0.3])

    if np.sum(W) > 1.0:
        print "Weights are wrong"
        sys.exit()

elif scheduler == 2:
    N_usrs = 50

else:
    print "Scheduler Error"
    sys.exit()



N_array = range(1, N_usrs)
class_durations = W * Tf
#class_durations = np.array([1.5, 3.5, 3.0])

# choose the scheduler to simulate



class_1_throughput = []
class_2_throughput = []
class_3_throughput = []

class_1_avg_delay = []
class_2_avg_delay = []
class_3_avg_delay = []

class_1_std_delay = []
class_2_std_delay = []
class_3_std_delay = []

class_1_max_delay = []
class_2_max_delay = []
class_3_max_delay = []

for N in N_array:

    RRM = Radio_Resource_Manager(RRM_id=0, N=N, N_MS=N_MS, downlink_bw=downlink_bw)
    RRM.assign_class_to_users()
    RRM.assign_SE_to_MS()
    RRM.randomize_usr_start_time(10)

    print class_durations

    for slt in range(T):
        #print "\nSLOT: %d" % (slt)
        # update user state
        RRM.update_user_state(current_slt=slt)

        # check if a user has a packet to add to scheduler queue
        RRM.update_user_queues(current_slt=slt)

        # call on the scheduler to schedule packets for transmission
        if scheduler == 0:
            RRM.priority_based_scheduler()
        elif scheduler == 1:
            RRM.WRR_scheduler(class_durations=class_durations, current_slt=slt)
        elif scheduler == 2:
            RRM.WRR_scheduler(class_durations=class_durations, current_slt=slt)

        # call in transmitter portion of radio to actually TX the packets
        RRM.call_transmitter(current_slt=slt)

    class_1_max_delay.append(RRM.max_delay_per_class[0])
    class_2_max_delay.append(RRM.max_delay_per_class[1])
    class_3_max_delay.append(RRM.max_delay_per_class[2])

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

    class_1_throughput.append(bps_1/1e3) # kbps
    class_2_throughput.append(bps_2/1e3) # kbps
    class_3_throughput.append(bps_3/1e3) # kbps


    # average delay per class
    #d_per_usr = RRM.delays_per_user
    #c1_delays = d_per_usr[c1_idx]
    #c2_delays = d_per_usr[c2_idx]
    #c3_delays = d_per_usr[c3_idx]

    c1_delays_per_user = np.array(RRM.packet_delays_per_class[0])
    c2_delays_per_user = np.array(RRM.packet_delays_per_class[1])
    c3_delays_per_user = np.array(RRM.packet_delays_per_class[2])

    c1_avg_delays = np.average(c1_delays_per_user)
    c2_avg_delays = np.average(c2_delays_per_user)
    c3_avg_delays = np.average(c3_delays_per_user)

    class_1_avg_delay.append(c1_avg_delays)
    class_2_avg_delay.append(c2_avg_delays)
    class_3_avg_delay.append(c3_avg_delays)

    c1_std_delays = np.std(c1_delays_per_user)
    c2_std_delays = np.std(c2_delays_per_user)
    c3_std_delays = np.std(c3_delays_per_user)

    class_1_std_delay.append(c1_std_delays)
    class_2_std_delay.append(c2_std_delays)
    class_3_std_delay.append(c3_std_delays)



plt.figure(1)
plt.plot(N_array, class_1_throughput, label='Class 1')
plt.plot(N_array, class_2_throughput, label='Class 2')
plt.plot(N_array, class_3_throughput, label='Class 3')
plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Throughput (Kbps)")
plt.title("Number of Users vs Throughput Per Class")
if scheduler == 0:
    plt.savefig("figures/Priority_Oriented_Throughput.png")
elif scheduler == 1:
    plt.savefig("figures/WRR_Throughput.png")
elif scheduler == 2:
    plt.savefig("figures/WRR_PFT_Throughput.png")

#plt.show()

plt.figure(2)
plt.plot(N_array, class_1_avg_delay, label='Class 1')
plt.plot(N_array, class_2_avg_delay, label='Class 2')
plt.plot(N_array, class_3_avg_delay, label='Class 3')
plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Average Delay (ms)")
plt.title("Number of Users vs Average Delay Per Class")

if scheduler == 0:
    plt.savefig("figures/Priority_Oriented_Avg_Delay.png")
elif scheduler == 1:
    plt.savefig("figures/WRR_Avg_Delay.png")
elif scheduler == 2:
    plt.savefig("figures/WRR_PFT_Avg_Delay.png")


plt.figure(3)
plt.plot(N_array, class_1_std_delay, label='Class 1')
plt.plot(N_array, class_2_std_delay, label='Class 2')
plt.plot(N_array, class_3_std_delay, label='Class 3')
plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Standard Deviation of Delay (ms)")
plt.title("Number of Users vs Standard Deviation of Delay Per Class")
if scheduler == 0:
    plt.savefig("figures/Priority_Oriented_Std_Delay.png")
elif scheduler == 1:
    plt.savefig("figures/WRR_Std_Delay.png")
elif scheduler == 2:
    plt.savefig("figures/WRR_PFT_Std_Delay.png")



