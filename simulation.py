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

# Choose the Scheduler to use
scheduler = 2

T = 10000         # number of simulations slots

Tf = 30 #90  # 1 msec slots
backoff = -0.0
c1_per = 28.0 - backoff/2
c2_per = 56.5 - backoff/2
c3_per = 15.5 + backoff

W = np.array([c1_per / 100, c2_per / 100, c3_per / 100])


c_sum = np.sum(np.array([c1_per, c2_per, c3_per]))
print c_sum
if c_sum != 100.0:
    print "Weights are wrong"
    sys.exit()


if scheduler == 0:
    N_usrs = 100

elif scheduler == 1:
    N_usrs = 70

elif scheduler == 2:
    N_usrs = 70

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

c1_pop_below_QOS = []
c2_pop_below_QOS = []

MS_throughput = []
for i in range(N_MS):
    MS_throughput.append([[], [], []])

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
            RRM.WRR_PFT_scheduler(class_durations=class_durations, current_slt=slt)

        # call in transmitter portion of radio to actually TX the packets
        RRM.call_transmitter(current_slt=slt)


    a = RRM.max_delay_per_class[0]
    if a == None:
        print "got 0"
        a = 0.0
    #class_1_max_delay.append(RRM.max_delay_per_class[0])
    class_1_max_delay.append(a)

    a = RRM.max_delay_per_class[1]
    if a == None:
        print "got 1"
        a = 0.0
    #class_2_max_delay.append(RRM.max_delay_per_class[1])
    class_2_max_delay.append(a)

    a = RRM.max_delay_per_class[2]
    if a == None:
        print "got 2"
        a = 0.0
    #class_3_max_delay.append(RRM.max_delay_per_class[2])
    class_3_max_delay.append(a)

    num_tx = RRM.transmissions_per_user
    c1_idx = RRM.class_n_usrs[0]
    c2_idx = RRM.class_n_usrs[1]
    c3_idx = RRM.class_n_usrs[2]

    if len(c1_idx) == 0:
        class_1_tx = 0.0
        bps_1 = 0.0
    else:
        class_1_tx = num_tx[c1_idx]
        class_1_tx = np.sum(class_1_tx)
        bps_1 = ((1.0*class_1_tx / (T*1e-3)) * RRM.class_lookup[0].get("packet_size")) / len(c1_idx)

    if len(c2_idx) == 0:
        class_2_tx = 0.0
        bps_2 = 0.0
    else:
        class_2_tx = num_tx[c2_idx]
        class_2_tx = np.sum(class_2_tx)
        bps_2 = ((1.0*class_2_tx / (T*1e-3)) * RRM.class_lookup[1].get("packet_size")) / len(c2_idx)

    if len(c3_idx) == 0:
        class_3_tx = 0.0
        bps_3 = 0.0
    else:
        class_3_tx = num_tx[c3_idx]
        class_3_tx = np.sum(class_3_tx)
        bps_3 = ((1.0*class_3_tx / (T*1e-3)) * RRM.class_lookup[2].get("packet_size")) / len(c3_idx)

    class_1_throughput.append(bps_1/1e3) # kbps
    class_2_throughput.append(bps_2/1e3) # kbps
    class_3_throughput.append(bps_3/1e3) # kbps


    # average delay per class
    #d_per_usr = RRM.delays_per_user
    #c1_delays = d_per_usr[c1_idx]
    #c2_delays = d_per_usr[c2_idx]
    #c3_delays = d_per_usr[c3_idx]

    c1_delays_per_class = np.array(RRM.packet_delays_per_class[0])
    if len(c1_delays_per_class) == 0:
        c1_delays_per_class = np.zeros(1)

    c2_delays_per_class = np.array(RRM.packet_delays_per_class[1])
    if len(c2_delays_per_class) == 0:
        c2_delays_per_class = np.zeros(1)

    c3_delays_per_class = np.array(RRM.packet_delays_per_class[2])
    if len(c3_delays_per_class) == 0:
        c3_delays_per_class = np.zeros(1)

    c1_avg_delays = np.average(c1_delays_per_class)
    c2_avg_delays = np.average(c2_delays_per_class)
    c3_avg_delays = np.average(c3_delays_per_class)

    class_1_avg_delay.append(c1_avg_delays)
    class_2_avg_delay.append(c2_avg_delays)
    class_3_avg_delay.append(c3_avg_delays)

    c1_std_delays = np.std(c1_delays_per_class)
    c2_std_delays = np.std(c2_delays_per_class)
    c3_std_delays = np.std(c3_delays_per_class)

    class_1_std_delay.append(c1_std_delays)
    class_2_std_delay.append(c2_std_delays)
    class_3_std_delay.append(c3_std_delays)

    # percent of packets with delay below
    c1_pop_below_QOS.append( 100.0*len(np.where(c1_delays_per_class < 60.0)[0]) / len(c1_delays_per_class))
    c2_pop_below_QOS.append( 100.0*len(np.where(c2_delays_per_class < 360.0)[0]) / len(c2_delays_per_class))

    for i in range(N_MS):
        ms_c1_t = (1.0*RRM.MS_received_pkts[i][0] / (T * 1e-3)) # packets/sec
        ms_c2_t = (1.0*RRM.MS_received_pkts[i][1] / (T * 1e-3)) # packets/sec
        ms_c3_t = (1.0*RRM.MS_received_pkts[i][2] / (T * 1e-3)) # packets/sec

        MS_throughput[i][0].append(ms_c1_t)
        MS_throughput[i][1].append(ms_c1_t)
        MS_throughput[i][2].append(ms_c1_t)


plt.figure(1)
plt.plot(N_array, class_1_throughput, label='Class 1', color='r')
plt.plot(N_array, class_2_throughput, label='Class 2', color='b')
plt.plot(N_array, class_3_throughput, label='Class 3', color='g')
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
plt.plot(N_array, class_1_avg_delay, label='Class 1', color='r')
plt.plot(N_array, class_2_avg_delay, label='Class 2', color='b')
plt.plot(N_array, class_3_avg_delay, label='Class 3', color='g')

# plot QOS limits
plt.plot(N_array, np.ones(len(N_array))*60.0, label='Class 1 QOS Limit', color='r', linestyle='--')
plt.plot(N_array, np.ones(len(N_array))*360.0, label='Class 2 QOS Limit', color='b', linestyle='--')
plt.plot(N_array, np.ones(len(N_array))*600.0, label='Class 3 QOS Limit', color='g', linestyle='--')

plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Average Delay (ms)")
plt.title("Number of Users vs Average Delay Per Class")
plt.ylim([0, 700])

if scheduler == 0:
    plt.savefig("figures/Priority_Oriented_Avg_Delay.png")
elif scheduler == 1:
    plt.savefig("figures/WRR_Avg_Delay.png")
elif scheduler == 2:
    plt.savefig("figures/WRR_PFT_Avg_Delay.png")

plt.figure(3)
plt.plot(N_array, class_1_std_delay, label='Class 1', color='r')
plt.plot(N_array, class_2_std_delay, label='Class 2', color='b')
plt.plot(N_array, class_3_std_delay, label='Class 3', color='g')
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


plt.figure(4)
plt.plot(N_array, c1_pop_below_QOS, label='Class 1 (Below 60 msec)', color='r')
plt.plot(N_array, c2_pop_below_QOS, label='Class 2 (Below 360 msec)', color='b')

plt.plot(N_array, np.ones(len(N_array))*90.0, label='90% Threshold', color='g', linestyle='--')

plt.legend()
plt.grid()
plt.xlabel("Number of Users")
plt.ylabel("Percent of Packets")
plt.title("Percent of Packets Below Class 1 and Class 2 QoS Delay Restriction")

if scheduler == 0:
    plt.savefig("figures/Priority_Oriented_POP_Delay.png")
elif scheduler == 1:
    plt.savefig("figures/WRR_POP_Delay.png")
elif scheduler == 2:
    plt.savefig("figures/WRR_PFT_POP_Delay.png")

# plot the throughput for per class for each MS
for i in range(N_MS):
    plt.figure((5 + i))

    filename = "MS_%d_Throughput_" % (i)
    s_name = ""
    if scheduler == 0:
        s_name = "PO.png"
    elif scheduler == 1:
        s_name = "WRR.png"
    elif scheduler == 2:
        s_name = "WRR_PFT.png"

    c1_throughput = (np.array(MS_throughput[i][0]) * RRM.class_lookup[0].get("packet_size"))/1e6
    c2_throughput = (np.array(MS_throughput[i][1]) * RRM.class_lookup[1].get("packet_size"))/1e6
    c3_throughput = (np.array(MS_throughput[i][2]) * RRM.class_lookup[2].get("packet_size"))/1e6

    plt.plot(N_array, c1_throughput, label='Class 1', color='r')
    plt.plot(N_array, c2_throughput, label='Class 2', color='g')
    plt.plot(N_array, c3_throughput, label='Class 3', color='b')
    plt.plot(N_array, np.ones(len(N_array))*RRM.MS_SE[i]/1e6, label='Max Throughput', color='k', linestyle='--')

    plt.legend()
    plt.grid()
    plt.xlabel("Number of Users")
    plt.ylabel("Throughput (Mbps)")
    title_n = "Number of Users vs Mobile Station %d Throughput Per Class" % (i)
    plt.title(title_n)
    path = "figures/"+filename+s_name
    plt.savefig(path)
