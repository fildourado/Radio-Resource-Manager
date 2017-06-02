import numpy as np
import random
import math
from collections import deque
import sys

"""
Create helpful data structures
"""

class_1 = {"priority": 1,               # priority
           "burst": 400.0,              # activity burst msec
           "pause": 600.0,              # pause msec
           "des_throughput": 50e3,      # bits/sec
           "max_queue_delay": 60.0,     # msec for 90% of packets
           "packet_size": 480}

class_2 = {"priority": 2,               # priority
           "burst": 1000.0,             # activity burst msec
           "pause": 4000.0,             # pause msec
           "des_throughput": 1e6,       # bits/sec
           "max_queue_delay": 360.0,    # msec for 90% of packets
           "packet_size": 1600}         # bits

class_3 = {"priority": 3,               # priority
           "des_throughput": 0.4e6,     # bits/sec
           "avg_queue_delay": 0.6e-3,   # average queue delay desired
           "packet_size": 1200}         # bits

packet_s = {"src": -1,  # User ID
            "dest": -1,  # MS ID
            "size": -1,  # bits
            "TOD": -1,  # Time of departure from User
            "TOA": -1}  # Time of arrival at MS


class Radio_Resource_Manager(object):
    """ 
    Radio Resource Manager
    """

    def __init__(self, RRM_id, N, N_MS, downlink_bw):
        print "\nInitializing Radio Resource Manager (ID %d)" % (RRM_id)

        ################################################################################################################
        # Input Parameters
        ################################################################################################################
        self.RRM_id = RRM_id                            # which RRM is this
        self.N = N                                      # number of users
        self.N_MS = N_MS                                # number of mobile stations
        self.downlink_bw = downlink_bw                  # downlink bw

        ################################################################################################################
        # Helpful Constants
        ################################################################################################################
        self.T_pkt_1 = (class_1.get("packet_size") / class_1.get("des_throughput"))*1e3
        self.T_pkt_2 = (class_2.get("packet_size") / class_2.get("des_throughput"))*1e3
        self.T_pkt_3 = (class_3.get("packet_size") / class_3.get("des_throughput"))*1e3

        self.N_burst_1 = class_1.get("burst") / self.T_pkt_1            # number of packets to Tx in burst mode
        self.N_burst_2 = class_2.get("burst") / self.T_pkt_2            # number of packets to Tx in burst mode

        self.class_lookup = [class_1, class_2, class_3]                 # quick class lookup table based on class ID
        self.N_burst_lookup = [self.N_burst_1, self.N_burst_2, -1]

        #print self.N_burst_1
        #print self.N_burst_2


        ################################################################################################################
        # State Tracking Variables
        ################################################################################################################
        self.usr_state = [0]*self.N
        self.MS_SE = [0]*self.N_MS                      # contains the spectral efficiencies assigned to each MS
        self.usr_start_time = [-1] * self.N

        self.delays_per_user = [0] * self.N
        self.transmissions_per_user = [0] * self.N

        # default to empty packet
        self.pkt_to_tx = self.get_new_packet(-1, -1, -1, -1)

        self.transmitting_to_MS = False
        self.tx_expiration = -1

        self.user_class_map = [0] * self.N  # tracks each users class ID

        # User queues and transmit queues - actually implemented as deques
        self.transmit_queue = deque()           # scheduler queued packets that need to be transmitted
        self.user_queues = []                   # stores the queue of each user
        self.create_user_deques()               #

        self.f = 0.0    # total throughput rate requested by N users
        self.N1 = 0     # number of class 1 users
        self.N2 = 0     # number of class 2 users
        self.N3 = 0     # number of class 3 users



    def reset(self):
        print "Resetting RRM"

    """
    Main Functions
    """

    #
    def update_user_state(self, current_slt):
        for usr in range(self.N):
            class_id = self.user_class_map[usr]
            print "\tUpdating usr %d class %d" % (usr, class_id)

            if class_id == 3:
                print "error: not implemented yet"
            else:
                # class 1 or 2 type user

                # if the user is idle, check if its time to burst
                if self.usr_state[usr] == 0:
                    if self.usr_start_time[usr] == current_slt:
                        self.usr_state[usr] = 1                                 # update user state to burst mode
                        class_id = self.user_class_map[usr] - 1                 # get the class id - 0 indexed for LUT
                        burst_period = self.class_lookup[class_id].get("burst") # get the duration of the burst period
                        N = self.N_burst_lookup[class_id]
                        if N == -1:
                            print "Error: Should never be looking at class ID 3 in this part of code"
                            sys.exit()

                        # create a Time-To-Transmit vector so packets will be sent at a specific time during a burst
                        self.burst_tx_time[usr] = current_slt + np.linspace(0, burst_period-1, N)

                    elif self.usr_start_time[usr] == -1:
                        print "Error: Should never have negative start time"
                        sys.exit()



                        # user is in burst period
            #else:




    def update_user_queues(self, current_slt):
        print "\nUpdating User Queues**"
        # check the current state of each user. If in a burst period, add a packet to its queue.
        for usr in range(self.N):

            # state of 0 is idle
            if self.usr_state[usr] == 0:
                # stay idle
                self.usr_state[usr] = 0

            # state of 1 is burst so add packet to queue
            else:
                class_id = self.user_class_map[usr]
                size = self.class_lookup[class_id]["packet_size"]
                # make a new packet
                new_pkt = self.get_new_packet(usr, self.get_rand_ms_id(), size, current_slt)

                # add the packet to the that users queue
                self.push_packet(usr, new_pkt)


    def priority_based_scheduler(self):
        print "\n**Priority Based Scheduler**"
        a = 0
        # check the user queues for who gets to transmit next
        #for usr in range(self.N):
         #   if self.user_queues[usr].empty():

    def call_transmitter(self, current_slt):
        print "\n**Calling Transmitter**"

        # if currently transmitting then check if done
        if self.transmitting_to_MS:
            if self.tx_expiration <= current_slt:
                print "TX Finished"

                self.transmitting_to_MS = False
                src = self.pkt_to_tx.get("src")
                print src
                TOD = self.pkt_to_tx.get("TOD")
                self.delays_per_user[src] += (current_slt - TOD)
                self.transmissions_per_user[src] += 1

            else:
                print "Currently TX'ing"

        else:
            # if transmit queue populates and transmitter idle, pop a packet off and begin transmitting
            if self.transmit_queue:
                print "Deque Not Empty"

                # Transmit as many buffers as we can in 1 msec

                self.pkt_to_tx = self.transmit_queue.pop()
                self.transmitting_to_MS = True

                ms_dest = self.pkt_to_tx.get("dest")
                print ms_dest

                usr_src = self.pkt_to_tx.get("src")
                print usr_src

                tx_rate = self.MS_SE[ms_dest] * self.downlink_bw  # bps
                print tx_rate

                print (self.pkt_to_tx.get("size") / tx_rate) * 1e3

                tx_duration = math.ceil( (self.pkt_to_tx.get("size") / tx_rate) * 1e3)        # msec



                print int(tx_duration)
                self.tx_expiration = current_slt + int(tx_duration)


            else:
                print "Deque Empty - Nothing to TX"


    ####################################################################################################################
    """
    Helper Functions
    """
    ####################################################################################################################

    def randomize_usr_start_time(self, max_num_slts):
        if max_num_slts == 0:
            self.usr_start_time = [0] * self.N
        else:
            self.usr_start_time = [random.randrange(0, max_num_slts, 1) for _ in range(self.N)]

    def push_packet(self, usr, pkt):
        self.user_queues[usr].appendleft(pkt)

    def pop_packet(self, usr):
        return self.user_queues[usr].pop()

    def assign_SE_to_MS(self):
        for ms_id in range(self.N_MS):
            self.MS_SE[ms_id] = self.get_random_MS_Spectral_Efficiency()


    def assign_class_to_users(self):
        self.f = self.N / ((0.3/20e3) + (0.4/200e3) + (0.3/400e3)) # bps
        self.N1 = int(round((0.3 * self.f)/(20e3)))
        self.N2 = int(round((0.4 * self.f) / (200e3)))
        self.N3 = int(round((0.3 * self.f) / (400e3)))

        #self.N1 = (0.3 * self.f)/(20e3)
        #self.N2 = (0.4 * self.f) / (200e3)
        #self.N3 = (0.3 * self.f) / (400e3)

        print "Number of users: %d" %(self.N)
        sum_of_users = self.N1 + self.N2 + self.N3
        diff = self.N - sum_of_users
        if diff > 0:
            if diff == 1:
                self.N2 += 1
                print "Off by 1, correcting"
                #print "User Class Distribution: 1/2/3 = %f/%f/%f " % (self.N1, self.N2, self.N3)
                print "User Class Distribution: 1/2/3 = %d/%d/%d " % (self.N1, self.N2, self.N3)

            else:
                print "Error: user class distribution no bueno, %d users required" % (self.N)
                sys.exit()
        elif diff< 0:
            if diff == -1:
                self.N2 += -1
                print "Off by -1, correcting"
                #print "User Class Distribution: 1/2/3 = %f/%f/%f " % (self.N1, self.N2, self.N3)
                print "User Class Distribution: 1/2/3 = %d/%d/%d " % (self.N1, self.N2, self.N3)
            else:
                print "Error: More users than %d? Not possible" % (self.N)
                sys.exit()
        else:
            #print "User Class Distribution: 1/2/3 = %f/%f/%f " % (self.N1, self.N2, self.N3)
            print "User Class Distribution: 1/2/3 = %d/%d/%d " % (self.N1, self.N2, self.N3)

        self.user_class_map = [1]*self.N1 + [2]*self.N2 + [3]*self.N3
        # print self.user_class_map
        random.shuffle(self.user_class_map)
        # print self.user_class_map

    def get_class_label(self):
        prob = random.random()
        class_label = -1
        if prob > .7:
            class_label = 3
        elif prob >.3:
            class_label = 2
        else:
            class_label = 1

        return class_label

    def get_random_MS_Spectral_Efficiency(self):
        prob = random.random()
        # for 40% of MS
        SE_1 = 0.2  # bps/Hz
        # for 30% of MS
        SE_2 = 1.0  # bps/Hz
        # for 30% of MS
        SE_3 = 2.0  # bps/Hz
        if prob > 0.6:
            return SE_1
        elif prob > 0.3:
            return SE_2
        else:
            return SE_3

    def get_rand_ms_id(self):
        return random.randint(0,(self.N_MS-1))

    def create_user_deques(self):
        for n in range(self.N):
            dq = deque()
            self.user_queues.append(dq)

    def get_new_packet(self, src, dest, size, TOD):
        new_pkt = packet_s.copy()
        new_pkt["src"] = src
        new_pkt["dest"] = dest
        new_pkt["size"] = size
        new_pkt["TOD"] = TOD
        return new_pkt
