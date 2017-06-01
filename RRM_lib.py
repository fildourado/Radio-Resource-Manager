import numpy as np
import random
import math
from collections import deque

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

packet = {"src": -1,  # User ID
          "dest": -1,  # MS ID
          "size": -1,  # bits
          "TOD": -1,  # Time of departure from User
          "TOA": -1}  # Time of arrival at MS


class Radio_Resource_Manager(object):
    """ 
    Radio Resource Manager
    """

    def __init__(self, RRM_id, N, N_MS, downlink_bw):
        print "Initializing Radio Resource Manager"

        # parameters
        self.RRM_id = RRM_id                            # which RRM is this
        self.N = N                                      # number of users
        self.N_MS = N_MS                                # number of mobile stations
        self.downlink_bw = downlink_bw                  # downlink bw


        self.T_pkt_1 = (class_1.get("packet_size") / class_1.get("des_throughput"))*1e3
        self.T_pkt_2 = (class_2.get("packet_size") / class_2.get("des_throughput"))*1e3
        self.T_pkt_3 = (class_3.get("packet_size") / class_3.get("des_throughput"))*1e3

        self.N_burst_1 = class_1.get("burst") / self.T_pkt_1
        self.N_burst_2 = class_2.get("burst") / self.T_pkt_2
        #self.N_burst_3 = class_1.get("burst") / self.T_pkt_1

        self.class_lookup = [class_1, class_2, class_3]
        self.N_burst_lookup = [self.N_burst_1, self.N_burst_2, -1]
        self.MS_SE = [0]*self.N_MS                      # contains the spectral efficiencies assigned to each MS


        print self.N_burst_1
        print self.N_burst_2

        # state tracking variables
        self.transmit_queue = deque()       # scheduler queued packets that need to be transmitted

        self.delays_per_user = [0] * self.N
        self.transmissions_per_user = [0] * self.N

        self.pkt_to_tx = packet.copy()
        self.transmitting_to_MS = False
        self.tx_expiration = -1

        self.user_class_map = [0] * self.N  # tracks each users class ID
        self.user_queues = []



        # init queues and lists
        self.create_user_deques()

    def reset(self):
        print "Resetting RRM"

    """
    Main Functions
    """

    #
    def update_user_state(self, current_slt):
        for usr in range(self.N):
            # if the user is idle, check if its time to burst
            if self.usr_state[usr] == 0:
                if self.usr_start_time[usr] == current_slt:
                    self.usr_state[usr] = 1
                    class_id = self.user_class_map[usr]
                    burst_period = self.class_lookup[class_id].get("burst")
                    N = self.N_burst_lookup[class_id]
                    self.burst_tx_time[usr] = current_slt + np.linspace(0, burst_period-1, N)

            # user is in burst period
            #else:




    def update_user_queues(self, current_slt):
        print "\nUpdating User Queues**"
        # check the current state of each user. If in a burst period, add a packet to its queue.
        for usr in range(self.N):

            # state of 0 is idle
            if self.usr_state[usr] == 0:
                self.usr_state[usr] = 0

            # state of 1 is burst so add packet to queue
            else:
                # make a new packet
                new_pkt = packet.copy()
                new_pkt["src"] = usr
                new_pkt["dest"] = self.get_rand_ms_id()
                new_pkt["size"] = self.class_lookup[self.user_class_map[usr]]["packet_size"]
                new_pkt["TOD"] = current_slt

                # add the packet to the deque
                self.user_queues[usr].appendleft(packet)


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



    """
    Helper Functions
    """
    def assign_SE_to_MS(self):
        for ms_id in range(self.N_MS):
            self.MS_SE[ms_id] = self.get_random_MS_Spectral_Efficiency()


    def assign_class_to_users(self):
        for usr in range(self.N):
            self.user_class_map[usr] = self.get_class_label()


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

    def get_new_packet(self):
        return packet.copy()