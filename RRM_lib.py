import numpy as np
import random
from collections import deque

"""
Create helpful data structures
"""
class_1 = {"priority": 1,  # priority
           "burst": 400.0,  # activity burst msec
           "pause": 600.0,  # pause msec
           "des_throughput": 50000.0,  #
           "max_queue_delay": 60.0,
           "packet_size": 480}

class_2 = {"priority": 2,  # priority
           "burst": 1000.0,  # activity burst msec
           "pause": 4000.0,  # pause msec
           "des_throughput": 1e6,  # QoS desired throughput
           "max_queue_delay": 360.0,  # msec for 90% fo packets
           "packet_size": 1600}  # bits

packet = {"src": -1,  # User ID
          "dest": -1,  # MS ID
          "size": -1,  # bits
          "TOD": -1,  # Time of departure from User
          "TOA": -1}  # Time of arrival at MS


class Radio_Resource_Manager(object):
    """ 
    Radio Resource Manager
    """

    def __init__(self, RRM_id, N, N_MS):
        print "Initializing Radio Resource Manager"

        # parameters
        self.RRM_id = RRM_id        # which RRM is this
        self.N = N                  # number of users
        self.N_MS = N_MS            # number of mobile stations


        # state tracking variables
        self.transmitting_to_MS = False
        self.user_queues = []


        # init queues and lists
        self.create_user_deques()

    def reset(self):
        print "Resetting RRM"



    """
    Main Functions
    """

    def assign_class_to_users(self):
        return packet


    def priority_based_scheduler(self):

        # check the user queues for who gets to transmit next

        # if currently transmitting to MS check if transmission has ended
        if self.transmitting_to_MS:
            self.transmitting_to_MS = False



    """
    Helper Functions
    """
    def get_class_label(self):
        prob = random.random()
        class_label = -1
        if prob > .7:
            class_label = 3
        elif prob >.4:
            class_label = 2
        else:
            class_label = 1

        return class_label

    def get_rand_ms_id(self):
        return random.randint(0,(self.N_MS-1))

    def create_user_deques(self):
        for n in range(self.N):
            dq = deque()
            self.user_queues.append(dq)
