import numpy as np
import rospy

from std_msgs.msg import UInt8
from geometry_msgs.msg import Twist

class Motor:
    def __init__(self):
         self.wheel_velocity_table = {}
         self.robot_movement_table = {}
         self.loadConfig()
         if (self.validate() == False):
             raise ValueError('Motor Configuation File is occupted')
         self.prepareTable()
         self.action_space = self.prepareActionSpace()

    def transformTwist(self, twist):
        min_diff = float('inf')
        for key, value in self.robot_movement_table.items():
            this_diff = self.computeTwistDifferent(value, twist)
            if min_diff > this_diff:
                min_diff = this_diff
                closest_command = key

        print self.robot_movement_table.get(closest_command)
        print closest_command
        return closest_command

    def computeTwistDifferent(self, target_twist, reference_twist):
        diff = 0
        diff += abs(target_twist.linear.x - reference_twist.linear.x )
        diff += abs(target_twist.linear.y - reference_twist.linear.y )
        diff += abs(target_twist.linear.z - reference_twist.linear.z )
        diff += abs(target_twist.angular.x - reference_twist.angular.x )
        diff += abs(target_twist.angular.y - reference_twist.angular.y )
        diff += abs(target_twist.angular.z - reference_twist.angular.z )
        return diff


    def stop(self):
        twist = self.robot_movement_table.get((64,192))
        return twist

    def getTwist(self, action_number):
        if (256 > action_number >= 0):
            action_key = self.action_space[action_number]
            print "A : {}".format(action_key)
            twist = self.robot_movement_table.get(action_key)
        else:
            print "Get Twist out of bound"
            twist = self.robot_movement_table.get((64,192))
        return twist

    def prepareActionSpace(self):
        action_space = []
        for m1 in range(64, 127+1):
            for m2 in range(192, 255+1):
                if (m1%4 == 0 and m2%4 == 0):
                    action_space.append((m1, m2))
        return action_space

    def loadConfig(self):
        config_list = []
        with open('motor_config.txt') as f:
                config_list = f.read().split(',')
        i = 0
        for item in config_list:
            self.wheel_velocity_table[i] = float(item)/100.0
            i = i+1

    def validate(self):
        valid = False

        for key, value in self.wheel_velocity_table.items():
            if (np.isnan(value) == False):
                valid = True
        return valid

    def prepareTable(self):
        vel_cmd = None
        for m1 in range(0, 127+1):
            for m2 in range(128, 255+1):
                vel_cmd = Twist()
                # should be m2 - m1
                vel_cmd.linear.x = ( self.wheel_velocity_table[m2] + self.wheel_velocity_table[m1] ) / 2.0
                vel_cmd.angular.z = ( self.wheel_velocity_table[m2] - self.wheel_velocity_table[m1] ) / 0.189
                self.robot_movement_table[(m1, m2)] = vel_cmd
                vel_cmd = None

class CmdNode:
    def __init__(self):
        rospy.init_node('cmd_vel_node', anonymous=True)
        self.pub = rospy.Publisher('robot_command', UInt8, queue_size = 10)
        rospy.Subscriber('/robot_cmd_vel', Twist, self.callback)
        self.motor = Motor()


    def callback(self, data):
        command1, command2 = self.motor.transformTwist(data)
        self.pub.publish(command1)
        self.pub.publish(command2)

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()

if __name__ == "__main__":
    node = CmdNode()
    node.run()
