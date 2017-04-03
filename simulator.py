import rospy
import numpy as np
import math

from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from stdr_msgs.srv import MoveRobot

from time import sleep

from motor import Motor


class Simulator():

    def __init__(self):
        rospy.init_node('learning_environment', anonymous=True)
        self.update_box = rospy.ServiceProxy('/stdr_server/updateBox', Empty)
        # self.reset = rospy.ServiceProxy('/stdr_server/reset', Empty)
        self.reset_robot = rospy.ServiceProxy('/robot0/replace', MoveRobot)
        self.vel_pub = rospy.Publisher('/robot0/cmd_vel', Twist, queue_size=2)
        self.motor = Motor();

    def reset(self):
        newPose = Pose2D()
        x_upper_limit = (15.5 - 0.7)
        x_lower_limit = 0.7
        y_upper_limit = (14.92 - 0.7)
        y_lower_limit = 0.7
        newPose.x = (x_upper_limit - x_lower_limit) * np.random.random_sample() + x_lower_limit
        newPose.y = (y_upper_limit - y_lower_limit) * np.random.random_sample() + y_lower_limit
        newPose.theta = 2*math.pi * np.random.random_sample()
        rospy.wait_for_service('/robot0/replace')
        try:
            self.reset_robot(newPose)
        except rospy.ServiceException as exc:
            print ('Service did not process request' + str(exc))


    def getState(self, target_pose):
        pose_data = []
        range_data = []
        raw_data = None
        raw_current_pose = None
        done = False
        collision = False
        while raw_data is None:
            try:
                raw_data = rospy.client.wait_for_message('/robot0/laser_0', LaserScan, timeout = 5)
            except Exception as e:
                raise

        while raw_current_pose is None:
            try:
                raw_current_pose = rospy.client.wait_for_message('/robot0/odom', Odometry, timeout = 5)
            except Exception as e:
                raise

        for i , item in enumerate(raw_data.ranges):
            if raw_data.ranges[i] == float('Inf'):
                range_data.append(round(5.00, 2))
            elif np.isnan(raw_data.ranges[i]):
                range_data.append(round(0.60, 2))
            else:
                range_data.append(round(raw_data.ranges[i], 2))
            if (0.4 > raw_data.ranges[i] > 0):
                done = True
                collision = True

        (roll, pitch, yaw) = euler_from_quaternion([raw_current_pose.pose.pose.orientation.x, \
            raw_current_pose.pose.pose.orientation.y , \
            raw_current_pose.pose.pose.orientation.z , \
            raw_current_pose.pose.pose.orientation.w])
        pose_data.append(raw_current_pose.pose.pose.position.x)
        pose_data.append(raw_current_pose.pose.pose.position.y)
        pose_data.append(yaw)
        """
            Check is the target position is reached
        """
        if ((target_pose[0] +0.05 > pose_data[0] > target_pose[0]-0.05) and \
            (target_pose[1] +0.05 > pose_data[1] > target_pose[1]-0.05 )):
                done = True

        return range_data + pose_data, done, collision

    def sendAction(self, action):
        cmd_vel = self.motor.getTwist(action)
        self.vel_pub.publish(cmd_vel)
        sleep(0.5)
        cmd_vel = self.motor.stop()
        self.vel_pub.publish(cmd_vel)
        # self.vel_pub

    def step(self, action, target_pose):

        rospy.wait_for_service('/stdr_server/updateBox')
        try:
            self.update_box()
        except rospy.ServiceException, e:
            print "/stdr_server/updateBox service call failed"

        self.sendAction(action)

        state, done, collision = self.getState(target_pose)

        if (done == True and collision == True):
            reward = -100
        elif (done == True and collision == False):
            reward = +200
        else:
            reward = -0.1

        return state, reward , done , {}






s = Simulator()
observation, done, collision = s.getState([3.76, 2.46, 1.23]);
# print len(observation)
# print len(s.motor.action_space)
s.reset()
