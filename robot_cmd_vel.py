import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Twist

from motor import Motor

def callback(data):
    print data


if __name__ == "__main__":
    m = Motor()
    rospy.init_node("robot_cmd_vel", anonymous=True)
    subscriber = rospy.Subscriber('robot_cmd_vel', Twist, callback)
    publisher = rospy.Publisher('robot_command', UInt8, queue_size=2)
    rospy.spin()
