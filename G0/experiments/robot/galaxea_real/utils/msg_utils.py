import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

def act_to_joint(action):
    joint_msg = JointState()
    joint_msg.position = action
    return joint_msg

def threshold_input(value, thres=0):
    """_summary_
    Thresholds the input value to zero if it is below a certain threshold.
    This is useful for ignoring small values in the input to avoid disturbances.
    Args:
        value (_type_): _description_
        thres (float, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if abs(value) > thres:
        return value
    else:
        return 0.0

def act_to_twist(action, action_thres=0):
    """
    Converts an action array into a ROS TwistStamped message with optional thresholding.
    This function creates a TwistStamped ROS message, sets its header timestamp to the current time,
    and assigns the linear and angular components based on the provided action values after applying
    a threshold via the 'threshold_input' function. It expects the action array to contain at least six
    elements, where the first three correspond to linear velocities (x, y, z) and the last three to 
    angular velocities (x, y, z).
    Args:
        action (iterable): A sequence of at least six numerical values representing the desired
            linear and angular velocities.
        action_thres (float, optional): The threshold value to be applied to the action inputs.
            Defaults to 0.
    Returns:
        TwistStamped: A ROS message with its timestamp set to the current time and its twist
            attributes (linear and angular) populated with the thresholded values from 'action'.
    """
    action_cmd_msg = TwistStamped()
    action_cmd_msg.header.stamp = rospy.Time.now()
    # action_cmd_msg.header.frame_id = "base_link"
    
    action_cmd_msg.twist.linear.x = threshold_input(action[0], action_thres)
    action_cmd_msg.twist.linear.y = threshold_input(action[1], action_thres)
    action_cmd_msg.twist.linear.z = threshold_input(action[2], action_thres)
    action_cmd_msg.twist.angular.x = threshold_input(action[3], action_thres)
    action_cmd_msg.twist.angular.y = threshold_input(action[4], action_thres)
    action_cmd_msg.twist.angular.z = threshold_input(action[5], action_thres)
    return action_cmd_msg
