from collections import deque
from dataclasses import dataclass, field
from functools import partial
import time
from typing import Dict, Literal

import cv2
import numpy as np
from experiments.robot.galaxea_real.utils import msg_utils
import rospy
from cv_bridge import CvBridge
from enum import Enum
from geometry_msgs.msg import TwistStamped
from loguru import logger
from sensor_msgs.msg import CompressedImage, JointState
from utils import log_utils, msg_utils

R1_LITE = "R1_LITE"
R1 = "R1"

@dataclass
class R1LiteTopicsConfig:
    joint_state_input: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/hdas/feedback_arm_left",
            "left_gripper": "/hdas/feedback_gripper_left",
            "right_arm": "/hdas/feedback_arm_right",
            "right_gripper": "/hdas/feedback_gripper_right",
            "torso": "/hdas/feedback_torso",
            "chassis": "/hdas/feedback_chassis"
        }
    )

    image_input: Dict[str, str] =  field(
        default_factory=lambda: {
            "head_rgb": "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "left_rgb": "/hdas/camera_wrist_left/color/image_raw/compressed",
            "right_rgb": "/hdas/camera_wrist_right/color/image_raw/compressed"
        }
    )

    joint_state_output: Dict[str, str] = field(
        default_factory=lambda: {
            "left_arm": "/motion_target/target_joint_state_arm_left",
            "left_gripper": "/motion_target/target_position_gripper_left",
            "right_arm": "/motion_target/target_joint_state_arm_right",
            "right_gripper": "/motion_target/target_position_gripper_right"
        }
    )

    twist_output: Dict[str, str] = field(
        default_factory=lambda: {
            "torso": "/motion_target/target_speed_torso",
            "chassis": "/motion_target/target_speed_chassis"
        }
    )

@dataclass
class GalaxeaInferfaceConfig:
    hardware: Literal['R1', 'R1_LITE'] = R1_LITE
    topic: R1LiteTopicsConfig = field(default_factory=R1LiteTopicsConfig)

    msg_time_diff_threshold: float = 0.1

    with_torso: bool = False
    with_chassis: bool = False

    camera_deque_length: int = 7
    deque_length: int = 200

    control_freq: int = 15
    dry_run: bool = False
    torso_chassis_thres: float = 0.01  # threshold for torso and chassis, I think 0.01 is a good value

class GalaxeaInterface:
    def __init__(self, config: GalaxeaInferfaceConfig):
        self.config = config

        if not rospy.core.is_initialized():
            rospy.init_node("galaxea_real")
            logger.info("Initialized ROS node")
        else:
            logger.info("ROS node already initialized")

        self.br = CvBridge()
        self.inputs_dict = {}
        self.last_camera_time = 0.
        self.lastest_instruction = ""
        self._init_topics()

        time.sleep(1)

    def _camera_callback(self, msg: CompressedImage, que: list, topic: str):
        img_cv_bgr = self.br.compressed_imgmsg_to_cv2(msg)
        if len(img_cv_bgr.shape) == 3 and img_cv_bgr.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
        elif len(img_cv_bgr.shape) == 3 and img_cv_bgr.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unexpected image format: {img_cv_bgr.shape}")
        # if self.config.hardware == R1_LITE and "head" in topic:
        #     img = img_cv[:, :img_cv.shape[1] // 2]
        img = img_cv
        que.append(
            dict(
                data=img,
                message_time=msg.header.stamp.to_sec(),
            )
        )

    def _joint_states_callback(self, msg: JointState, que: list, topic: str):
        que.append(
            dict(
                position=np.array(msg.position, dtype=np.float32),
                velocity=np.array(msg.velocity, dtype=np.float32),
                message_time=msg.header.stamp.to_sec()
            )
        )

    def _init_topics(self):
        self.subscribers = {}
        self.publishers = {}
        config_topic = self.config.topic
        for topic_dict, topic_type, topic_callback, deque_maxlen in zip(
            [config_topic.joint_state_input, config_topic.image_input],
            [JointState, CompressedImage],
            [self._joint_states_callback, self._camera_callback],
            [self.config.deque_length, self.config.camera_deque_length],
        ):
            for topic, topic_name in topic_dict.items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self.inputs_dict[topic] = deque(
                        maxlen=deque_maxlen
                    )
                    self.subscribers[topic] = rospy.Subscriber(
                        topic_name, topic_type, partial(
                            topic_callback, 
                            que=self.inputs_dict[topic], 
                            topic=topic,
                        )
                    )
                    logger.info(f"Subscriber {topic} created.")

        for topic_dict, topic_type in zip(
            [config_topic.joint_state_output, config_topic.twist_output],
            [JointState, TwistStamped],
        ):
            for topic, topic_name in topic_dict.items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self.publishers[topic] = rospy.Publisher(
                        topic_name, topic_type, queue_size=self.config.deque_length
                    )
                    logger.info(f"Publisher {topic} created.")

    def find_nearest_message(self, topic_name, timestamp):
        min_diff = 100.0
        nearest_msg = None
        data_queue = self.inputs_dict[topic_name].copy()
        for msg in data_queue:
            diff = np.abs(msg['message_time'] - timestamp)
            if diff < min_diff:
                nearest_msg = msg
                min_diff = diff
        return nearest_msg
    
    def lookup_by_camera_under_tolerance(self, camera_timestamp, threshold):
        msgs = {}
        for topic, que in self.inputs_dict.items():
            if topic == "head":
                msgs[topic] = que[-1]
            else:
                msg = self.find_nearest_message(topic, camera_timestamp)
                if msg is None:
                    logger.warning(f'Channel: {topic} does not have any message')
                    msgs.clear()
                    return None
                time_diff = np.abs(msg['message_time'] - camera_timestamp)
                if time_diff > threshold:    
                    if topic not in ['left_rgb', 'right_rgb']:
                        logger.warning(f'Channel: {topic} does not have messages that statisfy threshold: {threshold} > {time_diff}')
                        msgs.clear()
                        return None
                msgs[topic] = msg
        return msgs
        
    def get_observations(self):
        if len(self.inputs_dict["head_rgb"]) == 0:
            logger.warning('No camera_head message')
            return None
        
        latest_camera_msg = self.inputs_dict["head_rgb"][-1]
        latest_camera_time = latest_camera_msg['message_time']

        if latest_camera_time <= self.last_camera_time:
            latest_camera_time_str = log_utils.get_readable_timestamp(latest_camera_time)
            last_camera_time_str = log_utils.get_readable_timestamp(self.last_camera_time)
            logger.warning(f'No new head camera message. last: {last_camera_time_str} latest: {latest_camera_time_str}')
            return None
        
        obs = self.lookup_by_camera_under_tolerance(latest_camera_time, self.config.msg_time_diff_threshold)

        if obs is None:
            logger.warning('Failed to get latest_msgs')
            return None

        self.last_camera_time = latest_camera_time

        return obs
    
    def  _publish_action(self, action_dict):
        config_topic = self.config.topic
        for topic_dict, topic_fn in zip(
            [config_topic.joint_state_output, config_topic.twist_output],
            [msg_utils.act_to_joint, msg_utils.act_to_twist]
        ):
            for topic, topic_name in topic_dict.items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag and topic in action_dict:
                    if not self.config.dry_run:
                        if 'torso' in topic or 'chassis' in topic:
                            self.publishers[topic].publish(topic_fn(action_dict[topic], self.config.torso_chassis_thres))
                        else:
                            self.publishers[topic].publish(topic_fn(action_dict[topic]))

    def step(self, action_dict):
        if rospy.is_shutdown():
            logger.error('ROS node is shut down. Cannot execute step.')
            return None
        
        self._publish_action(action_dict)
        obs = self.get_observations()
        rospy.sleep(1. / self.config.control_freq)
        return obs
    
    def is_close(self):
        return rospy.is_shutdown()

    def get_latest_instruction(self):
        return self.lastest_instruction

if __name__ == "__main__":
    interface = GalaxeaInterface(GalaxeaInferfaceConfig())
    time.sleep(1)
    for i in range(10):
        time.sleep(0.1)
        obs = interface.get_observations()
        for k, v in obs.items():
            if 'data' in v:
                print(k, v['data'].shape, end=" ")
            else:
                print(k, v['position'].shape, v['velocity'].shape, end=" ")
        print("")