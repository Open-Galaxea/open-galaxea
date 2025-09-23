from enum import IntEnum

# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    ABS_EEF_POS_BIMANUAL = 5    # 2 x [absolute EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)]
    REL_EEF_POS_BIMANUAL = 6    # 2 x [relative EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)]
    R1_LITE = 7             # 2 x [QPOS (6) + gripper (1)] + Torso Velocity (6) + Chassis Velocity (6)
    R1_LITE_RELJOINT = 8    # 2 x [QPOS (6, relative to proprio) + gripper (1)] + Torso Velocity (6) + Chassis Velocity (6)
    # fmt: on