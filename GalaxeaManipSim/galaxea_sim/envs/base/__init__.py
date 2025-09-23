import gymnasium as gym

from galaxea_sim.robots.r1 import R1Robot
from galaxea_sim.robots.r1_pro import R1ProRobot
from galaxea_sim.robots.r1_lite import R1LiteRobot

gym.register(
    id='R1Base-v0',
    entry_point='galaxea_sim.envs.base.bimanual_manipulation:BimanualManipulationEnv',
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1Robot, 
    )
)

gym.register(
    id='R1ProBase-v0',
    entry_point='galaxea_sim.envs.base.bimanual_manipulation:BimanualManipulationEnv',
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1ProRobot, 
    )
)

gym.register(
    id='R1LiteBase-v0',
    entry_point='galaxea_sim.envs.base.bimanual_manipulation:BimanualManipulationEnv',
    disable_env_checker=True,
    order_enforce=False,
    kwargs=dict(
        robot_class=R1LiteRobot, 
    )
)