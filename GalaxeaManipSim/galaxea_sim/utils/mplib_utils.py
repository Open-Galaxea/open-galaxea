import os

import mplib.sapien_utils
import sapien
from .sapien_utils import get_actor_mesh

import mplib
import numpy as np
from mplib.pymp import Pose
from tempfile import TemporaryDirectory

def get_planner_mask(num_dofs, move_group_joint_indices, other_move_group_joint_indices):
    mask = np.ones(num_dofs, dtype=np.bool_)
    for i in move_group_joint_indices:
        if i not in other_move_group_joint_indices:
            mask[i] = False
    return mask

def get_sim2mplib_mapping(sim_joint_names, mplib_joint_names):
    mapping = np.zeros(len(mplib_joint_names), dtype=np.int32)
    for i, name in enumerate(mplib_joint_names):
        mapping[i] = sim_joint_names.index(name)
    return lambda x: x[mapping]

def planner_attach_obj(planner: mplib.sapien_utils.conversion.SapienPlanner, obj: sapien.Entity, touch_links):
    planner.planning_world.attach_object(
        obj, planner.robot.name, planner.move_group_link_id, 
        touch_links=touch_links
    )
        
def planner_detach_obj(planner: mplib.sapien_utils.conversion.SapienPlanner, obj):
    if obj is not None:
        planner.planning_world.detach_object(obj, also_remove=True)
        
def disable_table_collision(planner: mplib.sapien_utils.conversion.SapienPlanner):
    table_name = None
    all_object_names = planner.planning_world.get_object_names()
    for obj_name in all_object_names:
        if "table" in obj_name:
            table_name = obj_name
            break
    acm = planner.planning_world.get_allowed_collision_matrix()
    if table_name is not None:
        for obj_name in all_object_names:
            if obj_name != table_name:
                acm.set_entry(table_name, obj_name, True)