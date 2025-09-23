import sapien
import numpy as np 

from galaxea_sim.utils.robotwin_utils import create_box
from .blocks_stack_easy import BlocksStackEasyEnv

class BlocksStackHardEnv(BlocksStackEasyEnv):
    def _setup_block3(self):
        rand_pos = self._rand_pose()
        while abs(rand_pos.p[1]) < 0.05 or np.sum(pow(rand_pos.p[:2] - np.array([-self.place_pos_x_offset, 0]), 2)) < 0.0225 \
            or np.linalg.norm(rand_pos.p[:2] - self.block1.get_pose().p[:2] + self.tabletop_center_in_world[:2]) < 0.06 \
            or np.linalg.norm(rand_pos.p[:2] - self.block2.get_pose().p[:2] + self.tabletop_center_in_world[:2]) < 0.06:
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.block3 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0,0,1),
            name="box"
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "block1"):
            self._scene.remove_actor(self.block1)
        if hasattr(self, "block2"):
            self._scene.remove_actor(self.block2)
        if hasattr(self, "block3"):
            self._scene.remove_actor(self.block3)
        self._setup_block1()
        self._setup_block2()
        self._setup_block3()
        if reset_info is not None:
            self.block1.set_pose(sapien.Pose(reset_info["block1_pose"][:3], reset_info["block1_pose"][3:]))
            self.block2.set_pose(sapien.Pose(reset_info["block2_pose"][:3], reset_info["block2_pose"][3:]))
            self.block3.set_pose(sapien.Pose(reset_info["block3_pose"][:3], reset_info["block3_pose"][3:]))

    def solution(self):
        substeps, last_arm = self.move_block(self.block1, 1)
        for substep in substeps:
            yield substep
        substeps, last_arm = self.move_block(self.block2, 2, last_arm)
        for substep in substeps:
            yield substep
        substeps, last_arm = self.move_block(self.block3, 3, last_arm)
        for substep in substeps:
            yield substep
    
    def _get_info(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        target_pose = [self.place_pos_x_offset + self.tabletop_center_in_world[0], 0]
        eps = [0.025, 0.025, 0.01]
        success = np.all(np.abs(block1_pose - np.array(target_pose + [self.table_height + self.block_half_size])) < eps) and \
               np.all(np.abs(block2_pose - np.array(target_pose + [self.table_height + self.block_half_size * 3])) < eps) and \
                np.all(np.abs(block3_pose - np.array(target_pose + [self.table_height + self.block_half_size * 5])) < eps)
        return dict(
            success=success,
        )
    
    def _get_reset_info(self):
        return dict(
            block1_pose=np.concatenate([self.block1.get_pose().p, self.block1.get_pose().q]),
            block2_pose=np.concatenate([self.block2.get_pose().p, self.block2.get_pose().q]),
            block3_pose=np.concatenate([self.block3.get_pose().p, self.block3.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    def get_object_dict(self):
        return dict(
            block1=np.concatenate([self.block1.get_pose().p, self.block1.get_pose().q]),
            block2=np.concatenate([self.block2.get_pose().p, self.block2.get_pose().q]),
            block3=np.concatenate([self.block3.get_pose().p, self.block3.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0