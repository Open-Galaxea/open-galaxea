from galaxea_sim.utils.robotwin_utils import create_obj
from galaxea_sim.utils.rand_utils import rand_pose
from .mug_hanging_easy import MugHangingEasyEnv

class MugHangingHardEnv(MugHangingEasyEnv):
    def _setup_rack(self):
        rack_pose = rand_pose(
            xlim=[0.05, 0.15], 
            ylim=[-0.35, -0.15],
            zlim=[0.005],
            rotate_rand=False,
            qpos=[0.31906068, 0.31906068, 0.63103113, 0.63103113]
        )
        self.rack, self.rack_data = create_obj(
            self._scene,
            pose=rack_pose,
            modelname="040_rack",
            is_static=True,
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )