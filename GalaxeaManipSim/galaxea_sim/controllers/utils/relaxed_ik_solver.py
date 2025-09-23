from galaxea_sim.controllers.utils.python_wrapper import RelaxedIKRust
import pathlib

class RelaxedIKSolver:
    def __init__(self, left_setting_file_path, right_setting_file_path, tolerances=None):
        """
        Initializes the RelaxedIKSolver with paths to configuration files for left and right solvers, and tolerance settings.
        :param left_setting_file_path: Path to the setting file for the left arm.
        :param right_setting_file_path: Path to the setting file for the right arm.
        :param tolerances: List of tolerances for the IK solver. If None, default tolerances will be used.
        """
        self.relaxed_ik_left = RelaxedIKRust(str(pathlib.Path(__file__).parent.parent / left_setting_file_path))
        self.relaxed_ik_right = RelaxedIKRust(str(pathlib.Path(__file__).parent.parent / right_setting_file_path))
        # Set default tolerances if not provided
        self.tolerances = tolerances if tolerances else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]  # Default tolerances

    def _convert_wxyz_to_xyzw(self, quat_wxyz):
        """
        Convert quaternion from wxyz format to xyzw format.
        
        :param quat_wxyz: Quaternion in wxyz format [w, x, y, z]
        :return: Quaternion in xyzw format [x, y, z, w]
        """
        return [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    def solve_position_left(self, target_pos, target_quat_wxyz):
        """
        Solves for the position of the left arm with internal tolerance.

        :param target_pos: The target position for the left arm.
        :param target_quat_wxyz: The target quaternion for the left arm in wxyz format.
        :return: IK solution for the left arm.
        """
        # Convert quaternion from wxyz to xyzw
        target_quat_xyzw = self._convert_wxyz_to_xyzw(target_quat_wxyz)
        return self.relaxed_ik_left.solve_position(target_pos, target_quat_xyzw, self.tolerances)

    def solve_position_right(self, target_pos, target_quat_wxyz):
        """
        Solves for the position of the right arm with internal tolerance.

        :param target_pos: The target position for the right arm.
        :param target_quat_wxyz: The target quaternion for the right arm in wxyz format.
        :return: IK solution for the right arm.
        """
        # Convert quaternion from wxyz to xyzw
        target_quat_xyzw = self._convert_wxyz_to_xyzw(target_quat_wxyz)
        return self.relaxed_ik_right.solve_position(target_pos, target_quat_xyzw, self.tolerances)

    def solve_position_both(self, target_pos_left, target_quat_wxyz_left, target_pos_right, target_quat_wxyz_right):
        """
        Solves for the positions of both the left and right arms.

        :param target_pos_left: The target position for the left arm.
        :param target_quat_wxyz_left: The target quaternion for the left arm in wxyz format.
        :param target_pos_right: The target position for the right arm.
        :param target_quat_wxyz_right: The target quaternion for the right arm in wxyz format.
        :return: Tuple of IK solutions for left and right arms.
        """
        # Convert quaternions from wxyz to xyzw
        target_quat_xyzw_left = self._convert_wxyz_to_xyzw(target_quat_wxyz_left)
        target_quat_xyzw_right = self._convert_wxyz_to_xyzw(target_quat_wxyz_right)
        
        left_ik_solution = self.solve_position_left(target_pos_left, target_quat_xyzw_left)
        right_ik_solution = self.solve_position_right(target_pos_right, target_quat_xyzw_right)
        return left_ik_solution, right_ik_solution
