import torch
from .rotation_conversions import quaternion_to_matrix, matrix_to_quaternion

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4


class RelativePoseTransformer:
    def __init__(self):
        pass
    
    def forward(self, pose, base_pose):
        assert pose.shape[-1] == 7
        pose_matrix = pose_to_matrix(pose)
        base_pose_matrix = pose_to_matrix(base_pose)
        pose_matrix = absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = matrix_to_pose(pose_matrix)
        return pose
    
    def backward(self, pose, base_pose):
        assert pose.shape[-1] == 7
        pose_matrix = pose_to_matrix(pose)
        base_pose_matrix = pose_to_matrix(base_pose)
        pose_matrix = relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = matrix_to_pose(pose_matrix)
        return pose
    
    
def pose_to_matrix(pose: torch.Tensor):
    position = pose[..., 0: 3]
    quaternion = pose[..., [6, 3, 4, 5]]
    rotation = quaternion_to_matrix(quaternion)
    matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
    matrix[..., 0: 3, 0: 3] = rotation
    matrix[..., 0: 3, 3] = position
    matrix[..., 3, 3] = 1
    return matrix


def matrix_to_pose(matrix: torch.Tensor):
    position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
    rotation = matrix[..., 0: 3, 0: 3]
    quaternion = matrix_to_quaternion(rotation)
    quaternion = quaternion[..., [1, 2, 3, 0]]
    pose = torch.cat([position, quaternion], dim=-1)
    return pose


def absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
    return torch.linalg.inv(base_pose_matrix) @ pose_matrix


def relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
    return base_pose_matrix @ pose_matrix
    
