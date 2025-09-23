import numpy as np
import sapien
import transforms3d as t3d

def rand_pose(
    xlim: np.ndarray | list,
    ylim: np.ndarray | list,
    zlim: np.ndarray | list,
    ylim_prop: bool = False,
    rotate_rand: bool = False,
    rotate_lim: list = [0, 0, 0],
    qpos: list = [1, 0, 0, 0],
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])

    while ylim_prop and abs(x) < 0.15 and y > 0:
        y = np.random.uniform(ylim[0], 0)

    z = np.random.uniform(zlim[0], zlim[1])

    rotate = qpos
    if rotate_rand:
        angles = [np.random.uniform(-rotate_lim[i], rotate_lim[i]) for i in range(3)]
        rotate_quat = t3d.euler.euler2quat(*angles)
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)

    return sapien.Pose([x, y, z], rotate)
