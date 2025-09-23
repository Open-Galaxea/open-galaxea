import numpy as np
import sapien
import trimesh

from typing import List

def get_joint_indices_by_key(joints, key):
    return [
        i for i, joint in enumerate(joints)
        if key in joint.get_name()
    ]
    
def get_link_by_name(links, name):
    for link in links:
        if link.get_name() == name:
            return link
    raise ValueError(f"Link {name} not found.")

def get_actor_meshes(actor: sapien.Entity):
    """Get actor (collision) meshes in the actor frame."""
    meshes = []
    for component in actor.components:
        if not isinstance(component, sapien.pysapien.physx.PhysxRigidBodyComponent): continue
        for col_shape in component.get_collision_shapes():
            if isinstance(col_shape, sapien.pysapien.physx.PhysxCollisionShapeBox):
                mesh = trimesh.creation.box(extents=2 * col_shape.half_size)
            else:
                raise TypeError(type(col_shape))
            mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
            meshes.append(mesh)
    return meshes

def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None

def get_actor_mesh(actor: sapien.Entity, to_world_frame=True, visual=False):
    mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh

def add_camera_to_scene(scene: sapien.Scene, name, pose, width=320*4, height=240*4, fovy_deg=35, near=0.1, far=100):
    camera = scene.add_camera(
        name=name,
        width=width,
        height=height,
        fovy=np.deg2rad(fovy_deg),
        near=near,
        far=far,
    )
    camera.entity.set_pose(pose)
    return camera

def add_mounted_camera_to_scene(scene: sapien.Scene, name, local_pose, mount, width=320, height=180, fovy_deg: float=90, near=0.001, far=100):
    
    camera = scene.add_mounted_camera(
        name=name,
        mount=mount,
        pose=local_pose,
        width=width,
        height=height,
        fovy=np.deg2rad(fovy_deg),
        near=near,
        far=far,
    )
    return camera