import json
import numpy as np
import sapien
import transforms3d as t3d

from galaxea_sim import ASSETS_DIR
from .rand_utils import rand_pose

def create_urdf_obj(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = 1.0,
    fix_root_link = True,
    tabletop_center_in_world: np.ndarray = np.array([0, 0, 0]),
):
    modeldir = ASSETS_DIR / "robotwin_models" / modelname
    json_file_path = modeldir / 'model_data.json'
    
    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    loader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    modeldir = ASSETS_DIR / "robotwin_models" / modelname
    object: sapien.pysapien.physx.PhysxArticulation = loader.load((modeldir / "mobility.urdf").as_posix())
    pose.set_p(pose.get_p() + tabletop_center_in_world)
    object.set_root_pose(pose)
    return object, model_data

def rand_create_urdf_obj(
    scene: sapien.Scene,
    modelname: str,
    xlim,
    ylim,
    zlim,
    ylim_prop = False,
    rotate_rand = False,
    rotate_lim = [0,0,0],
    qpos = [1,0,0,0],
    scale = 1.0,
    fix_root_link = True,
    tabletop_center_in_world: np.ndarray = np.array([0, 0, 0]),
):
    
    obj_pose = rand_pose(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        ylim_prop=ylim_prop,
        rotate_rand=rotate_rand,
        rotate_lim=rotate_lim,
        qpos=qpos
    )

    return create_urdf_obj(
        scene,
        pose=obj_pose,
        modelname=modelname,
        scale=scale,
        fix_root_link=fix_root_link,
        tabletop_center_in_world=tabletop_center_in_world,
    )

def create_visual_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

# create box
def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=(1, 1, 1),
    name="",
) -> sapien.Entity:
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(
            half_size=half_size, material=scene.create_physical_material(0.5, 0.5, 0.6)
        )
    )

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)

    return entity


def create_table(
    scene: sapien.Scene,
    pose: sapien.Pose,
    length: float,
    width: float,
    height: float,
    thickness: float = 0.1,
    color: tuple = (1, 1, 1), 
    name: str = "table",
    is_static: bool = True
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    
    builder = scene.create_actor_builder()
    body_type = "static" if is_static else "dynamic"
    builder.set_physx_body_type(body_type)

    # Create tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])
    tabletop_half_size = (width / 2, length / 2, thickness / 2)
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size, material=scene.create_physical_material(0.8, 0.8, 0.6))
    builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=color)

    # Create table legs
    leg_spacing = 0.1
    leg_half_size = (thickness / 2, thickness / 2, height / 2)
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (width / 2 - leg_spacing / 2)
            y = j * (length / 2 - leg_spacing / 2)
            leg_pose = sapien.Pose([x, y, -height / 2])
            builder.add_box_collision(pose=leg_pose, half_size=leg_half_size)
            builder.add_box_visual(pose=leg_pose, half_size=leg_half_size, material=color)

    table = builder.build(name=name)
    table.set_pose(pose)
    return table

# create obj model
def create_obj(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = None,
    convex = False,
    is_static = False,
    model_id = None,
    model_z_val = False,
    tabeltop_center_in_world: np.ndarray = np.array([0, 0, 0]),
):
    modeldir = ASSETS_DIR / "robotwin_models" / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / 'model_data.json'
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f'model_data{model_id}.json'

    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        if scale is None: scale = model_data["scale"]
    except:
        model_data = None
        
    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if model_z_val:
        pose.set_p(pose.get_p()[:2].tolist() + [(t3d.quaternions.quat2mat(pose.get_q()) @ (np.array(model_data["extents"]) * scale))[2]/2])
        
    if convex==True:
        builder.add_multiple_convex_collisions_from_file(
            filename = file_name.as_posix(),
            scale= scale
        )
    else:
        builder.add_nonconvex_collision_from_file(
            filename = file_name.as_posix(),
            scale = scale
        )
        
    pose.set_p(pose.get_p() + tabeltop_center_in_world)
    
    builder.add_visual_from_file(
        filename=file_name.as_posix(),
        scale= scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)
    return mesh, model_data

# create glb model
def create_glb(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = (1,1,1),
    convex = False,
    is_static = False,
    model_id = None,
    model_z_val = False,
    tabeltop_center_in_world: np.ndarray = np.array([0, 0, 0])
):
    modeldir = ASSETS_DIR / "robotwin_models" / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / 'model_data.json'
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f'model_data{model_id}.json'

    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        if scale is not None: scale = model_data["scale"]
    except:
        model_data = None
        
    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if model_z_val:
        pose.set_p(pose.get_p()[:2].tolist() + [(t3d.quaternions.quat2mat(pose.get_q()) @ (np.array(model_data["extents"]) * scale))[2]/2])
        
    if convex==True:
        builder.add_multiple_convex_collisions_from_file(
            filename = file_name.as_posix(),
            decomposition="coacd",
            scale=scale
        )
    else:
        builder.add_nonconvex_collision_from_file(
            filename = file_name.as_posix(),
            scale = scale
        )
    
    builder.add_visual_from_file(
        filename=file_name.as_posix(),
        scale= scale)
    mesh = builder.build(name=modelname)
    pose.set_p(pose.get_p() + tabeltop_center_in_world)
    mesh.set_pose(pose)
    return mesh, model_data

def rand_create_glb(
    scene: sapien.Scene,
    modelname: str,
    xlim: np.ndarray | list,
    ylim: np.ndarray | list,
    zlim: np.ndarray | list,
    ylim_prop: bool = False,
    rotate_rand: bool = False,
    rotate_lim: list = [0, 0, 0],
    qpos: list = [1, 0, 0, 0],
    scale: tuple = (1, 1, 1),
    convex: bool = False,
    is_static: bool = False,
    model_id: int = None,
    model_z_val: bool = False,
    tabeltop_center_in_world: np.ndarray = np.array([0, 0, 0]),
) -> tuple:
    """Randomly create a GLB model in the scene with specified parameters."""
    
    obj_pose = rand_pose(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        ylim_prop=ylim_prop,
        rotate_rand=rotate_rand,
        rotate_lim=rotate_lim,
        qpos=qpos
    )

    return create_glb(
        scene=scene,
        pose=obj_pose,
        modelname=modelname,
        scale=scale,
        convex=convex,
        is_static=is_static,
        model_id=model_id,
        model_z_val=model_z_val,
        tabeltop_center_in_world=tabeltop_center_in_world
    )
    
def get_grasp_pose_w_labeled_direction(actor: sapien.Entity, actor_data, grasp_matrix=np.eye(4), pre_dis=0., id=0):
    actor_matrix = actor.get_pose().to_transformation_matrix()
    local_contact_matrix = np.asarray(actor_data['contact_pose'][id])
    trans_matrix = np.asarray(actor_data['trans_matrix'])
    local_contact_matrix[:3,3] *= actor_data['scale']
    global_contact_pose_matrix = actor_matrix  @ local_contact_matrix @ trans_matrix @ grasp_matrix @ np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
    global_contact_pose_matrix_q = global_contact_pose_matrix[:3,:3]
    global_grasp_pose_p = global_contact_pose_matrix[:3,3] + global_contact_pose_matrix_q @ np.array([-0.12-pre_dis,0,0]).T
    global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
    res_pose = list(global_grasp_pose_p)+list(global_grasp_pose_q)
    return np.array(res_pose)

def get_target_pose_from_goal_point_and_direction(actor: sapien.Entity, actor_data, endpose: sapien.pysapien.physx.PhysxArticulationLinkComponent, target_pose=None, target_grasp_qpose=None):
    actor_matrix = actor.get_pose().to_transformation_matrix()
    local_target_matrix = np.asarray(actor_data['target_pose'])
    local_target_matrix[:3,3] *= actor_data['scale']
    res_matrix = np.eye(4)
    res_matrix[:3,3] = (actor_matrix  @ local_target_matrix)[:3,3] - endpose.get_entity_pose().p
    res_matrix[:3,3] = np.linalg.inv(t3d.quaternions.quat2mat(endpose.get_entity_pose().q) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])) @ res_matrix[:3,3]
    res_pose = list(target_pose - t3d.quaternions.quat2mat(target_grasp_qpose) @ res_matrix[:3,3]) + target_grasp_qpose
    return res_pose

def get_actor_goal_pose(actor: sapien.Entity, actor_data):
    actor_matrix = actor.get_pose().to_transformation_matrix()
    local_target_matrix = np.asarray(actor_data['target_pose'])
    local_target_matrix[:3,3] *= actor_data['scale']
    return (actor_matrix @ local_target_matrix)[:3,3]

def get_grasp_pose_w_given_direction(actor: sapien.Entity, actor_data, grasp_qpos: list, pre_dis, id=0):
    actor_matrix = actor.get_pose().to_transformation_matrix()
    local_contact_matrix = np.asarray(actor_data['contact_pose'][id])
    local_contact_matrix[:3,3] *= actor_data['scale']
    grasp_matrix= t3d.quaternions.quat2mat(grasp_qpos)
    global_contact_pose_matrix = actor_matrix @ local_contact_matrix
    global_grasp_pose_p = global_contact_pose_matrix[:3,3] + grasp_matrix @ np.array([-0.12-pre_dis,0,0]).T
    res_pose = list(global_grasp_pose_p) + grasp_qpos
    return res_pose