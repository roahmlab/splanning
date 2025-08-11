import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.urdf_base import STEP_TYPE, RENDERER
from environments.urdf_obstacle import KinematicUrdfWithObstacles
import numpy as np
from collections import OrderedDict
from typing import List

def normal_vec_to_basis(norm_vec: np.ndarray) -> np.ndarray:
    """Creates a 3D basis for any defining normal vector (to an arbitrary hyperplane).
    
    Args:
        norm_vec (np.ndarray): The normal vector.

    Returns:
        np.ndarray: The basis as column vectors.
    """
    # first normalize the vector
    norm_vec = np.array(norm_vec, dtype=float).squeeze()
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    # Helper function for simple basis with unitary elements
    def simple_basis(order):
        ret = np.eye(3)
        idx = (np.arange(3) + order) % 3
        return ret[:,idx]

    # Try to project [1, 0, 0]
    if (proj := np.dot([1.0, 0, 0], norm_vec)):
        # Use this vector to create an orthogonal component
        rej = np.array([1.0, 0, 0]) - (norm_vec * proj)
        # Case for normal vector of [1, 0, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(1)
    # If not, try to project [0, 1, 0] and do the same
    elif (proj := np.dot([0, 1.0, 0], norm_vec)):
        rej = np.array([0, 1.0, 0]) - (norm_vec * proj)
        # Case for normal vector of [0, 1, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(2)
    else:
        # Otherwise, we are dealing with normal vector of [0, 0, 1],
        # so just create the identity as the basis
        return simple_basis(3)
    
    # Find a third orthogonal vector
    cross = np.cross(rej, norm_vec)
    # Just for simplicity, we treat the cross as x, the rej as y, and the vec as z
    # in order to keep a properly left-handed basis
    cross = cross / np.linalg.norm(cross)
    rej = rej / np.linalg.norm(rej)
    return np.column_stack((cross, rej, norm_vec))

class KinematicUrdfWithObstaclesMultiview(KinematicUrdfWithObstacles):
    def __init__(self,
                 camera_dist_scale_range = [0.5, 1],
                 camera_center_noise_scale = 0.0,
                 n_extra_cams = 1, # number of additional cameras
                 top_front_cams = False,
                 add_render_floor = True,
                 render_floor_height = -0.5,
                 add_render_walls = True,
                 render_wall_dist = 2.5,
                 render_wall_height = 8,
                 **kwargs):
        if kwargs.get('renderer', RENDERER.PYRENDER_OFFSCREEN) != RENDERER.PYRENDER_OFFSCREEN:
            raise ValueError('renderer must be pyrender-offscreen!')
        super().__init__(renderer=RENDERER.PYRENDER_OFFSCREEN, **kwargs)

        self.camera_dist_scale_range = camera_dist_scale_range
        self.camera_center_noise_scale = camera_center_noise_scale
        self.n_extra_cams = n_extra_cams

        self.camera_map = OrderedDict()
        self.scene_robot_meshes = []
        self.centroid = None
        self.scale = None

        self.top_front_cams = top_front_cams

        self.add_render_floor = add_render_floor
        self.render_floor_height = render_floor_height
        self.add_render_walls = add_render_walls
        self.render_wall_dist = render_wall_dist
        self.render_wall_height = render_wall_height

    def reset(self,
              camera_poses: List = None,
              **kwargs):
        # Setup the initial environment
        super().reset(**kwargs)
        # Cameras don't exist to observations incomplete. So we need to create the cameras first

        # create the cameras
        self._create_cameras(camera_poses)
        # get the robot meshes
        self._extract_scene_robot_meshes()

        return self.get_observations()

    def add_obstacle(self, position, size):
        super().add_obstacle(position, size)

        # recreate the cameras
        self._create_cameras()
        # get the robot meshes
        self._extract_scene_robot_meshes()

        return self.get_observations()

    def _extract_scene_robot_meshes(self):
        # extract all the pyrender robot meshes
        self.scene_robot_meshes = []
        def node_helper(node):
            for child in node.children:
                node_helper(child)
            if node.mesh is not None:
                self.scene_robot_meshes.append(node.mesh)
            return lambda: None
        for node in self.scene_map.values():
            node_helper(node)

    def _create_cameras(self, camera_poses: List = None):
        # Create the pyrender scene
        render_fps = self.render_fps
        render_size = self.render_size
        render_mesh = self.render_mesh
        renderer_kwargs = self.renderer_kwargs
        self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)

        # Generate the goal position
        import pyrender
        from pyrender.constants import DEFAULT_Z_FAR
        if camera_poses is None:
            if len(self.camera_map) == 0:
                camera_poses = []
                i = 0
                while i < self.n_extra_cams:
                    pose = self._randomize_camera_pose()
                    loc = pose[:3,3]
                    if loc[2] < -0.5 \
                        or loc[0] < -2.5 or loc[0] > 2.5 \
                        or loc[1] < -2.5 or loc[1] > 2.5:
                        continue
                    camera_poses.append(pose)
                    i += 1
                # camera_poses = [self._randomize_camera_pose() for _ in range(self.n_extra_cams)]
            else:
                camera_poses = [pose for pose in self.camera_map.values()]
        # if camera_poses is None:
        #     if len(self.camera_map) == 0:
        #         camera_poses = [self._randomize_camera_pose() for _ in range(self.n_extra_cams)]
        #     elif self.top_front_cams:
        #         camera_poses = self._generate_cams_top_front()
        #     else:
        #         camera_poses = [pose for pose in self.camera_map.values()]
        # reset the camera map
        self.camera_map = OrderedDict()
        # Iterate through each of the camera poses
        for i in range(self.n_extra_cams):
            pose = camera_poses[i]
            ## camera node
            camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.01, zfar=DEFAULT_Z_FAR)
            camera_node = pyrender.Node(name=f'CAMERA{i}', camera=camera, matrix=pose)
            self.scene.add_node(camera_node)
            self.camera_map[camera_node] = pose
    
    def _generate_cams_top_front(self):
        pass

    def _randomize_camera_pose(self):
        if self.scene is None:
            raise RuntimeError('Must call _create_pyrender_scene before randomizing camera!')
        # Generate randomized normal vector in the uppder z plane
        # Generate basis based on this normal vector
        # Generate randomized centroid offset & distance
        # move the basis based on the new centroid and distance
        from pyrender.constants import DEFAULT_SCENE_SCALE
        from scipy.spatial.transform import Rotation as R
        normal = 1-self.np_random.random(3)*np.array([2, 2, 1])
        normal = normal / np.linalg.norm(normal)
        pose = np.eye(4)
        # make a random rotation basis
        basis1 = R.from_rotvec(self.np_random.uniform(low=-np.pi, high=np.pi) * np.array([0, 0, 1])).as_matrix()
        basis2 = normal_vec_to_basis(normal)
        pose[:3,:3] = basis2 @ basis1
        center = self.centroid + self.np_random.uniform(low=-self.camera_center_noise_scale, high=self.camera_center_noise_scale, size=3)
        scale_scale = self.np_random.uniform(low=self.camera_dist_scale_range[0], high=self.camera_dist_scale_range[1])
        scale = self.scale    
        if scale == 0.0:
            scale = DEFAULT_SCENE_SCALE
        hfov = np.pi / 6.0
        dist = scale*scale_scale / (2.0 * np.tan(hfov))
        pose[:3,3] = center + normal * dist
        return pose

    def render(self,
               hide_robot = False,
               render_extra_cameras = True,
               render_frames = None,
               render_fps = None,
               render_size = None,
               render_mesh = None,
               renderer = None,
               renderer_kwargs = None):
        render_frames = render_frames if render_frames is not None else self.render_frames
        render_fps = render_fps if render_fps is not None else self.render_fps
        render_size = render_size if render_size is not None else self.render_size
        render_mesh = render_mesh if render_mesh is not None else self.render_mesh
        renderer = renderer if renderer is not None else self.renderer
        renderer_kwargs = renderer_kwargs if renderer_kwargs is not None else self.renderer_kwargs

        if renderer != RENDERER.PYRENDER_OFFSCREEN:
            raise ValueError('renderer must be pyrender-offscreen!')
        if self.scene is None:
            raise RuntimeError('Must call reset before rendering!')

        if hide_robot:
            # iterate through the robot scene map to hide the meshes
            for mesh in self.scene_robot_meshes:
                mesh.is_visible = False
        
        q_render = self._interpolate_q(self.last_trajectory, render_frames)
        q_render = self._wrap_cont_joints(q_render)
        fk = self.robot.visual_trimesh_fk_batch(q_render)
        
        # Clear any prior calls
        [clear() for clear in self.last_clear_calls]
        self.last_clear_calls = []

        # Call any onetime render callbacks and make list of callbacks to call each timestep
        step_callback_list = []
        full_callback_list = []
        for callback, needs_time in self.render_callbacks.values():
            if needs_time:
                step_callback_list.append(callback)
            else:
                full_callback_list.append(callback)
        # Create the render callback for each timestep
        def step_render_callback(timestep):
            cleanup_list = []
            for callback in step_callback_list:
                cleanup = callback(self, timestep)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]
        
        def full_render_callback():
            cleanup_list = []
            for callback in full_callback_list:
                cleanup = callback(self)
                if cleanup is not None:
                    cleanup_list.append(cleanup)
            return lambda: [cleanup() for cleanup in cleanup_list]

        # Generate the render
        self.last_clear_calls.append(full_render_callback())
        step_cleanup_callback = None
        n_extra_cameras = self.n_extra_cams if render_extra_cameras else 0
        color_frames = [[None]*(n_extra_cameras+1)]*render_frames
        depth_frames = [[None]*(n_extra_cameras+1)]*render_frames
        for i in range(render_frames):
            for mesh, node in self.scene_map.items():
                pose = fk[mesh][i]
                node.matrix = pose
            timestep = float(i+1)/render_frames * self.t_step
            if step_cleanup_callback is not None:
                step_cleanup_callback()
            step_cleanup_callback = step_render_callback(timestep)
            main_camera = self.scene.main_camera_node
            color_frames[i][0], depth_frames[i][0] = self.scene_viewer.render(self.scene)
            for cam_id in range(n_extra_cameras):
                self.scene.main_camera_node = list(self.camera_map.keys())[cam_id]
                color_frames[i][cam_id+1], depth_frames[i][cam_id+1] = self.scene_viewer.render(self.scene)
            self.scene.main_camera_node = main_camera
        if step_cleanup_callback is not None:
            self.last_clear_calls.append(step_cleanup_callback)

        # iterate through the robot scene map to restore the meshes
        for mesh in self.scene_robot_meshes:
            mesh.is_visible = True
        return color_frames, depth_frames
    
    def get_observations(self):
        observations = super().get_observations()
        observations['extra_cameras'] = {cam.name: pose for cam, pose in self.camera_map.items()}
        if self.scene is not None:
            observations['extra_cameras']['BASECAM'] = self.scene.main_camera_node.matrix
        return observations
    
    def _create_pyrender_scene(self, *args, **kwargs):
        import pyrender
        # add the obstacles
        obs_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.4,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 0.4, 0.4, 1.0),
        )
        super()._create_pyrender_scene(*args, obs_mat=obs_mat, obs_wireframe=False, **kwargs)

        # Save the centroid as just the robots and obstacles
        self.centroid = self.scene.centroid
        self.scale = self.scene.scale
        
        # Create a plane for the ground
        if self.add_render_floor:
            ground_node = self.scene.add(self._create_floor(), name='ground')

        # Create 4 walls
        if self.add_render_walls:
            walls_node = self.scene.add(self._create_walls(), name='walls')

    def _create_floor(self):
        import trimesh, pyrender
        plane_width = self.render_wall_dist * 2
        ground_transform = trimesh.transformations.translation_matrix([0,0,self.render_floor_height-0.005])
        ground = trimesh.primitives.Box([plane_width, plane_width, 0.01], transform=ground_transform)
        ground_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(0.4, 0.4, 0.4, 1.0),
        )
        ground = pyrender.Mesh.from_trimesh(ground, material=ground_mat)
        return ground
    
    def _create_walls(self):
        import trimesh, pyrender
        # Create four planes for the walls
        wall_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(0.7, 0.8, 0.8, 1.0),
        )
        walls = []
        for i in range(4):
            angle = i * (np.pi / 2) # Calculate the rotation angle for each wall
            wall_transform = trimesh.transformations.rotation_matrix(angle, [0,0,1])
            wall_transform = wall_transform @ trimesh.transformations.translation_matrix(
                [0, self.render_wall_dist + 0.005, self.render_wall_height / 2 + self.render_floor_height])
            walls.append(
                trimesh.primitives.Box([
                    self.render_wall_dist * 2 + 0.02,
                    0.01,
                    self.render_wall_height
                ], transform=wall_transform))
        
        walls = pyrender.Mesh.from_trimesh(walls, material=wall_mat)
        return walls

if __name__ == '__main__':

    # Load robot
    import os
    import zonopyrobots as zpr

    print('Loading Robot')
    # This is hardcoded for now
    rob = zpr.ZonoArmRobot.load(zpr.robots.urdfs.KinovaGen3)
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    test = KinematicUrdfWithObstaclesMultiview(
        robot = rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=True,
        use_bb_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        n_obs=15,
        render_fps=30,
        render_frames=10,
        n_extra_cams=20,
        viz_goal=False,)
    # test.render()
    test.reset()

    # test.add_obstacle(np.array([0, 0, -0.26]), np.array([0.5, 0.5, 0.5]))
    # test.reset(
    #     qpos=np.array([ 3.1098, -0.9964, -0.2729, -2.3615,  0.2724, -1.6465, -0.5739]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([-1.9472,  1.4003, -1.3683, -1.1298,  0.7062, -1.0147, -1.1896]),
    #     obs_pos=[
    #         np.array([ 0.3925, -0.7788,  0.2958]),
    #         np.array([0.3550, 0.3895, 0.3000]),
    #         np.array([-0.0475, -0.1682, -0.7190]),
    #         np.array([0.3896, 0.5005, 0.7413]),
    #         np.array([0.4406, 0.1859, 0.1840]),
    #         np.array([ 0.1462, -0.6461,  0.7416]),
    #         np.array([-0.4969, -0.5828,  0.1111]),
    #         np.array([-0.0275,  0.1137,  0.6985]),
    #         np.array([ 0.4139, -0.1155,  0.7733]),
    #         np.array([ 0.5243, -0.7838,  0.4781])
    #         ])
    # test.reset(
    #     qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
    #     obs_pos=[
    #         np.array([0.65,-0.46,0.33]),
    #         np.array([0.5,-0.43,0.3]),
    #         np.array([0.47,-0.45,0.15]),
    #         np.array([-0.3,0.2,0.23]),
    #         np.array([0.3,0.2,0.31])
    #         ])
    # Save a pickle with the images
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(4,4)
    plt.ion()
    if True:
        im, depth = test.render(hide_robot=True)
        for i in range(0,len(im),10):
            axarr[0][0].imshow(im[i][0])
            axarr[1][0].imshow(im[i][1])
            axarr[2][0].imshow(im[i][2])
            axarr[3][0].imshow(im[i][3])
            axarr[0][1].imshow(im[i][0+4])
            axarr[1][1].imshow(im[i][1+4])
            axarr[2][1].imshow(im[i][2+4])
            axarr[3][1].imshow(im[i][3+4])
            axarr[0][2].imshow(im[i][0+8])
            axarr[1][2].imshow(im[i][1+8])
            axarr[2][2].imshow(im[i][2+8])
            axarr[3][2].imshow(im[i][3+8])
            axarr[0][3].imshow(im[i][0+12])
            axarr[1][3].imshow(im[i][1+12])
            axarr[2][3].imshow(im[i][2+12])
            axarr[3][3].imshow(im[i][3+12])
            plt.draw()
            plt.pause(1)
        import pickle
        data = {'rgb': im[0], 'depth': depth[0], 'poses': test.get_observations()['extra_cameras']}
        with open('multicam_obs.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # INGESTING THIS TRASH FORMAT IS LIKE:
        ####
# import pickle

# with open('multicam_obs.pickle', 'rb') as handle:
#     data = pickle.load(handle)

# # the first set of rgb and depth are the base camera
# basecam_pose = data['poses']['BASECAM']
# basecam_rgb = data['rgb'][0]
# basecam_depth = data['depth'][0]

# # each camera then follows
# for i in range(len(data['rgb'])-1):
#     pose = data['poses'][f'CAMERA{i}']
#     rgb = data['rgb'][i+1]
#     depth = data['depth'][i+1]
        ####
    for i in range(200):
        a = test.step(np.random.random(7)-0.5)
        # test.render()
        # im, depth = test.render(hide_robot=False)
        # for i in range(0,len(im),10):
        #     axarr[0][0].imshow(im[i][0])
        #     axarr[1][0].imshow(im[i][1])
        #     axarr[2][0].imshow(im[i][2])
        #     axarr[3][0].imshow(im[i][3])
        #     axarr[0][1].imshow(im[i][0+4])
        #     axarr[1][1].imshow(im[i][1+4])
        #     axarr[2][1].imshow(im[i][2+4])
        #     axarr[3][1].imshow(im[i][3+4])
        #     axarr[0][2].imshow(im[i][0+8])
        #     axarr[1][2].imshow(im[i][1+8])
        #     axarr[2][2].imshow(im[i][2+8])
        #     axarr[3][2].imshow(im[i][3+8])
        #     axarr[0][3].imshow(im[i][0+12])
        #     axarr[1][3].imshow(im[i][1+12])
        #     axarr[2][3].imshow(im[i][2+12])
        #     axarr[3][3].imshow(im[i][3+12])
        #     plt.draw()
        #     plt.pause(0.1)
        test.render(render_fps=5)
        # print(a)

    print("hi")

    # env = Locked_Arm_3D(n_obs=3,T_len=50,interpolate=True,locked_idx=[1,2],locked_qpos = [0,0])
    # for _ in range(3):
    #     for _ in range(10):
    #         env.step(torch.rand(env.dof))
    #         env.render()
    #         #env.reset()