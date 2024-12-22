# import cv2
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
import sys


# Load dm_control model
model = mujoco.Physics.from_xml_path('assets/banana.xml')
model_bgr = mujoco.Physics.from_xml_path('assets/banana_bgr.xml')

# Load the robot arm chain from the URDF file
my_chain = ikpy.chain.Chain.from_urdf_file("assets/a1_right.urdf")
    
class RRT:
    class Node:
        def __init__(self, q):
            self.q = q
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.03, path_resolution=0.003, goal_sample_rate=5, max_iter=1000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, model):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, model):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].q) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, model):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def get_nearest_node_index(self, node_list, rnd_node):
        """
        Find the index of the nearest node to the random node.
        
        Args:
            node_list: List of nodes in the RRT tree.
            rnd_node: Randomly generated node.
        
        Returns:
            Index of the nearest node in the node list.
        """
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q)) for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q) - np.array(from_node.q))
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        delta_q = (np.array(to_node.q) - np.array(from_node.q)) / distance

        for i in range(num_steps):
            new_q = new_node.q + delta_q * self.path_resolution
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q)

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        else:
            rand_q = self.end.q
        return self.Node(rand_q)

    def check_collision(self, node, model):
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, goal_ind):
        path = [self.end.q]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q)
            node = node.parent
        path.append(self.start.q)
        return path[::-1]

    def calc_dist_to_goal(self, q):
        return np.linalg.norm(np.array(self.end.q) - np.array(q))

def get_depth(sim):
    # depth is a float array, in meters.
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2*depth[depth <= 1].mean()
    # Scale to [0, 255]
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def check_collision_with_dm_control(model, joint_config):
    """
    Function to check if a given joint configuration results in a collision using dm_control's collision detection.
    Args:
        model: dm_control Mujoco model
        joint_config: List of joint angles to check for collision
    Returns:
        True if collision-free, False if there is a collision
    """
    model.data.qpos[0:6] = joint_config  # Set joint positions
    model.forward()  # Update the simulation state

    # Check for collisions
    contacts = model.data.ncon  # Number of contacts (collisions)
    # contacts=0
    return contacts == 0 or check_gripper_collision(model) # True if no contacts (collision-free)

def check_gripper_collision(model):
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_banana_right = ("a1_right/a1_8_gripper_finger_touch_right", "banana_collision") in all_contact_pairs
    touch_banana_left = ("a1_right/a1_8_gripper_finger_touch_left", "banana_collision") in all_contact_pairs
    return touch_banana_left or touch_banana_right

def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """
    Function to apply the RRT-generated path (list of joint configurations) to the dm_control simulation,
    while recording the frames into a video.
    
    Args:
        model: dm_control Mujoco model
        path: List of joint configurations generated by the RRT planner
        video_name: Name of the output video file
    """
    # Setup for video recording
    width, height = 640, 480  # Resolution of each camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  # Two 640x480 images side by side

    # set initial joint angles
    model.data.qpos[0:6] = start
    model.forward()

    # Apply the path to the simulation and record the video
    for q in path:
        # Check joint limits
        
        # print(f"{q=}")

        # model.data.qpos[:] = q  # Set joint angles
        model.data.ctrl[0:6] = q  # Set joint angles
        
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)

        # Step the simulation forward to the next state
        model.step()

    # 这里为了稳定悬空
    for i in range(30):
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
        model.step()

    start_joints_down=model.data.qpos[0:6]
    target_position_down=target_position
    #TODO adjust the downward path, this is the final grasping position
    target_position_down[2]= target_position[2]
    target_orientation_euler_down=target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    
    # 生成插值因子，num 表示插值的数量，比如 10 表示插值 10 次
    num_interpolations = 18
    t_values = np.linspace(0, 1, num=num_interpolations)
    #TODO generate joint angle trajectory, based on the start and end position, no need to consider time since it is simulation
    interpolated_lists_down = 
    if interpolated_lists_down:
        print("down path found")
        # Apply the path to the simulation and record the video

        #TODO apply the path to pick up motion and 
        for q in interpolated_lists_down:

            
            # Render from both cameras and concatenate side by side
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            # Convert frame from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            # Write the frame to the video
            out.write(frame_bgr)

            # Step the simulation forward to the next state
            model.step()
    


    for i in range(30):
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
        model.step()        
    # #现在已经找到了路径并渲染了 目前需要关闭夹爪 并渲染


    #TODO pick up path   
    start_joints_up =
    target_position_up =   
    target_position_up[2] = 
    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")
    #TODO 生成多维的插值列表
    interpolated_lists_up = 
    if interpolated_lists_up:
        print("up path found")
        # Apply the path to the simulation and record the video
        for q in interpolated_lists_up:
            # Check joint limits

            # model.data.qpos[:] = q  # Set joint angles
            model.data.ctrl[0:6] = q  # Set joint angles
            
            # Render from both cameras and concatenate side by side
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            
            # Convert frame from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video
            out.write(frame_bgr)

            # Step the simulation forward to the next state
            model.step()

    # Release the video writer
    out.release()
    print(f"Video saved as {video_name}")

def close_gripper():
    # 夹爪闭合控制
    model.data.ctrl[6]=-0.15
    model.data.ctrl[7]=0.15

def open_gripper():
    # 夹爪打开控制
    model.data.ctrl[6]=0
    model.data.ctrl[7]=0


# Example usage:
start = [0., 0., 0., 0., 0., 0.]  # Start joint angles
#TODO target in Cartesian x=0.3, y=0.4
target_position 
target_orientation_euler 
target_orientation

# IK
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")

# goal and joint limits
goal=joint_angles[1:]
print("goal",goal)
joint_limits = [(-3, 3)] * 6  # Example joint limits
joint_limits[2] = (-3, 0) # elbow
joint_limits[3] = (-1.5, 1.5) # forearm_roll

#TODO Initialize RRT (assuming you have the RRT class set up) and Generate RRT Path


# Apply the path to the MuJoCo simulation and record video
if rrt_path:
    print("Path found!")

    #TODO 打开夹爪
    
    # Apply RRT Path 
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_2.mp4")
else:
    print("No path found!")