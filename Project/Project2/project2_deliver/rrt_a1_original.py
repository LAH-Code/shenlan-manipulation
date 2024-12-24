from dm_control import mujoco   #模拟物理环境
import cv2
import numpy as np
import random

# Load dm_control model
model = mujoco.Physics.from_xml_path('assets/a1_rrt.xml')

# RRT 伪代码
# 1. 初始化树，将起始节点加入树中
# 2. 重复以下步骤直到达到最大迭代次数：
#     a. 生成随机节点
#     b. 找到树中距离随机节点最近的节点
#     c. 从最近的节点向随机节点扩展一段距离
#     d. 如果路径上没有碰撞，将新节点加入树中
#     e. 如果新节点距离目标节点小于一定阈值，尝试连接到目标节点
# 3. 如果找到路径，从终点回溯到起点，得到路径

class RRT:
    class Node:
        def __init__(self, q):      
        # 初始化Node实例
            self.q = q
            self.path_q = []
            self.parent = None        

    def __init__(self, start, goal, joint_limits, expand_dis=0.1, path_resolution=0.01, goal_sample_rate=5, max_iter=1000):
        # 属于外部的RRT类，用于初始化RRT类的实例
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
            rnd_node = self.get_random_node()          # 生成随机节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # 找到最近的节点的索引
            nearest_node = self.node_list[nearest_ind]   #获取距离随机节点最近的节点

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            # 通过扩展距离self.expand_dis，从nearest_node到rnd_node生成一个新节点new_node
            if self.check_collision(new_node, model):   # 检查新节点是否与障碍物碰撞
                self.node_list.append(new_node)         # 如果没有碰撞，将新节点添加到节点列表中

            if self.calc_dist_to_goal(self.node_list[-1].q) <= self.expand_dis: # 检查最后一个节点是否接近目标
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis) # 生成从最后一个节点到目标的路径
                if self.check_collision(final_node, model):
                    return self.generate_final_course(len(self.node_list) - 1) #生成并返回最终路径

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
        # 计算node_list中每个节点到随机节点rnd_node的距离
        # np.array(node.q)：将节点的关节配置转换为numpy数组
        # np.linalg.norm：计算两个节点之间的欧氏距离
        min_index = dlist.index(min(dlist))
        return min_index
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
         Generate a new node by moving from the 'from_node' towards the 'to_node' by a distance of 'extend_length'.
         extend_length默认无穷大
        """
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q) - np.array(from_node.q))
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        delta_q = (np.array(to_node.q) - np.array(from_node.q)) / distance

        for i in range(num_steps):
            new_q = new_node.q + delta_q * self.path_resolution
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            # 将新节点的位置限制在关节限制范围内
            new_node.path_q.append(new_node.q)
            # 将新节点的位置添加到路径中

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            # random.randint(0, 100)：生成0到100之间的随机整数
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
            # random.uniform(joint_min, joint_max)：生成joint_min和joint_max之间的随机浮点数
        else:
            rand_q = self.end.q
        return self.Node(rand_q)
            # rand_q为目标节点的关节配置

    def check_collision(self, node, model):
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, goal_ind):
        # goal_ind：目标节点的索引
        path = [self.end.q]     # 初始化路径列表，将目标节点的关节配置添加到路径中
        node = self.node_list[goal_ind]  # 获取目标节点
        while node.parent is not None:   # 循环遍历节点的父节点，直到起始节点（其父节点为None）
            path.append(node.q)          # 将当前节点的关节配置添加到路径中
            node = node.parent           # 将当前节点更新为其父节点
        path.append(self.start.q)        # 将起始节点的关节配置添加到路径中
        return path[::-1]                # 返回反转后的路径列表，使路径从起点到终点

    def calc_dist_to_goal(self, q):
        return np.linalg.norm(np.array(self.end.q) - np.array(q))


def check_collision_with_dm_control(model, joint_config):
    """
    Function to check if a given joint configuration results in a collision using dm_control's collision detection.
    Args:
        model: dm_control Mujoco model
        joint_config: List of joint angles to check for collision
    Returns:
        True if collision-free, False if there is a collision
    """
    model.data.qpos[:] = joint_config  # Set joint positions
    model.forward()  # Update the simulation state

    # Check for collisions
    contacts = model.data.ncon  # Number of contacts (collisions)
    # 获取当前模拟状态下的接触点数量（碰撞数量）
    return contacts == 0  # True if no contacts (collision-free)



def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion.mp4"):
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
    model.data.qpos[:] = start
    model.forward()

    # Apply the path to the simulation and record the video
    for q in path:
        # Check joint limits
        
        # print(f"{q=}")

        # model.data.qpos[:] = q  # Set joint angles
        model.data.ctrl[:] = q  # Set joint angles
        
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


# Example usage:
start = [0.5, 1.3, -0.8, -1.5, 0.5, -0.45]  # Start joint angles
goal = [-0.5, 1.3, -0.8, 1.5, 0.5, 0.45]  # Goal joint angles
joint_limits = [(-3, 3)] * 6  # Example joint limits
# 创建一个包含6个元素的列表，每个元素都是一个元组(-3, 3)，表示每个关节的角度限制
# joint_limits = [(-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3)]
joint_limits[2] = (-3, 0) # elbow
joint_limits[3] = (-1.5, 1.5) # forearm_roll

# Initialize RRT (assuming you have the RRT class set up)
rrt = RRT(start, goal, joint_limits)
rrt_path = rrt.planning(model)  # Generate the RRT path

# Apply the path to the MuJoCo simulation and record video
if rrt_path:
    print("Path found!")
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_ori.mp4")
else:
    print("No path found!")