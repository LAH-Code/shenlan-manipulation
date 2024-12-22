# 深蓝学院 Project 2 
# 基于RRT算法的GalaXea A1机械臂路径规划

## 完成此Project后，你将：

- 理解RRT路径规划算法在6DOF机械臂上如何实现
- 了解原版RRT的缺点，并懂得如何提升算法
- 学会如何使用IK求解关节角度
- 学会如何利用RRT在Mujoco仿真中抓取物体并且生成数据

---

## 你的任务

- 利用RRT（及其可能的升级版）来实现A1机械臂抓取物体数据生成
- 任务铺垫：
  - 起始机械臂位置![start](/media/start.png)
  - 目标机械臂位置![end](/media/end.png)
  - 如果没有路径规划机械臂就会撞上障碍物![no_plan](/media/no_plan.gif)
  - 用RRT路径规划之后![with_plan](/media/with_plan.gif)

---

## 任务步骤

### Step 1: 理解RRT算法，跑通代码
- 阅读`rrt_a1.py`, 理解算法是怎么代码实现的
- 启动Project1中配置好的conda环境：`conda activate act_a1`
- 跑rrt直到成功找到path：`python3 rrt_a1.py`
- 查看生成path的效果：`rrt_robot_motion.mp4`

### Step 2: 提升RRT算法 
- 问题：目前RRT找到一条成功路径的概率并不大，或者说找到一条成功路径所需要的时间很长
- 解决方案：基于现有的RRT进行算法升级，提升规划效率，如使用RRT*或者RRT-Connect算法等等

### Step 3: 基于模板实现物体抓取
- RRT作为路径规划算法，可以实现已知目标位姿的物体抓取。
- 我们提供了抓取的代码模板，可以参照其实现抓取，请安装两个额外依赖`pip install ikpy transformations` 
- 使用` rrt_a1_advance.py`, 填上里面` #TODO`的部分，就可以生成 `rrt_robot_motion_2.mp4`类似的效果

### Step 4: 实现物体抓取并且录制数据
- 在Mujoco中搭建用于机械臂抓取的物体场景
- 当前路径目标点的定义是在机械臂的6维关节空间内，也就是说我们现在的目的地其实是一个关节状态，但我们想要的是给定物体的位姿然后让机械臂直接去抓取。你需要参考`ik.py`中的代码，使用IK求解目标末端位姿下的关节角度。使用 `python3 ik.py`即可生成示例IK视频，粗坐标为实际末端位姿，细坐标为目标位姿，若IK求解成功，两者将完美重合，如下图：![ik](/media/ik.gif)
- 结合前两点和第3步，搭建自己的规划路径并且抓取物体的数据录制管线，以备后续训练使用

### Step 5: 完善数据采集的pipeline
- 使用不同场景（多个位置，多种物体）完成抓取
- 完成数据录制
- 能否与proj 1结合，完成不同场景和物体的抓取，并且能够有效学到避障