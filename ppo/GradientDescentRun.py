import torch
import torch.optim as optim
from DataLoaderRun import get_pos_from_args
from smplx import SMPL
import trimesh
import os
from math import acos as arccos
# 假设 beta 初始化为一个随机值，这里需要根据您的实际情况来初始化
beta = torch.nn.Parameter(torch.load("D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-betas.pt"), requires_grad=True)
poses = torch.zeros(1, 24, 3)  # 姿势保持固定
trans = torch.zeros(1, 3)

# 定义smpl_model
smpl_model = SMPL(model_path="D:/2024create/packedwork/SMPL_NEUTRAL.pkl",
                      gender='neutral', batch_size=1)
# 定义优化器
optimizer = optim.Adam([beta], lr=0.1)

# 定义损失函数，这里使用MSE损失
criterion = torch.nn.MSELoss()

# 最大迭代次数
max_iters = 100

# 定义早停条件
tolerance = 1e-4  # 损失函数变化的阈值
patience = 5  # 容忍连续多个epoch没有改善
trigger_times = 3  # 连续没有改善的epoch数
best_loss = float('inf')

# 优化循环
for iteration in range(max_iters):
    optimizer.zero_grad()  # 清空梯度
    # 使用 SMPL 模型和当前的 beta 计算人体关节点位置
    returns = get_pos_from_args(beta)
    smpl_joint_positions, smpl_ankle_poses, smpl_l_foot_position, smpl_r_foot_position\
        = returns[0], returns[1], returns[2].squeeze(0), returns[3].squeeze(0)
    smpl_l_ankle_pose = smpl_ankle_poses[0]
    smpl_l_ankle_pose.requires_grad = True
    smpl_r_ankle_pose = smpl_ankle_poses[1]
    smpl_r_ankle_pose.requires_grad = True
    print(smpl_l_ankle_pose)
    print(smpl_r_ankle_pose)
    print(beta.data)

    # 获取机器人的关节点位置
    robot_joint_positions = ([0.,           0.,             0.1265],  # HeadPitch_position 15

                             [0.,           0.05,           -0.085],  # LHipPitch_position 1
                             [0.,           0.05,           -0.185],  # LKneePitch_position 4
                             [0.,           0.05,           -0.2879],  # LAnklePitch_position 7

                             [0.,           0.098,          0.1],  # LShoulderPitch_position 16
                             [0.10260061,  0.0711092,       0.1],  # LElbowYaw_position 18
                             [0.15417259,   0.04941281,     0.1],  # LWristYaw_position 20

                             [0.,         -0.05,            -0.085],  # RHipPitch_position 2
                             [0.,           -0.05,          -0.185],  # RKneePitch_position 5
                             [0.,         -0.05,            -0.2879],  # RAnklePitch_position 8

                             [0.,           -0.098,         0.1],  # RShoulderPitch_position 17
                             [0.10260061, -0.0711092,       0.1],  # RElbowYaw_position 19
                             [0.15417259,   -0.04941281,    0.1])  # RWristYaw_position 21
    ROBOT_JOINT_NAMES = [
        'head', 'left_hip', 'left_knee', 'left_ankle', 'left_shoulder', 'left_elbow', 'left_wrist',
                'right_hip', 'right_knee', 'right_ankle', 'right_shoulder', 'right_elbow', 'right_wrist'
    ]

    LKneePitch_index = ROBOT_JOINT_NAMES.index('left_knee')
    LAnklePitch_index = ROBOT_JOINT_NAMES.index('left_ankle')
    RKneePitch_index = ROBOT_JOINT_NAMES.index('right_knee')
    RAnklePitch_index = ROBOT_JOINT_NAMES.index('right_ankle')

    LKneePitch_position = torch.tensor(robot_joint_positions[LKneePitch_index], requires_grad=True)
    LAnklePitch_position = torch.tensor(robot_joint_positions[LAnklePitch_index], requires_grad=True)
    RKneePitch_position = torch.tensor(robot_joint_positions[RKneePitch_index], requires_grad=True)
    RAnklePitch_position = torch.tensor(robot_joint_positions[RAnklePitch_index], requires_grad=True)

    robot_joint_positions=torch.tensor(robot_joint_positions)
    robot_joint_positions.requires_grad=True
    # 计算 Loss (即机器人关节点和人体关节点的差异)
    loss_pos = criterion(smpl_joint_positions, robot_joint_positions)

    l_v_prime = smpl_l_foot_position - LAnklePitch_position
    #print(smpl_l_foot_position.shape)
    #print(LAnklePitch_position.shape)
    l_v = ((LAnklePitch_position - LKneePitch_position) / torch.norm(LAnklePitch_position - LKneePitch_position, p=2)
           * torch.norm(l_v_prime, p=2))

    theta_l = arccos(torch.dot(l_v_prime, l_v) / (torch.norm(l_v_prime, p=2) * torch.norm(l_v, p=2)))
    x_l = theta_l * (l_v[1]*l_v_prime[2] - l_v[2]*l_v_prime[1])
    y_l = theta_l * (l_v[2]*l_v_prime[0] - l_v[0]*l_v_prime[2])
    z_l = theta_l * (l_v[0]*l_v_prime[1] - l_v[1]*l_v_prime[0])
    robot_l_ankle_pose = torch.tensor([x_l, y_l, z_l], requires_grad=True)

    loss_poses_l = criterion(smpl_l_ankle_pose , robot_l_ankle_pose)

    r_v_prime = smpl_r_foot_position - RKneePitch_position
    r_v = ((RAnklePitch_position - RKneePitch_position) / torch.norm(RAnklePitch_position - RKneePitch_position, p=2)
           * torch.norm(r_v_prime, p=2))

    theta_r = arccos(torch.dot(r_v_prime, r_v) / (torch.norm(r_v_prime, p=2) * torch.norm(r_v, p=2)))
    x_r = theta_r * (r_v[1]*r_v_prime[2] - r_v[2]*r_v_prime[1])
    y_r = theta_r * (r_v[2]*r_v_prime[0] - r_v[0]*r_v_prime[2])
    z_r = theta_r * (r_v[0]*r_v_prime[1] - r_v[1]*r_v_prime[0])
    robot_r_ankle_pose = torch.tensor([x_r, y_r, z_r], requires_grad=True)

    loss_poses_r = criterion(smpl_r_ankle_pose , robot_r_ankle_pose)

    loss_poses = loss_poses_l + loss_poses_r

    loss = loss_pos + loss_poses

    # 反向传播，计算 beta 的梯度
    loss.backward()
    print(beta.grad)

    # 更新 beta
    optimizer.step()

    # 打印损失值，监控训练过程

    print(f'Iteration {iteration}, Loss: {loss.item()}')

    # 可以在这里加入早停策略或者其他的稳定判断条件
    if loss.item() < best_loss:
        best_loss = loss.item()
        trigger_times = 0  # 重置触发次数
    else:
        trigger_times += 1  # 增加连续没有改善的epoch数

    if trigger_times >= patience:
        print(f'Early stopping triggered at iteration {iteration}.')
        break

# 最终的 beta_robot 为优化后的 beta
beta_robot = beta.detach().numpy()
beta_prime_robot = torch.tensor(beta_robot)
# 保存优化后的 beta 参数
# torch.save(beta_robot, "path_to_save_your_beta.pt")
torch.save(beta_robot, "D:/2024create/packedwork/data/cmu/CMU/01/01_01_poses-Robot-betas.pt")
result = smpl_model(betas=beta_prime_robot,body_pose=torch.zeros(1, 23, 3),  # pose parameters
                    global_orient=torch.zeros(1, 1, 3),  # global orientation
                    transl=torch.zeros(1, 3))
mesh = trimesh.Trimesh(vertices=result.vertices[0].cpu().numpy(),
                                   faces=smpl_model.faces)
mesh.export('-smpl_mesh_example.obj')

