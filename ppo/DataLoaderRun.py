

def get_pos_from_npz_file():
    # 这个函数的用途是实现论文中对静止状态的人体SMPL模型进行类人机器人的betas优化，直到找到betas_prime作为类人机器人的身体姿态参数向量。
    # 因为静止状态的SMPL人体模型的poses和trans是不变的，所以只需要截取poses和trans的第一层数据即可，betas照常取前10维。
    import numpy as np
    import torch
    from smplx import SMPL

    # 初始化 SMPL 模型
    smpl_model = SMPL(model_path="D:/2024create/packedwork/SMPL_NEUTRAL.pkl",
                      gender='neutral', batch_size=1)

    betas = []
    poses = []
    trans = []

    data = np.load("D:2024create\packedwork\data\ACCAD\Female1General_c3d\A1 - Stand_poses.npz")
    # for keys in data.keys():
    #     if keys == 'betas' or keys == 'poses' or keys == 'trans':
    #         print("keys: ", keys)
    #         print(data[keys][0])
    #         print(data[keys][0].shape)

    betas.extend(data['betas'][:10].reshape(1, -1))  # 粗暴简单直接截取前10维，可能会出一些问题，但是我在AMASS数据集中没有找到只有10维的betas的数据集。
    poses.extend(data['poses'][0, :72].reshape(1, -1))  # 这里截取前72维是没问题的，因为后面73到156都是补充的手掌或面部等关节信息，在前72维重合。
    trans.extend(data['trans'][0, :].reshape(1, -1))

    torch.save(torch.tensor(np.asarray(betas, dtype=np.float32)),
               'D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-betas.pt')
    torch.save(torch.tensor(np.asarray(poses, dtype=np.float32)),
               'D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-poses.pt')
    torch.save(torch.tensor(np.asarray(trans, dtype=np.float32)),
               'D:/2024create/packedwork/data/cmu/CMU/01/01_01_poses-trans.pt')

    betas_tensor = torch.load("D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-betas.pt")
    poses_tensor = torch.load("D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-poses.pt")
    trans_tensor = torch.load("D:/2024create/packedwork/data/cmu/CMU/01/01_01_poses-trans.pt")

    # 输入 betas（体型参数）和 poses（姿态参数）
    # betas = torch.randn(1, 10)  # 10维体型参数
    # poses = torch.randn(1, 72)  # 72维姿态参数（24个关节，每个关节3维）
    # trans = torch.randn(1, 3)

    output = smpl_model.forward(betas=torch.zeros(1, 10),  # shape parameters
                    body_pose=torch.zeros(1, 23, 3),  # pose parameters
                    global_orient=torch.zeros(1, 1, 3),  # global orientation
                    transl=torch.zeros(1, 3))

    # 从输出中获取关节位置
    joint_positions = output.joints  # (1, 24, 3) - 每个关节的三维位置

    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
        'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
        'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]

    # 提取特定关节，例如LAnklePitch
    HeadPitch_index=JOINT_NAMES.index('head')
    LHipPitch_index = JOINT_NAMES.index('left_hip')
    LKneePitch_index = JOINT_NAMES.index('left_knee')
    LAnklePitch_index = JOINT_NAMES.index('left_ankle')
    LShoulderPitch_index = JOINT_NAMES.index('left_shoulder')
    LElbowYaw_index=JOINT_NAMES.index('left_elbow')
    LWristYaw_index=JOINT_NAMES.index('left_wrist')
    RHipPitch_index = JOINT_NAMES.index('right_hip')
    RKneePitch_index = JOINT_NAMES.index('right_knee')
    RAnklePitch_index = JOINT_NAMES.index('right_ankle')
    RShoulderPitch_index = JOINT_NAMES.index('right_shoulder')
    RElbowYaw_index = JOINT_NAMES.index('right_elbow')
    RWristYaw_index = JOINT_NAMES.index('right_wrist')

    HeadPitch_position = joint_positions[:, HeadPitch_index, :]
    LHipPitch_position = joint_positions[:, LHipPitch_index, :]
    LKneePitch_position = joint_positions[:, LKneePitch_index, :]
    LAnklePitch_position = joint_positions[:, LAnklePitch_index, :]
    LShoulderPitch_position = joint_positions[:, LShoulderPitch_index, :]
    LElbowYaw_position = joint_positions[:, LElbowYaw_index, :]
    LWristYaw_position = joint_positions[:, LWristYaw_index, :]
    RHipPitch_position = joint_positions[:, RHipPitch_index, :]
    RKneePitch_position = joint_positions[:, RKneePitch_index, :]
    RAnklePitch_position = joint_positions[:, RAnklePitch_index, :]
    RShoulderPitch_position = joint_positions[:, RShoulderPitch_index, :]
    RElbowYaw_position = joint_positions[:, RElbowYaw_index, :]
    RWristYaw_position = joint_positions[:, RWristYaw_index, :]

    return torch.cat([HeadPitch_position, LHipPitch_position, LKneePitch_position, LAnklePitch_position, LShoulderPitch_position,
                      LElbowYaw_position,LWristYaw_position,RHipPitch_position,RKneePitch_position,
                        RAnklePitch_position, RShoulderPitch_position, RElbowYaw_position, RWristYaw_position], dim=0)


def get_pos_from_args(betas):
    """

    Args:
        betas: (1, 10) betas
        poses: (1, 72) poses[i]
        trans: (1, 3) trans[i]

    Returns: the positions of joints required.

    """
    from smplx import SMPL
    import torch
    model = SMPL(model_path="D:/2024create/packedwork/SMPL_NEUTRAL.pkl",
                 gender='neutral', batch_size=1)

    betas_tensor = torch.load("D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-betas.pt")
    poses_tensor = torch.load("D:/2024create\packedwork\data\cmu\CMU/01/01_01_poses-poses.pt")
    trans_tensor = torch.load("D:/2024create/packedwork/data/cmu/CMU/01/01_01_poses-trans.pt")
    poses_tensor = poses_tensor.view(1, 24, 3)
    print("poses_tensor",poses_tensor)
    print("trans_tensor",trans_tensor)
    body_pose = poses_tensor[:, 1:, :]
    global_orient = poses_tensor[:, 0, :].view(1,1,3)
    trans_tensor = trans_tensor.view(1, 3)
    #print(global_orient.shape)
    #print(body_pose.shape)
    output = model.forward(betas=betas,  # shape parameters
                           body_pose=body_pose,  # pose parameters
                           global_orient=global_orient,  # global orientation
                           transl=trans_tensor)

    # 从输出中获取关节位置
    joint_positions = output.joints  # (1, 24, 3) - 每个关节的三维位置
    body_poses = output.body_pose

    #print(output.body_pose.shape)
    #print("The answer of whether body_poses equals to poses_tensor is:", body_poses == poses_tensor)

    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
        'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
        'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]

    # 提取特定关节，例如LAnklePitch
    HeadPitch_index = JOINT_NAMES.index('head')
    LHipPitch_index = JOINT_NAMES.index('left_hip')
    LKneePitch_index = JOINT_NAMES.index('left_knee')
    LAnklePitch_index = JOINT_NAMES.index('left_ankle')
    LFootPitch_index = JOINT_NAMES.index('left_foot')  # 用来和机器人的knee、ankle做三元组计算ankle pose
    LShoulderPitch_index = JOINT_NAMES.index('left_shoulder')
    LElbowYaw_index = JOINT_NAMES.index('left_elbow')
    LWristYaw_index = JOINT_NAMES.index('left_wrist')
    RHipPitch_index = JOINT_NAMES.index('right_hip')
    RKneePitch_index = JOINT_NAMES.index('right_knee')
    RAnklePitch_index = JOINT_NAMES.index('right_ankle')
    RFootPitch_index = JOINT_NAMES.index('right_foot')  # 同
    RShoulderPitch_index = JOINT_NAMES.index('right_shoulder')
    RElbowYaw_index = JOINT_NAMES.index('right_elbow')
    RWristYaw_index = JOINT_NAMES.index('right_wrist')

    HeadPitch_position = joint_positions[:, HeadPitch_index, :]
    LHipPitch_position = joint_positions[:, LHipPitch_index, :]
    LKneePitch_position = joint_positions[:, LKneePitch_index, :]
    LAnklePitch_position = joint_positions[:, LAnklePitch_index, :]
    LShoulderPitch_position = joint_positions[:, LShoulderPitch_index, :]
    LElbowYaw_position = joint_positions[:, LElbowYaw_index, :]
    LWristYaw_position = joint_positions[:, LWristYaw_index, :]
    RHipPitch_position = joint_positions[:, RHipPitch_index, :]
    RKneePitch_position = joint_positions[:, RKneePitch_index, :]
    RAnklePitch_position = joint_positions[:, RAnklePitch_index, :]
    RShoulderPitch_position = joint_positions[:, RShoulderPitch_index, :]
    RElbowYaw_position = joint_positions[:, RElbowYaw_index, :]
    RWristYaw_position = joint_positions[:, RWristYaw_index, :]

    LFootPitch_position = joint_positions[:, LFootPitch_index, :]
    RFootPitch_position = joint_positions[:, RFootPitch_index, :]

    # HeadPitch_pose = body_poses[:, HeadPitch_index, :]
    # LHipPitch_pose = body_poses[:, LHipPitch_index, :]
    # LKneePitch_pose = body_poses[:, LKneePitch_index, :]
    LAnklePitch_pose = body_poses[:, LAnklePitch_index-1, :]
    # LShoulderPitch_pose = body_poses[:, LShoulderPitch_index, :]
    # LElbowYaw_pose = body_poses[:, LElbowYaw_index, :]
    # LWristYaw_pose = body_poses[:, LWristYaw_index, :]
    # RHipPitch_pose = body_poses[:, RHipPitch_index, :]
    # RKneePitch_pose = body_poses[:, RKneePitch_index, :]
    RAnklePitch_pose = body_poses[:, RAnklePitch_index-1, :]
    # RShoulderPitch_pose = body_poses[:, RShoulderPitch_index, :]
    # RElbowYaw_pose = body_poses[:, RElbowYaw_index, :]
    # RWristYaw_pose = body_poses[:, RWristYaw_index, :]

    return (
        torch.cat([
                        HeadPitch_position, LHipPitch_position, LKneePitch_position, LAnklePitch_position,
                        LShoulderPitch_position, LElbowYaw_position, LWristYaw_position,
                        RHipPitch_position, RKneePitch_position, RAnklePitch_position,
                        RShoulderPitch_position, RElbowYaw_position, RWristYaw_position], dim=0),
        # torch.cat([
        #                 HeadPitch_pose, LHipPitch_pose, LKneePitch_pose, LAnklePitch_pose,
        #                 LShoulderPitch_pose, LElbowYaw_pose, LWristYaw_pose,
        #                 RHipPitch_pose, RKneePitch_pose, RAnklePitch_pose,
        #                 RShoulderPitch_pose, RElbowYaw_pose, RWristYaw_pose], dim=0),
        torch.cat([
                        LAnklePitch_pose,
                        RAnklePitch_pose], dim=0),
        LFootPitch_position,
        RFootPitch_position
    )


# def get_robot_joint_positions():
#     from ppo_walking import supervisor, Environment
#     import torch
#     robot = supervisor
#     env = Environment(robot)
#     return torch.cat([torch.tensor(env.left_foot), torch.tensor(env.right_foot),
#                       torch.tensor(env.l_ankle_pitch_pos), torch.tensor(env.r_ankle_pitch_pos),
#                       torch.tensor(env.l_knee_pitch_pos), torch.tensor(env.r_knee_pitch_pos),
#                       torch.tensor(env.l_hip_pitch_pos), torch.tensor(env.r_hip_pitch_pos),
#                       torch.tensor(env.l_shoulder_pitch_pos), torch.tensor(env.r_shoulder_pitch_pos)]).view(10, 3)


if __name__ == "__main__":
    pos = get_pos_from_npz_file()
    print(pos)
    print(pos.shape)
