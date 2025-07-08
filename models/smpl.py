import pickle
import os
import numpy as np
import chumpy as ch
import torch
from torch import nn
import config
import constants
from smplx.body_models import ModelOutput

def load_model(pkl_path):
    """
    加载SMPL .pkl模型，支持 posedirs 为 numpy，shapedirs 为 chumpy。
    返回模型对象（包含 pose, betas, r 等属性）。
    """
    if os.path.isdir(pkl_path):
        pkl_path = os.path.join(pkl_path, 'SMPL_NEUTRAL.pkl')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    for key, val in data.items():
        if 'chumpy' in str(type(val)):
            data[key] = np.array(val)
        if 'scipy.sparse' in str(type(val)):
            data[key] = val.toarray()

    class SMPLModel:
        def __init__(self, model_data):
            for key in model_data:
                setattr(self, key, model_data[key])
            self.pose = np.zeros(self.kintree_table.shape[1] * 3)
            self.betas = np.zeros(self.shapedirs.shape[-1])

        @property
        def r(self):
            v_shaped = self.v_template + np.tensordot(self.shapedirs, self.betas, axes=[2, 0])

            pose = self.pose.reshape((-1, 3))  # shape (24, 3)
            R = [angle_axis_to_rotation_matrix(p) for p in pose]

            ident = np.eye(3)
            pose_feature = np.concatenate([(R[i] - ident).ravel() for i in range(1, len(R))])

            posedirs = self.posedirs.reshape(-1, self.posedirs.shape[-1])
            pose_offsets = posedirs.dot(pose_feature).reshape(self.v_template.shape)
            v_posed = v_shaped + pose_offsets

            J = self.J_regressor.dot(v_shaped)
            G = [None] * 24
            G[0] = with_zeros(np.hstack((R[0], J[0].reshape(3, 1))))
            for i in range(1, 24):
                G[i] = G[self.kintree_table[0, i]].dot(
                    with_zeros(np.hstack((R[i], (J[i] - J[self.kintree_table[0, i]]).reshape(3, 1))))
                )
            G = np.stack(G, axis=0)

            G[:, :3, 3] -= np.matmul(G[:, :3, :3], J.reshape(24, 3, 1)).squeeze(-1)
            T = np.tensordot(self.weights, G, axes=[1, 0])
            v_posed_homo = np.hstack([v_posed, np.ones((v_posed.shape[0], 1))])
            v_homo = np.matmul(T, v_posed_homo[:, :, None])
            verts = v_homo[:, :3, 0]
            return verts

    return SMPLModel(data)

def angle_axis_to_rotation_matrix(angle_axis):
    theta = np.linalg.norm(angle_axis)
    if theta < 1e-8:
        return np.eye(3)
    axis = angle_axis / theta
    x, y, z = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([
        [cos_theta + x*x*(1 - cos_theta), x*y*(1 - cos_theta) - z*sin_theta, x*z*(1 - cos_theta) + y*sin_theta],
        [y*x*(1 - cos_theta) + z*sin_theta, cos_theta + y*y*(1 - cos_theta), y*z*(1 - cos_theta) - x*sin_theta],
        [z*x*(1 - cos_theta) - y*sin_theta, z*y*(1 - cos_theta) + x*sin_theta, cos_theta + z*z*(1 - cos_theta)]
    ])
    return R

def rotation_matrix_to_angle_axis(rotation_matrix):
    R = rotation_matrix
    cos_angle = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.acos(torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6))

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)
    axis = axis / (2 * torch.sin(angle).unsqueeze(-1) + 1e-8)

    return axis * angle.unsqueeze(-1)

def with_zeros(mat):
    return np.vstack((mat, np.array([[0.0, 0.0, 0.0, 1.0]])))

class SMPL(nn.Module):
    def __init__(self, model_path, batch_size=1):
        super(SMPL, self).__init__()
        self.batch_size = batch_size
        self.model = load_model(model_path)
        self.faces = self.model.f

        num_verts = self.model.v_template.shape[0]
        self.register_buffer('J_regressor', torch.tensor(self.model.J_regressor, dtype=torch.float32))

        J_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        if J_extra.shape[1] != num_verts:
            J_regressor_new = np.zeros((J_extra.shape[0], num_verts), dtype=np.float32)
            J_regressor_new[:, :J_extra.shape[1]] = J_extra
            J_extra = J_regressor_new
        self.register_buffer('J_regressor_extra', torch.tensor(J_extra, dtype=torch.float32))

        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES if
                  constants.JOINT_MAP[i] < self.J_regressor.shape[0] + self.J_regressor_extra.shape[0]]
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, betas, body_pose, global_orient, trans=None, **kwargs):
        assert betas.shape[0] == 1, "当前仅支持 batch_size=1"

        rotmat = torch.cat([global_orient, body_pose], dim=1).view(-1, 3, 3)
        pose_axis_angle = rotation_matrix_to_angle_axis(rotmat).view(1, -1)
        pose = pose_axis_angle[0].detach().cpu().numpy()
        betas_np = betas[0].detach().cpu().numpy()

        self.model.pose[:] = pose
        self.model.betas[:] = betas_np

        verts = torch.tensor(self.model.r, dtype=torch.float32).unsqueeze(0)

        joints = torch.matmul(self.J_regressor, verts)
        extra_joints = torch.matmul(self.J_regressor_extra, verts)
        joints = torch.cat([joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]

        return ModelOutput(
            vertices=verts,
            joints=joints,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            full_pose=torch.cat([global_orient, body_pose], dim=1)
        )




