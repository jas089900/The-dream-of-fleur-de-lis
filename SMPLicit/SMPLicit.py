import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from .SMPL import SMPL
from .SMPLicit_options import FitOptions as Options
from .smplicit_core_test import Model
from os.path import dirname, join

# CIHP Human Parsing 对应 label（仅保留已有模型）:
# 5 (hair), 6 (upper-clothes), 7 (skirt), 9 (pants), 11 (shoes)

CIHP_LABEL_TO_MODEL_INDEX = {
    5: 3,  # hair
    6: 0,  # upperclothes
    7: 2,  # skirt
    9: 1,  # pants
    11: 4  # shoes
}

class SMPLicit(nn.Module):
    def __init__(self,opt):
        super(SMPLicit, self).__init__()
        _opt = opt

        # 修正路径
        project_root = dirname(dirname(__file__))
        _opt.path_checkpoints = join(project_root, 'checkpoints')
        _opt.path_cluster_files = join(project_root, 'clusters') + '/'
        _opt.path_SMPL = getattr(
            _opt, 'path_SMPL', join(project_root, 'SMPLicit', 'utils', 'neutral_smpl_with_cocoplus_reg.txt')
        )

        # 按顺序加载模型：upperclothes, pants, skirt, hair, shoes
        self.models = [
            Model(  # ID 0 - upperclothes
                join(_opt.path_checkpoints, 'upperclothes.pth'),
                _opt.upperbody_n_z_cut, _opt.upperbody_n_z_style, _opt.upperbody_num_clusters,
                join(_opt.path_cluster_files, _opt.upperbody_clusters),
                _opt.upperbody_b_min, _opt.upperbody_b_max,
                _opt.upperbody_resolution, thresh=_opt.upperbody_thresh_occupancy
            ),
            Model(  # ID 1 - pants
                join(_opt.path_checkpoints, 'pants.pth'),
                _opt.pants_n_z_cut, _opt.pants_n_z_style, _opt.pants_num_clusters,
                join(_opt.path_cluster_files, _opt.pants_clusters),
                _opt.pants_b_min, _opt.pants_b_max,
                _opt.pants_resolution, thresh=_opt.pants_thresh_occupancy
            ),
            Model(  # ID 2 - skirt
                join(_opt.path_checkpoints, 'skirt.pth'),
                _opt.skirt_n_z_cut, _opt.skirt_n_z_style, _opt.skirt_num_clusters,
                join(_opt.path_cluster_files, _opt.skirt_clusters),
                _opt.skirt_b_min, _opt.skirt_b_max,
                _opt.skirt_resolution, thresh=_opt.skirt_thresh_occupancy
            ),
            Model(  # ID 3 - hair
                join(_opt.path_checkpoints, 'hair.pth'),
                _opt.hair_n_z_cut, _opt.hair_n_z_style, _opt.hair_num_clusters,
                join(_opt.path_cluster_files, _opt.hair_clusters),
                _opt.hair_b_min, _opt.hair_b_max,
                _opt.hair_resolution, thresh=_opt.hair_thresh_occupancy
            ),
            Model(  # ID 4 - shoes
                join(_opt.path_checkpoints, 'shoes.pth'),
                _opt.shoes_n_z_cut, _opt.shoes_n_z_style, _opt.shoes_num_clusters,
                join(_opt.path_cluster_files, _opt.shoes_clusters),
                _opt.shoes_b_min, _opt.shoes_b_max,
                _opt.shoes_resolution, thresh=_opt.shoes_thresh_occupancy
            )
        ]

        self.SMPL_Layer = SMPL(_opt.path_SMPL, obj_saveable=True)
        self.smpl_faces = self.SMPL_Layer.faces

        self.Astar_pose = torch.zeros(1, 72)
        self.Astar_pose[0, 5] = 0.04
        self.Astar_pose[0, 8] = -0.04

        self._opt = _opt
        self.step = 1000

    def reconstruct(self, model_labels=[6], Zs=[np.zeros(18)], pose=np.zeros(72), beta=np.zeros(10)):
        for i in range(len(Zs)):
            if not torch.is_tensor(Zs[i]):
                Zs[i] = torch.FloatTensor(Zs[i])
        if not torch.is_tensor(pose):
            pose = torch.FloatTensor(pose)
        if not torch.is_tensor(beta):
            beta = torch.FloatTensor(beta)

        pose = pose.view(1, -1)
        beta = beta.view(1, -1)

        posed_smpl = self.SMPL_Layer.forward(beta=beta, theta=pose, get_skin=True)[0][0].cpu().data.numpy()
        J, unposed_smpl = self.SMPL_Layer.skeleton(beta, require_body=True)
        Astar_smpl = self.SMPL_Layer.forward(beta=beta, theta=self.Astar_pose, get_skin=True)[0][0]

        inference_mesh = trimesh.Trimesh(
            vertices=unposed_smpl[0].cpu().numpy(),
            faces=self.smpl_faces.cpu().numpy(),
            process=False
        )
        inference_lowerbody = trimesh.Trimesh(
            vertices=Astar_smpl.cpu().numpy(),
            faces=self.smpl_faces.cpu().numpy(),
            process=False
        )

        out_meshes = [trimesh.Trimesh(posed_smpl, self.smpl_faces.cpu().numpy(), process=False)]
        for i, label in enumerate(model_labels):
            if label not in CIHP_LABEL_TO_MODEL_INDEX:
                continue
            id_ = CIHP_LABEL_TO_MODEL_INDEX[label]
            if id_ in [1, 2]:  # pants or skirt
                mesh = self.models[id_].reconstruct(Zs[i].cpu().data.numpy(), inference_lowerbody)
                mesh = self.pose_mesh_lowerbody(mesh, pose, beta, J, unposed_smpl)
            else:
                mesh = self.models[id_].reconstruct(Zs[i].cpu().data.numpy(), inference_mesh)
                if id_ == 4:  # shoes
                    mesh = self.get_right_shoe(mesh)
                mesh = self.pose_mesh(mesh, pose, J, unposed_smpl)
            out_meshes.append(mesh)

        return out_meshes

    def get_right_shoe(self, mesh):
        right_vshoe = mesh.vertices.copy()
        right_vshoe[:, 0] *= -1
        fshoe = mesh.faces
        mesh = trimesh.Trimesh(np.concatenate((mesh.vertices, right_vshoe)),
                               np.concatenate((fshoe, fshoe[:, ::-1] + len(right_vshoe))))
        return mesh

    def pose_mesh(self, mesh, pose, J, v):
        step = self.step
        iters = len(mesh.vertices) // step
        if len(mesh.vertices) % step != 0:
            iters += 1
        for i in range(iters):
            in_verts = torch.FloatTensor(mesh.vertices[i * step:(i + 1) * step]).unsqueeze(0)
            _, out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, J, v, in_verts)
            mesh.vertices[i * step:(i + 1) * step] = out_verts.cpu().data.numpy()

        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5)
        return mesh

    def pose_mesh_lowerbody(self, mesh, pose, beta, J, v):
        step = self.step
        iters = len(mesh.vertices) // step
        if len(mesh.vertices) % step != 0:
            iters += 1
        for i in range(iters):
            in_verts = torch.FloatTensor(mesh.vertices[i * step:(i + 1) * step])
            out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, self.Astar_pose, pose, beta, J, v)
            mesh.vertices[i * step:(i + 1) * step] = out_verts.cpu().data.numpy()

        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5)
        return mesh

    def forward(self, model_id, z_cut, z_style, points):
        return self.models[model_id]._G.forward(z_cut, z_style, points)






