import argparse
import os
import numpy as np

class FitOptions:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            description="Image fitting options for SMPLicit"
        )
        self._initialized = False

    def initialize(self):


        # 添加命令行参数
        self._parser.add_argument('--output_folder', type=str, default='./outputs', help='结果保存目录')
        self._parser.add_argument('--image_folder', type=str, default='data/images/', help='输入图像文件夹')
        self._parser.add_argument('--smpl_prediction_folder', type=str, default='data/smpl_prediction/', help='SMPL 预测结果文件夹')
        self._parser.add_argument('--cloth_segmentation_folder', type=str, default='data/cloth_segmentation/', help='服装分割图像文件夹')
        self._parser.add_argument('--instance_segmentation_folder', type=str, default='data/instance_segmentation/', help='实例分割图像文件夹')
        self._parser.add_argument('--image_extension', type=str, default='.jpg', help='图像扩展名（.png, .jpg 等）')
        self._parser.add_argument('--z_dim', type=int, default=18, help='服装潜变量维度（默认 18）')
        self._parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
        self._parser.add_argument('--lr_decayed', type=float, default=0.0003, help='衰减后的学习率')
        self._parser.add_argument('--step', type=int, default=10000, help='步长')
        self._parser.add_argument('--iterations', type=int, default=200, help='优化迭代次数')
        self._parser.add_argument('--index_samples', type=int, default=100, help='采样点数')
        self._parser.add_argument('--is_train', action='store_true', help='训练模式开关')
        self._parser.add_argument('--do_videos', action='store_true', help='是否生成视频')
        self._parser.add_argument('--resolution', type=int, default=64, help='分辨率')
        self._parser.add_argument('--clusters', type=str, default='', help='聚类文件名')
        self._parser.add_argument('--num_clusters', type=int, default=500, help='聚类数量')
        self._parser.add_argument('--clamp_value', type=float, default=0.5, help='截断阈值')
        self._parser.add_argument('--num_params_style', type=int, default=12, help='风格参数维度 (默认 upperbody_n_z_style)')
        self._parser.add_argument('--num_params_shape', type=int, default=6, help='形状参数维度 (默认 upperbody_n_z_cut)')
        self._parser.add_argument('--other_labels', type=str, default='', help='额外的 parsing label 列表')
        self._parser.add_argument('--repose', action='store_true', help='是否重新姿态化')
        self._parser.add_argument('--b_min', type=str, default='', help='边界最小值')
        self._parser.add_argument('--b_max', type=str, default='', help='边界最大值')
        self._parser.add_argument('--color', type=str, default='', help='可视化颜色')
        self._parser.add_argument('--upperbody_loadepoch', type=int, default=11)
        self._parser.add_argument('--upperbody_clusters', type=str, default='indexs_clusters_tshirt_smpl.npy')
        self._parser.add_argument('--upperbody_num_clusters', type=int, default=500)
        self._parser.add_argument('--upperbody_n_z_cut', type=int, default=6)
        self._parser.add_argument('--upperbody_n_z_style', type=int, default=12)
        self._parser.add_argument('--upperbody_resolution', type=int, default=128)
        self._parser.add_argument('--upperbody_thresh_occupancy', type=float, default=-0.055)

        # Pants options:
        self._parser.add_argument('--pants_loadepoch', type=int, default=60)
        self._parser.add_argument('--pants_clusters', type=str, default='clusters_lowerbody.npy')
        self._parser.add_argument('--pants_num_clusters', type=int, default=500)
        self._parser.add_argument('--pants_n_z_cut', type=int, default=6)
        self._parser.add_argument('--pants_n_z_style', type=int, default=12)
        self._parser.add_argument('--pants_resolution', type=int, default=128)
        self._parser.add_argument('--pants_thresh_occupancy', type=float, default=-0.08)

        # Skirts options:
        self._parser.add_argument('--skirt_loadepoch', type=int, default=40)
        self._parser.add_argument('--skirt_clusters', type=str, default='clusters_lowerbody.npy')
        self._parser.add_argument('--skirt_num_clusters', type=int, default=500)
        self._parser.add_argument('--skirt_n_z_cut', type=int, default=6)
        self._parser.add_argument('--skirt_n_z_style', type=int, default=12)
        self._parser.add_argument('--skirt_resolution', type=int, default=128)
        self._parser.add_argument('--skirt_thresh_occupancy', type=float, default=-0.05)

        # Hair options:
        self._parser.add_argument('--hair_loadepoch', type=int, default=20000)
        self._parser.add_argument('--hair_clusters', type=str, default='clusters_hairs.npy')
        self._parser.add_argument('--hair_num_clusters', type=int, default=500)
        self._parser.add_argument('--hair_n_z_cut', type=int, default=6)
        self._parser.add_argument('--hair_n_z_style', type=int, default=12)
        self._parser.add_argument('--hair_resolution', type=int, default=512)
        self._parser.add_argument('--hair_thresh_occupancy', type=float, default=-2.0)
        #self._parser.add_argument('--hair_thresh_occupancy', type=float, default=-1.8)

        # Shoes options
        self._parser.add_argument('--shoes_loadepoch', type=int, default=20000)
        self._parser.add_argument('--shoes_clusters', type=str, default='clusters_shoes.npy')
        self._parser.add_argument('--shoes_n_z_cut', type=int, default=0)
        self._parser.add_argument('--shoes_n_z_style', type=int, default=4)
        self._parser.add_argument('--shoes_resolution', type=int, default=128)
        self._parser.add_argument('--shoes_thresh_occupancy', type=float, default=-0.04)
        self._parser.add_argument('--shoes_num_clusters', type=int, default=100)

        path_SMPLicit = '/home/elf/SMPLicit-main/SMPLicit/'
        self._parser.add_argument('--path_checkpoints', type=str, default=path_SMPLicit + 'checkpoints/')
        self._parser.add_argument('--path_cluster_files', type=str, default=path_SMPLicit + 'clusters/')
        self._parser.add_argument('--path_SMPL', type=str,
                                  default=path_SMPLicit + 'utils/neutral_smpl_with_cocoplus_reg.txt')

        self._opt = None

    def parse(self):
        self.initialize()
        self._opt = self._parser.parse_args()

        self._opt.upperbody_b_min = [-0.8, -0.4, -0.3]
        self._opt.upperbody_b_max = [0.8, 0.6, 0.3]
        self._opt.pants_b_min = [-0.3, -1.2, -0.3]
        self._opt.pants_b_max = [0.3, 0.0, 0.3]
        self._opt.skirt_b_min = [-0.3, -1.2, -0.3]
        self._opt.skirt_b_max = [0.3, 0.0, 0.3]
        self._opt.hair_b_min = [-0.35, -0.42, -0.33]
        self._opt.hair_b_max = [0.35, 0.68, 0.37]
        self._opt.shoes_b_min = [-0.1, -1.4, -0.2]
        self._opt.shoes_b_max = [0.25, -0.6, 0.3]

        return self._opt







