import os
import glob
import pickle
import time
import numpy as np
import cv2
import torch
import trimesh
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
from SMPLicit.SMPLicit_options import FitOptions
from SMPLicit import SMPLicit
import pyopencl as cl
import imageio
import pyrender
from pyrender.constants import RenderFlags

pyrender.offscreen = True

fitoptions = FitOptions()
_opt = fitoptions.parse()
opt = FitOptions().parse()
def get_device():
    return torch.device("cpu")


class SMPLProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.device = get_device()
        self._init_models()
        self.sdf_calculator = SDFCalculator()  # 初始化SDF计算器

    def _init_models(self):
        """初始化深度学习模型"""
        # SMPLicit模型
        self.smplicit = SMPLicit(self.opt)   # 默认就在 CPU 上

        if hasattr(self.opt, 'smplicit_weights'):
            state_dict = torch.load(self.opt.smplicit_weights)
            self.smplicit.load_state_dict(state_dict)

        self.smpl_faces = self.smplicit.SMPL_Layer.faces


    def process_image(self, img_path):
        """处理单个图像的主流程"""
        try:
            data = self._load_data(img_path)
            results = []

            for idx, pred in enumerate(data['smpl_predictions']):
                try:
                    result = self._process_prediction(pred, data)
                    results.append(result)
                except Exception as e:
                    print(f"处理第{idx}个预测时出错: {str(e)}")
                    continue

            return results
        except Exception as e:
            print(f"处理图像{img_path}时发生严重错误: {str(e)}")
            return []

    def _load_data(self, img_path):
        """加载输入数据（增加错误处理）"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {img_path}")

        # 加载SMPL预测
        smpl_path = os.path.join(
            self.opt.smpl_prediction_folder,
            f"{base_name}_prediction_result.pkl"
        )
        if not os.path.exists(smpl_path):
            raise FileNotFoundError(f"SMPL预测文件不存在: {smpl_path}")

        with open(smpl_path, 'rb') as f:
            smpl_data = pickle.load(f)

        # 加载分割数据
        seg_path = os.path.join(
            self.opt.cloth_segmentation_folder,
            f"{base_name}.png"
        )
        segmentation = cv2.imread(seg_path, 0) if os.path.exists(seg_path) else None

        return {
            'image': image,
            'smpl_predictions': smpl_data['pred_output_list'],
            'segmentation': segmentation,
            'base_name': base_name
        }

    def _process_prediction(self, pred, data):
        """处理单个SMPL预测结果，适配SMPLicit接口正确使用"""
        # 1. Shape 参数 (beta)
        if 'pred_betas' in pred:
            beta_smpl = torch.FloatTensor(pred['pred_betas']).to(self.device)
            if beta_smpl.dim() > 1 and beta_smpl.shape[0] == 1:
                beta_smpl = beta_smpl.squeeze(0)  # [1,10] -> [10]
        else:
            beta_smpl = torch.zeros(10, device=self.device)
        beta_np = beta_smpl.cpu().numpy()  # ndarray (10,)

        # 2. Pose 参数
        if 'pred_body_pose' in pred:
            pose_tensor = torch.FloatTensor(pred['pred_body_pose']).to(self.device)
            if pose_tensor.dim() > 1 and pose_tensor.shape[0] == 1:
                pose_tensor = pose_tensor.squeeze(0)  # [1,72] or [1,69] -> [72] or [69]
            pose_np = pose_tensor.cpu().numpy()  # ndarray (69,) or (72,)

            if pose_np.shape[0] == 69:
                # 前面补全全局旋转
                pose_np = np.concatenate([np.zeros(3, dtype=pose_np.dtype), pose_np], axis=0)
            elif pose_np.shape[0] != 72:
                raise ValueError(f"错误的 pose 长度: {pose_np.shape[0]}，应为 72")
        else:
            pose_np = np.zeros(72, dtype=np.float32)

        # 3. Cloth latent Z
        model_index = getattr(self.opt, 'index_cloth', 2)
        z_cut_dim, z_style_dim = 0, 0

        if model_index ==5:
            z_cut_dim = self.smplicit._opt.upperbody_n_z_cut
            z_style_dim = self.smplicit._opt.upperbody_n_z_style
        elif model_index == 9:
            z_cut_dim = self.smplicit._opt.pants_n_z_cut
            z_style_dim = self.smplicit._opt.pants_n_z_style
        elif model_index in [18,19]:
            z_cut_dim = self.smplicit._opt.shoes_n_z_cut
            z_style_dim = self.smplicit._opt.shoes_n_z_style
        elif model_index == 12:
            z_cut_dim = self.smplicit._opt.skirts_n_z_cut
            z_style_dim = self.smplicit._opt.skirts_n_z_style
        elif model_index == 2:
            z_cut_dim = self.smplicit._opt.hair_n_z_cut
            z_style_dim = self.smplicit._opt.hair_n_z_style
        else:
            raise ValueError(f"未定义的 model_index: {model_index}，请在分支中添加支持")


        # … 其他 model_index 分支 …
        total_z_dim = z_cut_dim + z_style_dim
        print(f"model_index: {model_index}, total_z_dim: {total_z_dim}")

        if 'cloth_params' in pred and len(pred['cloth_params']) == total_z_dim:
            cloth_z = torch.FloatTensor(pred['cloth_params']).to(self.device)
        else:
            cloth_z = torch.randn(total_z_dim, device=self.device)
        z_np = cloth_z.cpu().numpy()  # ndarray (total_z_dim,)

        # 调试信息
        print(f"调试 - beta形状: {beta_smpl.shape}, pose形状: {pose_np.shape}, Z形状: {z_np.shape}")

        # 4. 调用 SMPLicit.reconstruct —— 传入 ndarray 而非 list
        with torch.no_grad():
            try:
                # 按 SMPLicit 源码参数顺序：id, thetas, betas, Zs
                body_mesh, clothed_mesh = self.smplicit.reconstruct(
                    np.array([model_index], dtype=np.int32),  # 1) model_ids (batch,)
                    pose_np[None, :],  # 2) thetas (batch, 72)
                    beta_np[None, :],  # 3) betas  (batch, 10)
                    z_np[None, :]  # 4) Zs     (batch, total_z_dim)
                )
            except Exception as e:
                print(f"重建失败: {e}")
                return {
                    'body_mesh': trimesh.Trimesh(),
                    'clothed_mesh': trimesh.Trimesh(),
                    'camera_params': pred.get('pred_camera', None),
                    'sdf_grid': np.zeros((64, 64, 64))
                }

        return {
            'body_mesh': body_mesh,
            'clothed_mesh': clothed_mesh,
            'camera_params': pred.get('pred_camera', None),
            'sdf_grid': None
        }

    def _generate_smpl_mesh(self, beta_smpl, pose):
        """生成 SMPL 人体网格（只使用 SMPL beta & pose）"""
        with torch.no_grad():
            verts, joints, _ = self.smplicit.SMPL_Layer(
                beta=beta_smpl.unsqueeze(0),
                theta=pose.unsqueeze(0),
                get_skin=True
            )
        verts = verts[0].cpu().numpy()
        return trimesh.Trimesh(
            vertices=verts,
            faces=self.smpl_faces,
            process=False
        )

    def _optimize_clothing(self, body_mesh, cloth_z, data):
        """基于 cloth_z 迭代优化服装网格"""
        for _ in range(self.opt.iterations):
            # 使用当前 cloth_z 重建网格
            verts = self.smplicit.models[self.opt.index_cloth].reconstruct(
                cloth_z.detach().cpu().numpy(),
                body_mesh.vertices
            )
            cloth_mesh = trimesh.Trimesh(
                vertices=verts,
                faces=self.smpl_faces,
                process=False
            )

            # 对齐到人体
            aligned = self._align_cloth_to_body(body_mesh, cloth_mesh)

            # 计算 loss 并反向
            loss = self._calculate_loss(aligned, data)
            loss.backward()

            # SGD 步骤更新 cloth_z
            with torch.no_grad():
                cloth_z -= self.opt.lr * cloth_z.grad
                cloth_z.grad.zero_()

        return aligned




    def _initialize_cloth_params(self, pred):
        """初始化服装参数"""
        # 从预测数据加载或随机初始化
        if 'cloth_params' in pred:
            return torch.nn.Parameter(
                torch.FloatTensor(pred['cloth_params']).to(self.device),
                requires_grad=True
            )
        else:
            return torch.nn.Parameter(
                torch.randn(self.opt.z_dim).to(self.device),
                requires_grad=True
            )

    def _generate_cloth_mesh(self, params, body_mesh):
        """生成服装网格"""
        with torch.no_grad():
            # 选择服装模型（示例使用第一个模型）
            model = self.smplicit.models[0]
            vertices = model.reconstruct(
                params.cpu().numpy(),
                body_mesh.vertices
            )
        return trimesh.Trimesh(vertices=vertices, faces=model.faces)

    def _align_cloth_to_body(self, body_mesh, cloth_mesh):
        """精确网格对齐"""
        # 计算变换矩阵
        body_verts = body_mesh.vertices
        cloth_verts = cloth_mesh.vertices

        # 计算包围盒
        body_min = np.min(body_verts, axis=0)
        body_max = np.max(body_verts, axis=0)
        cloth_min = np.min(cloth_verts, axis=0)
        cloth_max = np.max(cloth_verts, axis=0)

        # 计算缩放和平移
        scale = (body_max - body_min) / (cloth_max - cloth_min + 1e-8)
        translation = body_min - cloth_min * scale

        # 应用变换
        transformed_verts = cloth_verts * scale + translation

        return trimesh.Trimesh(
            vertices=transformed_verts,
            faces=cloth_mesh.faces,
            process=False
        )

    def _calculate_loss(self, cloth_mesh, data):
        """计算优化损失"""
        # 实现具体的损失计算逻辑
        return torch.tensor(0.0, requires_grad=True)  # 示例返回值

    def _update_parameters(self, params):
        """参数更新"""
        # 实现优化器更新步骤
        pass

    def process_image(self, img_path):
        """处理单个图像的主流程（添加视频生成）"""
        try:
            data = self._load_data(img_path)
            results = []

            for idx, pred in enumerate(data['smpl_predictions']):
                try:
                    result = self._process_prediction(pred, data)
                    results.append(result)
                except Exception as e:
                    print(f"处理第{idx}个预测时出错: {str(e)}")
                    continue

            # 新增视频生成部分
            if results:
                self._generate_visualization(data['image'], results, img_path)

            return results
        except Exception as e:
            print(f"处理图像{img_path}时发生严重错误: {str(e)}")
            return []

    def _generate_visualization(self, original_image, results, img_path):
        """生成可视化视频"""
        # 配置输出路径
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join(self.opt.output_folder, base_name)
        video_path = os.path.join(output_dir, f"{base_name}_demo.mp4")

        # 准备网格数据
        render_meshes = []

        # 添加人体网格（灰色）
        body_mesh = results[0]['body_mesh']
        render_meshes.append((body_mesh, [0.8, 0.8, 0.8]))  # 灰色

        # 添加服装网格（红色）
        if 'clothed_mesh' in results[0]:
            clothed_mesh = results[0]['clothed_mesh']
            render_meshes.append((clothed_mesh, [0.9, 0.2, 0.2]))  # 红色

        # 初始化渲染器
        renderer = VideoRenderer(video_path, resolution=(1280, 720), fps=24)

        # 渲染并保存
        renderer.render_rotation(render_meshes)

    def _postprocess(self, mesh, sdf_grid):
        """后处理网格"""
        # 应用SDF平滑
        vertices = mesh.vertices
        sdf_values = self._query_sdf(vertices, sdf_grid)

        # 移除外部顶点
        mask = sdf_values < self.opt.sdf_threshold
        return mesh.update_vertices(mask)

    def _query_sdf(self, points, sdf_grid):
        """查询SDF值"""
        # 标准化坐标到网格空间
        normalized = (points - self.sdf_calculator.bmin) / \
                     (self.sdf_calculator.bmax - self.sdf_calculator.bmin)
        indices = np.floor(normalized * self.sdf_calculator.grid_size).astype(int)

        # 提取SDF值
        return sdf_grid[
            indices[:, 0],
            indices[:, 1],
            indices[:, 2]
        ]


class VideoRenderer:
    def __init__(self, output_path, resolution=(1280, 720), fps=24):
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        self._init_renderer()

    def _init_renderer(self):
        """初始化渲染器"""
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1]
        )

        # 配置基础光照
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.78)

    def _create_scene(self, meshes):
        """创建包含所有网格的场景"""
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

        # 添加网格
        for mesh, color in meshes:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color + [1.0],
                metallicFactor=0.2,
                roughnessFactor=0.6
            )
            trimesh_obj = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(trimesh_obj)

        # 添加光照和相机
        scene.add(self.light, pose=np.eye(4))
        return scene

    def render_rotation(self, meshes_with_colors, output_path=None):
        """渲染360度旋转动画"""
        output_path = output_path or self.output_path
        frames = []

        # 生成旋转角度序列
        angles = np.linspace(0, 2 * np.pi, 60)

        for idx, angle in enumerate(angles):
            # 创建新场景
            scene = self._create_scene(meshes_with_colors)

            # 设置相机位姿
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [0, 1.5, 3.5]  # 调整相机距离
            camera_pose[:3, :3] = cv2.Rodrigues(np.array([0, angle, 0]))[0]
            scene.add(self.camera, pose=camera_pose)

            # 渲染帧
            color, depth = self.renderer.render(
                scene,
                flags=RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
            )
            frames.append(color)

        # 保存视频
        imageio.mimwrite(output_path, frames, fps=self.fps, macro_block_size=None)
        print(f"视频已保存至：{output_path}")
# SDF计算器（优化版）
class SDFCalculator:
    def __init__(self, grid_size=64, bmin=-1.0, bmax=1.0):
        self.grid_size = grid_size
        self.bmin = np.array([bmin, bmin, bmin])  # 确保是数组形式
        self.bmax = np.array([bmax, bmax, bmax])  # 确保是数组形式
        try:
            self._init_opencl()
            self.use_gpu = True
        except Exception as e:
            print(f"无法初始化OpenCL，将使用CPU模式: {str(e)}")
            self.use_gpu = False

    def _init_opencl(self):
        """安全初始化OpenCL环境"""
        import pyopencl as cl
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self._build_kernels()

    def _build_kernels(self):
        """编译OpenCL内核"""
        import pyopencl as cl
        # 检查内核文件是否存在
        kernel_path = 'kernels/sdf.cl'
        if not os.path.exists(kernel_path):
            print(f"警告：找不到内核文件 {kernel_path}，跳过GPU加速")
            self.use_gpu = False
            return

        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        self.program = cl.Program(self.ctx, kernel_code).build()

    def compute(self, vertices, faces):
        """自动选择计算模式"""
        if self.use_gpu:
            try:
                return self._compute_gpu(vertices, faces)
            except Exception as e:
                print(f"GPU计算失败，回退到CPU: {str(e)}")
                return self._compute_cpu(vertices, faces)
        else:
            return self._compute_cpu(vertices, faces)

    def _compute_gpu(self, vertices, faces):
        """GPU加速计算"""
        import pyopencl as cl
        import numpy as np
        # 转换数据格式
        vertices_cl = np.ascontiguousarray(vertices, dtype=np.float32)
        faces_cl = np.ascontiguousarray(faces, dtype=np.int32)

        # 创建缓冲区
        mem_flags = cl.mem_flags
        v_buf = cl.Buffer(self.ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vertices_cl)
        f_buf = cl.Buffer(self.ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=faces_cl)
        sdf_buf = cl.Buffer(self.ctx, mem_flags.WRITE_ONLY,
                            size=self.grid_size ** 3 * np.dtype(np.float32).itemsize)

        # 执行内核
        self.program.compute_sdf(
            self.queue,
            (self.grid_size, self.grid_size, self.grid_size),
            None,
            v_buf,
            f_buf,
            sdf_buf,
            np.int32(len(vertices)),
            np.int32(len(faces)),
            np.float32(self.bmin[0]),
            np.float32(self.bmax[0])
        )

        # 读取结果
        sdf_grid = np.empty((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        cl.enqueue_copy(self.queue, sdf_grid, sdf_buf)
        return sdf_grid

    def _compute_cpu(self, vertices, faces):
        """CPU回退计算 - 简化版"""
        import numpy as np
        from scipy.spatial import KDTree

        # 创建KD树
        try:
            kdtree = KDTree(vertices)
        except Exception as e:
            print(f"创建KDTree失败: {str(e)}")
            return np.zeros((self.grid_size, self.grid_size, self.grid_size))

        # 创建网格点
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        # 创建采样点
        x = np.linspace(self.bmin[0], self.bmax[0], self.grid_size)
        y = np.linspace(self.bmin[1], self.bmax[1], self.grid_size)
        z = np.linspace(self.bmin[2], self.bmax[2], self.grid_size)

        # 为了提高效率，我们只计算边界附近的SDF
        # 这是一个简化版本，实际应用中可能需要更精确的计算
        for i in range(self.grid_size):
            print(f"SDF计算进度: {i}/{self.grid_size}", end="\r")
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    point = np.array([x[i], y[j], z[k]])
                    dist, _ = kdtree.query(point)
                    grid[i, j, k] = dist

        return grid


# 主程序执行
if __name__ == "__main__":

    processor = SMPLProcessor(_opt)

    # 获取输入文件
    input_files = glob.glob(os.path.join(opt.image_folder, f"*{opt.image_extension}"))

    # 处理所有图像
    for img_path in input_files:
        print(f"\n正在处理: {os.path.basename(img_path)}")
        start_time = time.time()

        try:

            results = processor.process_image(img_path)

            output_dir = os.path.join(opt.output_folder, os.path.splitext(os.path.basename(img_path))[0])
            os.makedirs(output_dir, exist_ok=True)

            for idx, result in enumerate(results):

                body_mesh_path = os.path.join(output_dir, f"body_{idx}.obj")
                result['body_mesh'].export(body_mesh_path)

                cloth_mesh_path = os.path.join(output_dir, f"cloth_{idx}.obj")
                result['clothed_mesh'].export(cloth_mesh_path)

                sdf_path = os.path.join(output_dir, f"sdf_{idx}.npy")
                np.save(sdf_path, result['sdf_grid'])

            print(f"处理完成，耗时: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"处理失败: {str(e)}")
            continue
