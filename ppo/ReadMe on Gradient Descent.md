## 环境配置
> conda env create -f environment.yml

上述环境为整个ppo项目的执行环境，下面列出 Gradient Descent模块所需环境

> python == 3.7.16  
> pytorch == 1.13.1  
> smplx == 0.1.28  
> numpy == 1.21.5  
> trimesh == 4.4.1
## 代码构成与注意事项
GradientDescent 模块主要由 GradientRun.py 和 DataLoaderRun.py 两个文件构成  
> GradientRun.py  \
>  DataLoaderRun.py  

请注意在上述文件中均含有绝对路径，请在运行前检查并改为可执行路径。

## 部分代码内容提示
第四十七行中机器人关节点位置使用的是前向运动学计算出的webot机器人t-pose  
数据，请注意webot自带的函数可能难以完成关节点三维坐标获取。  

第九行中的beta取自CMU数据集中的一个非t-pose动作，用作随机的beta初值

84-108行的作用是基于SMPL模型的foot节点模拟出了一个机器人的foot节点从而  
计算机器人ankle姿态，将smpl模型的ankle节点姿态与机器人的ankle节点姿态  
构建出一个损失函数，与原有损失函数相加进行优化，以改进机器人smpl模型脚部姿态。

## 关联的dataloader文件
实际使用到的是第九十四行的get_pos_from_args函数，请注意修改绝对路径。