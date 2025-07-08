import pickle

# 如果你没安装 chumpy，请先运行 pip 安装：
# pip install chumpy

import chumpy as ch

model_path = 'data/smpl/SMPL_NEUTRAL.pkl'  # 替换成你真实的路径
with open(model_path, 'rb') as f:
    model = pickle.load(f, encoding='latin1')

# 检查 posedirs 和 shapedirs
print('posedirs 类型:', type(model['posedirs']))
print('posedirs 形状:', model['posedirs'].shape)
print('shapedirs 类型:', type(model['shapedirs']))
print('shapedirs 形状:', model['shapedirs'].shape)
