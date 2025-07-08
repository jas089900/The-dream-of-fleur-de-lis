import numpy as np
import json

npz_data = np.load('SMPL_NEUTRAL.npz')
json_data = {}

for key in npz_data.files:
    value = npz_data[key]
    # 转换为普通列表（确保可以 JSON 序列化）
    json_data[key] = value.tolist()

with open('neutral_smpl_with_cocoplus_reg.json', 'w') as f:
    json.dump(json_data, f)

print("✔ Converted to neutral_smpl_with_cocoplus_reg.json")

