import numpy as np
import json

def convert_smpl_npz(input_path, output_path):
    data = np.load(input_path, allow_pickle=True)

    # 创建新的数据结构
    model = {
        'v_template': data['v_template'].tolist(),
        'shapedirs': data['shapedirs'].tolist(),
        'posedirs': data['posedirs'].tolist(),
        'J_regressor': data['J_regressor'].tolist(),
        'weights': data['weights'].tolist(),
        'kintree_table': data['kintree_table'].tolist(),
        'f': data['f'].tolist(),
        'J': data['J'].tolist(),
        'J_regressor_prior': data['J_regressor_prior'].tolist(),
        'weights_prior': data['weights_prior'].tolist(),

        # 缺失字段填充：用 J_regressor 替代 cocoplus_regressor
        'cocoplus_regressor': data['J_regressor'].tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(model, f)

    print(f"Converted and saved to {output_path}")

if __name__ == '__main__':
    input_npz = 'SMPL_NEUTRAL.npz'  # 或替换为完整路径
    output_txt = 'neutral_smpl_with_cocoplus_reg.txt'
    convert_smpl_npz(input_npz, output_txt)

