import torch

def inspect_pt_file(file_path):
    print(f"--- 正在检查文件: {file_path} ---")
    
    # 1. 加载 .pt 文件 (map_location='cpu' 防止没有GPU环境报错)
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 判断数据类型
    print(f"数据类型: {type(data)}")

    # 3. 如果是字典类型 (最常见的情况，通常包含多个字段)
    if isinstance(data, dict):
        print(f"文件包含 {len(data.keys())} 个键 (Keys): {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Tensor形状: {value.shape}, 数据类型: {value.dtype}")
            else:
                print(f"  - Key: '{key}', 类型: {type(value)}")

        print("\n--- point_ids 数据 ---")
        print("前 15 个点的 ID:", data['point_ids'][:15].tolist()) # .tolist() 可以让输出格式更干净

        print("\n--- mask 数据 ---")
        print("前 15 个掩码值:", data['mask'][:15].tolist())
                
    # 4. 如果直接是一个张量 (Tensor)
    elif isinstance(data, torch.Tensor):
        print(f"文件是一个单纯的张量。形状: {data.shape}, 数据类型: {data.dtype}")
        
    else:
        print("其他未知格式。")
    print("-" * 40 + "\n")
    


inspect_pt_file("/rcp-scratch/iccluster040_scratch/students/moudden/bachelor_project/renders/edifici_339/3D/edifici_339_DINOv3_filled_features.pt")