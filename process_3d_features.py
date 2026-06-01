import os
import torch
import numpy as np
import pandas as pd
from plyfile import PlyData
from tqdm import tqdm

DATA_ROOT = "/rcp-scratch/iccluster040_scratch/students/moudden/bachelor_project/renders"  
OUTPUT_FILE = "./timeatlas_3d_vectors.parquet"
LEVELS = [0, 1, 2]     

def process_all_buildings():
    # A flat list to store all entity records
    records = []
    
    if not os.path.exists(DATA_ROOT):
        print(f"[Error] DATA_ROOT not found: {DATA_ROOT}")
        return

    edifici_dirs = [d for d in os.listdir(DATA_ROOT) 
                    if d.startswith("edifici_") and os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    print(f"Found {len(edifici_dirs)} building models. Starting processing...")

    for edifici_id in tqdm(edifici_dirs, desc="Processing Buildings"):
        
        pt_filename = f"{edifici_id}_DINOv3_fused_features.pt"
        pt_path = os.path.join(DATA_ROOT, edifici_id, "3D", pt_filename)
        
        if not os.path.exists(pt_path):
            continue
            
        pt_data = torch.load(pt_path, map_location='cpu')
        feat_bank = pt_data['feat_bank'].numpy()  # Shape: [N_valid, 1024]
        point_ids = pt_data['point_ids'].numpy()  # Shape: [N_valid]
        
        ply_paths = {lvl: os.path.join(DATA_ROOT, edifici_id, "3D", f"partition_level_{lvl}.ply") for lvl in LEVELS}
        if not all(os.path.exists(p) for p in ply_paths.values()):
            continue

        total_points = len(PlyData.read(ply_paths[0])['vertex'].data['label'])
        df_points = pd.DataFrame({'point_index': np.arange(total_points)})
        
        for lvl in LEVELS:
            plydata = PlyData.read(ply_paths[lvl])
            df_points[f'label_l{lvl}'] = np.array(plydata['vertex'].data['label'])

        feat_dict = {int(pid): vec for pid, vec in zip(point_ids, feat_bank)}

        # Helper function to extract and append records safely
        def extract_and_append(group_df, entity_id, level_num):
            point_indices = group_df['point_index'].tolist()
            valid_vectors = [feat_dict[idx] for idx in point_indices if idx in feat_dict]
            
            # Only store entities that have valid visual features
            if len(valid_vectors) > 0:
                mean_vector = np.mean(valid_vectors, axis=0).astype(np.float32).tolist()
                
                records.append({
                    "edifici_id": edifici_id,
                    "entity_id": entity_id,
                    "level": level_num,
                    "vector": mean_vector,
                    "point_indices": point_indices
                })

        # --- Process Level 0 Entities ---
        for l0_val, group in df_points.groupby('label_l0'):
            extract_and_append(group, f"L0_{l0_val}", 0)

        # --- Process Level 1 Entities ---
        for (l0_val, l1_val), group in df_points.groupby(['label_l0', 'label_l1']):
            extract_and_append(group, f"L0_{l0_val}_L1_{l1_val}", 1)

        # --- Process Level 2 Entities ---
        for (l0_val, l1_val, l2_val), group in df_points.groupby(['label_l0', 'label_l1', 'label_l2']):
            extract_and_append(group, f"L0_{l0_val}_L1_{l1_val}_L2_{l2_val}", 2)

    print(f"\nAggregated {len(records)} valid entities. Converting to DataFrame...")
    df_output = pd.DataFrame(records)
    
    print(f"Compressing and saving to Parquet format...")
    df_output.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    
    print(f"Success! Parquet file seamlessly saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    process_all_buildings()