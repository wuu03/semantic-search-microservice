import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Dataset Class (Mirroring eval.py)
# ==========================================
class COCOSemanticDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, max_samples=10):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())[:max_samples]
        self.cat_ids = sorted(self.coco.getCatIds())
        self.id_map = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.classes = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in self.cat_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_pil = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        
        w_orig, h_orig = img_pil.size
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.id_map:
                mask[self.coco.annToMask(ann) > 0] = self.id_map[cat_id]
        
        mask_pil = Image.fromarray(mask)
        mask_512 = mask_pil.resize((512, 512), resample=Image.NEAREST)
        
        # Save raw image for visualization
        img_np = np.array(img_pil.resize((512, 512)))

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            # Default to basic [0, 1] tensor for normalization experiments
            img_tensor = T.Compose([T.Resize((512, 512)), T.ToTensor()])(img_pil)

        return img_tensor, torch.from_numpy(np.array(mask_512)).long(), img_np

# ==========================================
# 2. Diagnostic Inference
# ==========================================
def run_debug(version="c-radio_v4-h", threshold=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DEBUG START] Version={version}, Device={device}, Threshold={threshold}")

    os.makedirs("debug_output", exist_ok=True)

    # 1. Base Dataset (No Norm for now, will apply in loop)
    img_dir = "coco_data/val2017"
    ann_file = "coco_data/annotations/instances_val2017.json"
    dataset = COCOSemanticDataset(img_dir, ann_file, transform=None, max_samples=3)
    
    print(f"Classes: {len(dataset.classes)} | Mapping: {dataset.classes[0]} -> ID 1")

    # 2. Load Model
    radseg = torch.hub.load('RADSeg-OVSS/RADSeg', 'radseg_encoder',
                            model_version=version, lang_model="siglip2-g",
                            device=device, predict=False)
    if hasattr(radseg, 'model'):
        radseg.model.eval()
    else:
        radseg.eval()

    # 3. Prep Prompts
    prompts = [f"a photo of a {name}" for name in dataset.classes]
    text_vecs = radseg.encode_prompts(prompts, onehot=False)
    text_vecs = F.normalize(text_vecs, dim=-1)

    # 4. Compare Normalizations
    norm_configs = [
        {"name": "Norm_None", "mean": None, "std": None},
        {"name": "Norm_0.5", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    ]

    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor_raw, gt_mask, img_np = dataset[i]
            
            for config in norm_configs:
                img_tensor = img_tensor_raw.clone()
                if config["mean"] is not None:
                    img_tensor = T.Normalize(mean=config["mean"], std=config["std"])(img_tensor)
                
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Inference
                scga_feat = radseg.encode_image_to_feat_map(img_tensor)
                visual_aligned = radseg.align_spatial_features_with_language(scga_feat, onehot=False)
                visual_aligned = F.normalize(visual_aligned, dim=1)

                B, C, Hf, Wf = visual_aligned.shape
                flat_feat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
                similarity = flat_feat @ text_vecs.t()

                max_vals, max_indices = torch.max(similarity, dim=-1)
                
                s_min, s_max = max_vals.min().item(), max_vals.max().item()
                above_thresh = (max_vals > threshold).sum().item()
                
                print(f"Sample {i} | {config['name']} | Sim Range: [{s_min:.3f}, {s_max:.3f}] | Above Thr: {above_thresh}/{max_vals.numel()}")

                # Pred
                preds = (max_indices + 1).reshape(Hf, Wf)
                preds[max_vals.reshape(Hf, Wf) < threshold] = 0
                preds_upscaled = F.interpolate(preds.unsqueeze(0).unsqueeze(0).float(), size=(512, 512), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)

                # Visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img_np)
                axes[0].set_title(f"Orig: Sample {i}")
                axes[1].imshow(gt_mask, cmap='tab20', vmin=0, vmax=80)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(preds_upscaled, cmap='tab20', vmin=0, vmax=80)
                axes[2].set_title(f"Pred ({config['name']})")
                heatmap = max_vals.reshape(Hf, Wf).cpu().numpy()
                im = axes[3].imshow(heatmap, cmap='hot', vmin=0, vmax=0.3)
                plt.colorbar(im, ax=axes[3])
                axes[3].set_title("Sim Heatmap")
                for ax in axes: ax.axis('off')
                
                # Count unique labels in Pred
                unique_pred = np.unique(preds_upscaled)
                print(f"  Pred Labels: {unique_pred}")

                save_path = f"debug_output/sample_{i}_{config['name']}.png"
                plt.savefig(save_path)
                plt.close()
                print(f"  Saved to {save_path}")

if __name__ == "__main__":
    run_debug()
