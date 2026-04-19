import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm


# ==========================================
# 1. 数据集加载类 (保持不变)
# ==========================================
class COCOSemanticDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, max_samples=1000):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())[:max_samples]
        self.cat_ids = sorted(self.coco.getCatIds())
        self.id_map = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.classes = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in self.cat_ids]

    def __len__(self):
        return len(self.ids)

    # def __getitem__(self, index):
    #     img_id = self.ids[index]
    #     img_info = self.coco.loadImgs(img_id)[0]
    #     img = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
    #     w, h = img.size
    #     ann_ids = self.coco.getAnnIds(imgIds=img_id)
    #     anns = self.coco.loadAnns(ann_id  s)
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     for ann in anns:
    #         cat_id = ann['category_id']
    #         if cat_id in self.id_map:
    #             mask[self.coco.annToMask(ann) > 0] = self.id_map[cat_id]
    #     if self.transform:
    #         img = self.transform(img)
    #     return img, torch.from_numpy(mask).long()

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')

        # 1. 生成原始 Mask
        w_orig, h_orig = img.size
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.id_map:
                mask[self.coco.annToMask(ann) > 0] = self.id_map[cat_id]

        # 2. 【关键修改】将 Mask 转为 PIL Image 并缩放到 512x512
        # 必须使用 Image.NEAREST，否则标签值会变模糊
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((512, 512), resample=Image.NEAREST)
        mask_resized = np.array(mask_pil)

        # 3. 应用图片转换
        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(mask_resized).long()


# ==========================================
# 2. 评测逻辑
# ==========================================
def run_evaluation(args):
    device = args.device
    print(f"\n🚀 Starting Evaluation: {args.version} on {device}")

    # 加载模型 (开启 amp=True 以大幅降低显存占用)
    radseg = torch.hub.load('RADSeg-OVSS/RADSeg', 'radseg_encoder',
                            model_version=args.version, lang_model="siglip2-g",
                            device=device, predict=False, amp=True)
    torch.cuda.empty_cache()

    if hasattr(radseg, 'model'):
        radseg.model.eval()
    else:
        radseg.eval()

    # 数据预处理 (恢复关键的 Normalize: SigLIP 和 RADIO v3-h/v4-h 期望输入在 [-1, 1] 范围)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = COCOSemanticDataset(args.img_dir, args.ann_file, transform=transform, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 预计算向量 (包裹在 no_grad 以节省显存)
    # 【修复重点】：物体使用 ensemble，背景使用 raw_prompt，最后合并并强制 L2 归一化
    with torch.no_grad():
        print(f"Encoding {len(dataset.classes)} objects with prompt ensembling...")
        obj_vecs = radseg.encode_labels(dataset.classes, onehot=False)
        
        bg_prompt = ["background, environment, surroundings, other unrelated objects"]
        bg_vec = radseg.encode_prompts(bg_prompt, onehot=False)
        
        text_vecs = torch.cat([obj_vecs, bg_vec], dim=0)
        # CRITICAL: 必须重新进行 L2 归一化，因为 encode_labels 均值化后模长变小了
        text_vecs = F.normalize(text_vecs, dim=-1)
    
    torch.cuda.empty_cache()

    # 指标初始化 (0背景 + 80类 = 81类)
    metric = MeanIoU(num_classes=len(dataset.classes) + 1).to(device)

    # 开始循环
    # with torch.no_grad():
    #     for imgs, gts in tqdm(loader, desc=f"Evaluating {args.version}"):
    #         imgs, gts = imgs.to(device), gts.to(device)
    #
    #         scga_feat = radseg.encode_image_to_feat_map(imgs)
    #         visual_aligned = radseg.align_spatial_features_with_language(scga_feat, onehot=False)
    #         visual_aligned = F.normalize(visual_aligned, dim=1)
    #
    #         B, C, Hf, Wf = visual_aligned.shape
    #         flat_feat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
    #         similarity = flat_feat @ text_vecs.t()
    #
    #         max_vals, max_indices = torch.max(similarity, dim=-1)
    #         preds = (max_indices + 1).reshape(B, Hf, Wf)
    #         preds[max_vals.reshape(B, Hf, Wf) < args.threshold] = 0
    #
    #         preds_res = F.interpolate(preds.unsqueeze(1).float(), size=gts.shape[1:], mode='nearest').squeeze(1).long()
    #         metric.update(preds_res, gts)

    with torch.no_grad():
        for imgs, gts in tqdm(loader, desc=f"Evaluating {args.version}"):
            imgs, gts = imgs.to(device), gts.to(device)

            # 1. 模型推理 (回归远端模式，不使用不支持的 use_feat_mlp)
            scga_feat = radseg.encode_image_to_feat_map(imgs)
            visual_aligned = radseg.align_spatial_features_with_language(scga_feat, onehot=False)
            visual_aligned = F.normalize(visual_aligned, dim=1)

            # 2. 计算余弦相似度并应用 Softmax 竞争
            B, C, Hf, Wf = visual_aligned.shape
            flat_feat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
            
            # 使用温度系数 100 放大差异并进行 Softmax
            logits = (flat_feat @ text_vecs.t()) * 100
            probs = F.softmax(logits, dim=-1) # (N_pixels, 81)

            # 3. 得到预测结果 (映射 80->0 表示背景，0-79->1-80 表示类)
            _, max_indices = torch.max(probs, dim=-1)
            preds = (max_indices + 1) % 81
            preds = preds.reshape(B, Hf, Wf)

            # 4. 【优化关键】将预测图 (比如 16x16) 直接放大到 512x512
            # 之前你可能是在放大到原图尺寸 (比如 4000x3000)，那是极慢的！
            preds_res = F.interpolate(
                preds.unsqueeze(1).float(),
                size=(512, 512),  # 统一放大到 512
                mode='nearest'
            ).squeeze(1).long()

            # 5. 指标更新
            metric.update(preds_res, gts)

            # 6. 【显存清理】
            torch.cuda.empty_cache()

    result_miou = metric.compute().item()
    print(f"\n✅ {args.version} mIoU: {result_miou:.4f}")

    # 结果存入本地文件
    with open("/tmp/eval_results.log", "a") as f:
        f.write(f"Version: {args.version} | mIoU: {result_miou:.4f} | Samples: {args.max_samples}\n")


# ==========================================
# 3. 命令行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="c-radio_v4-h", help="radio_v3-h or c-radio_v4-h")
    parser.add_argument("--img_dir", type=str, default="coco_data/val2017")
    parser.add_argument("--ann_file", type=str, default="coco_data/annotations/instances_val2017.json")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_evaluation(args)
