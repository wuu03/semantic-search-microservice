import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from radseg.radseg import RADSegEncoder


def patch_siglip2_offline_loading():
    try:
        from transformers import AutoModel, AutoProcessor
    except Exception:
        return

    local_snapshot = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/"
        "snapshots/a713301b217d38485fb2204c808367d10bc3cc40"
    )
    if not os.path.isdir(local_snapshot):
        return

    original_model_from_pretrained = AutoModel.from_pretrained
    original_processor_from_pretrained = AutoProcessor.from_pretrained

    def patched_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "google/siglip2-giant-opt-patch16-384":
            pretrained_model_name_or_path = local_snapshot
            kwargs.setdefault("local_files_only", True)
        return original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def patched_processor_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "google/siglip2-giant-opt-patch16-384":
            pretrained_model_name_or_path = local_snapshot
            kwargs.setdefault("local_files_only", True)
        return original_processor_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoModel.from_pretrained = patched_model_from_pretrained
    AutoProcessor.from_pretrained = patched_processor_from_pretrained


class COCOSemanticDataset(Dataset):
    def __init__(self, img_dir, ann_file, image_size=512, max_samples=1000):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.image_size = image_size
        self.ids = list(self.coco.imgs.keys())[:max_samples]
        self.cat_ids = sorted(self.coco.getCatIds())
        self.id_map = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        self.classes = [self.coco.loadCats(cat_id)[0]["name"] for cat_id in self.cat_ids]
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        width, height = img.size
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            cat_id = ann["category_id"]
            mapped_id = self.id_map.get(cat_id)
            if mapped_id is None:
                continue
            mask[self.coco.annToMask(ann) > 0] = mapped_id

        mask = Image.fromarray(mask).resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )

        return self.image_transform(img), torch.from_numpy(np.array(mask)).long()


def compute_mean_iou(confusion_matrix, include_background=False):
    intersection = torch.diag(confusion_matrix).float()
    pred_area = confusion_matrix.sum(dim=0).float()
    target_area = confusion_matrix.sum(dim=1).float()
    union = pred_area + target_area - intersection

    valid_classes = union > 0
    if not include_background and valid_classes.numel() > 0:
        valid_classes[0] = False

    if valid_classes.sum() == 0:
        return 0.0, torch.empty(0, device=confusion_matrix.device), valid_classes

    ious = intersection[valid_classes] / union[valid_classes].clamp_min(1e-6)
    return ious.mean().item(), ious, valid_classes


def update_confusion_matrix(confusion_matrix, preds, gts, num_classes):
    valid_mask = (gts >= 0) & (gts < num_classes)
    if not torch.any(valid_mask):
        return

    inds = num_classes * gts[valid_mask] + preds[valid_mask]
    confusion_matrix += torch.bincount(
        inds, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def encode_text_embeddings(radseg, classes, use_prompt_templates=True, include_background_prompt=False):
    with torch.no_grad():
        if use_prompt_templates:
            text_vecs = radseg.encode_labels(classes, onehot=False)
        else:
            prompts = [f"a photo of a {name}" for name in classes]
            text_vecs = radseg.encode_prompts(prompts, onehot=False)

        text_vecs = F.normalize(text_vecs.float(), dim=-1)

        if include_background_prompt:
            bg_vec = radseg.encode_prompts(
                ["background, environment, surroundings, unlabeled region"],
                onehot=False,
            )
            bg_vec = F.normalize(bg_vec.float(), dim=-1)
            text_vecs = torch.cat([bg_vec, text_vecs], dim=0)

    return text_vecs


def run_evaluation(args):
    device = args.device
    print(f"\nStarting evaluation for {args.version} on {device}")

    patch_siglip2_offline_loading()
    lang_model = args.lang_model or "siglip2-g"

    radseg = RADSegEncoder(
        model_version=args.version,
        lang_model=lang_model,
        device=device,
        predict=False,
        amp=args.amp,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        scra_scaling=args.scra_scaling,
        scga_scaling=args.scga_scaling,
    )

    if hasattr(radseg, "model"):
        radseg.model.eval()
    else:
        radseg.eval()

    dataset = COCOSemanticDataset(
        args.img_dir,
        args.ann_file,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    text_vecs = encode_text_embeddings(
        radseg,
        dataset.classes,
        use_prompt_templates=not args.use_raw_prompts,
        include_background_prompt=args.include_background_prompt,
    ).to(device)

    num_classes = len(dataset.classes) + 1
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )

    with torch.no_grad():
        for imgs, gts in tqdm(loader, desc=f"Evaluating {args.version}"):
            imgs = imgs.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            feat_map = radseg.encode_image_to_feat_map(imgs)
            visual_aligned = radseg.align_spatial_features_with_language(
                feat_map,
                onehot=False,
                use_feat_mlp=args.use_feat_mlp,
            )
            visual_aligned = F.normalize(visual_aligned.float(), dim=1)

            batch_size, channels, feat_h, feat_w = visual_aligned.shape
            flat_feat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, channels)
            similarity = flat_feat @ text_vecs.t()

            max_vals, max_indices = torch.max(similarity, dim=-1)

            if args.include_background_prompt:
                preds = max_indices.reshape(batch_size, feat_h, feat_w).long()
            else:
                preds = (max_indices + 1).reshape(batch_size, feat_h, feat_w).long()
                if args.threshold is not None:
                    preds[max_vals.reshape(batch_size, feat_h, feat_w) < args.threshold] = 0

            preds = F.interpolate(
                preds.unsqueeze(1).float(),
                size=gts.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()

            update_confusion_matrix(confusion_matrix, preds, gts, num_classes)

    miou, class_ious, valid_classes = compute_mean_iou(
        confusion_matrix, include_background=args.include_background_in_miou
    )
    num_valid = int(valid_classes.sum().item())

    print(f"\nValid classes in mIoU: {num_valid} / {num_classes}")
    print(f"{args.version} mIoU: {miou:.4f}")

    os.makedirs(os.path.dirname(args.output_log) or ".", exist_ok=True)
    with open(args.output_log, "a", encoding="utf-8") as f:
        f.write(
            f"version={args.version} "
            f"miou={miou:.4f} "
            f"samples={args.max_samples} "
            f"valid_classes={num_valid} "
            f"threshold={args.threshold} "
            f"use_feat_mlp={args.use_feat_mlp} "
            f"use_raw_prompts={args.use_raw_prompts} "
            f"include_background_prompt={args.include_background_prompt} "
            f"include_background_in_miou={args.include_background_in_miou}\n"
        )

    if args.print_class_iou:
        valid_class_indices = torch.nonzero(valid_classes, as_tuple=False).flatten().tolist()
        for class_idx, class_iou in zip(valid_class_indices, class_ious.tolist()):
            class_name = "background" if class_idx == 0 else dataset.classes[class_idx - 1]
            print(f"{class_idx:>3} {class_name:<20} IoU={class_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="c-radio_v4-h")
    parser.add_argument("--img_dir", type=str, default="coco_data/val2017")
    parser.add_argument(
        "--ann_file",
        type=str,
        default="coco_data/annotations/instances_val2017.json",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_log", type=str, default="eval_results.log")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lang_model", type=str, default=None)
    parser.add_argument("--slide_crop", type=int, default=336)
    parser.add_argument("--slide_stride", type=int, default=224)
    parser.add_argument("--scra_scaling", type=float, default=10.0)
    parser.add_argument("--scga_scaling", type=float, default=10.0)
    parser.add_argument("--use_raw_prompts", action="store_true")
    parser.add_argument("--use_feat_mlp", action="store_true")
    parser.add_argument("--include_background_prompt", action="store_true")
    parser.add_argument("--include_background_in_miou", action="store_true")
    parser.add_argument("--print_class_iou", action="store_true")

    run_evaluation(parser.parse_args())
