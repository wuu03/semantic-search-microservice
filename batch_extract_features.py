import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def spherical_kmeans(features, num_clusters=100, num_iters=100, tol=1e-4):
    features = features.float()
    x = F.normalize(features, p=2, dim=-1)
    num_points, feat_dim = x.shape

    if num_points <= num_clusters:
        return x, torch.arange(num_points, device=x.device)

    indices = torch.randperm(num_points, device=x.device)[:num_clusters]
    centers = x[indices]

    for _ in range(num_iters):
        sim = torch.matmul(x, centers.transpose(0, 1))
        labels = torch.argmax(sim, dim=-1)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, feat_dim), x)
        new_centers = F.normalize(new_centers, p=2, dim=-1)

        center_shift = (centers * new_centers).sum(dim=-1).mean()
        centers = new_centers
        if 1.0 - center_shift < tol:
            break

    return centers, labels


def _recompute_centers(features, labels, num_clusters):
    feat_dim = features.shape[-1]
    centers = torch.zeros((num_clusters, feat_dim), device=features.device, dtype=features.dtype)
    centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, feat_dim), features)
    counts = torch.bincount(labels, minlength=num_clusters).to(features.device)
    valid = counts > 0
    if valid.any():
        centers[valid] = centers[valid] / counts[valid].unsqueeze(1)
        centers[valid] = F.normalize(centers[valid], p=2, dim=-1)
    return centers, counts


def adaptive_spherical_kmeans(
    features,
    max_clusters=100,
    min_cluster_pixels=8,
    merge_similarity=0.985,
    num_iters=100,
    tol=1e-4,
):
    centers, labels = spherical_kmeans(
        features,
        num_clusters=max_clusters,
        num_iters=num_iters,
        tol=tol,
    )

    num_points = features.shape[0]
    if num_points == 0:
        return centers, labels

    centers, counts = _recompute_centers(features, labels, centers.shape[0])

    active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return centers[:0], labels

    if min_cluster_pixels > 1 and active_ids.numel() > 1:
        small_ids = active_ids[counts[active_ids] < min_cluster_pixels]
        large_ids = active_ids[counts[active_ids] >= min_cluster_pixels]
        if small_ids.numel() > 0 and large_ids.numel() > 0:
            for small_id in small_ids.tolist():
                pixel_mask = labels == small_id
                if not pixel_mask.any():
                    continue
                sims = torch.matmul(features[pixel_mask], centers[large_ids].transpose(0, 1))
                best_large = large_ids[torch.argmax(sims, dim=1)]
                labels[pixel_mask] = best_large
            centers, counts = _recompute_centers(features, labels, centers.shape[0])
            active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()

    if merge_similarity is not None and active_ids.numel() > 1:
        while True:
            active_centers = centers[active_ids]
            sim = torch.matmul(active_centers, active_centers.transpose(0, 1))
            sim.fill_diagonal_(-1.0)
            max_sim = torch.max(sim)
            if max_sim < merge_similarity:
                break

            merge_pair = torch.nonzero(sim == max_sim, as_tuple=False)[0]
            left_idx = active_ids[merge_pair[0]].item()
            right_idx = active_ids[merge_pair[1]].item()

            if counts[left_idx] < counts[right_idx]:
                left_idx, right_idx = right_idx, left_idx

            labels[labels == right_idx] = left_idx
            centers, counts = _recompute_centers(features, labels, centers.shape[0])
            active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
            if active_ids.numel() <= 1:
                break

    active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    remapped_labels = labels.clone()
    final_centers = []
    for new_cluster_id, old_cluster_id in enumerate(active_ids.tolist()):
        remapped_labels[labels == old_cluster_id] = new_cluster_id
        final_centers.append(centers[old_cluster_id])

    if final_centers:
        final_centers = torch.stack(final_centers, dim=0)
    else:
        final_centers = centers[:0]

    return final_centers, remapped_labels


class FeatureBatchExtractor:
    def __init__(
        self,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        device="cuda",
        num_clusters=100,
        min_cluster_pixels=8,
        merge_similarity=0.985,
    ):
        self.device = device
        self.num_clusters = num_clusters
        self.min_cluster_pixels = min_cluster_pixels
        self.merge_similarity = merge_similarity

        print(f"Loading RADSeg model {model_version}...")
        self.radseg = torch.hub.load(
            "RADSeg-OVSS/RADSeg",
            "radseg_encoder",
            model_version=model_version,
            lang_model=lang_model,
            device=self.device,
            predict=False,
        )
        if hasattr(self.radseg, "model"):
            self.radseg.model.eval()
        else:
            self.radseg.eval()

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def process_tensor(self, img_tensor):
        img_tensor = img_tensor.to(self.device)

        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        _, channels, height_fm, width_fm = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, channels)
        dense_flat = F.normalize(dense_flat, dim=-1)

        centers, labels = adaptive_spherical_kmeans(
            dense_flat,
            max_clusters=self.num_clusters,
            min_cluster_pixels=self.min_cluster_pixels,
            merge_similarity=self.merge_similarity,
        )
        label_map = labels.reshape(height_fm, width_fm)

        clusters = [
            {
                "cluster_id": int(cluster_idx),
                "v": centers[cluster_idx].detach().cpu().tolist(),
            }
            for cluster_idx in range(centers.shape[0])
        ]

        return {
            "clusters": clusters,
            "feature_map_size": [int(height_fm), int(width_fm)],
            "cluster_id_map": label_map.detach().cpu().tolist(),
        }


class FastImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_name = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, img_name, True
        except Exception:
            return torch.zeros((3, 224, 224)), img_name, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract clustered RADSeg features from images.")
    parser.add_argument("--input_dir", type=str, default="images/", help="Directory with images")
    parser.add_argument("--output_file", type=str, default="tmp/features.jsonl", help="Output JSONL file")
    parser.add_argument("--num_clusters", type=int, default=100, help="Maximum number of clusters per image")
    parser.add_argument(
        "--min_cluster_pixels",
        type=int,
        default=8,
        help="Clusters smaller than this are reassigned to larger clusters",
    )
    parser.add_argument(
        "--merge_similarity",
        type=float,
        default=0.985,
        help="Merge clusters whose cosine similarity exceeds this threshold",
    )
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h")

    args = parser.parse_args()

    extractor = FeatureBatchExtractor(
        num_clusters=args.num_clusters,
        model_version=args.model_version,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
    )

    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} not found.")
        raise SystemExit(1)

    image_paths = [
        os.path.join(args.input_dir, name)
        for name in os.listdir(args.input_dir)
        if name.casefold().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    dataset = FastImageDataset(image_paths, extractor.transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as read_file:
            for line in read_file:
                try:
                    processed_ids.add(json.loads(line)["image_id"])
                except Exception:
                    continue
        print(f"Resuming: {len(processed_ids)} images already processed. Skipping them.")

    with open(args.output_file, "a", encoding="utf-8") as write_file:
        for tensor, img_names, valid_flags in tqdm(dataloader):
            img_name = img_names[0]

            if img_name in processed_ids:
                continue

            is_valid = valid_flags[0].item()
            if not is_valid:
                print(f"Invalid or corrupted image skipped: {img_name}")
                continue

            try:
                cluster_result = extractor.process_tensor(tensor)
                data = {
                    "image_id": img_name,
                    "clusters": cluster_result["clusters"],
                    "feature_map_size": cluster_result["feature_map_size"],
                    "cluster_id_map": cluster_result["cluster_id_map"],
                }
                write_file.write(json.dumps(data) + "\n")
                write_file.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"Error processing {img_name}: {exc}")

    print(f"Success! Features saved to {args.output_file}")
