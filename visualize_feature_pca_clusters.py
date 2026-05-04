import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from batch_extract_features import adaptive_spherical_kmeans
from vl_backends import create_backend


def pca_rgb(features_2d: torch.Tensor) -> np.ndarray:
    """Project flattened features to RGB with torch PCA."""
    x = features_2d.float()
    x = x - x.mean(dim=0, keepdim=True)

    # q must be <= min(n, d)
    u, s, v = torch.pca_lowrank(x, q=3, center=False)
    rgb = x @ v[:, :3]

    rgb = rgb.cpu().numpy()
    rgb = rgb / (rgb.std(axis=0, keepdims=True) + 1e-6)
    rgb = 1.0 / (1.0 + np.exp(-1.5 * rgb))
    return np.clip(rgb, 0.0, 1.0)


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    num_clusters = int(label_map.max()) + 1 if label_map.size else 0
    cmap = plt.get_cmap("tab20", max(num_clusters, 1))
    rgb = cmap(label_map / max(num_clusters - 1, 1))[..., :3]
    return rgb


def resize_overlay(rgb_map: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    resized = cv2.resize(
        (rgb_map * 255.0).astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Visualize single-image PCA patch features and clustering.")
    parser.add_argument("image_path", type=str, help="Path to one image")
    parser.add_argument("--backend", type=str, default="tips", choices=["tips", "talk2dino", "radseg"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_clusters", type=int, default=20, help="Maximum requested clusters")
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--merge_similarity", type=float, default=0.985)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h")
    parser.add_argument("--lang_model", type=str, default="siglip2-g")
    parser.add_argument("--output_path", type=str, default="scratch/pca_cluster_viz.png")
    args = parser.parse_args()

    backend = create_backend(
        backend_name=args.backend,
        device=args.device,
        model_id=args.model_id,
        model_version=args.model_version,
        lang_model=args.lang_model,
    )

    image = Image.open(args.image_path).convert("RGB")
    model_input = backend.transform(image).unsqueeze(0).to(args.device) if backend.transform else image

    with torch.no_grad():
        feature_map = backend.encode_image_to_feature_map(model_input)

    _, channels, height_fm, width_fm = feature_map.shape
    dense_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    dense_flat = F.normalize(dense_flat, dim=-1)

    pca_rgb_flat = pca_rgb(dense_flat)
    pca_rgb_map = pca_rgb_flat.reshape(height_fm, width_fm, 3)

    centers, labels = adaptive_spherical_kmeans(
        dense_flat,
        max_clusters=args.num_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
    )
    label_map = labels.reshape(height_fm, width_fm).detach().cpu().numpy()
    cluster_rgb_map = colorize_labels(label_map)

    requested = args.num_clusters
    final_clusters = centers.shape[0]

    original_np = np.asarray(image).astype(np.float32) / 255.0
    pca_overlay = resize_overlay(pca_rgb_map, image.width, image.height)
    cluster_overlay = resize_overlay(cluster_rgb_map, image.width, image.height)

    pca_blend = np.clip(original_np * 0.45 + pca_overlay * 0.55, 0.0, 1.0)
    cluster_blend = np.clip(original_np * 0.45 + cluster_overlay * 0.55, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pca_blend)
    axes[1].set_title(f"PCA overlay\n{height_fm}x{width_fm} feature map")
    axes[1].axis("off")

    axes[2].imshow(cluster_blend)
    axes[2].set_title(
        f"Cluster overlay\nrequested={requested}, final={final_clusters}\n"
        f"min_pixels={args.min_cluster_pixels}, merge_sim={args.merge_similarity}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"feature_map shape: {(1, channels, height_fm, width_fm)}")
    print(f"requested clusters: {requested}")
    print(f"final clusters: {final_clusters}")
    print(f"saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
