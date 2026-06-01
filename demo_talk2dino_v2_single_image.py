import argparse
import importlib
import json
import math
import sys
from pathlib import Path
import os

import clip
# import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from torchvision.io import read_image
import torchvision.transforms as T


DEFAULT_IMAGE_PATH = r"D:\RADSeg\1a2b81a5-845f-5cb1-b4b2-0e0e3df27bf2.jpeg"
DEFAULT_MODEL_ID = "lorebianchi98/Talk2DINO-ViTB"
DEFAULT_QUERIES = ["water", "building", "road"]
DEFAULT_NEGATIVE = "background"
DEFAULT_ANYUP_MODEL = "anyup_multi_backbone"
DEFAULT_PALETTE = [
    [255, 0, 0],
    [0, 255, 255],
    [255, 255, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
]


def is_talk2dinov3_model(model_id: str) -> bool:
    return "talk2dinov3" in model_id.lower()


def snapshot_model_local_dir(model_id: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_cache_dir = os.path.join(current_dir, ".cache")
    local_dir = os.path.join(local_cache_dir, "hf_models", model_id.replace("/", "__").replace("-", "_"))
    try:
        return snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except Exception:
        if (local_dir / "config.json").exists():
            return str(local_dir)
        raise


def patch_talk2dino_loading():
    original = PreTrainedModel.mark_tied_weights_as_initialized

    def patched(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original(self, loading_info)

    PreTrainedModel.mark_tied_weights_as_initialized = patched
    return original


def patch_clip_loading(model_id: str):
    local_dir = snapshot_model_local_dir(model_id)
    config_path = Path(local_dir) / "config.json"
    clip_model_name = "ViT-B/16"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        clip_model_name = config.get("clip_model_name", clip_model_name)

    preloaded_clip = clip.load(clip_model_name, device="cpu")
    original_clip_load = clip.load

    def patched_clip_load(name, device="meta", *args, **kwargs):
        if name == clip_model_name:
            return preloaded_clip
        return original_clip_load(name, device="cpu", *args, **kwargs)

    clip.load = patched_clip_load
    return original_clip_load


def load_talk2dino_model(model_id: str, device: str):
    original_patch = patch_talk2dino_loading()
    original_clip_patch = patch_clip_loading(model_id)
    local_dir = snapshot_model_local_dir(model_id)
    try:
        if is_talk2dinov3_model(model_id):
            package_name = model_id.replace("/", "__").replace("-", "_")
            init_path = Path(local_dir) / "__init__.py"
            if not init_path.exists():
                init_path.write_text("", encoding="utf-8")
            parent_dir = str(Path(local_dir).parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            talk2dino_module = importlib.import_module(f"{package_name}.modeling_talk2dino")
            talk2dino_cls = getattr(talk2dino_module, "Talk2DINO")
            config_module = importlib.import_module(f"{package_name}.configuration_talk2dino")
            config_cls = getattr(config_module, "Talk2DINOConfig")
            config = config_cls.from_pretrained(local_dir)
            model = talk2dino_cls(config)
            checkpoint = load_file(str(Path(local_dir) / "model.safetensors"))
            remapped = {
                key.replace(".weight_1", ".gamma_1").replace(".weight_2", ".gamma_2"): value
                for key, value in checkpoint.items()
            }
            missing, _unexpected = model.load_state_dict(remapped, strict=False, assign=True)
            real_missing = [key for key in missing if "pamr" not in key]
            if real_missing:
                print(f"[talk2dinov3_load] Unexpected missing keys: {real_missing}")
        else:
            model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
    finally:
        PreTrainedModel.mark_tied_weights_as_initialized = original_patch
        clip.load = original_clip_patch

    return model.to(device).eval()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tiny Hugging Face Talk2DINO v2 single-image query demo."
    )
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Independent text queries, for example: water building road",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional output path. Defaults to scratch/<image_stem>_talk2dino_v2_demo.png",
    )
    parser.add_argument(
        "--negative_query",
        default=DEFAULT_NEGATIVE,
        help="Generic negative class used for competitive mask when needed. Default: background",
    )
    parser.add_argument(
        "--use_anyup",
        action="store_true",
        help="Also upsample Talk2DINO features with AnyUp and visualize high-resolution heatmaps.",
    )
    parser.add_argument(
        "--anyup_entrypoint",
        default=DEFAULT_ANYUP_MODEL,
        help="torch.hub AnyUp entrypoint, for example: anyup or anyup_multi_backbone",
    )
    parser.add_argument(
        "--anyup_use_natten",
        action="store_true",
        help="Load the NATTEN-based AnyUp variant.",
    )
    parser.add_argument(
        "--anyup_q_chunk_size",
        type=int,
        default=None,
        help="Optional q_chunk_size passed to AnyUp for lower memory usage.",
    )
    parser.add_argument(
        "--anyup_output_size",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional output size for AnyUp feature upsampling, for example: --anyup_output_size 384 384",
    )
    return parser


def compute_similarity_map(model, image, query: str, device: str) -> np.ndarray:
    with torch.no_grad():
        text_embed = model.encode_text(query)
        image_embed = model.encode_image(image)

    if not isinstance(text_embed, torch.Tensor):
        text_embed = torch.as_tensor(text_embed)
    if not isinstance(image_embed, torch.Tensor):
        image_embed = torch.as_tensor(image_embed)

    text_embed = text_embed.to(device)
    image_embed = image_embed.to(device)

    if text_embed.dim() == 1:
        text_embed = text_embed.unsqueeze(0)

    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    similarity = (image_embed @ text_embed.T).squeeze(0, -1)
    if similarity.dim() != 1:
        raise ValueError(f"Expected flattened patch similarity, got shape {tuple(similarity.shape)}")

    num_tokens = similarity.shape[0]
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square grid.")

    return similarity.reshape(side, side).detach().cpu().numpy()


def extract_lr_feature_map(model, image, device: str) -> torch.Tensor:
    with torch.no_grad():
        image_embed = model.encode_image(image)

    if not isinstance(image_embed, torch.Tensor):
        image_embed = torch.as_tensor(image_embed)
    image_embed = image_embed.to(device)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    if image_embed.dim() != 3:
        raise ValueError(f"Expected Talk2DINO patch tokens with shape (B, N, C), got {tuple(image_embed.shape)}")

    batch_size, num_tokens, channels = image_embed.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square grid.")

    return image_embed.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2)


def build_hr_image_tensor(image: Image.Image, device: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def compute_anyup_similarity_map(
    model,
    upsampler,
    image: Image.Image,
    lr_features: torch.Tensor,
    query: str,
    device: str,
    q_chunk_size=None,
    output_size=None,
) -> np.ndarray:
    hr_image = build_hr_image_tensor(image, device)

    with torch.no_grad():
        text_embed = model.encode_text(query)
        kwargs = {}
        if q_chunk_size is not None:
            kwargs["q_chunk_size"] = q_chunk_size
        if output_size is not None:
            kwargs["output_size"] = output_size
        hr_features = upsampler(hr_image, lr_features, **kwargs)

    if not isinstance(text_embed, torch.Tensor):
        text_embed = torch.as_tensor(text_embed)
    if not isinstance(hr_features, torch.Tensor):
        hr_features = torch.as_tensor(hr_features)

    text_embed = text_embed.to(device)
    hr_features = hr_features.to(device)

    if text_embed.dim() == 1:
        text_embed = text_embed.unsqueeze(0)

    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    hr_features = hr_features / hr_features.norm(dim=1, keepdim=True).clamp_min(1e-8)

    similarity = torch.einsum("bchw,bc->bhw", hr_features, text_embed)
    return similarity.squeeze(0).detach().cpu().numpy()


def build_competitive_visuals_from_similarity_maps(similarity_maps, palette):
    stacked = np.stack(similarity_maps, axis=0)
    competitive_mask = stacked.argmax(axis=0)
    qualitative_overlay = plot_qualitative_mask(
        np.zeros((*competitive_mask.shape, 3), dtype=np.uint8),
        competitive_mask,
        palette,
    )
    return competitive_mask, qualitative_overlay


def build_palette(num_queries: int):
    palette = [color[:] for color in DEFAULT_PALETTE]
    while len(palette) < num_queries:
        palette.append([int(x) for x in np.random.randint(0, 255, size=3)])
    return palette[:num_queries]


def plot_qualitative_mask(image_np: np.ndarray, mask_np: np.ndarray, palette):
    qualitative = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        qualitative[mask_np == idx] = np.array(color, dtype=np.uint8)
    return qualitative


def compute_competitive_mask(model, image_path: Path, queries, device: str):
    img = read_image(str(image_path)).to(device).float().unsqueeze(0)

    with torch.no_grad():
        text_tokens = model.build_dataset_class_tokens("sub_imagenet_template", queries)
        text_tokens = text_tokens.to(device)
        text_emb = model.build_text_embedding(text_tokens)
        mask, _ = model.generate_masks(
            img,
            img_metas=None,
            text_emb=text_emb,
            classnames=queries,
            apply_pamr=True,
        )
        mask = mask.argmax(dim=1)

    return mask.squeeze(0).detach().cpu().numpy()


# def render_demo(
#     image_path: Path,
#     queries,
#     negative_query: str,
#     model_id: str,
#     output_path: Path,
#     use_anyup: bool,
#     anyup_entrypoint: str,
#     anyup_use_natten: bool,
#     anyup_q_chunk_size,
#     anyup_output_size,
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     print(f"Loading model: {model_id}")

#     model = load_talk2dino_model(model_id, device)

#     image = Image.open(image_path).convert("RGB")
#     image_np = np.array(image)
#     lr_features = extract_lr_feature_map(model, image, device)
#     competitive_queries = list(queries)
#     competitive_label_names = list(queries)
#     if len(competitive_queries) == 1 and negative_query:
#         competitive_queries.append(negative_query)

#     palette = build_palette(len(competitive_queries))
#     competitive_mask = compute_competitive_mask(model, image_path, competitive_queries, device)
#     qualitative_overlay = plot_qualitative_mask(image_np, competitive_mask, palette)
#     upsampler = None
#     if use_anyup:
#         print(f"Loading AnyUp: {anyup_entrypoint} (use_natten={anyup_use_natten})")
#         upsampler = torch.hub.load(
#             "wimmerth/anyup",
#             anyup_entrypoint,
#             use_natten=anyup_use_natten,
#         ).to(device).eval()

#     num_rows = 4 if use_anyup else 2
#     fig, axes = plt.subplots(num_rows, len(queries) + 1, figsize=(5 * (len(queries) + 1), 4.8 * num_rows))
#     if not isinstance(axes, np.ndarray):
#         axes = np.array([axes])
#     if axes.ndim == 1:
#         axes = axes.reshape(1, -1)

#     axes[0, 0].imshow(image)
#     axes[0, 0].set_title("Original")
#     axes[0, 0].axis("off")

#     axes[1, 0].imshow(image)
#     axes[1, 0].imshow(qualitative_overlay, alpha=0.6, interpolation="nearest")
#     if len(competitive_queries) == len(queries):
#         axes[1, 0].set_title("Competitive mask overlay")
#     else:
#         axes[1, 0].set_title(f"Competitive mask overlay\n({queries[0]} vs {negative_query})")
#     axes[1, 0].axis("off")

#     anyup_similarity_maps = []

#     for idx, query in enumerate(queries, start=1):
#         similarity = compute_similarity_map(model, image, query, device)
#         axes[0, idx].imshow(image)
#         axes[0, idx].imshow(similarity, cmap="magma", alpha=0.5, interpolation="bilinear")
#         axes[0, idx].set_title(
#             f"{query}\nmin={similarity.min():.4f} max={similarity.max():.4f} mean={similarity.mean():.4f}"
#         )
#         axes[0, idx].axis("off")

#         single_mask = np.zeros_like(competitive_mask)
#         single_mask[competitive_mask == (idx - 1)] = 1
#         single_overlay = np.zeros((*competitive_mask.shape, 3), dtype=np.uint8)
#         single_overlay[single_mask == 1] = np.array(palette[idx - 1], dtype=np.uint8)
#         axes[1, idx].imshow(image)
#         axes[1, idx].imshow(single_overlay, alpha=0.6, interpolation="nearest")
#         if len(competitive_queries) == len(queries):
#             axes[1, idx].set_title(f"{query} region")
#         else:
#             axes[1, idx].set_title(f"{query} wins vs {negative_query}")
#         axes[1, idx].axis("off")

#         if use_anyup and upsampler is not None:
#             hr_similarity = compute_anyup_similarity_map(
#                 model,
#                 upsampler,
#                 image,
#                 lr_features,
#                 query,
#                 device,
#                 q_chunk_size=anyup_q_chunk_size,
#                 output_size=tuple(anyup_output_size) if anyup_output_size is not None else None,
#             )
#             anyup_similarity_maps.append(hr_similarity)
#             axes[2, idx].imshow(image)
#             axes[2, idx].imshow(hr_similarity, cmap="magma", alpha=0.5, interpolation="bilinear")
#             axes[2, idx].set_title(
#                 f"AnyUp {query}\nmin={hr_similarity.min():.4f} max={hr_similarity.max():.4f} mean={hr_similarity.mean():.4f}"
#             )
#             axes[2, idx].axis("off")

#     if use_anyup:
#         axes[2, 0].imshow(image)
#         axes[2, 0].set_title("Original (AnyUp heatmaps)")
#         axes[2, 0].axis("off")

#         anyup_competitive_maps = list(anyup_similarity_maps)
#         if len(queries) == 1 and negative_query:
#             anyup_negative = compute_anyup_similarity_map(
#                 model,
#                 upsampler,
#                 image,
#                 lr_features,
#                 negative_query,
#                 device,
#                 q_chunk_size=anyup_q_chunk_size,
#                 output_size=tuple(anyup_output_size) if anyup_output_size is not None else None,
#             )
#             anyup_competitive_maps.append(anyup_negative)

#         anyup_competitive_mask, anyup_qualitative_overlay = build_competitive_visuals_from_similarity_maps(
#             anyup_competitive_maps, palette
#         )

#         axes[3, 0].imshow(image)
#         axes[3, 0].imshow(anyup_qualitative_overlay, alpha=0.6, interpolation="nearest")
#         if len(competitive_queries) == len(queries):
#             axes[3, 0].set_title("AnyUp competitive mask overlay")
#         else:
#             axes[3, 0].set_title(f"AnyUp competitive mask overlay\n({queries[0]} vs {negative_query})")
#         axes[3, 0].axis("off")

#         for idx, query in enumerate(queries, start=1):
#             single_mask = np.zeros_like(anyup_competitive_mask)
#             single_mask[anyup_competitive_mask == (idx - 1)] = 1
#             single_overlay = np.zeros((*anyup_competitive_mask.shape, 3), dtype=np.uint8)
#             single_overlay[single_mask == 1] = np.array(palette[idx - 1], dtype=np.uint8)
#             axes[3, idx].imshow(image)
#             axes[3, idx].imshow(single_overlay, alpha=0.6, interpolation="nearest")
#             if len(competitive_queries) == len(queries):
#                 axes[3, idx].set_title(f"AnyUp {query} region")
#             else:
#                 axes[3, idx].set_title(f"AnyUp {query} wins vs {negative_query}")
#             axes[3, idx].axis("off")

#     fig.suptitle(f"Talk2DINO v2 single-image query demo\n{image_path.name}", fontsize=14)
#     fig.tight_layout()
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved visualization to: {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("scratch") / f"{image_path.stem}_talk2dino_v2_demo.png"

    # render_demo(
    #     image_path=image_path,
    #     queries=args.queries,
    #     negative_query=args.negative_query,
    #     model_id=args.model_id,
    #     output_path=output_path,
    #     use_anyup=args.use_anyup,
    #     anyup_entrypoint=args.anyup_entrypoint,
    #     anyup_use_natten=args.anyup_use_natten,
    #     anyup_q_chunk_size=args.anyup_q_chunk_size,
    #     anyup_output_size=args.anyup_output_size,
    # )


if __name__ == "__main__":
    main()
