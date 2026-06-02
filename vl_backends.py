import importlib
import json
import math
import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
local_cache_dir = os.path.join(current_dir, ".cache")

os.environ["HF_HOME"] = os.path.join(local_cache_dir, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(local_cache_dir, "torch")

import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

from demo_talk2dino_v2_single_image import (
    build_hr_image_tensor,
    extract_lr_feature_map,
    load_talk2dino_model,
    patch_clip_loading,
    patch_talk2dino_loading,
)


class VisionLanguageBackend:
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = None
        self.image_embedding_type = "mean_pooled_dense"

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_to_feature_map(self, image_input):
        raise NotImplementedError

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        if feature_map is None:
            feature_map = self.encode_image_to_feature_map(image_input)
        embedding = feature_map.mean(dim=(2, 3))
        return F.normalize(embedding, dim=-1)


class RADSegBackend(VisionLanguageBackend):
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
        super().__init__(device=device)
        print(f"Loading RADSeg model {model_version}...")
        self.model = torch.hub.load(
            "RADSeg-OVSS/RADSeg",
            "radseg_encoder",
            model_version=model_version,
            lang_model=lang_model,
            device=self.device,
            predict=False,
        )
        if hasattr(self.model, "model"):
            self.model.model.eval()
        else:
            self.model.eval()

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode_prompts(prompts, onehot=False)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, image_input) -> torch.Tensor:
        feat_map = self.model.encode_image_to_feat_map(image_input.to(self.device))
        aligned = self.model.align_spatial_features_with_language(feat_map, onehot=False)
        return F.normalize(aligned, dim=1)

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        # image_input = image_input.to(self.device)
        
        # if hasattr(self.model, "model"):
        #     summary, _ = self.model.model(image_input)
        # else:
        #     summary, _ = self.model(image_input)
        feat_map = self.model.encode_image_to_feat_map(image_input.to(self.device))
        aligned = self.model.align_spatial_features_with_language(feat_map, onehot=False)

        pooled = F.adaptive_avg_pool2d(aligned, (1, 1))
        
        global_vector = pooled.view(pooled.size(0), -1)
            
        return F.normalize(global_vector, dim=-1)


class TIPSBackend(VisionLanguageBackend):
    def __init__(self, model_id="google/tipsv2-l14", device="cuda"):
        super().__init__(device=device)
        self.image_embedding_type = "native_image_embedding"
        print(f"Loading TIPS model {model_id}...")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        # Official model card: resize, ToTensor only, no ImageNet normalization.
        self.transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
            ]
        )

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode_text(prompts)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        return F.normalize(embeddings.to(self.device), dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, image_input) -> torch.Tensor:
        outputs = self.model.encode_image(image_input.to(self.device))
        patch_tokens = outputs.patch_tokens
        if not isinstance(patch_tokens, torch.Tensor):
            patch_tokens = torch.as_tensor(patch_tokens)
        patch_tokens = patch_tokens.to(self.device)

        if patch_tokens.dim() != 3:
            raise ValueError(f"Unsupported TIPS patch token shape: {tuple(patch_tokens.shape)}")

        batch_size, num_tokens, channels = patch_tokens.shape
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"TIPS patch token count {num_tokens} is not a square grid.")

        feature_map = patch_tokens.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2)
        return F.normalize(feature_map, dim=1)

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        outputs = self.model.encode_image(image_input.to(self.device))
        for attr_name in ("image_embeds", "image_features", "pooler_output"):
            value = getattr(outputs, attr_name, None)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                return F.normalize(value.to(self.device), dim=-1)
        return super().encode_image_embedding(image_input, feature_map=feature_map)


class TIPSAnyUpBackend(TIPSBackend):
    def __init__(
        self,
        model_id="google/tipsv2-l14",
        device="cuda",
        anyup_entrypoint="anyup_multi_backbone",
        anyup_use_natten=False,
        anyup_q_chunk_size=None,
        anyup_output_size=(384, 384),
    ):
        super().__init__(model_id=model_id, device=device)

        print(f"Loading AnyUp: {anyup_entrypoint} (use_natten={anyup_use_natten})")
        self.upsampler = torch.hub.load(
            "wimmerth/anyup",
            anyup_entrypoint,
            use_natten=anyup_use_natten,
        ).to(self.device).eval()

        self.anyup_q_chunk_size = anyup_q_chunk_size
        self.anyup_output_size = tuple(anyup_output_size) if anyup_output_size is not None else None
        self.transform = None

    @torch.no_grad()
    def encode_image_to_feature_map(self, image_input) -> torch.Tensor:
        if not isinstance(image_input, Image.Image):
            raise ValueError("TIPSAnyUpBackend expects a PIL.Image input.")

        transformed = TIPSBackend.encode_image_to_feature_map(
            self,
            self.transform_for_tips(image_input).unsqueeze(0),
        )
        hr_image = build_hr_image_tensor(image_input, self.device)
        kwargs = {}
        if self.anyup_q_chunk_size is not None:
            kwargs["q_chunk_size"] = self.anyup_q_chunk_size
        if self.anyup_output_size is not None:
            kwargs["output_size"] = self.anyup_output_size
        hr_features = self.upsampler(hr_image, transformed, **kwargs)
        return F.normalize(hr_features.to(self.device), dim=1)

    def transform_for_tips(self, image_input):
        return T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
            ]
        )(image_input).to(self.device)

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        if not isinstance(image_input, Image.Image):
            return super().encode_image_embedding(image_input, feature_map=feature_map)
        image_tensor = self.transform_for_tips(image_input).unsqueeze(0)
        return TIPSBackend.encode_image_embedding(self, image_tensor, feature_map=feature_map)


class Talk2DINOBackend(VisionLanguageBackend):
    def __init__(self, model_id="lorebianchi98/Talk2DINO-ViTL", device="cuda"):
        super().__init__(device=device)
        self.image_embedding_type = "native_cls_token"
        print(f"Loading Talk2DINO model {model_id}...")
        if "talk2dinov3" in model_id.lower():
            self.model = load_talk2dino_model(model_id, self.device)
            self.transform = None
            return

        package_name = model_id.replace("/", "__").replace("-", "_")
        local_model_dir = snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(local_cache_dir, "hf_models", package_name),
            local_dir_use_symlinks=False,
        )
        init_path = os.path.join(local_model_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as handle:
                handle.write("")
        parent_dir = os.path.dirname(local_model_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        talk2dino_module = importlib.import_module(f"{package_name}.modeling_talk2dino")
        talk2dino_cls = getattr(talk2dino_module, "Talk2DINO")
        if not hasattr(talk2dino_cls, "all_tied_weights_keys"):
            talk2dino_cls.all_tied_weights_keys = {}

        clip_model_name = "ViT-B/16"
        with open(os.path.join(local_model_dir, "config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
        clip_model_name = config.get("clip_model_name", clip_model_name)

        preloaded_clip = clip.load(clip_model_name, device="cpu")
        original_clip_load = clip.load

        def patched_clip_load(name, device="meta", *args, **kwargs):
            if name == clip_model_name:
                return preloaded_clip
            return original_clip_load(name, device="cpu", *args, **kwargs)

        clip.load = patched_clip_load
        try:
            self.model = talk2dino_cls.from_pretrained(
                local_model_dir,
                low_cpu_mem_usage=False,
            )
        finally:
            clip.load = original_clip_load

        self.model = self.model.to(self.device).eval()
        self.transform = None

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        if not prompts:
            return torch.empty((0, 1024), device=self.device) 

        tokens = self.model.build_dataset_class_tokens("sub_imagenet_template", prompts)

        text_embeds = self.model.build_text_embedding(tokens).float()

        if text_embeds.ndim == 3:
            text_embeds = text_embeds.mean(dim=1)

        return F.normalize(text_embeds.to(self.device), dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, image_input) -> torch.Tensor:
        image_embed = self.model.encode_image(image_input)
        if not isinstance(image_embed, torch.Tensor):
            image_embed = torch.as_tensor(image_embed)
        image_embed = image_embed.to(self.device)

        if image_embed.dim() == 4:
            if image_embed.shape[1] <= 8 and image_embed.shape[-1] > image_embed.shape[1]:
                feature_map = image_embed.permute(0, 3, 1, 2)
            else:
                feature_map = image_embed
        elif image_embed.dim() == 3:
            batch_size, num_tokens, channels = image_embed.shape
            side = int(math.isqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f"Talk2DINO patch token count {num_tokens} is not a square grid.")
            feature_map = image_embed.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported Talk2DINO image embedding shape: {tuple(image_embed.shape)}")

        return F.normalize(feature_map, dim=1)

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        return encode_talk2dino_cls_token(self.model, image_input, self.device)


class Talk2DINOAnyUpBackend(VisionLanguageBackend):
    def __init__(
        self,
        model_id="lorebianchi98/Talk2DINO-ViTB",
        device="cuda",
        anyup_entrypoint="anyup_multi_backbone",
        anyup_use_natten=False,
        anyup_q_chunk_size=None,
        anyup_output_size=(384, 384),
    ):
        super().__init__(device=device)
        self.image_embedding_type = "native_cls_token"
        print(f"Loading Talk2DINO + AnyUp model {model_id}...")
        self.model = load_talk2dino_model(model_id, self.device)

        print(f"Loading AnyUp: {anyup_entrypoint} (use_natten={anyup_use_natten})")
        self.upsampler = torch.hub.load(
            "wimmerth/anyup",
            anyup_entrypoint,
            use_natten=anyup_use_natten,
        ).to(self.device).eval()

        self.anyup_q_chunk_size = anyup_q_chunk_size
        self.anyup_output_size = tuple(anyup_output_size) if anyup_output_size is not None else None
        self.transform = None

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        outputs = []
        for prompt in prompts:
            text_embed = self.model.encode_text(prompt)
            if not isinstance(text_embed, torch.Tensor):
                text_embed = torch.as_tensor(text_embed)
            if text_embed.dim() == 1:
                text_embed = text_embed.unsqueeze(0)
            outputs.append(text_embed.to(self.device))
        embeddings = torch.cat(outputs, dim=0)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, image_input) -> torch.Tensor:
        if not isinstance(image_input, Image.Image):
            raise ValueError("Talk2DINOAnyUpBackend expects a PIL.Image input.")

        lr_features = extract_lr_feature_map(self.model, image_input, self.device)
        hr_image = build_hr_image_tensor(image_input, self.device)
        kwargs = {}
        if self.anyup_q_chunk_size is not None:
            kwargs["q_chunk_size"] = self.anyup_q_chunk_size
        if self.anyup_output_size is not None:
            kwargs["output_size"] = self.anyup_output_size
        hr_features = self.upsampler(hr_image, lr_features, **kwargs)
        hr_features = hr_features.to(self.device)
        return F.normalize(hr_features, dim=1)

    @torch.no_grad()
    def encode_image_embedding(self, image_input, feature_map=None) -> torch.Tensor | None:
        return encode_talk2dino_cls_token(self.model, image_input, self.device)


def encode_talk2dino_cls_token(model, image_input, device) -> torch.Tensor:
    if isinstance(image_input, torch.Tensor):
        image_tensor = image_input.to(device)
    elif isinstance(image_input, Image.Image):
        image_tensor = model.image_transforms(image_input).to(device)
    else:
        raise ValueError("Talk2DINO CLS extraction expects a PIL.Image or torch.Tensor input.")

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    features = model.model.forward_features(image_tensor)
    if isinstance(features, dict):
        if "x_norm_clstoken" in features:
            cls_token = features["x_norm_clstoken"]
        elif "x_norm_patchtokens" in features:
            cls_token = features["x_norm_patchtokens"].mean(dim=1)
        else:
            tensor_values = [value for value in features.values() if isinstance(value, torch.Tensor)]
            if not tensor_values:
                raise ValueError("Talk2DINO forward_features returned no tensor values.")
            cls_token = tensor_values[0][:, 0, :] if tensor_values[0].dim() == 3 else tensor_values[0]
    else:
        cls_token = features[:, 0, :] if features.dim() == 3 else features

    if cls_token.dim() == 1:
        cls_token = cls_token.unsqueeze(0)

    return F.normalize(cls_token.to(device), dim=-1)


def create_backend(
    backend_name,
    device="cuda",
    model_version="c-radio_v4-h",
    lang_model="siglip2-g",
    model_id=None,
    anyup_entrypoint="anyup_multi_backbone",
    anyup_use_natten=False,
    anyup_q_chunk_size=None,
    anyup_output_size=(384, 384),
):
    backend_name = backend_name.lower()
    if backend_name == "radseg":
        return RADSegBackend(model_version=model_version, lang_model=lang_model, device=device)
    if backend_name == "tips":
        resolved_model_id = model_id or "google/tipsv2-l14"
        return TIPSBackend(model_id=resolved_model_id, device=device)
    if backend_name == "tips_anyup":
        resolved_model_id = model_id or "google/tipsv2-l14"
        return TIPSAnyUpBackend(
            model_id=resolved_model_id,
            device=device,
            anyup_entrypoint=anyup_entrypoint,
            anyup_use_natten=anyup_use_natten,
            anyup_q_chunk_size=anyup_q_chunk_size,
            anyup_output_size=anyup_output_size,
        )
    if backend_name == "talk2dino":
        resolved_model_id = model_id or "lorebianchi98/Talk2DINO-ViTL"
        return Talk2DINOBackend(model_id=resolved_model_id, device=device)
    if backend_name == "talk2dino_anyup":
        resolved_model_id = model_id or "lorebianchi98/Talk2DINO-ViTB"
        return Talk2DINOAnyUpBackend(
            model_id=resolved_model_id,
            device=device,
            anyup_entrypoint=anyup_entrypoint,
            anyup_use_natten=anyup_use_natten,
            anyup_q_chunk_size=anyup_q_chunk_size,
            anyup_output_size=anyup_output_size,
        )
    raise ValueError(f"Unsupported backend '{backend_name}'")
