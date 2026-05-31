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
from transformers import AutoModel


class VisionLanguageBackend:
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = None

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_to_feature_map(self, image_input):
        raise NotImplementedError
    
    def encode_image_global(self, image_input) -> torch.Tensor:
        raise NotImplementedError


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
    def encode_image_global(self, image_input) -> torch.Tensor:
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


class Talk2DINOBackend(VisionLanguageBackend):
    def __init__(self, model_id="lorebianchi98/Talk2DINO-ViTL", device="cuda"):
        super().__init__(device=device)
        print(f"Loading Talk2DINO model {model_id}...")
        package_name = model_id.replace("/", "__").replace("-", "_")
        local_model_dir = snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(os.getcwd(), "scratch", "hf_models", package_name),
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


def create_backend(
    backend_name,
    device="cuda",
    model_version="c-radio_v4-h",
    lang_model="siglip2-g",
    model_id=None,
):
    backend_name = backend_name.lower()
    if backend_name == "radseg":
        return RADSegBackend(model_version=model_version, lang_model=lang_model, device=device)
    if backend_name == "tips":
        resolved_model_id = model_id or "google/tipsv2-l14"
        return TIPSBackend(model_id=resolved_model_id, device=device)
    if backend_name == "talk2dino":
        resolved_model_id = model_id or "lorebianchi98/Talk2DINO-ViTL"
        return Talk2DINOBackend(model_id=resolved_model_id, device=device)
    raise ValueError(f"Unsupported backend '{backend_name}'")
