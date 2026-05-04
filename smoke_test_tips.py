import argparse

import torch
from PIL import Image

from vl_backends import create_backend


def main():
    parser = argparse.ArgumentParser(description="Minimal TIPS smoke test for text/image embedding shapes.")
    parser.add_argument("image_path", type=str, help="Path to a test image")
    parser.add_argument("--prompt", type=str, default="a historic cathedral", help="Test text prompt")
    parser.add_argument("--model_id", type=str, default="google/tipsv2-l14", help="TIPS model id")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()

    backend = create_backend(
        backend_name="tips",
        device=args.device,
        model_id=args.model_id,
    )

    image = Image.open(args.image_path).convert("RGB")
    image_input = backend.transform(image).unsqueeze(0).to(args.device) if backend.transform else image

    with torch.no_grad():
        text_emb = backend.encode_text([args.prompt])
        feature_map = backend.encode_image_to_feature_map(image_input)

    print(f"text_emb shape: {tuple(text_emb.shape)}")
    print(f"feature_map shape: {tuple(feature_map.shape)}")
    print(f"text_emb dtype: {text_emb.dtype}")
    print(f"feature_map dtype: {feature_map.dtype}")


if __name__ == "__main__":
    main()
