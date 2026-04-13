import os
import argparse
import time
import json
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

def spherical_kmeans(features, num_clusters=100, num_iters=100, tol=1e-4):
    x = F.normalize(features, p=2, dim=-1)
    N, D = x.shape
    if N <= num_clusters:
        # Not enough pixels to cluster, fallback to returning all
        return x, torch.arange(N, device=x.device)
    
    indices = torch.randperm(N, device=x.device)[:num_clusters]
    centers = x[indices]

    for i in range(num_iters):
        sim = torch.matmul(x, centers.transpose(0, 1))
        labels = torch.argmax(sim, dim=-1)
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
        new_centers = F.normalize(new_centers, p=2, dim=-1)

        center_shift = (centers * new_centers).sum(dim=-1).mean()
        centers = new_centers
        if 1.0 - center_shift < tol:
            break
    return centers, labels

class FeatureBatchExtractor:
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda", num_clusters=100):
        self.device = device
        self.num_clusters = num_clusters
        print(f"Loading RADSeg model {model_version}...")
        self.radseg = torch.hub.load(
            'RADSeg-OVSS/RADSeg', 'radseg_encoder',
            model_version=model_version, lang_model=lang_model, device=self.device, predict=False
        )
        if hasattr(self.radseg, 'model'):
            self.radseg.model.eval()
        else:
            self.radseg.eval()
            
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def process_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract dense features
        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)
        
        B, C, H_f, W_f = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        dense_flat = F.normalize(dense_flat, dim=-1)
        
        # Get cluster centers (size: num_clusters * C)
        centers, labels = spherical_kmeans(dense_flat, num_clusters=self.num_clusters)
        # return the centers purely like search_demo.py (it relies on K centers)
        return centers.cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Extract Image Features on Remote GPU")
    parser.add_argument("--input_dir", type=str, default="images/", help="Directory with images")
    parser.add_argument("--output_file", type=str, default="features.jsonl", help="Output JSONL file")
    parser.add_argument("--num_clusters", type=int, default=100, help="Number of clusters per image")
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h")
    
    args = parser.parse_args()
    
    extractor = FeatureBatchExtractor(num_clusters=args.num_clusters, model_version=args.model_version)
    
    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} not found.")
        exit(1)
        
    image_files = [f for f in os.listdir(args.input_dir) if f.casefold().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {args.input_dir}")
    
    # Process images and append to jsonl
    with open(args.output_file, 'w') as f:
        for img_name in tqdm(image_files):
            img_path = os.path.join(args.input_dir, img_name)
            try:
                clusters = extractor.process_image(img_path)
                data = {
                    "image_id": img_name,
                    "clusters": clusters
                }
                f.write(json.dumps(data) + '\n')
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
    print(f"Success! Features saved to {args.output_file}")
