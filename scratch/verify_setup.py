
import torch
import torch.hub

print("Checking model loading...")
# Try loading the model once to ensure it's cached and works
try:
    model = torch.hub.load('RADSeg-OVSS/RADSeg', 'radseg_encoder',
                           model_version='c-radio_v4-h', lang_model="siglip2-g",
                           device='cuda', predict=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

print("Checking data access...")
from pycocotools.coco import COCO
import os

img_dir = "coco_data/val2017"
ann_file = "coco_data/annotations/instances_val2017.json"

if os.path.exists(img_dir) and os.path.exists(ann_file):
    print("Data paths exist.")
    coco = COCO(ann_file)
    print(f"COCO loaded. Number of images: {len(coco.imgs)}")
else:
    print(f"Data paths MISSING: img_dir: {os.path.exists(img_dir)}, ann_file: {os.path.exists(ann_file)}")
