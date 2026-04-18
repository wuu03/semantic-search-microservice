import os

img_dir = "coco_data/val2017"
ann_file = "coco_data/annotations/instances_val2017.json"

print(f"Checking paths from CWD: {os.getcwd()}")
print(f"img_dir exists: {os.path.exists(img_dir)} ({os.path.abspath(img_dir)})")
print(f"ann_file exists: {os.path.exists(ann_file)} ({os.path.abspath(ann_file)})")

if os.path.exists(img_dir) and os.path.exists(ann_file):
    print("SUCCESS: Relative paths are correct.")
else:
    print("FAILURE: Relative paths are incorrect or files are missing.")
