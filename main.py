# uvicorn main:app --host 0.0.0.0 --port 8001
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, APIRouter, Response, File, UploadFile, Form, Query
from pydantic import BaseModel
from typing import Optional, List
import io
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
from redis import Redis
import cv2
import os
import json
from dotenv import load_dotenv
import laspy

from vl_backends import create_backend

app = FastAPI(title="TimeAtlas Vector Encoding API", version="1.0")

backend = None

load_dotenv()
redis_host = os.getenv("SERVER_IP", "127.0.0.1")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_client = Redis.from_url(f"redis://{redis_host}:{redis_port}/0")

RENDERS_ROOT = Path(os.getenv("RENDERS_ROOT", "./renders"))
ASSETS_ROOT = Path(os.getenv("ASSETS_ROOT", "./assets"))

_tile_cache = {}

class EncodeRequest(BaseModel):
    query: str
    negative_text: Optional[str] = "background"

class EncodeResponse(BaseModel):
    positive_vector: List[float]
    negative_vectors: List[List[float]]

class UnifiedEncodeResponse(BaseModel):
    query_type: str  
    vector: List[float]
    negative_vectors: Optional[List[List[float]]] = None

class OBBEncodeRequest(BaseModel):
    tile_id: str
    bbox_min: List[float]
    bbox_max: List[float]
    rotation: float = 0.0

@app.on_event("startup")
async def load_model():
    global backend
    print("Loading Model...")
    try:
        backend = create_backend(
            backend_name="talk2dino",
            device="cuda", 
            model_id="lorebianchi98/Talk2DINOv3-ViTL",
            # model_version="c-radio_v4-h",
            # lang_model="siglip2-g"
        )
        print("Service Ready.")
    except Exception as e:
        print(f"Failed to load model: {e}")

# @app.post("/api/encode", response_model=EncodeResponse)
# async def encode_text(request: EncodeRequest):
#     if backend is None:
#         raise HTTPException(status_code=503, detail="Loading model... Please try again later.")

#     try:
#         neg_prompts = [p.strip() for p in request.negative_text.split(",") if p.strip()] if request.negative_text else ["background"]
#         prompts = [request.query] + neg_prompts

#         with torch.no_grad():
#             embeddings = backend.encode_text(prompts)
#             if embeddings.dim() == 1:
#                 embeddings = embeddings.unsqueeze(0)
#             embeddings = F.normalize(embeddings, dim=-1)

#         text_vectors = embeddings.detach().cpu().numpy().tolist()

#         return {
#             "positive_vector": text_vectors[0],
#             "negative_vectors": text_vectors[1:]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/encode", response_model=UnifiedEncodeResponse)
async def encode(
    query_text: Optional[str] = Form(None, description="Text Query"),
    query_image: Optional[UploadFile] = File(None, description="Image Query"),
    negative_text: Optional[str] = Form("background", description="Negative Text")
):
    if backend is None:
        raise HTTPException(status_code=503, detail="Loading model... Please try again later.")

    if query_image is not None and query_image.filename != "":
        try:
            contents = await query_image.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            neg_prompts = [p.strip() for p in negative_text.split(",") if p.strip()] if negative_text else ["background"]

            patch_size = 16 
            original_w, original_h = image.size
            
            new_w = (original_w // patch_size) * patch_size
            new_h = (original_h // patch_size) * patch_size
            
            if original_w != new_w or original_h != new_h:
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            if backend.transform is not None:
                image_tensor = backend.transform(image).unsqueeze(0)
            else:
                import torchvision.transforms as T
                image_tensor = T.ToTensor()(image).unsqueeze(0)

            with torch.no_grad():
                embedding = backend.encode_image_embedding(image_tensor)
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                neg_embeddings = backend.encode_text(neg_prompts)
                if neg_embeddings.dim() == 1:
                    neg_embeddings = neg_embeddings.unsqueeze(0)
                embedding = F.normalize(embedding, p=2, dim=-1)
                neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)
                
            image_vector = embedding.detach().cpu().numpy().tolist()[0]
            neg_vectors = neg_embeddings.detach().cpu().numpy().tolist()
            
            return {
                "query_type": "image",
                "vector": image_vector,
                "negative_vectors": neg_vectors
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    elif query_text is not None and query_text.strip() != "":
        try:
            neg_prompts = [p.strip() for p in negative_text.split(",") if p.strip()] if negative_text else ["background"]
            prompts = [query_text] + neg_prompts

            with torch.no_grad():
                embeddings = backend.encode_text(prompts)
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                embeddings = F.normalize(embeddings, dim=-1)

            text_vectors = embeddings.detach().cpu().numpy().tolist()

            return {
                "query_type": "text",
                "vector": text_vectors[0],
                "negative_vectors": text_vectors[1:]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Must provide either 'query_text' or 'query_image'.")
    
@app.get("/api/mask/{image_id}")
async def get_cluster_mask(image_id: str, clusters: str,
    w: Optional[int] = Query(None, description="Original image width from frontend"), 
    h: Optional[int] = Query(None, description="Original image height from frontend")
):
    key = f"image_fm:{image_id}" 
    payload = redis_client.hgetall(key)
    if not payload:
        return Response(status_code=404)

    height = int(payload[b"height"])
    width = int(payload[b"width"])
    dtype_name = payload[b"dtype"].decode("utf-8")
    decoded = zlib.decompress(payload[b"data"])
    cluster_id_map = np.frombuffer(decoded, dtype=np.dtype(dtype_name)).reshape(height, width)

    if w is not None and h is not None and w > 0 and h > 0:
        cluster_id_map = cv2.resize(
            cluster_id_map, 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        )
        height, width = h, w

    mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    kernel_size = max(3, int(min(height, width) * 0.005))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    cluster_list = clusters.split(",")
    for item in cluster_list:
        parts = item.split(":")
        if len(parts) != 2: continue
        
        cid = int(parts[0])
        score = float(parts[1]) 
        
        alpha = int(max(0, min(1, score)) * 150) 
        
        target_pixels = (cluster_id_map == cid)

        target_uint8 = target_pixels.astype(np.uint8)
        
        dilated = cv2.dilate(target_uint8, kernel, iterations=1)
        
        border_pixels = (dilated - target_uint8) > 0

        mask_rgba[target_pixels] = [255, 50, 50, alpha] 

        mask_rgba[border_pixels] = [200, 0, 0, 255]

    img = Image.fromarray(mask_rgba, 'RGBA')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

def get_tile_data(tile_id: str):
    if tile_id in _tile_cache:
        return _tile_cache[tile_id]

    tile_3d = RENDERS_ROOT / tile_id / "3D"
    
    if not tile_3d.exists():
        raise HTTPException(status_code=404, detail=f"Tile directory not found: {tile_3d}")

    meta_path = ASSETS_ROOT / f"{tile_id}_tile_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        offset = np.array(meta["center_offset"], dtype=np.float32)
    else:
        offset_file = tile_3d / "center_offset.npy"
        if not offset_file.exists():
            raise HTTPException(status_code=404, detail="center_offset not found.")
        offset = np.load(str(offset_file)).astype(np.float32)

    las_path = tile_3d / f"{tile_id}.las"
    if not las_path.exists():
        raise HTTPException(status_code=404, detail=f"LAS file not found: {las_path}")
        
    print(f"Loading LAS for {tile_id}...")
    las = laspy.read(str(las_path))
    xyz_zup = np.vstack([las.x, las.y, las.z]).T.astype(np.float32) - offset
    
    xyz_viewer = xyz_zup.copy()
    xyz_viewer[:, [1, 2]] = xyz_viewer[:, [2, 1]]

    chosen_feat_path = None
    for suffix in ["_DINOv3_filled_features.pt", "_DINOv3_fused_features.pt"]:
        candidate = tile_3d / f"{tile_id}{suffix}"
        if candidate.exists():
            chosen_feat_path = candidate
            break
            
    if not chosen_feat_path:
        raise HTTPException(status_code=404, detail=f"No DINOv3 feature files found for {tile_id}.")

    print(f"Loading features from {chosen_feat_path.name}...")
    saved = torch.load(str(chosen_feat_path), map_location="cpu", weights_only=False)
    
    features_tensor = saved['feat_bank'].float()
    point_ids = saved['point_ids']
    if not isinstance(point_ids, torch.Tensor):
        point_ids = torch.tensor(point_ids, dtype=torch.long)
        
    idx = point_ids.long().numpy()
    xyz_mapped = xyz_viewer[idx]
    
    features_mapped = F.normalize(features_tensor, dim=1).numpy()

    _tile_cache[tile_id] = {
        'xyz': xyz_mapped,
        'features': features_mapped
    }
    
    print(f"Tile {tile_id} cached successfully: {len(idx)} valid points.")
    return _tile_cache[tile_id]

@app.post("/api/encode-obb")
async def encode_obb(request: OBBEncodeRequest):
    tile_id = request.tile_id
    bbox_min = np.array(request.bbox_min)
    bbox_max = np.array(request.bbox_max)
    rot = request.rotation

    negative_text = []
    neg_prompts = [p.strip() for p in negative_text.split(",") if p.strip()] if negative_text else ["background"]
    
    try:
        tile_data = get_tile_data(tile_id)
        coords = tile_data['xyz']
        features = tile_data['features']

        cx, cy, cz = (bbox_min + bbox_max) / 2.0
        hx, hy, hz = (bbox_max - bbox_min) / 2.0
        
        cos_rot, sin_rot = np.cos(-rot), np.sin(-rot)
        
        dx, dy, dz = coords[:, 0] - cx, coords[:, 1] - cy, coords[:, 2] - cz
        
        rx = cos_rot * dx - sin_rot * dz
        rz = sin_rot * dx + cos_rot * dz
        
        mask = (np.abs(rx) <= hx) & (np.abs(dy) <= hy) & (np.abs(rz) <= hz)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            raise HTTPException(status_code=400, detail="No points found within the specified OBB.")
            
        selected_features = features[valid_indices]
        mean_vector = np.mean(selected_features, axis=0)
        mean_vector_tensor = F.normalize(torch.tensor(mean_vector), dim=-1)

        with torch.no_grad():
            neg_embeddings = backend.encode_text(neg_prompts)
            if neg_embeddings.dim() == 1:
                neg_embeddings = neg_embeddings.unsqueeze(0)
            neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)
            
        neg_vectors = neg_embeddings.detach().cpu().numpy().tolist()
        
        return {
            "query_type": "3d_obb",
            "vector": mean_vector_tensor.tolist(),
            "negative_vectors": neg_vectors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cluster-points")
async def get_cluster_points(tile: str, clusters: str):
    entity_ids = [c.strip() for c in clusters.split(",") if c.strip()]
    if not entity_ids:
        raise HTTPException(status_code=400, detail="No clusters provided.")
        
    try:
        tile_data = get_tile_data(tile)
        coords = tile_data['xyz']
        
        result_dict = {}
        for entity_id in entity_ids:
            redis_key = f"edifici_fm:{tile}:{entity_id}"
            hash_values = redis_client.hmget(redis_key, ["dtype", "data"])
            
            if not hash_values or hash_values[0] is None:
                continue
                
            dtype_bytes, compressed_data = hash_values
            
            try:
                dtype_str = dtype_bytes.decode('utf-8')
                
                decompressed_bytes = zlib.decompress(compressed_data)
                
                indices = np.frombuffer(decompressed_bytes, dtype=dtype_str)
                
                entity_xyz = coords[indices].tolist()
                result_dict[entity_id] = entity_xyz
                
            except Exception as inner_e:
                print(f"Failed to parse Entity {entity_id}: {str(inner_e)}")
                continue #
            
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting cluster points: {str(e)}")