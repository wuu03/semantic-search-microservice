# uvicorn api.main:app --host 0.0.0.0 --port 8001
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, APIRouter, Response
from pydantic import BaseModel
from typing import Optional, List
import io
import zlib
import numpy as np
from PIL import Image
from redis import Redis

from vl_backends import create_backend

app = FastAPI(title="TimeAtlas Vector Encoding API", version="1.0")

backend = None

redis_client = Redis.from_url("redis://localhost:6379/0")

class EncodeRequest(BaseModel):
    query: str
    negative_text: Optional[str] = "background"

class EncodeResponse(BaseModel):
    positive_vector: List[float]
    negative_vectors: List[List[float]]

@app.on_event("startup")
async def load_model():
    global backend
    print("Loading Model...")
    try:
        backend = create_backend(
            backend_name="radseg",
            device="cuda", 
            model_id=None,
            model_version="c-radio_v4-h",
            lang_model="siglip2-g"
        )
        print("Service Ready.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/api/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Loading model... Please try again later.")

    try:
        neg_prompts = [p.strip() for p in request.negative_text.split(",") if p.strip()] if request.negative_text else ["background"]
        prompts = [request.query] + neg_prompts

        with torch.no_grad():
            embeddings = backend.encode_text(prompts)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            embeddings = F.normalize(embeddings, dim=-1)

        text_vectors = embeddings.detach().cpu().numpy().tolist()

        return {
            "positive_vector": text_vectors[0],
            "negative_vectors": text_vectors[1:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/mask/{image_id}")
async def get_cluster_mask(image_id: str, clusters: str):
    key = f"radseg_k10_fm:{image_id}" 
    payload = redis_client.hgetall(key)
    if not payload:
        return Response(status_code=404)

    height = int(payload[b"height"])
    width = int(payload[b"width"])
    dtype_name = payload[b"dtype"].decode("utf-8")
    decoded = zlib.decompress(payload[b"data"])
    cluster_id_map = np.frombuffer(decoded, dtype=np.dtype(dtype_name)).reshape(height, width)

    mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    cluster_list = clusters.split(",")
    for item in cluster_list:
        parts = item.split(":")
        if len(parts) != 2: continue
        
        cid = int(parts[0])
        score = float(parts[1]) 
        
        alpha = int(max(0, min(1, score)) * 150) 
        
        target_pixels = (cluster_id_map == cid)
        mask_rgba[target_pixels] = [255, 50, 50, alpha] 

    img = Image.fromarray(mask_rgba, 'RGBA')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")