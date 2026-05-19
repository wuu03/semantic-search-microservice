# uvicorn api.main:app --host 0.0.0.0 --port 8001
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, APIRouter, Response, File, UploadFile, Form, Query
from pydantic import BaseModel
from typing import Optional, List
import io
import zlib
import numpy as np
from PIL import Image
from redis import Redis
import cv2

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

class UnifiedEncodeResponse(BaseModel):
    query_type: str  
    vector: List[float]
    negative_vectors: Optional[List[List[float]]] = None

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
                embedding = backend.encode_image_global(image_tensor)
                
            image_vector = embedding.detach().cpu().numpy().tolist()[0]
            
            return {
                "query_type": "image",
                "vector": image_vector
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
    key = f"radseg_k10_fm:{image_id}" 
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

    kernel = np.ones((3, 3), np.uint8)
    
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