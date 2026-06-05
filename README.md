# Semantic Search Microservice

> **Part of the [Time Atlas](https://timeatlas.eu/) platform** — an interactive historical-geographical platform integrating spatial and temporal dimensions to make historical records searchable across space and time.

This repository contains the Python microservice that powers the multimodal semantic search capability of the Time Atlas platform. It exposes a set of HTTP endpoints for encoding queries (text, images, image regions, and 3D point cloud selections) into dense vector representations, and for rendering score-proportional visual overlays over search results.

> **Note on repository scope:** The frontend (Nuxt 3) and backend API gateway (Laravel) code that integrates this microservice into the Time Atlas platform reside in the project's private monorepo. For access, please contact the Time Atlas team at EPFL.

---

## Table of Contents

- [Project Context](#project-context)
- [Role in the System](#role-in-the-system)
- [Architecture](#architecture)
- [Endpoints](#endpoints)
- [Models](#models)
- [Data Storage](#data-storage)
- [Repository Structure](#repository-structure)
- [External Data](#external-data)
- [Setup](#setup)
- [Related Components](#related-components)

---

## Project Context

Time Atlas aggregates over 182,000 historical records — including photographs, cadastral documents, paintings, and 3D architectural reconstructions — anchored to geographic coordinates and historical time periods. The platform's original retrieval system was limited to keyword-based search, which cannot accommodate the visual and geometric nature of image and 3D data, nor support cross-modal queries such as searching by uploaded image or by selecting a region of a point cloud.

This microservice was developed as part of a full-stack integration effort to add **multimodal semantic search** to the platform, supporting five query modalities:

| Modality | Description |
|----------|-------------|
| Text query | Natural language search over all historical records |
| Image upload | Find visually similar records by uploading an image |
| Image region crop | Search using a user-selected sub-region of a displayed image |
| POI anchor record | Use an existing record as a search anchor via its global embedding |
| 3D OBB selection | Search using a spatial bounding box drawn over a 3D point cloud model |

---

## Role in the System

The microservice sits between the Laravel API gateway and the Elasticsearch vector index, serving as the **stateless encoder engine** in the search pipeline:

```
Nuxt 3 frontend
      │
      ▼
Nitro BFF proxy
      │
      ▼
Laravel API gateway  ──────────────────────►  Elasticsearch
      │                  script_score query     (vector index)
      │                  + geo ranking
      ▼
FastAPI microservice   ◄──────────────────── Redis
  (this repository)        feature map cache
```

For each incoming search request, the Laravel gateway forwards the query payload to this service, which:

1. Encodes the input into a normalised dense vector using the appropriate model
2. Returns the query vector (and optional negative contrast vectors) to Laravel
3. Laravel then executes the Elasticsearch `script_score` query using the returned vector

For visualisation requests, the service additionally reads pre-computed feature maps from Redis and renders RGBA PNG overlays that are streamed back to the frontend.

---

## Architecture

The service is implemented with **FastAPI** and designed to be entirely stateless with respect to request handling. Model weights are loaded into GPU memory once at startup and held warm for the lifetime of the process, ensuring that per-request latency reflects only forward pass cost rather than model loading overhead.

3D tile data (LAS point clouds and DINOv3 feature tensors) is loaded lazily on first access and cached in a process-level dictionary, avoiding repeated disk reads for frequently queried tiles.

```
main.py
├── /api/encode              # Text and image encoding (Talk2DINOv3)
├── /api/encode-obb          # 3D oriented bounding box encoding
├── /api/cluster-points      # 3D entity point coordinate lookup
├── /api/heatmap/{image_id}  # Score-proportional RGBA heatmap rendering
└── /api/mask/{image_id}     # Binary cluster mask rendering (fallback)
```

---

## Endpoints

### `POST /api/encode`

Encodes a text string or image into a dense query vector.

**Request:** multipart/form-data
| Field | Type | Description |
|-------|------|-------------|
| `query_image` | file (optional) | Image to encode |
| `query_text` | string (optional) | Text query to encode |
| `negative_text` | string | Negative contrast prompt (default: `"background"`) |

At least one of `query_image` or `query_text` must be present.

**Response:**
```json
{
  "query_type": "image" | "text",
  "vector": [float, ...],
  "negative_vectors": [[float, ...], ...]
}
```

---

### `POST /api/encode-obb`

Encodes a 3D oriented bounding box selection by mean-pooling the DINOv3 features of all points within the specified region.

**Request:** JSON
```json
{
  "tile_id": "edifici_XXXX",
  "bbox_min": [x, y, z],
  "bbox_max": [x, y, z],
  "rotation": [[...], [...], [...]]
}
```

**Response:** same structure as `/api/encode` with `query_type: "3d_obb"`

---

### `GET /api/cluster-points`

Returns the 3D coordinates of points belonging to specified entity clusters within a tile, used by the frontend 3D viewer for point highlight rendering.

**Query params:** `tile` (edifici ID), `clusters` (comma-separated entity IDs)
 
**Response:**
```json
{
  "L0_3": [[x, y, z], ...],
  "L1_7": [[x, y, z], ...]
}
```
 
Each key is an entity ID; the value is the list of XYZ coordinates of all points belonging to that entity, read from the Redis index (`edifici_fm:{tile}:{entity_id}`) and looked up against the tile's full point array.

---

### `GET /api/heatmap/{image_id}`

Renders a score-proportional RGBA heatmap overlay for an image result, using the TURBO colormap with Gaussian smoothing.

**Query params:** `clusters`, `w`, `h`, `g_max`, `g_min`

**Response:** `image/png`

---

### `GET /api/mask/{image_id}`

Renders a binary cluster mask overlay highlighting the regions corresponding to specified cluster IDs.

**Query params:** `clusters`, `w`, `h`

**Response:** `image/png`

---

## Models

| Model | Role |
|-------|------|
| **Talk2DINOv3** | Primary encoder for all modalities. Maps text and image inputs into a shared 1024-dimensional embedding space via DINOv3 patch features aligned to language. Used for text encoding, image encoding, and 3D point-level feature extraction. Semantic clustering of image regions is also performed using Talk2DINOv3 patch features. |

Model weights are loaded at service startup. The service requires a CUDA-capable GPU for production inference.

If you use Talk2DINOv3 in your work, please cite the original authors:
 
```bibtex
@inproceedings{barsellotti2025talking,
  title={Talking to dino: Bridging self-supervised vision backbones
         with language for open-vocabulary segmentation},
  author={Barsellotti, Luca and Bianchi, Lorenzo and Messina, Nicola and Carrara, Fabio and
          Cornia, Marcella and Baraldi, Lorenzo and Falchi, Fabrizio and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22025--22035},
  year={2025}
}
```

### Earlier experimental models (not used in current pipeline)
 
| Model | Notes |
|-------|-------|
| **RADSeg** | Region-aware semantic segmentation model evaluated during an earlier phase of the project. Source files retained in the repository for reference; see `radseg/` directory and [Third-party code](#third-party-code) section. |
| **Perception Encoder (PE) + MiniLM** | Initial patch-based architecture used in the earliest prototype, operating on separate visual and text embedding spaces with Z-Score normalization for score fusion. Superseded by the unified Talk2DINOv3 approach. |

---

## Data Storage

The service relies on two external data stores, both populated during the **offline indexing phase**:

### Redis

Pre-computed feature maps are stored in Redis for fast access at query time, avoiding repeated model inference during result rendering:

| Key pattern | Contents |
|-------------|----------|
| `image_fm:{image_id}` | Cluster-to-pixel mapping for image records (zlib-compressed numpy array) |
| `edifici_fm:{edifici_id}` | Entity-to-point-index mapping for 3D models |

### Disk (tile feature files)

3D tile data is stored as files on disk:

| File | Contents |
|------|----------|
| `{edifici_id}_DINOv3_fused_features.pt` | Per-point DINOv3 feature vectors for a building tile |
| `partition_level_{0,1,2}.ply` | Point cloud with semantic partition labels at three levels of granularity |
| `{edifici_id}_DINOv3_global_embedding.npy` | Pre-computed global embedding for the full building model |

The offline indexing pipeline reads these files, computes cluster vectors via mean pooling, and writes all embeddings to Elasticsearch. See the [External Data](#external-data) section for details on required input file formats.

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Redis instance
- ElasticSearch instance
- Access to pre-computed feature files (`.pt`, `.ply`, `.npy`)

### Installation

```bash
git clone https://github.com/wuu03/semantic-search-microservice.git
cd semantic-search-microservice
```

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and set the following:

```env
SERVER_IP=localhost
REDIS_PORT=6379
RENDERS_ROOT=/path/to/3d/tile/features
ASSETS_ROOT=/path/to/3d/tile/models
```

### Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The service will load model weights on startup. First startup may take 30–60 seconds depending on GPU and model size.

---

## Repository Structure

```
semantic-search-microservice/
│
├── main.py                      # FastAPI application — encoding and rendering endpoints
├── vl_backends.py               # Vision-language model backend wrappers (Talk2DINOv3)
├── hubconf.py                   # Model hub configuration
│
├── # Indexing and data ingestion
├── index_features_to_es.py      # Unified vector indexing pipeline → Elasticsearch
├── process_3d_features.py       # 3D entity feature aggregation → Parquet
├── load_featuremaps_to_redis.py # Image cluster-to-pixel maps → Redis
├── load_3d_indices_to_redis.py  # 3D entity-to-point indices → Redis
│
├── # Utilities
└── test_es_visual_search.py     # End-to-end search pipeline test script
│
└── # Environment and dependencies
    └── requirements.txt         # Pip dependencies
```

### Authorship note

The following files were created or updated by the author as part of the integration work:
`main.py` (API endpoints and rendering logic), `index_features_to_es.py`, `process_3d_features.py`, `load_featuremaps_to_redis.py`, `load_3d_indices_to_redis.py`.

Model backend files (`vl_backends.py`, `test_es_visual_search.py`, `batch_extract_features.py`, `demo_talk2dino_v2_single_image.py`) were provided by collaborating team members responsible for model configuration and evaluation (some modified), and are included here to make the service self-contained and runnable.

---
 
### Third-party code
 
The following files originate from the [RADSeg](https://github.com/RADSeg-OVSS/RADSeg) official code release (Alama et al., CVPR 2026, arXiv:2511.19704) and were provided to this project by collaborating team members, with minor modifications for integration purposes:
 
| File | Origin |
|------|--------|
| `radseg/base.py` | RADSeg official release |
| `radseg/prompt_templates.py` | RADSeg official release |
| `radseg/radseg.py` | RADSeg official release (modified) |
| `radseg/sam_utils.py` | RADSeg official release |
| `hubconf.py` | RADSeg / AM-RADIO official release |
 
If you use RADSeg in your work, please cite the original authors:
 
```bibtex
@inproceedings{alama2026radseg,
  title={RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot
         Open-Vocabulary Segmentation Using Agglomerative Models},
  author={Alama, Omar and Jariwala, Darshil and Bhattacharya, Avigyan and Kim,
          Seungchan and Wang, Wenshan and Scherer, Sebastian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9294--9304},
  year={2026}
}
```

---

## External Data

Several large data files are required to run the full pipeline but are not tracked in this repository. The table below describes each category, its format, and how it is produced or obtained.

### Image features (for indexing)

| File | Format | Description | Produced by |
|------|--------|-------------|-------------|
| `features_*.jsonl` | JSONL | One record per image, containing `image_id`, `image_embedding` (global), `metadata_embedding`, and `clusters` (list of `{cluster_id, v}` dicts) | RADSeg + Talk2DINOv3 encoding pipeline (team) |
| `images_metadata.csv` | CSV | Maps image filenames to historical record UUIDs | Exported from Time Atlas database |

The JSONL file is the primary input to both `index_features_to_es.py` (vector indexing) and `load_featuremaps_to_redis.py` (feature map ingestion). Each line represents one image record with the following fields:

```json
{
  "image_id": "example_image.jpg",
  "image_embedding": [float, ...],
  "metadata_embedding": [float, ...],
  "clusters": [
    { "cluster_id": 0, "v": [float, ...] },
    { "cluster_id": 1, "v": [float, ...] }
  ],
  "cluster_id_map": [[int, ...], ...],
  "feature_map_size": [height, width]
}
```

- `image_embedding` — global Talk2DINOv3 embedding for the full image (float[1024])
- `metadata_embedding` — embedding derived from the record's textual metadata (float[1024])
- `clusters` — per-cluster mean-pooled vectors produced by RADSeg semantic segmentation
- `cluster_id_map` — 2D pixel-level array mapping each pixel to its cluster ID; used by `load_featuremaps_to_redis.py` to build the heatmap overlay lookup table stored in Redis
- `feature_map_size` — `[height, width]` of the cluster ID map in feature space

### 3D model features and point cloud data
 
All 3D-related files are organised per building under a shared `RENDERS_ROOT` directory, with additional viewer assets under `ASSETS_ROOT`. Both paths are configured via environment variables.
 
**Per-building files** (under `RENDERS_ROOT/{edifici_id}/3D/`):
 
| File | Format | Used by | Description |
|------|--------|---------|-------------|
| `{edifici_id}.las` | LAS | `get_tile_data`, `/api/encode-obb`, `/api/cluster-points` | Raw point cloud with XYZ coordinates; offset-corrected and axis-remapped at load time for viewer alignment |
| `{edifici_id}_DINOv3_filled_features.pt` | PyTorch tensor | `get_tile_data` (preferred) | Per-point DINOv3 features with interpolated coverage for points lacking direct encoder output |
| `{edifici_id}_DINOv3_fused_features.pt` | PyTorch tensor | `get_tile_data` (fallback), `process_3d_features.py` | Per-point DINOv3 feature vectors; used when filled variant is absent |
| `partition_level_{0,1,2}.ply` | PLY | `process_3d_features.py` | Point cloud with semantic partition labels at three granularity levels |
| `{edifici_id}_DINOv3_global_embedding.npy` | NumPy array | `index_features_to_es.py` | Pre-computed global embedding for the full building model |
| `center_offset.npy` | NumPy array | `get_tile_data` (fallback) | XYZ centroid offset for coordinate normalisation; used only when tile meta JSON is absent |
 
The `filled` feature variant is loaded preferentially over `fused` when both are present. Loaded tile data — XYZ coordinates, mapped feature vectors, and the full point array — is held in a process-level cache keyed by `tile_id` to avoid repeated disk reads across requests.
 
**Viewer asset files** (under `ASSETS_ROOT/`):
 
| File | Format | Used by | Description |
|------|--------|---------|-------------|
| `{edifici_id}_tile_meta.json` | JSON | `get_tile_data` | Tile metadata including `center_offset` array; preferred over `center_offset.npy` when present |
| `{edifici_id}_*.ply` | PLY | Frontend 3D viewer | Downsampled point cloud model served to the browser for interactive rendering |
 
**Intermediate output** (produced by `process_3d_features.py`):
 
| File | Format | Description |
|------|--------|-------------|
| `clustered_3d_vectors.parquet` | Parquet (snappy) | Aggregated entity-level feature vectors across all buildings; input to `index_features_to_es.py` |
 
Each row in the Parquet file represents one entity at one granularity level:
 
| Column | Type | Description |
|--------|------|-------------|
| `edifici_id` | string | Building identifier |
| `entity_id` | string | Entity label, e.g. `L0_3`, `L1_7` |
| `level` | int | Granularity level (0 = finest, 2 = coarsest) |
| `vector` | float[1024] | Mean-pooled DINOv3 feature vector for the entity |
| `point_indices` | list[int] | Indices of 3D points belonging to this entity |
| `children_l0` | list[str] | L0 child entity IDs (present for level ≥ 1) |
| `children_l1` | list[str] | L1 child entity IDs (present for level 2 only) |
 
---

## Related Components

This microservice is one component of a larger integration. The full system comprises:

| Component | Technology | Location |
|-----------|-----------|----------|
| **This microservice** | FastAPI / Python | This repository |
| **API gateway** | Laravel (PHP) | Private — contact Time Atlas team |
| **Frontend** | Nuxt 3 / Vue | Private — contact Time Atlas team |
| **Vector index** | Elasticsearch | Deployed in Time Atlas backend Docker |
| **Offline indexing pipeline** | Python | This repository |
| **3D point cloud viewer** | Three.js | Private — contact Time Atlas team |

For access to the frontend and backend integration code, or for questions about the broader Time Atlas platform, please contact the Time Atlas research team at EPFL.

---

## Acknowledgements

This microservice was developed as part of a master-level integration project at EPFL. The semantic encoder models (Talk2DINOv3, RADSeg) were configured and evaluated by collaborating team members. The Time Atlas platform is a research initiative of the [Digital Humanities Lab](https://www.epfl.ch/labs/dhlab/) at EPFL.
