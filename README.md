# Unified Image Search Stack

This branch keeps a single, compact workflow for postcard image search with three interchangeable vision-language backends:

- `radseg`
- `tips`
- `talk2dino`

The core pipeline is:

1. Extract clustered dense features from images
2. Store cluster vectors in Elasticsearch
3. Store `cluster_id_map` feature maps in Redis
4. Run text search with localization overlays
5. Evaluate search quality with metadata-derived weak labels and per-query visual reports

## Kept entrypoints

- [D:\RADSeg\vl_backends.py](D:\RADSeg\vl_backends.py): unified backend wrappers
- [D:\RADSeg\batch_extract_features.py](D:\RADSeg\batch_extract_features.py): batch clustered feature extraction
- [D:\RADSeg\index_features_to_es.py](D:\RADSeg\index_features_to_es.py): index cluster vectors into Elasticsearch
- [D:\RADSeg\load_featuremaps_to_redis.py](D:\RADSeg\load_featuremaps_to_redis.py): load feature maps into Redis
- [D:\RADSeg\test_es_visual_search.py](D:\RADSeg\test_es_visual_search.py): search demo with heatmap/cluster visualization
- [D:\RADSeg\evaluate_search_with_metadata.py](D:\RADSeg\evaluate_search_with_metadata.py): quantitative evaluation + query reports
- [D:\RADSeg\docker-compose.yml](D:\RADSeg\docker-compose.yml): single local stack for Elasticsearch, Redis, Kibana

## 1. Start local services

```bash
docker compose up -d
```

Services:

- Elasticsearch: `http://localhost:9200`
- Redis: `redis://localhost:6379/0`
- Kibana: `http://localhost:5601`

## 2. Extract features

Example with `radseg`:

```bash
python batch_extract_features.py \
  --backend radseg \
  --input_dir images \
  --output_file features_radseg.jsonl \
  --num_clusters 10 \
  --min_cluster_pixels 16 \
  --merge_similarity 0.97 \
  --device cuda
```

Example with `tips`:

```bash
python batch_extract_features.py \
  --backend tips \
  --model_id google/tipsv2-b14 \
  --input_dir images \
  --output_file features_tips.jsonl \
  --num_clusters 10 \
  --min_cluster_pixels 16 \
  --merge_similarity 0.97 \
  --device cuda
```

Example with `talk2dino`:

```bash
python batch_extract_features.py \
  --backend talk2dino \
  --input_dir images \
  --output_file features_talk2dino.jsonl \
  --num_clusters 10 \
  --min_cluster_pixels 16 \
  --merge_similarity 0.97 \
  --device cuda
```

Each JSONL row contains:

- `image_id`
- `feature_map_size`
- `cluster_id_map`
- `clusters: [{cluster_id, v}]`

## 3. Load feature maps into Redis

```bash
python load_featuremaps_to_redis.py \
  --jsonl features_radseg.jsonl \
  --redis_url redis://localhost:6379/0 \
  --key_prefix radseg_fm
```

## 4. Index vectors into Elasticsearch

```bash
python index_features_to_es.py \
  --jsonl features_radseg.jsonl \
  --es_host http://localhost:9200 \
  --index radseg_images \
  --recreate
```

The vector dimension is inferred automatically from the JSONL file.

## 5. Run search demo

```bash
python test_es_visual_search.py "arched bridge over water" \
  --backend radseg \
  --device cuda \
  --es_host http://localhost:9200 \
  --es_index radseg_images \
  --redis_url redis://localhost:6379/0 \
  --redis_key_prefix radseg_fm \
  --image_root images \
  --negative_text "background, sky, clouds, text, border, trees, road, people" \
  --top_k 6 \
  --candidate_k 120 \
  --temperature 10
```

Notes:

- If `--output_path` is omitted, a file is auto-generated under `scratch/`
- `--result_mode image` returns top images
- `--result_mode cluster` returns top clusters, allowing multiple clusters from one image

## 6. Evaluate search quality

The evaluator uses [D:\RADSeg\images_metadata.csv](D:\RADSeg\images_metadata.csv) as weak supervision.

It produces:

- image-level metrics
- cluster-level metrics
- query-level ground truth files
- per-query visualization images
- a query browser report in Markdown and HTML

Example:

```bash
python evaluate_search_with_metadata.py \
  --es_host http://localhost:9200 \
  --es_index radseg_images \
  --backend radseg \
  --device cuda \
  --redis_url redis://localhost:6379/0 \
  --redis_key_prefix radseg_fm \
  --image_root images \
  --candidate_k 120 \
  --temperature 10 \
  --negative_text "background, sky, clouds, text, border, trees, road, people" \
  --visualize_queries all \
  --visualize_result_mode image \
  --visualize_top_k 6
```

Outputs go to:

```text
scratch/eval_<backend>_<index>/
```

Important files:

- `summary.md`
- `summary.json`
- `image_level_metrics.csv`
- `cluster_level_metrics.csv`
- `image_level_top_hits.csv`
- `cluster_level_top_hits.csv`
- `query_ground_truth.json`
- `query_ground_truth_pairs.csv`
- `query_browser_report.html`
- `visualizations/...`

## Recommended search behavior

This codebase supports two query families:

1. General concept queries
   - `river`
   - `bridge`
   - `church`
   - `castle`

2. Specific landmark queries
   - `Piazza San Marco`
   - `Charles Bridge`
   - `Schloss Pyrmont`

For product use, specific landmark queries should eventually combine metadata filtering with visual reranking.
