# python index_features_to_es.py --jsonl features_radseg_k10.jsonl --es_host http://localhost:9200 --index radseg_k10_images --metadata_csv images_metadata.csv --laravel_json historical_metadata.json --recreate
import argparse
import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


def infer_vector_dim(jsonl_path: Path) -> int:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_embedding = record.get("image_embedding")
            if image_embedding:
                return len(image_embedding)
            metadata_embedding = record.get("metadata_embedding")
            if metadata_embedding:
                return len(metadata_embedding)
            for cluster in record.get("clusters", []):
                vector = cluster["v"] if isinstance(cluster, dict) else cluster
                return len(vector)
    raise ValueError(f"No vectors found in {jsonl_path}")

def load_image_uuid_mapping(metadata_csv: Path | None) -> dict:
    if metadata_csv is None or not metadata_csv.exists():
        return {}

    metadata_by_image = {}
    with metadata_csv.open("r", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle)
        
        try:
            headers = next(reader)
        except StopIteration:
            return {}

        if not headers:
            return {}

        for row in reader:
            if not row:
                continue
            
            image_id_raw = row[0].strip()
            if not image_id_raw:
                continue
            
            unified_key = Path(image_id_raw).stem

            last_col_val = ""
            for item in reversed(row):
                item_stripped = item.strip()
                if item_stripped:  
                    last_col_val = item_stripped
                    break
            
            if not last_col_val or last_col_val == image_id_raw:
                last_col_val = "unknown_uuid"
            
            metadata_by_image[unified_key] = last_col_val
            
    return metadata_by_image


def load_edifici_mapping(mapping_json: Path | None) -> dict:
    if mapping_json and mapping_json.exists():
        with mapping_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_laravel_metadata(laravel_json: Path | None) -> dict:
    if laravel_json and laravel_json.exists():
        with laravel_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def create_index(es: Elasticsearch, index_name: str, dims: int, recreate: bool) -> None:
    if recreate and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    if es.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "embedding_type": {"type": "keyword"},
                "image_id": {"type": "keyword"},
                "edifici_id": {"type": "keyword"},
                "cluster_id": {"type": "integer"},
                "entity_id": {"type": "keyword"},
                "level": {"type": "integer"},
                "historical_record_uuid": {"type": "keyword"},
                "dataset_slug": {"type": "keyword"},
                "date_range": {"type": "date_range"},
                "poi_locations": {"type": "geo_point"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)

def generate_image_actions(jsonl_path: Path, index_name: str, image_uuid_mapping: dict, laravel_metadata: dict):
    if not jsonl_path.exists():
        return

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_id = record["image_id"]
            unified_key = Path(image_id).stem
            hr_uuid = image_uuid_mapping.get(unified_key)
            if hr_uuid == "unknown_uuid":
                hr_uuid = None

            base_meta = {"historical_record_uuid": hr_uuid}
            if hr_uuid and hr_uuid in laravel_metadata:
                record_meta = laravel_metadata[hr_uuid]
                base_meta["dataset_slug"] = record_meta.get("dataset_slug")
                base_meta["date_range"] = record_meta.get("date_range")
                base_meta["poi_locations"] = record_meta.get("poi_locations")
            
            base_meta = {k: v for k, v in base_meta.items() if v is not None}

            image_embedding = record.get("image_embedding")
            if image_embedding:
                yield {
                    "_index": index_name,
                    "_id": f"img_global__{image_id}",
                    "_source": {
                        "embedding_type": "image_global",
                        "image_id": image_id,
                        "cluster_id": -1,
                        "vector": image_embedding,
                        **base_meta
                    },
                }

            metadata_embedding = record.get("metadata_embedding")
            if metadata_embedding:
                yield {
                    "_index": index_name,
                    "_id": f"img_meta__{image_id}",
                    "_source": {
                        "embedding_type": "image_metadata",
                        "image_id": image_id,
                        "cluster_id": -2,
                        "vector": metadata_embedding,
                        **base_meta
                    },
                }

            for fallback_cluster_id, cluster in enumerate(record.get("clusters", [])):
                if isinstance(cluster, dict):
                    cluster_id = int(cluster.get("cluster_id", fallback_cluster_id))
                    vector = cluster["v"]
                else:
                    cluster_id = fallback_cluster_id
                    vector = cluster

                yield {
                    "_index": index_name,
                    "_id": f"img_cluster__{image_id}_{cluster_id}",
                    "_source": {
                        "embedding_type": "image_cluster",
                        "image_id": image_id,
                        "cluster_id": cluster_id,
                        "vector": vector,
                        **base_meta
                    },
                }


def generate_3d_actions(parquet_path: Path, npy_dir: Path | None, index_name: str, edifici_mapping: dict, laravel_metadata: dict):
    processed_global_edifici = set()

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            edifici_id = row["edifici_id"]
            entity_id = row["entity_id"]
            level = int(row["level"])
            cluster_vec = row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"])
            mapping_val = edifici_mapping.get(edifici_id)
            hr_uuid = mapping_val if isinstance(mapping_val, str) else (mapping_val.get("uuid") if isinstance(mapping_val, dict) else None)

            base_meta = {"historical_record_uuid": hr_uuid}
            if hr_uuid and hr_uuid in laravel_metadata:
                record_meta = laravel_metadata[hr_uuid]
                base_meta["dataset_slug"] = record_meta.get("dataset_slug")
                base_meta["date_range"] = record_meta.get("date_range")
                base_meta["poi_locations"] = record_meta.get("poi_locations")
            
            base_meta = {k: v for k, v in base_meta.items() if v is not None}

            if npy_dir and edifici_id not in processed_global_edifici:
                npy_path = npy_dir / f"{edifici_id}_DINOv3_global_embedding.npy"
                if not npy_path.exists():
                    npy_path = npy_dir / edifici_id / "3D" / f"{edifici_id}_DINOv3_global_embedding.npy"

                if npy_path.exists():
                    try:
                        global_vec = np.load(npy_path).tolist()
                        yield {
                            "_index": index_name,
                            "_id": f"3d_global__{edifici_id}",
                            "_source": {
                                "embedding_type": "3d_global",
                                "edifici_id": edifici_id,
                                "vector": global_vec,
                                **base_meta
                            },
                        }
                    except Exception as e:
                        print(f"Error loading npy for {edifici_id}: {e}")
                processed_global_edifici.add(edifici_id)

            yield {
                "_index": index_name,
                "_id": f"3d_cluster__{edifici_id}_{entity_id}",
                "_source": {
                    "embedding_type": "3d_cluster",
                    "edifici_id": edifici_id,
                    "entity_id": entity_id,
                    "level": level,
                    "vector": cluster_vec,
                    **base_meta
                },
            }


def count_total_expected_actions(image_jsonl: Path | None, parquet_path: Path | None) -> int:
    total = 0
    if image_jsonl and image_jsonl.exists():
        with image_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                record = json.loads(line)
                total += len(record.get("clusters", []))
                if record.get("image_embedding"): total += 1
                if record.get("metadata_embedding"): total += 1
                
    if parquet_path and parquet_path.exists():
        df_meta = pd.read_parquet(parquet_path, columns=["edifici_id"])
        total += len(df_meta)
        total += df_meta["edifici_id"].nunique()
        
    return total

def main():
    parser = argparse.ArgumentParser(description="TimeAtlas Unified Multimodal Vector Indexer")
    parser.add_argument("--image_jsonl", type=str, default=None, help="Path to image features JSONL")
    parser.add_argument("--3d_parquet", dest="parquet_3d", type=str, default=None, help="Path to aggregated 3D parquet file")
    parser.add_argument("--3d_npy_dir", dest="npy_dir_3d", type=str, default=None, help="Directory containing edifici global .npy files")
    
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to messy metadata CSV (image_id -> UUID)")
    parser.add_argument("--edifici_mapping", type=str, default=None, help="Path to 3D mapping JSON (edifici_id -> UUID)")
    parser.add_argument("--laravel_json", type=str, default=None, help="Path to historical_record.json (UUID -> spatiotemporal data)")
    
    parser.add_argument("--es_host", default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument("--index", default="multimodal_embedding", help="Target ES Index Name")
    parser.add_argument("--dims", type=int, default=1024, help="Fallback vector dimension if jsonl is missing")
    parser.add_argument("--batch_size", type=int, default=1000, help="Bulk API batch chunk size")
    parser.add_argument("--recreate", action="store_true", help="Recreate index before importing")
    args = parser.parse_args()

    img_path = Path(args.image_jsonl) if args.image_jsonl else None
    actual_dims = args.dims
    if img_path and img_path.exists():
        try:
            actual_dims = infer_vector_dim(img_path)
            print(f"Dynamically inferred vector dimension from JSONL: {actual_dims}")
        except Exception as e:
            print(f"Warning: Could not infer dimension ({e}). Using default: {actual_dims}")

    es = Elasticsearch(args.es_host, verify_certs=False, request_timeout=120)
    if not es.ping():
        raise SystemExit(f"Connection failed to ES at {args.es_host}")
    create_index(es, args.index, actual_dims, recreate=args.recreate)

    print("Loading mapping dictionaries...")
    image_uuid_map = load_image_uuid_mapping(Path(args.metadata_csv) if args.metadata_csv else None)
    edifici_uuid_map = load_edifici_mapping(Path(args.edifici_mapping) if args.edifici_mapping else None)
    laravel_metadata = load_laravel_metadata(Path(args.laravel_json) if args.laravel_json else None)
    
    print(f" - Loaded {len(image_uuid_map)} Image->UUID mappings (using tolerant CSV parser).")
    print(f" - Loaded {len(edifici_uuid_map)} 3D->UUID mappings.")
    print(f" - Loaded {len(laravel_metadata)} UUID->Spatiotemporal metadata entries.")

    pq_path = Path(args.parquet_3d) if args.parquet_3d else None
    total_actions = count_total_expected_actions(img_path, pq_path)
    if total_actions == 0:
        print("No data found to index. Exiting.")
        return

    def master_action_generator():
        if img_path and img_path.exists():
            yield from generate_image_actions(img_path, args.index, image_uuid_map, laravel_metadata)
        if pq_path and pq_path.exists():
            npy_dir_path = Path(args.npy_dir_3d) if args.npy_dir_3d else None
            yield from generate_3d_actions(pq_path, npy_dir_path, args.index, edifici_uuid_map, laravel_metadata)

    success_count, failed_count = 0, 0
    with tqdm(total=total_actions, desc="Pushing Vectors to ES") as pbar:
        for ok, result in helpers.streaming_bulk(
            client=es,
            actions=master_action_generator(),
            chunk_size=args.batch_size,
            max_retries=3,
            raise_on_error=False,
        ):
            if ok:
                success_count += 1
            else:
                failed_count += 1
            pbar.update(1)

    print(f"\nIndexing finished. Success: {success_count} | Failed: {failed_count}")


if __name__ == "__main__":
    main()
