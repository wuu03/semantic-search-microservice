# python index_features_to_es.py --jsonl features_radseg_k10.jsonl --es_host http://localhost:9200 --index radseg_k10_images --metadata_csv images_metadata.csv --laravel_json historical_metadata.json --recreate
import argparse
import csv
import json
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


METADATA_FIELDS = [
    "final_country",
    "final_city",
    "final_place",
    "date",
    "description",
    "transcription",
    "landmarks_identified",
    "historical_record_uuid",
]


def infer_vector_dim(jsonl_path: Path) -> int:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for cluster in record.get("clusters", []):
                vector = cluster["v"] if isinstance(cluster, dict) else cluster
                return len(vector)
    raise ValueError(f"No vectors found in {jsonl_path}")


def count_vectors(jsonl_path: Path) -> int:
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += len(json.loads(line).get("clusters", []))
    return total


def load_metadata(metadata_csv: Path | None) -> dict:
    if metadata_csv is None:
        return {}
    if not metadata_csv.exists():
        raise ValueError(f"Metadata CSV not found: {metadata_csv}")

    metadata_by_image = {}
    with metadata_csv.open("r", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle)
        
        try:
            headers = next(reader)
        except StopIteration:
            return {}

        if not headers:
            return {}

        last_col_name = "unknown_uuid" 
        for col in reversed(headers):
            col_stripped = col.strip()
            if col_stripped:
                last_col_name = col_stripped
                break

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
            
            metadata_by_image[unified_key] = {
                last_col_name: last_col_val
            }
            
    return metadata_by_image

def load_laravel_metadata(json_path: Path | None) -> dict:
    if json_path is None or not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def create_index(es: Elasticsearch, index_name: str, dims: int, recreate: bool) -> None:
    if recreate and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    if es.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "cluster_id": {"type": "integer"},
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


def generate_actions(jsonl_path: Path, index_name: str, metadata_by_image: dict | None = None, laravel_metadata: dict | None = None):
    metadata_by_image = metadata_by_image or {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id_raw = record["image_id"]
            unified_key = Path(image_id_raw).stem

            for fallback_cluster_id, cluster in enumerate(record.get("clusters", [])):
                if isinstance(cluster, dict):
                    cluster_id = int(cluster.get("cluster_id", fallback_cluster_id))
                    vector = cluster["v"]
                else:
                    cluster_id = fallback_cluster_id
                    vector = cluster

                source = {
                    "image_id": image_id_raw, 
                    "cluster_id": cluster_id,
                    "vector": vector,
                }
                
                source.update(metadata_by_image.get(unified_key, {}))

                hr_uuid = source.get("historical_record_uuid")
                if hr_uuid and hr_uuid in laravel_metadata:
                    source.update(laravel_metadata[hr_uuid])

                yield {
                    "_index": index_name,
                    "_id": f"{image_id_raw}_{cluster_id}",
                    "_source": source,
                }


def main():
    parser = argparse.ArgumentParser(description="Index cluster vectors from feature JSONL into Elasticsearch.")
    parser.add_argument("--jsonl", required=True, help="Path to feature JSONL")
    parser.add_argument("--es_host", default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--index", default="tips_images", help="Index name")
    parser.add_argument("--metadata_csv", default=None, help="Optional image metadata CSV to copy into every cluster document")
    parser.add_argument("--laravel_json", default=None, help="Path to Laravel exported metadata JSON")
    parser.add_argument("--batch_size", type=int, default=500, help="Bulk indexing batch size")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate index before indexing")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"Feature file not found: {jsonl_path}")

    es = Elasticsearch(args.es_host, verify_certs=False, request_timeout=60)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")

    dims = infer_vector_dim(jsonl_path)
    total_vectors = count_vectors(jsonl_path)
    metadata_by_image = load_metadata(Path(args.metadata_csv) if args.metadata_csv else None)
    laravel_metadata = load_laravel_metadata(Path(args.laravel_json) if args.laravel_json else None)
    print(f"Vector dimension: {dims}")
    print(f"Total vectors: {total_vectors}")
    if metadata_by_image:
        print(f"Metadata rows loaded: {len(metadata_by_image)}")

    create_index(es, args.index, dims, recreate=args.recreate)

    success_count = 0
    failed_count = 0
    for ok, result in tqdm(
        helpers.streaming_bulk(
            client=es,
            actions=generate_actions(jsonl_path, args.index, metadata_by_image=metadata_by_image, laravel_metadata=laravel_metadata),
            chunk_size=args.batch_size,
            max_retries=3,
            raise_on_error=False,
        ),
        total=total_vectors,
        desc="Indexing vectors",
    ):
        if ok:
            success_count += 1
        else:
            failed_count += 1

    print(f"Indexed: {success_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
