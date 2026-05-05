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
        for row in csv.DictReader(handle):
            image_id = (row.get("image_filename") or "").strip()
            if not image_id:
                continue
            metadata = {field: (row.get(field) or "").strip() for field in METADATA_FIELDS}
            metadata["metadata_text"] = " ".join(value for value in metadata.values() if value)
            metadata_by_image[image_id] = metadata
    return metadata_by_image


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
                "final_country": {"type": "keyword"},
                "final_city": {"type": "keyword"},
                "final_place": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "date": {"type": "keyword"},
                "description": {"type": "text"},
                "transcription": {"type": "text"},
                "landmarks_identified": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "historical_record_uuid": {"type": "keyword"},
                "metadata_text": {"type": "text"},
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


def generate_actions(jsonl_path: Path, index_name: str, metadata_by_image: dict | None = None):
    metadata_by_image = metadata_by_image or {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["image_id"]
            for fallback_cluster_id, cluster in enumerate(record.get("clusters", [])):
                if isinstance(cluster, dict):
                    cluster_id = int(cluster.get("cluster_id", fallback_cluster_id))
                    vector = cluster["v"]
                else:
                    cluster_id = fallback_cluster_id
                    vector = cluster

                source = {
                    "image_id": image_id,
                    "cluster_id": cluster_id,
                    "vector": vector,
                }
                source.update(metadata_by_image.get(image_id, {}))

                yield {
                    "_index": index_name,
                    "_id": f"{image_id}_{cluster_id}",
                    "_source": source,
                }


def main():
    parser = argparse.ArgumentParser(description="Index cluster vectors from feature JSONL into Elasticsearch.")
    parser.add_argument("--jsonl", required=True, help="Path to feature JSONL")
    parser.add_argument("--es_host", default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--index", default="tips_images", help="Index name")
    parser.add_argument("--metadata_csv", default=None, help="Optional image metadata CSV to copy into every cluster document")
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
            actions=generate_actions(jsonl_path, args.index, metadata_by_image=metadata_by_image),
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
