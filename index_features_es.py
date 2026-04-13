import json
import argparse
from elasticsearch import Elasticsearch, helpers

def create_index(es, index_name, dim):
    mapping = {
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "cluster_id": {"type": "integer"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    if es.indices.exists(index=index_name):
        print(f"Deleting existing index {index_name}...")
        es.indices.delete(index=index_name)
    print(f"Creating index '{index_name}' with dim {dim}...")
    es.indices.create(index=index_name, body=mapping)

def yield_docs(index_name, jsonl_file):
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_id = data["image_id"]
            # Each cluster center is a separate document
            for idx, vec in enumerate(data["clusters"]):
                yield {
                    "_index": index_name,
                    "_source": {
                        "image_id": image_id,
                        "cluster_id": idx,
                        "vector": vec
                    }
                }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Insert JSONL Features into Elasticsearch")
    parser.add_argument("--jsonl", type=str, default="features.jsonl", help="Input JSONL from remote GPU")
    parser.add_argument("--index", type=str, default="radseg_images", help="Elasticsearch index name")
    parser.add_argument("--dim", type=int, default=1152, help="Vector dimension size")
    args = parser.parse_args()

    # Disable strict cert checking for local test container
    es = Elasticsearch("http://localhost:9200")
    
    if not es.ping():
        print("Error: Could not connect to Elasticsearch at localhost:9200. Is Docker running?")
        exit(1)

    create_index(es, args.index, dim=args.dim)

    print(f"Streaming data from {args.jsonl} to Elasticsearch...")
    success, failed = helpers.bulk(es, yield_docs(args.index, args.jsonl), chunk_size=1000)
    print(f"Done! Successfully indexed {success} vectors.")
