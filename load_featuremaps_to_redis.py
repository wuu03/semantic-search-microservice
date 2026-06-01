# python load_featuremaps_to_redis.py --jsonl test_features.jsonl --redis_url redis://SERVER_IP:6379/0 --key_prefix image_fm --overwrite
import argparse
import json
import zlib

import numpy as np
from redis import Redis
from tqdm import tqdm


def choose_dtype(max_cluster_id):
    if max_cluster_id <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_cluster_id <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def encode_cluster_id_map(cluster_id_map, compression_level):
    array = np.asarray(cluster_id_map)
    if array.ndim != 2:
        raise ValueError(f"cluster_id_map must be 2D, got shape {array.shape}")

    max_cluster_id = int(array.max()) if array.size > 0 else 0
    dtype = choose_dtype(max_cluster_id)
    array = array.astype(dtype, copy=False)

    payload = zlib.compress(array.tobytes(order="C"), level=compression_level)
    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "dtype": np.dtype(dtype).name,
        "encoding": "zlib",
        "max_cluster_id": max_cluster_id,
        "payload": payload,
    }


def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def flush_pipeline(pipeline):
    pipeline.execute()
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Load feature maps from JSONL into Redis.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to the feature JSONL file")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--key_prefix", type=str, default="fm", help="Redis key prefix for feature maps")
    parser.add_argument("--batch_size", type=int, default=500, help="Pipeline batch size")
    parser.add_argument(
        "--compression_level",
        type=int,
        default=6,
        help="zlib compression level for serialized feature maps",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Redis entries for the same image_id",
    )
    args = parser.parse_args()

    redis_client = Redis.from_url(args.redis_url)
    redis_client.ping()

    total_lines = count_lines(args.jsonl)
    pipeline = redis_client.pipeline(transaction=False)

    imported = 0
    skipped_existing = 0
    skipped_invalid = 0
    queued = 0

    with open(args.jsonl, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total_lines, desc="Importing feature maps"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            image_id = record.get("image_id")
            cluster_id_map = record.get("cluster_id_map")
            feature_map_size = record.get("feature_map_size")
            clusters = record.get("clusters", [])

            if not image_id or cluster_id_map is None or feature_map_size is None:
                skipped_invalid += 1
                continue

            key = f"{args.key_prefix}:{image_id}"
            if not args.overwrite and redis_client.exists(key):
                skipped_existing += 1
                continue

            encoded = encode_cluster_id_map(cluster_id_map, compression_level=args.compression_level)
            expected_height, expected_width = feature_map_size
            if encoded["height"] != expected_height or encoded["width"] != expected_width:
                skipped_invalid += 1
                continue

            pipeline.hset(
                key,
                mapping={
                    "image_id": image_id,
                    "height": encoded["height"],
                    "width": encoded["width"],
                    "dtype": encoded["dtype"],
                    "encoding": encoded["encoding"],
                    "cluster_count": len(clusters),
                    "max_cluster_id": encoded["max_cluster_id"],
                    "data": encoded["payload"],
                },
            )
            queued += 1
            imported += 1

            if queued >= args.batch_size:
                flush_pipeline(pipeline)
                pipeline = redis_client.pipeline(transaction=False)
                queued = 0

    if queued > 0:
        flush_pipeline(pipeline)

    print(f"Imported: {imported}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped invalid: {skipped_invalid}")


if __name__ == "__main__":
    main()
