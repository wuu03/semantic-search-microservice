# python load_3d_indices_to_redis.py --parquet ./timeatlas_3d_vectors.parquet --redis_url redis://localhost:6379/0 --key_prefix edifici_fm --overwrite
import argparse
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from redis import Redis
from tqdm import tqdm


def choose_dtype(max_index):
    if max_index <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def flush_pipeline(pipeline):
    try:
        results = pipeline.execute()
        if any(not r for r in results if isinstance(r, bool)):
            print(" Warning: Some Redis commands in cluster pipeline might have skipped or failed.")
    except Exception as e:
        print(f"[-] Execute pipeline failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="TimeAtlas 3D Entity Point Indices to Redis Importer")
    parser.add_argument("--parquet", type=str, required=True, help="Path to aggregated 3D parquet file")
    parser.add_argument("--redis_url", default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--key_prefix", default="edifici_fm", help="Redis key prefix for 3D feature maps")
    parser.add_argument("--batch_size", type=int, default=500, help="Pipeline batch size")
    parser.add_argument("--compression_level", type=int, default=6, help="zlib compression level (1-9)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing keys in Redis")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"Parquet file not found: {parquet_path}")

    print(f"Connecting to Redis at {args.redis_url}...")
    redis_client = Redis.from_url(args.redis_url)
    try:
        redis_client.ping()
    except Exception as e:
        raise SystemExit(f"Could not connect to Redis: {e}")

    print("Reading 3D Parquet data...")
    df = pd.read_parquet(parquet_path, columns=["edifici_id", "entity_id", "level", "point_indices"])
    total_rows = len(df)
    print(f"Loaded {total_rows} entity records from Parquet.")

    pipeline = redis_client.pipeline(transaction=False)
    queued = 0
    imported = 0
    skipped_existing = 0
    skipped_invalid = 0

    with tqdm(total=total_rows, desc="Injecting 3D Indices to Redis") as pbar:
        for _, row in df.iterrows():
            pbar.update(1)
            edifici_id = row["edifici_id"]
            entity_id = row["entity_id"]
            level = int(row["level"])
            indices = row["point_indices"]

            if indices is None or len(indices) == 0:
                skipped_invalid += 1
                continue
                
            indices_array = np.asarray(indices)
            point_count = len(indices_array)

            key = f"{args.key_prefix}:{edifici_id}:{entity_id}"

            if not args.overwrite and redis_client.exists(key):
                skipped_existing += 1
                continue

            max_idx = int(indices_array.max()) if indices_array.size > 0 else 0
            dtype = choose_dtype(max_idx)
            indices_array = indices_array.astype(dtype, copy=False)

            compressed_payload = zlib.compress(indices_array.tobytes(order="C"), level=args.compression_level)

            pipeline.hset(
                key,
                mapping={
                    "edifici_id": edifici_id,
                    "entity_id": entity_id,
                    "level": level,
                    "point_count": point_count,
                    "dtype": np.dtype(dtype).name,
                    "encoding": "zlib",
                    "data": compressed_payload,
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

    print(f"\n[+] 3D Import Finished.")
    print(f"    - Successfully Imported: {imported}")
    print(f"    - Skipped (Existing): {skipped_existing}")
    print(f"    - Skipped (Invalid/Empty): {skipped_invalid}")


if __name__ == "__main__":
    main()