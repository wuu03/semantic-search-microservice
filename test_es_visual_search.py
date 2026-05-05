import argparse
import csv
import math
import os
import re
import unicodedata
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
from PIL import Image
from redis import Redis

from vl_backends import create_backend


GENERAL_QUERY_TERMS = {
    "architecture",
    "building",
    "bridge",
    "canal",
    "castle",
    "cathedral",
    "church",
    "city",
    "harbor",
    "lake",
    "mountain",
    "park",
    "river",
    "road",
    "square",
    "station",
    "street",
    "tower",
    "town",
    "village",
    "water",
}

METADATA_SEARCH_FIELDS = [
    "landmarks_identified",
    "final_place",
    "description",
    "final_city",
    "final_country",
    "transcription",
]


def slugify_for_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "query"


def build_default_output_path(backend, es_index, query_text, result_mode):
    os.makedirs("scratch", exist_ok=True)
    safe_query = slugify_for_filename(query_text)
    suffix = "cluster_mode" if result_mode == "cluster" else "heatmap"
    filename = f"{backend}_{es_index}_{safe_query}_{suffix}.png"
    return os.path.join("scratch", filename)


def normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


class MetadataQueryRouter:
    """Route broad visual concepts through vector search and named places through metadata-filtered reranking."""

    def __init__(self, metadata_csv, max_images=500):
        self.metadata_csv = metadata_csv
        self.max_images = max_images
        self.rows = []
        if metadata_csv and os.path.exists(metadata_csv):
            with open(metadata_csv, "r", encoding="latin1", newline="") as handle:
                self.rows = list(csv.DictReader(handle))

    def route(self, query_text, mode="auto"):
        if mode == "off" or not self.rows:
            return {
                "query_type": "general",
                "candidate_image_ids": None,
                "reason": "metadata filtering disabled or metadata CSV missing",
                "support": 0,
                "examples": [],
            }

        scored = self._score_metadata_matches(query_text)
        candidates = [item for item in scored if item["score"] >= 3.0]
        query_tokens = set(tokenize(query_text))
        is_generic = query_tokens and query_tokens.issubset(GENERAL_QUERY_TERMS)
        should_filter = mode == "force" or (bool(candidates) and not is_generic)

        if not should_filter:
            return {
                "query_type": "general",
                "candidate_image_ids": None,
                "reason": "broad visual concept; using pure vector search",
                "support": len(candidates),
                "examples": candidates[:5],
            }

        candidate_image_ids = [item["image_id"] for item in candidates[: self.max_images]]
        return {
            "query_type": "specific",
            "candidate_image_ids": candidate_image_ids,
            "reason": "named place/landmark matched metadata; filtering candidates before vector rerank",
            "support": len(candidates),
            "examples": candidates[:5],
        }

    def _score_metadata_matches(self, query_text):
        query_norm = normalize_text(query_text)
        query_tokens = set(tokenize(query_text))
        if not query_tokens:
            return []

        scored = []
        for row in self.rows:
            image_id = (row.get("image_filename") or "").strip()
            if not image_id:
                continue

            score = 0.0
            matched_fields = []
            for field in METADATA_SEARCH_FIELDS:
                value = row.get(field) or ""
                value_norm = normalize_text(value)
                if not value_norm:
                    continue

                field_tokens = set(tokenize(value_norm))
                if query_norm and query_norm in value_norm:
                    score += 6.0 if field in {"landmarks_identified", "final_place"} else 3.0
                    matched_fields.append(field)
                elif query_tokens.issubset(field_tokens):
                    score += 4.0 if field in {"landmarks_identified", "final_place"} else 2.0
                    matched_fields.append(field)
                else:
                    overlap = len(query_tokens & field_tokens)
                    if overlap:
                        score += overlap / max(len(query_tokens), 1)

            if score >= 1.0:
                scored.append(
                    {
                        "image_id": image_id,
                        "score": score,
                        "matched_fields": sorted(set(matched_fields)),
                        "final_place": row.get("final_place", ""),
                        "landmarks_identified": row.get("landmarks_identified", ""),
                    }
                )

        return sorted(scored, key=lambda item: item["score"], reverse=True)


class TextSearchVisualizer:
    def __init__(
        self,
        backend_name,
        es_host,
        es_index,
        redis_url,
        redis_key_prefix,
        image_root,
        device="cpu",
        vector_field="vector",
        image_id_field="image_id",
        cluster_id_field="cluster_id",
        model_id=None,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
    ):
        self.es = Elasticsearch(es_host)
        self.redis = Redis.from_url(redis_url)
        self.image_root = image_root
        self.redis_key_prefix = redis_key_prefix
        self.es_index = es_index
        self.vector_field = vector_field
        self.image_id_field = image_id_field
        self.cluster_id_field = cluster_id_field
        self.device = device

        print(f"Loading {backend_name} text encoder on {device}...")
        self.backend = create_backend(
            backend_name=backend_name,
            device=device,
            model_id=model_id,
            model_version=model_version,
            lang_model=lang_model,
        )

    @torch.no_grad()
    def encode_prompts(self, prompts):
        embeddings = self.backend.encode_text(prompts)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return F.normalize(embeddings, dim=-1)

    @staticmethod
    def normalize_negative_prompts(negative_text):
        prompts = []
        if negative_text:
            if isinstance(negative_text, str):
                prompts = [part.strip() for part in negative_text.split(",") if part.strip()]
            elif isinstance(negative_text, list):
                prompts = [str(part).strip() for part in negative_text if str(part).strip()]

        if not prompts:
            prompts = ["background"]
        return prompts

    def knn_search_candidates(self, query_vector, candidate_k):
        response = self.es.search(
            index=self.es_index,
            knn={
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": candidate_k,
                "num_candidates": max(candidate_k * 4, 100),
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=candidate_k,
        )
        return response["hits"]["hits"]

    def score_candidate_query(self, positive_vector, negative_vectors, temperature, candidate_query, size):
        script_source = f"""
double pos = cosineSimilarity(params.positive_vector, '{self.vector_field}');
double numer = Math.exp(pos * params.temperature);
double denom = numer;
for (neg in params.negative_vectors) {{
  double negScore = cosineSimilarity(neg, '{self.vector_field}');
  denom += Math.exp(negScore * params.temperature);
}}
return numer / denom;
"""
        response = self.es.search(
            index=self.es_index,
            query={
                "script_score": {
                    "query": candidate_query,
                    "script": {
                        "source": script_source,
                        "params": {
                            "positive_vector": positive_vector,
                            "negative_vectors": negative_vectors,
                            "temperature": float(temperature),
                        },
                    },
                }
            },
            _source=[self.image_id_field, self.cluster_id_field, "final_place", "landmarks_identified"],
            size=size,
        )
        return response["hits"]["hits"]

    def direct_search_with_negatives(
        self,
        query_text,
        negative_text,
        candidate_k,
        top_k,
        temperature,
        result_mode="image",
        metadata_candidate_image_ids=None,
    ):
        negative_prompts = self.normalize_negative_prompts(negative_text)
        prompts = [query_text] + negative_prompts
        text_vectors = self.encode_prompts(prompts)

        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        if metadata_candidate_image_ids:
            candidate_query = {"terms": {self.image_id_field: metadata_candidate_image_ids}}
            hits = self.score_candidate_query(
                positive_vector=positive_vector,
                negative_vectors=negative_vectors,
                temperature=temperature,
                candidate_query=candidate_query,
                size=max(candidate_k, top_k * 10),
            )
        else:
            preselected_hits = self.knn_search_candidates(query_vector=positive_vector, candidate_k=candidate_k)
            candidate_ids = [hit["_id"] for hit in preselected_hits]
            if not candidate_ids:
                return [], negative_prompts
            hits = self.score_candidate_query(
                positive_vector=positive_vector,
                negative_vectors=negative_vectors,
                temperature=temperature,
                candidate_query={"ids": {"values": candidate_ids}},
                size=candidate_k,
            )

        scored_hits = []
        for hit in hits:
            source = hit.get("_source", {})
            cluster_id = source.get(self.cluster_id_field, 0)
            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(cluster_id),
                    "score": float(hit.get("_score", 0.0)),
                    "raw_score": float(hit.get("_score", 0.0)),
                    "final_place": source.get("final_place", ""),
                    "landmarks_identified": source.get("landmarks_identified", ""),
                }
            )

        if result_mode == "cluster":
            results = sorted(scored_hits, key=lambda item: item["score"], reverse=True)
            return results[:top_k], negative_prompts

        best_per_image = {}
        for item in scored_hits:
            image_id = item["image_id"]
            if image_id not in best_per_image or item["score"] > best_per_image[image_id]["score"]:
                best_per_image[image_id] = item

        results = sorted(best_per_image.values(), key=lambda item: item["score"], reverse=True)
        return results[:top_k], negative_prompts

    def load_feature_map(self, image_id):
        key = f"{self.redis_key_prefix}:{image_id}"
        payload = self.redis.hgetall(key)
        if not payload:
            raise KeyError(f"Redis key not found: {key}")

        height = int(payload[b"height"])
        width = int(payload[b"width"])
        dtype_name = payload[b"dtype"].decode("utf-8")
        encoding = payload[b"encoding"].decode("utf-8")
        data = payload[b"data"]

        if encoding != "zlib":
            raise ValueError(f"Unsupported encoding '{encoding}' for {key}")

        decoded = zlib.decompress(data)
        return np.frombuffer(decoded, dtype=np.dtype(dtype_name)).reshape(height, width)

    @staticmethod
    def cluster_color(cluster_id, total_clusters):
        cmap = plt.get_cmap("tab20", max(total_clusters, 1))
        return np.array(cmap(cluster_id / max(total_clusters - 1, 1))[:3], dtype=np.float32)

    def make_overlay(self, image, cluster_id_map, cluster_ids):
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]

        image_np = np.asarray(image).astype(np.float32) / 255.0
        overlay = image_np.copy()
        total_clusters = int(cluster_id_map.max()) + 1

        for cluster_id in cluster_ids:
            mask = (cluster_id_map == cluster_id).astype(np.uint8)
            if mask.sum() == 0:
                continue

            mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = self.cluster_color(int(cluster_id), total_clusters)
            color_img = np.broadcast_to(color.reshape(1, 1, 3), image_np.shape)
            overlay[mask_resized] = overlay[mask_resized] * 0.45 + color_img[mask_resized] * 0.55

            edges = cv2.Canny((mask_resized.astype(np.uint8) * 255), 50, 150) > 0
            overlay[edges] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return np.clip(overlay, 0.0, 1.0)

    def resolve_image_path(self, image_id):
        image_path = os.path.join(self.image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    def visualize_results(self, results, query_text, negative_prompts, output_path=None, result_mode="image"):
        if not results:
            print("No matches found.")
            return

        if result_mode == "cluster":
            grouped = {}
            for result in results:
                grouped.setdefault(result["image_id"], []).append(result)
            display_items = list(grouped.items())
        else:
            display_items = [(result["image_id"], [result]) for result in results]

        cols = min(3, len(display_items))
        rows = math.ceil(len(display_items) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")

        for idx, (image_id, image_results) in enumerate(display_items):
            ax = axes[idx // cols, idx % cols]
            image_path = self.resolve_image_path(image_id)
            image = Image.open(image_path).convert("RGB")
            cluster_id_map = self.load_feature_map(image_id)
            cluster_ids = [item["cluster_id"] for item in image_results]
            overlay = self.make_overlay(image, cluster_id_map, cluster_ids)

            ax.imshow(overlay)
            neg_text = ", ".join(negative_prompts)
            cluster_text = ", ".join(
                f"{item['cluster_id']}:{item['score']:.4f}" for item in image_results[:6]
            )
            ax.set_title(
                f"{image_id}\n"
                f"clusters={cluster_text}\n"
                f"query='{query_text}' vs [{neg_text}]",
                fontsize=11,
            )
            ax.axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test Elasticsearch text search with Redis-backed feature-map visualization.")
    parser.add_argument("query", type=str, help="Positive text query")
    parser.add_argument("--backend", type=str, default="tips", choices=["tips", "talk2dino", "radseg"])
    parser.add_argument("--model_id", type=str, default="google/tipsv2-b14")
    parser.add_argument("--negative_text", type=str, default="background", help="Comma-separated negative prompts")
    parser.add_argument("--top_k", type=int, default=6, help="Number of unique images to visualize")
    parser.add_argument("--candidate_k", type=int, default=120, help="Number of candidate clusters retrieved from ES before direct negative scoring")
    parser.add_argument("--temperature", type=float, default=10.0, help="Softmax temperature for reranking")
    parser.add_argument("--result_mode", type=str, default="image", choices=["image", "cluster"], help="Rank by best image or by cluster hits")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--es_index", type=str, default="tips_images", help="Elasticsearch index name")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--redis_key_prefix", type=str, default="tips_fm", help="Redis key prefix used for feature maps")
    parser.add_argument("--image_root", type=str, default="images", help="Directory containing original images")
    parser.add_argument("--metadata_csv", type=str, default="images_metadata.csv", help="Optional metadata CSV for specific landmark/place query filtering")
    parser.add_argument("--metadata_filter", type=str, default="auto", choices=["auto", "off", "force"], help="Use metadata filtering for specific queries")
    parser.add_argument("--metadata_max_images", type=int, default=500, help="Maximum metadata-matched images allowed into vector reranking")
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h", help="RADSeg model version")
    parser.add_argument("--lang_model", type=str, default="siglip2-g", help="RADSeg language model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for text encoding")
    parser.add_argument("--vector_field", type=str, default="vector", help="ES vector field name")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the matplotlib figure")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = build_default_output_path(
            backend=args.backend,
            es_index=args.es_index,
            query_text=args.query,
            result_mode=args.result_mode,
        )
        print(f"Auto output path: {args.output_path}")

    visualizer = TextSearchVisualizer(
        backend_name=args.backend,
        es_host=args.es_host,
        es_index=args.es_index,
        redis_url=args.redis_url,
        redis_key_prefix=args.redis_key_prefix,
        image_root=args.image_root,
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        vector_field=args.vector_field,
        model_id=args.model_id,
    )

    if not visualizer.es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")
    visualizer.redis.ping()

    router = MetadataQueryRouter(args.metadata_csv, max_images=args.metadata_max_images)
    route = router.route(args.query, mode=args.metadata_filter)
    print(
        f"Query route: {route['query_type']} "
        f"support={route['support']} reason={route['reason']}"
    )
    for example in route["examples"][:3]:
        print(
            "  metadata match: "
            f"image={example['image_id']} score={example['score']:.2f} "
            f"fields={','.join(example['matched_fields']) or '-'} "
            f"place={example['final_place']} landmark={example['landmarks_identified']}"
        )

    results, negative_prompts = visualizer.direct_search_with_negatives(
        query_text=args.query,
        negative_text=args.negative_text,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        temperature=args.temperature,
        result_mode=args.result_mode,
        metadata_candidate_image_ids=route["candidate_image_ids"],
    )

    print(f"Top {len(results)} results for '{args.query}':")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image={result['image_id']} cluster={result['cluster_id']} "
            f"score={result['score']:.4f} raw_es={result['raw_score']:.4f} "
            f"place={result.get('final_place', '')}"
        )

    visualizer.visualize_results(
        results=results,
        query_text=args.query,
        negative_prompts=negative_prompts,
        output_path=args.output_path,
        result_mode=args.result_mode,
    )


if __name__ == "__main__":
    main()
