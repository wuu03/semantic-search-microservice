import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from test_es_visual_search import TextSearchVisualizer
from vl_backends import create_backend


COMMON_CONCEPTS = [
    {
        "benchmark_type": "concept",
        "query": "bridge",
        "label": "bridge",
        "positive_patterns": [r"\bbridge\b", r"\bdrawbridge\b", r"\bviaduct\b", r"\barched bridge\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "river",
        "label": "river",
        "positive_patterns": [r"\briver\b", r"\bcanal\b", r"\bwaterfront\b", r"\briverside\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "water",
        "label": "water",
        "positive_patterns": [r"\bwater\b", r"\blake\b", r"\briver\b", r"\bcanal\b", r"\bharbor\b", r"\bharbour\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "castle",
        "label": "castle",
        "positive_patterns": [r"\bcastle\b", r"\bfortress\b", r"\bpalace\b", r"\bschloss\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "church",
        "label": "church",
        "positive_patterns": [r"\bchurch\b", r"\bcathedral\b", r"\bchapel\b", r"\bbasilica\b", r"\babbey\b", r"\bkirche\b", r"\bdomkirche\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "tower",
        "label": "tower",
        "positive_patterns": [r"\btower\b", r"\bspire\b", r"\bbell tower\b", r"\bbelfry\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "street",
        "label": "street",
        "positive_patterns": [r"\bstreet\b", r"\bavenue\b", r"\broad\b", r"\bboulevard\b", r"\blane\b", r"\bmarket street\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "square",
        "label": "square",
        "positive_patterns": [r"\bsquare\b", r"\bplaza\b", r"\bmarket square\b", r"\bplatz\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "park",
        "label": "park",
        "positive_patterns": [r"\bpark\b", r"\bgarden\b"],
    },
    {
        "benchmark_type": "concept",
        "query": "arched bridge over water",
        "label": "bridge_over_water",
        "positive_patterns": None,
    },
]


def fix_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        repaired = text.encode("latin1").decode("utf-8")
        bad_original = text.count("Ã") + text.count("Â")
        bad_repaired = repaired.count("Ã") + repaired.count("Â")
        if bad_repaired < bad_original:
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def normalize_negative_prompts(negative_text):
    prompts = []
    if negative_text:
        prompts = [part.strip() for part in negative_text.split(",") if part.strip()]
    return prompts or ["background"]


def precision_at_k(items, relevant_set, k):
    top_items = items[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for item in top_items if item in relevant_set)
    return hits / float(k)


def recall_at_k(items, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in items[:k] if item in relevant_set)
    return hits / float(len(relevant_set))


def reciprocal_rank(items, relevant_set):
    for idx, item in enumerate(items, start=1):
        if item in relevant_set:
            return 1.0 / float(idx)
    return 0.0


def average_precision(items, relevant_set, k=None):
    if not relevant_set:
        return 0.0
    ranked = items if k is None else items[:k]
    hit_count = 0
    precisions = []
    for idx, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hit_count += 1
            precisions.append(hit_count / float(idx))
    if not precisions:
        return 0.0
    return sum(precisions) / float(len(relevant_set))


def ndcg_at_k(items, relevant_set, k):
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def slugify_for_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "benchmark"


def extract_landmark_names(value):
    text = fix_text(value)
    if not text:
        return []
    matches = re.findall(r"\s*([^,(][^()]*)\s*\(https?://[^)]+\)", text)
    if matches:
        return [match.strip(" ,") for match in matches if match.strip(" ,")]
    return []


def build_combined_text(row):
    parts = [
        row.get("description_clean", ""),
        row.get("transcription_clean", ""),
        row.get("landmarks_clean", ""),
        row.get("final_place_clean", ""),
        row.get("final_city_clean", ""),
        row.get("final_country_clean", ""),
    ]
    return " ".join(part for part in parts if part).lower()


class SearchScorer:
    def __init__(
        self,
        backend_name,
        es_host,
        es_index,
        device,
        model_id=None,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        vector_field="vector",
        image_id_field="image_id",
        cluster_id_field="cluster_id",
        candidate_k=120,
        negative_text="background, sky, clouds, text, border, trees, road, people",
        temperature=10.0,
    ):
        self.es = Elasticsearch(es_host)
        self.es_index = es_index
        self.vector_field = vector_field
        self.image_id_field = image_id_field
        self.cluster_id_field = cluster_id_field
        self.candidate_k = candidate_k
        self.temperature = float(temperature)
        self.negative_prompts = normalize_negative_prompts(negative_text)

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

    def knn_candidates(self, query_vector):
        response = self.es.search(
            index=self.es_index,
            knn={
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": self.candidate_k,
                "num_candidates": max(self.candidate_k * 4, 100),
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=self.candidate_k,
        )
        return response["hits"]["hits"]

    def search(self, query_text, result_mode="image", top_k=50):
        prompts = [query_text] + self.negative_prompts
        text_vectors = self.encode_prompts(prompts)
        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        preselected_hits = self.knn_candidates(positive_vector)
        candidate_ids = [hit["_id"] for hit in preselected_hits]
        if not candidate_ids:
            return []

        script_source = f"""
double pos = cosineSimilarity(params.positive_vector, '{self.vector_field}');
double numer = Math.exp(pos * params.temperature);
double denom = numer;
double bestNeg = -1000.0;
for (neg in params.negative_vectors) {{
  double negScore = cosineSimilarity(neg, '{self.vector_field}');
  if (negScore > bestNeg) bestNeg = negScore;
  denom += Math.exp(negScore * params.temperature);
}}
double score = numer / denom;
return score;
"""

        response = self.es.search(
            index=self.es_index,
            query={
                "script_score": {
                    "query": {"ids": {"values": candidate_ids}},
                    "script": {
                        "source": script_source,
                        "params": {
                            "positive_vector": positive_vector,
                            "negative_vectors": negative_vectors,
                            "temperature": self.temperature,
                        },
                    },
                }
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=self.candidate_k,
        )

        scored_hits = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(source.get(self.cluster_id_field, 0)),
                    "score": float(hit.get("_score", 0.0)),
                }
            )

        if result_mode == "cluster":
            return sorted(scored_hits, key=lambda item: item["score"], reverse=True)[:top_k]

        best_per_image = {}
        for item in scored_hits:
            image_id = item["image_id"]
            if image_id not in best_per_image or item["score"] > best_per_image[image_id]["score"]:
                best_per_image[image_id] = item
        return sorted(best_per_image.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def get_indexed_image_ids(es, index_name):
    image_ids = set()
    for hit in helpers.scan(
        client=es,
        index=index_name,
        query={"query": {"match_all": {}}},
        _source=["image_id"],
        size=1000,
    ):
        image_ids.add(hit["_source"]["image_id"])
    return image_ids


def prepare_metadata(metadata_csv, indexed_image_ids):
    df = pd.read_csv(metadata_csv, encoding="latin1")
    df["image_filename"] = df["image_filename"].map(fix_text)
    df["final_country_clean"] = df["final_country"].map(fix_text)
    df["final_city_clean"] = df["final_city"].map(fix_text)
    df["final_place_clean"] = df["final_place"].map(fix_text)
    df["description_clean"] = df["description"].map(fix_text)
    df["transcription_clean"] = df["transcription"].map(fix_text)
    df["landmarks_clean"] = df["landmarks_identified"].map(fix_text)
    df["landmark_names"] = df["landmarks_identified"].map(extract_landmark_names)
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    df = df[df["image_filename"].isin(indexed_image_ids)].copy()
    return df


def build_structured_queries(df, min_city_count, min_place_count, min_landmark_count, top_n_per_type):
    query_specs = []

    city_counts = df["final_city_clean"].value_counts()
    city_counts = city_counts[city_counts.index.str.strip() != ""]
    for value, count in city_counts[city_counts >= min_city_count].head(top_n_per_type).items():
        relevant = set(df.loc[df["final_city_clean"] == value, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": "city",
                "query": value,
                "label": value,
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    place_counts = df["final_place_clean"].value_counts()
    place_counts = place_counts[place_counts.index.str.strip() != ""]
    for value, count in place_counts[place_counts >= min_place_count].head(top_n_per_type).items():
        relevant = set(df.loc[df["final_place_clean"] == value, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": "place",
                "query": value,
                "label": value,
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    landmark_counter = Counter()
    landmark_to_images = defaultdict(set)
    for _, row in df.iterrows():
        image_id = row["image_filename"]
        for landmark_name in row["landmark_names"]:
            if not landmark_name:
                continue
            landmark_counter[landmark_name] += 1
            landmark_to_images[landmark_name].add(image_id)

    for landmark_name, count in landmark_counter.most_common():
        if count < min_landmark_count:
            break
        query_specs.append(
            {
                "benchmark_type": "landmark",
                "query": landmark_name,
                "label": landmark_name,
                "relevant_images": landmark_to_images[landmark_name],
                "support": len(landmark_to_images[landmark_name]),
            }
        )
        if sum(1 for item in query_specs if item["benchmark_type"] == "landmark") >= top_n_per_type:
            break

    return query_specs


def build_concept_queries(df):
    query_specs = []
    text = df["combined_text"]

    for concept in COMMON_CONCEPTS:
        if concept["label"] == "bridge_over_water":
            has_bridge = text.str.contains(r"\bbridge\b|\bdrawbridge\b|\bviaduct\b|\barched bridge\b", regex=True)
            has_water = text.str.contains(r"\briver\b|\bcanal\b|\bwater\b|\blake\b|\bharbor\b|\bharbour\b", regex=True)
            mask = has_bridge & has_water
        else:
            pattern = "|".join(concept["positive_patterns"])
            mask = text.str.contains(pattern, regex=True)

        relevant = set(df.loc[mask, "image_filename"])
        query_specs.append(
            {
                "benchmark_type": concept["benchmark_type"],
                "query": concept["query"],
                "label": concept["label"],
                "relevant_images": relevant,
                "support": len(relevant),
            }
        )

    return query_specs


def evaluate_image_level(query_spec, image_results):
    ranked_images = [item["image_id"] for item in image_results]
    relevant = query_spec["relevant_images"]
    return {
        "precision@1": precision_at_k(ranked_images, relevant, 1),
        "precision@5": precision_at_k(ranked_images, relevant, 5),
        "precision@10": precision_at_k(ranked_images, relevant, 10),
        "recall@1": recall_at_k(ranked_images, relevant, 1),
        "recall@5": recall_at_k(ranked_images, relevant, 5),
        "recall@10": recall_at_k(ranked_images, relevant, 10),
        "mrr": reciprocal_rank(ranked_images, relevant),
        "ap@10": average_precision(ranked_images, relevant, k=10),
        "ndcg@10": ndcg_at_k(ranked_images, relevant, 10),
    }


def evaluate_cluster_level(query_spec, cluster_results):
    relevant = query_spec["relevant_images"]
    ranked_cluster_images = [item["image_id"] for item in cluster_results]
    unique_ranked_images = []
    seen = set()
    for image_id in ranked_cluster_images:
        if image_id not in seen:
            seen.add(image_id)
            unique_ranked_images.append(image_id)

    top10 = cluster_results[:10]
    top20 = cluster_results[:20]
    rel_top10 = [item for item in top10 if item["image_id"] in relevant]
    rel_top20 = [item for item in top20 if item["image_id"] in relevant]
    top20_image_counts = Counter(item["image_id"] for item in rel_top20)
    repeated_relevant_images = sum(1 for _, count in top20_image_counts.items() if count >= 2)

    return {
        "cluster_precision@10": len(rel_top10) / 10.0 if top10 else 0.0,
        "cluster_precision@20": len(rel_top20) / 20.0 if top20 else 0.0,
        "image_recall_from_clusters@10": recall_at_k(unique_ranked_images, relevant, 10),
        "image_recall_from_clusters@20": recall_at_k(unique_ranked_images, relevant, 20),
        "cluster_mrr": reciprocal_rank(ranked_cluster_images, relevant),
        "relevant_unique_images@20": len(top20_image_counts),
        "repeated_relevant_images@20": repeated_relevant_images,
        "mean_relevant_cluster_score@20": float(np.mean([item["score"] for item in rel_top20])) if rel_top20 else 0.0,
        "mean_nonrelevant_cluster_score@20": float(np.mean([item["score"] for item in top20 if item["image_id"] not in relevant])) if top20 else 0.0,
    }


def export_ground_truth(report_dir, query_specs, metadata_df):
    metadata_lookup = metadata_df.set_index("image_filename")
    gt_json_path = os.path.join(report_dir, "query_ground_truth.json")
    gt_csv_path = os.path.join(report_dir, "query_ground_truth_pairs.csv")

    exported_queries = []
    gt_rows = []

    for query_spec in sorted(query_specs, key=lambda item: (item["benchmark_type"], item["query"])):
        relevant_images = sorted(query_spec["relevant_images"])
        relevant_items = []
        for image_id in relevant_images:
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""

            item = {
                "image_id": image_id,
                "final_city": final_city,
                "final_place": final_place,
                "landmarks_identified": landmarks_identified,
                "description": description,
            }
            relevant_items.append(item)

            gt_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query": query_spec["query"],
                    "label": query_spec["label"],
                    "support": query_spec["support"],
                    "image_id": image_id,
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )

        exported_queries.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                "relevant_images": relevant_images,
                "relevant_items": relevant_items,
            }
        )

    with open(gt_json_path, "w", encoding="utf-8") as handle:
        json.dump(exported_queries, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(gt_rows).to_csv(gt_csv_path, index=False, encoding="utf-8")
    return gt_json_path, gt_csv_path


def write_query_browser_report(report_dir, image_df, cluster_df, gt_pairs_df, image_hits_df, cluster_hits_df, visualizations_dir):
    md_path = os.path.join(report_dir, "query_browser_report.md")
    html_path = os.path.join(report_dir, "query_browser_report.html")

    image_lookup = {
        (row["benchmark_type"], row["query"]): row
        for _, row in image_df.iterrows()
    }
    cluster_lookup = {
        (row["benchmark_type"], row["query"]): row
        for _, row in cluster_df.iterrows()
    }

    query_keys = sorted(image_lookup.keys(), key=lambda item: (item[0], item[1].lower()))

    md_lines = ["# Query Browser Report", ""]
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Query Browser Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.45;}",
        "h1,h2,h3{margin-top:1.2em;}",
        ".card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:20px 0;}",
        ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 16px;margin:12px 0;}",
        ".metric{background:#f7f7f7;padding:8px 10px;border-radius:8px;}",
        "table{border-collapse:collapse;width:100%;margin:10px 0 16px 0;}",
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top;}",
        "th{background:#f0f0f0;}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:4px;}",
        "</style></head><body>",
        "<h1>Query Browser Report</h1>",
    ]

    for benchmark_type, query in query_keys:
        img_row = image_lookup[(benchmark_type, query)]
        clu_row = cluster_lookup[(benchmark_type, query)]

        gt_subset = gt_pairs_df[(gt_pairs_df["benchmark_type"] == benchmark_type) & (gt_pairs_df["query"] == query)].copy()
        image_hits_subset = image_hits_df[(image_hits_df["benchmark_type"] == benchmark_type) & (image_hits_df["query"] == query)].copy()
        cluster_hits_subset = cluster_hits_df[(cluster_hits_df["benchmark_type"] == benchmark_type) & (cluster_hits_df["query"] == query)].copy()

        image_hits_subset = image_hits_subset.sort_values("rank").head(10)
        cluster_hits_subset = cluster_hits_subset.sort_values("rank").head(10)
        gt_subset = gt_subset.head(12)

        query_slug = slugify_for_filename(query)
        vis_rel = None
        if visualizations_dir:
            vis_path = os.path.join(visualizations_dir, benchmark_type, f"{query_slug}_heatmap.png")
            if os.path.exists(vis_path):
                vis_rel = os.path.relpath(vis_path, report_dir).replace("\\", "/")

        md_lines.append(f"## {benchmark_type}: `{query}`")
        md_lines.append("")
        md_lines.append(f"- Support: `{int(img_row['support'])}`")
        md_lines.append(f"- Image metrics: `R@1={img_row['recall@1']:.4f}`, `R@5={img_row['recall@5']:.4f}`, `R@10={img_row['recall@10']:.4f}`, `MRR={img_row['mrr']:.4f}`, `nDCG@10={img_row['ndcg@10']:.4f}`")
        md_lines.append(f"- Cluster metrics: `P@10={clu_row['cluster_precision@10']:.4f}`, `P@20={clu_row['cluster_precision@20']:.4f}`, `ImageRecall@10={clu_row['image_recall_from_clusters@10']:.4f}`, `ClusterMRR={clu_row['cluster_mrr']:.4f}`")
        if vis_rel:
            md_lines.append(f"- Visualization: [{vis_rel}]({vis_rel})")
            md_lines.append("")
            md_lines.append(f"![{query}]({vis_rel})")
        md_lines.append("")

        html_parts.append(f"<div class='card'><h2>{benchmark_type}: <code>{query}</code></h2>")
        html_parts.append(f"<p><strong>Support:</strong> {int(img_row['support'])}</p>")
        html_parts.append("<div class='metrics'>")
        for label, value in [
            ("Recall@1", img_row["recall@1"]),
            ("Recall@5", img_row["recall@5"]),
            ("Recall@10", img_row["recall@10"]),
            ("MRR", img_row["mrr"]),
            ("nDCG@10", img_row["ndcg@10"]),
            ("Cluster P@10", clu_row["cluster_precision@10"]),
            ("Cluster P@20", clu_row["cluster_precision@20"]),
            ("ImageRecallFromClusters@10", clu_row["image_recall_from_clusters@10"]),
            ("Cluster MRR", clu_row["cluster_mrr"]),
        ]:
            html_parts.append(f"<div class='metric'><strong>{label}</strong><br>{float(value):.4f}</div>")
        html_parts.append("</div>")
        if vis_rel:
            html_parts.append(f"<h3>Search visualization</h3><img src='{vis_rel}' alt='{query}'>")

        def _table_html(df, cols, title):
            html = [f"<h3>{title}</h3>"]
            if df.empty:
                html.append("<p><em>None</em></p>")
                return "".join(html)
            html.append("<table><thead><tr>")
            for col in cols:
                html.append(f"<th>{col}</th>")
            html.append("</tr></thead><tbody>")
            for _, row in df.iterrows():
                html.append("<tr>")
                for col in cols:
                    value = row[col]
                    if isinstance(value, float):
                        if col == "score":
                            display = f"{value:.4f}"
                        else:
                            display = f"{value:.4f}" if not pd.isna(value) else ""
                    else:
                        display = str(value)
                    html.append(f"<td>{display}</td>")
                html.append("</tr>")
            html.append("</tbody></table>")
            return "".join(html)

        html_parts.append(
            _table_html(
                gt_subset[["image_id", "final_city", "final_place", "landmarks_identified"]]
                if not gt_subset.empty else gt_subset,
                ["image_id", "final_city", "final_place", "landmarks_identified"],
                "Ground truth examples (first 12)",
            )
        )
        html_parts.append(
            _table_html(
                image_hits_subset[["rank", "image_id", "cluster_id", "score", "is_relevant", "final_city", "final_place"]]
                if not image_hits_subset.empty else image_hits_subset,
                ["rank", "image_id", "cluster_id", "score", "is_relevant", "final_city", "final_place"],
                "Top image-level hits (first 10)",
            )
        )
        html_parts.append(
            _table_html(
                cluster_hits_subset[["rank", "image_id", "cluster_id", "score", "is_relevant", "final_city", "final_place"]]
                if not cluster_hits_subset.empty else cluster_hits_subset,
                ["rank", "image_id", "cluster_id", "score", "is_relevant", "final_city", "final_place"],
                "Top cluster-level hits (first 10)",
            )
        )
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("".join(html_parts))
    return md_path, html_path


def write_markdown_report(path, config, image_rows, cluster_rows):
    def macro(rows, key):
        values = [row[key] for row in rows if not pd.isna(row[key])]
        return float(np.mean(values)) if values else 0.0

    image_df = pd.DataFrame(image_rows)
    cluster_df = pd.DataFrame(cluster_rows)

    lines = []
    lines.append("# Metadata Search Evaluation")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Backend: `{config['backend']}`")
    lines.append(f"- Index: `{config['es_index']}`")
    lines.append(f"- Candidate K: `{config['candidate_k']}`")
    lines.append(f"- Negative prompts: `{config['negative_text']}`")
    lines.append(f"- Temperature: `{config['temperature']}`")
    if config.get("visualizations_dir"):
        lines.append(f"- Visualizations: `{config['visualizations_dir']}`")
    lines.append("")
    lines.append("## Image-Level Macro Metrics")
    lines.append("")
    for benchmark_type in sorted(image_df["benchmark_type"].unique()):
        subset = image_df[image_df["benchmark_type"] == benchmark_type]
        lines.append(f"### {benchmark_type}")
        lines.append("")
        lines.append(f"- Queries: `{len(subset)}`")
        lines.append(f"- Recall@1: `{subset['recall@1'].mean():.4f}`")
        lines.append(f"- Recall@5: `{subset['recall@5'].mean():.4f}`")
        lines.append(f"- Recall@10: `{subset['recall@10'].mean():.4f}`")
        lines.append(f"- MRR: `{subset['mrr'].mean():.4f}`")
        lines.append(f"- nDCG@10: `{subset['ndcg@10'].mean():.4f}`")
        lines.append("")
    lines.append("## Cluster-Level Macro Metrics")
    lines.append("")
    for benchmark_type in sorted(cluster_df["benchmark_type"].unique()):
        subset = cluster_df[cluster_df["benchmark_type"] == benchmark_type]
        lines.append(f"### {benchmark_type}")
        lines.append("")
        lines.append(f"- Queries: `{len(subset)}`")
        lines.append(f"- Cluster Precision@10: `{subset['cluster_precision@10'].mean():.4f}`")
        lines.append(f"- Cluster Precision@20: `{subset['cluster_precision@20'].mean():.4f}`")
        lines.append(f"- Image Recall From Clusters@10: `{subset['image_recall_from_clusters@10'].mean():.4f}`")
        lines.append(f"- Image Recall From Clusters@20: `{subset['image_recall_from_clusters@20'].mean():.4f}`")
        lines.append(f"- Cluster MRR: `{subset['cluster_mrr'].mean():.4f}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate image-level and cluster-level text search using metadata-derived weak labels.")
    parser.add_argument("--metadata_csv", default="images_metadata.csv")
    parser.add_argument("--es_host", default="http://localhost:9200")
    parser.add_argument("--es_index", required=True)
    parser.add_argument("--backend", default="radseg", choices=["radseg", "tips", "talk2dino"])
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--model_version", default="c-radio_v4-h")
    parser.add_argument("--lang_model", default="siglip2-g")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vector_field", default="vector")
    parser.add_argument("--negative_text", default="background, sky, clouds, text, border, trees, road, people")
    parser.add_argument("--candidate_k", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--image_top_k", type=int, default=20)
    parser.add_argument("--cluster_top_k", type=int, default=50)
    parser.add_argument("--min_city_count", type=int, default=10)
    parser.add_argument("--min_place_count", type=int, default=5)
    parser.add_argument("--min_landmark_count", type=int, default=5)
    parser.add_argument("--top_n_per_type", type=int, default=10)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--redis_url", default="redis://localhost:6379/0")
    parser.add_argument("--redis_key_prefix", default="fm")
    parser.add_argument("--image_root", default="images")
    parser.add_argument("--visualize_queries", choices=["all", "concept", "structured", "none"], default="all")
    parser.add_argument("--visualize_result_mode", choices=["image", "cluster"], default="image")
    parser.add_argument("--visualize_top_k", type=int, default=6)
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face cache only")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.report_dir is None:
        args.report_dir = os.path.join("scratch", f"eval_{args.backend}_{args.es_index}")
    os.makedirs(args.report_dir, exist_ok=True)

    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")

    indexed_image_ids = get_indexed_image_ids(es, args.es_index)
    print(f"Indexed images in {args.es_index}: {len(indexed_image_ids)}")

    metadata_df = prepare_metadata(args.metadata_csv, indexed_image_ids)
    print(f"Metadata rows aligned to index: {len(metadata_df)}")

    structured_queries = build_structured_queries(
        metadata_df,
        min_city_count=args.min_city_count,
        min_place_count=args.min_place_count,
        min_landmark_count=args.min_landmark_count,
        top_n_per_type=args.top_n_per_type,
    )
    concept_queries = build_concept_queries(metadata_df)
    all_queries = [query for query in structured_queries + concept_queries if query["support"] > 0]

    print(f"Total benchmark queries: {len(all_queries)}")

    gt_json_path, gt_csv_path = export_ground_truth(args.report_dir, all_queries, metadata_df)

    scorer = SearchScorer(
        backend_name=args.backend,
        es_host=args.es_host,
        es_index=args.es_index,
        device=args.device,
        model_id=args.model_id,
        model_version=args.model_version,
        lang_model=args.lang_model,
        vector_field=args.vector_field,
        candidate_k=args.candidate_k,
        negative_text=args.negative_text,
        temperature=args.temperature,
    )

    visualizations_dir = None
    visualizer = None
    if args.visualize_queries != "none":
        visualizations_dir = os.path.join(args.report_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        visualizer = TextSearchVisualizer(
            backend_name=args.backend,
            es_host=args.es_host,
            es_index=args.es_index,
            redis_url=args.redis_url,
            redis_key_prefix=args.redis_key_prefix,
            image_root=args.image_root,
            device=args.device,
            vector_field=args.vector_field,
            model_id=args.model_id,
            model_version=args.model_version,
            lang_model=args.lang_model,
        )

    image_rows = []
    cluster_rows = []
    cluster_hit_rows = []
    image_hit_rows = []
    metadata_lookup = metadata_df.set_index("image_filename")

    for query_spec in tqdm(all_queries, desc="Evaluating queries"):
        image_results = scorer.search(query_spec["query"], result_mode="image", top_k=args.image_top_k)
        cluster_results = scorer.search(query_spec["query"], result_mode="cluster", top_k=args.cluster_top_k)

        image_metrics = evaluate_image_level(query_spec, image_results)
        cluster_metrics = evaluate_cluster_level(query_spec, cluster_results)

        image_rows.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                **image_metrics,
            }
        )
        cluster_rows.append(
            {
                "benchmark_type": query_spec["benchmark_type"],
                "query": query_spec["query"],
                "label": query_spec["label"],
                "support": query_spec["support"],
                **cluster_metrics,
            }
        )

        if visualizer is not None:
            should_render = (
                args.visualize_queries == "all"
                or (args.visualize_queries == "concept" and query_spec["benchmark_type"] == "concept")
                or (args.visualize_queries == "structured" and query_spec["benchmark_type"] != "concept")
            )
            if should_render:
                query_slug = slugify_for_filename(query_spec["query"])
                mode_suffix = "cluster_mode" if args.visualize_result_mode == "cluster" else "heatmap"
                query_dir = os.path.join(visualizations_dir, query_spec["benchmark_type"])
                os.makedirs(query_dir, exist_ok=True)
                output_path = os.path.join(query_dir, f"{query_slug}_{mode_suffix}.png")
                render_results = cluster_results if args.visualize_result_mode == "cluster" else image_results
                visualizer.visualize_results(
                    render_results[: args.visualize_top_k],
                    query_text=query_spec["query"],
                    negative_prompts=visualizer.normalize_negative_prompts(args.negative_text),
                    output_path=output_path,
                    result_mode=args.visualize_result_mode,
                )

        relevant = query_spec["relevant_images"]
        for rank, item in enumerate(image_results, start=1):
            image_id = item["image_id"]
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"][:240]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""
            image_hit_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query": query_spec["query"],
                    "support": query_spec["support"],
                    "rank": rank,
                    "image_id": image_id,
                    "cluster_id": item["cluster_id"],
                    "score": item["score"],
                    "is_relevant": int(image_id in relevant),
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )
        for rank, item in enumerate(cluster_results, start=1):
            image_id = item["image_id"]
            if image_id in metadata_lookup.index:
                row = metadata_lookup.loc[image_id]
                final_city = row["final_city_clean"]
                final_place = row["final_place_clean"]
                landmarks_identified = row["landmarks_clean"]
                description = row["description_clean"][:240]
            else:
                final_city = ""
                final_place = ""
                landmarks_identified = ""
                description = ""
            cluster_hit_rows.append(
                {
                    "benchmark_type": query_spec["benchmark_type"],
                    "query": query_spec["query"],
                    "support": query_spec["support"],
                    "rank": rank,
                    "image_id": image_id,
                    "cluster_id": item["cluster_id"],
                    "score": item["score"],
                    "is_relevant": int(image_id in relevant),
                    "final_city": final_city,
                    "final_place": final_place,
                    "landmarks_identified": landmarks_identified,
                    "description": description,
                }
            )

    image_df = pd.DataFrame(image_rows).sort_values(["benchmark_type", "query"])
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["benchmark_type", "query"])
    image_hits_df = pd.DataFrame(image_hit_rows).sort_values(["query", "rank"])
    cluster_hits_df = pd.DataFrame(cluster_hit_rows).sort_values(["query", "rank"])
    gt_pairs_df = pd.read_csv(gt_csv_path, encoding="utf-8")

    image_csv = os.path.join(args.report_dir, "image_level_metrics.csv")
    cluster_csv = os.path.join(args.report_dir, "cluster_level_metrics.csv")
    image_hits_csv = os.path.join(args.report_dir, "image_level_top_hits.csv")
    cluster_hits_csv = os.path.join(args.report_dir, "cluster_level_top_hits.csv")
    summary_json = os.path.join(args.report_dir, "summary.json")
    summary_md = os.path.join(args.report_dir, "summary.md")

    image_df.to_csv(image_csv, index=False, encoding="utf-8")
    cluster_df.to_csv(cluster_csv, index=False, encoding="utf-8")
    image_hits_df.to_csv(image_hits_csv, index=False, encoding="utf-8")
    cluster_hits_df.to_csv(cluster_hits_csv, index=False, encoding="utf-8")

    config = {
        "backend": args.backend,
        "es_host": args.es_host,
        "es_index": args.es_index,
        "candidate_k": args.candidate_k,
        "image_top_k": args.image_top_k,
        "cluster_top_k": args.cluster_top_k,
        "negative_text": args.negative_text,
        "temperature": args.temperature,
        "visualizations_dir": visualizations_dir,
        "ground_truth_json": gt_json_path,
        "ground_truth_csv": gt_csv_path,
        "indexed_images": len(indexed_image_ids),
        "aligned_metadata_rows": len(metadata_df),
        "structured_queries": len(structured_queries),
        "concept_queries": len(concept_queries),
        "total_queries": len(all_queries),
    }

    summary = {
        "config": config,
        "image_level_macro": image_df.groupby("benchmark_type")[["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]].mean().round(4).to_dict(orient="index"),
        "cluster_level_macro": cluster_df.groupby("benchmark_type")[["cluster_precision@10", "cluster_precision@20", "image_recall_from_clusters@10", "image_recall_from_clusters@20", "cluster_mrr"]].mean().round(4).to_dict(orient="index"),
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown_report(summary_md, config, image_rows, cluster_rows)
    query_report_md, query_report_html = write_query_browser_report(
        args.report_dir,
        image_df,
        cluster_df,
        gt_pairs_df,
        image_hits_df,
        cluster_hits_df,
        visualizations_dir,
    )

    print(f"Saved image-level metrics to {image_csv}")
    print(f"Saved cluster-level metrics to {cluster_csv}")
    print(f"Saved image-level hit analysis to {image_hits_csv}")
    print(f"Saved cluster hit analysis to {cluster_hits_csv}")
    print(f"Saved ground truth JSON to {gt_json_path}")
    print(f"Saved ground truth CSV to {gt_csv_path}")
    print(f"Saved summary to {summary_json}")
    print(f"Saved markdown report to {summary_md}")
    print(f"Saved query browser markdown to {query_report_md}")
    print(f"Saved query browser HTML to {query_report_html}")


if __name__ == "__main__":
    main()
