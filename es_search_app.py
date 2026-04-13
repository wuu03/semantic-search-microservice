import argparse
import torch
import warnings
from elasticsearch import Elasticsearch

class TextEncoder:
    def __init__(self, model_version="c-radio_v3-h", lang_model="siglip2-g", device="cpu"):
        self.device = device
        # Suppress typical torch hub warnings locally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"Loading Text Encoder ({lang_model}) onto {device}...")
            self.radseg = torch.hub.load(
                'RADSeg-OVSS/RADSeg', 'radseg_encoder',
                model_version=model_version, lang_model=lang_model, device=self.device, predict=False
            )
        
        if hasattr(self.radseg, 'model'):
            self.radseg.model.eval()
        else:
            self.radseg.eval()

    @torch.no_grad()
    def encode(self, text):
        # We pass only the query_text, skipping negative_text for simplicity.
        vec = self.radseg.encode_prompts([text], onehot=False)
        # Assuming the shape is (C, N) where N=1 prompt
        # We need to flatten to a standard 1D python list
        return vec[:, 0].cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch Text-to-Image Search")
    parser.add_argument("query", type=str, help="Text to search for")
    parser.add_argument("--index", type=str, default="radseg_images", help="ES index")
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h")
    
    args = parser.parse_args()

    # 1. Connect to ES
    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        print("Error: Could not connect to Elasticsearch at localhost:9200")
        exit(1)

    # 2. Encode text locally (CPU mostly fine)
    encoder = TextEncoder(model_version=args.model_version, device="cpu")
    print(f"Encoding query: '{args.query}'...")
    query_vector = encoder.encode(args.query)

    print("Executing ANN Search...")
    # 3. Retrieve kNN using ES Collapse to deduplicate image_ids
    # This means if an image has multiple matching clusters, we only retrieve it once!
    
    # NOTE: Elasticsearch collapse works well with standard search, and 8.x supports it with kNN!
    body = {
        "knn": {
            "field": "vector",
            "query_vector": query_vector,
            "k": 100,           # Find overall top-100 closest clusters across db
            "num_candidates": 1000
        },
        "_source": ["image_id", "cluster_id"],
        "collapse": {
            "field": "image_id" # Return only the top matching cluster per image
        },
        "size": 5               # Final output is top 5 unique images
    }

    try:
        response = es.search(index=args.index, body=body)
    except Exception as e:
        print(f"Elasticsearch query error: {e}")
        exit(1)

    # 4. Display Results
    hits = response["hits"]["hits"]
    if not hits:
        print("No matches found.")
    else:
        print(f"\n=== Top Images for '{args.query}' ===")
        for idx, hit in enumerate(hits):
            score = hit['_score']
            img_id = hit['_source']['image_id']
            matched_cluster = hit['_source']['cluster_id']
            print(f"{idx+1}. Image: {img_id} (Score: {score:.4f}, Best Cluster: #{matched_cluster})")

