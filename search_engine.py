import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import os
from PIL import Image
import torchvision.transforms as T


# ============================================================
# 配置
# ============================================================
ES_HOST = "http://localhost:9200"
INDEX_NAME = "radseg_features"


def load_radseg():
    """加载 RADSeg 模型"""
    print("正在加载 RADSeg 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(
        'RADSeg-OVSS/RADSeg', 'radseg_encoder',
        model_version="c-radio_v4-h", lang_model="siglip2-g",
        device=device, predict=False
    )
    model.eval()
    print(f"模型加载完成 (设备: {device})")
    return model, device


def encode_text_query(model, text):
    """
    将文字查询编码为 1152 维向量。
    
    RADSeg 的正确接口是 encode_prompts()，不是 encode_text()。
    encode_prompts 内部会：
      1. tokenizer 分词
      2. lang_adaptor.encode_text 编码
      3. L2 归一化
    返回的向量和图像聚类向量在同一个语义空间中，可以直接用余弦相似度比较。
    """
    with torch.no_grad():
        # encode_prompts 接受 List[str]，返回 [N, 1152] 的归一化向量
        text_features = model.encode_prompts([text], onehot=False)
    # 取出第 0 个（我们只传了一句话），转成 Python list 给 ES 用
    return text_features[0].cpu().numpy().tolist()


def search_es(es, query_vector, top_k=10):
    """
    用一个 1152 维向量在 ES 中做 KNN 搜索。
    
    返回格式:
    [
        {"image_id": "001.jpg", "score": 0.92, "matched_cluster": 7,
         "cx": 0.3, "cy": 0.5, "bbox": [0.1, 0.2, 0.5, 0.8]},
        ...
    ]
    """
    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "cluster_vector",
            "query_vector": query_vector,
            "k": top_k * 20,
            "num_candidates": 500
        },
        source=["image_id", "cx", "cy", "bbox"],
        size=top_k * 20
    )

    # 按 image_id 聚合：每张图只保留得分最高的那个聚类
    best_per_image = {}
    for hit in response['hits']['hits']:
        src = hit['_source']
        img_id = src['image_id']
        score = hit['_score']
        cluster_idx = int(hit['_id'].rsplit('_', 1)[-1])

        if img_id not in best_per_image or score > best_per_image[img_id]['score']:
            best_per_image[img_id] = {
                "image_id": img_id,
                "score": score,
                "matched_cluster": cluster_idx,
                "cx": src.get("cx", 0.5),
                "cy": src.get("cy", 0.5),
                "bbox": src.get("bbox", [0, 0, 1, 1])
            }

    results = sorted(best_per_image.values(), key=lambda x: x['score'], reverse=True)
    return results[:top_k]


def text_search(model, es, query_text, top_k=10):
    """
    文搜图的完整流程：
    文字 → 向量 → ES KNN → Top K 结果
    """
    print(f"查询文字: \"{query_text}\"")
    
    # Step 1: 文字 → 向量
    query_vector = encode_text_query(model, query_text)
    print(f"文字向量维度: {len(query_vector)}")
    
    # Step 2: 向量 → ES 搜索
    results = search_es(es, query_vector, top_k=top_k)
    
    return results


def main():
    # 连接 ES
    es = Elasticsearch(
        hosts=[ES_HOST],
        verify_certs=False,
        request_timeout=30
    )
    try:
        info = es.info()
        print(f"已连接 Elasticsearch {info['version']['number']}")
    except Exception as e:
        print(f"ES 连接失败: {e}")
        return

    # 加载模型
    model, device = load_radseg()

    # 交互式搜索
    while True:
        print("\n" + "=" * 50)
        query = input("请输入搜索文字 (输入 'q' 退出): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue

        results = text_search(model, es, query, top_k=10)

        print(f"\n--- 搜索结果 (Top {len(results)}) ---")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['image_id']}  "
                  f"得分: {r['score']:.4f}  "
                  f"命中聚类: #{r['matched_cluster']}")


if __name__ == "__main__":
    main()
