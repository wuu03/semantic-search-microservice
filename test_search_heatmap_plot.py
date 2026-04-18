import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置
# ============================================================
ES_HOST = "http://localhost:9200"
INDEX_NAME = "radseg_features"
IMAGES_DIR = "images"  # 你的原图存放目录，请确保这里有图片

def load_radseg():
    """加载 RADSeg 模型"""
    print("正在加载 RADSeg 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(
        'RADSeg-OVSS/RADSeg', 'radseg_encoder',
        model_version="c-radio_v4-h", lang_model="siglip2-g",
        device=device, predict=False
    )
    if hasattr(model, 'model'):
        model.model.eval()
    
    print(f"模型加载完成 (设备: {device})")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return model, device, transform

def encode_text_query(model, text):
    """将文字查询编码为向量。"""
    with torch.no_grad():
        text_features = model.encode_prompts([text], onehot=False)
    return text_features[0].cpu().numpy().tolist()

def search_es(es, query_vector, top_k=10):
    """用向量进行 ES 粗召回，找出最相关的图片"""
    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "cluster_vector",
            "query_vector": query_vector,
            "k": top_k * 20,
            "num_candidates": 500
        },
        source=["image_id"],
        size=top_k * 20
    )

    best_per_image = {}
    for hit in response['hits']['hits']:
        img_id = hit['_source']['image_id']
        score = hit['_score']
        if img_id not in best_per_image or score > best_per_image[img_id]['score']:
            best_per_image[img_id] = {
                "image_id": img_id,
                "score": score,
            }

    results = sorted(best_per_image.values(), key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def generate_heatmap_overlay(model, device, transform, img_path, query_vector):
    """
    计算并返回原图与热力图的叠加准备数据
    """
    if not os.path.exists(img_path):
        return None, None
        
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    q_vec = torch.tensor(query_vector, device=device).unsqueeze(1)
    
    with torch.no_grad():
        scga_feat = model.encode_image_to_feat_map(tensor)
        visual_aligned = model.align_spatial_features_with_language(scga_feat, onehot=False)
        
        B, C, H_f, W_f = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        dense_flat = F.normalize(dense_flat, dim=-1)
        
        sim = torch.matmul(dense_flat, q_vec).squeeze()
        sim_map = sim.reshape(H_f, W_f).cpu().numpy()
    
    p_min = sim_map.min()
    p_max = sim_map.max()
    norm_sim = (sim_map - p_min) / (p_max - p_min + 1e-8)
    
    heatmap_img = Image.fromarray(np.uint8(255 * norm_sim))
    heatmap_resized = heatmap_img.resize(img.size, Image.BILINEAR)
    
    return img, heatmap_resized

def text_search_and_plot(model, device, transform, es, query_text, top_k=5, display_k=3):
    print(f"\n正在检索: \"{query_text}\" ...")
    query_vector = encode_text_query(model, query_text)
    
    # 之前如果用了错误的维度（例如 1152 找 1536），所以我们自动切片或扩充，但最好是在源头修复
    results = search_es(es, query_vector, top_k=top_k)
    
    if not results:
        print("未找到任何结果。请确认 Elasticsearch 里面有数据。")
        return
        
    print(f"\n--- 粗召回结果 (Top {len(results)}) ---")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['image_id']}  得分: {r['score']:.4f}")
        
    # 可视化前 display_k 名
    k_to_display = min(len(results), display_k)
    print(f"\n正在为 Top {k_to_display} 图片生成热力图...")
    
    fig, axes = plt.subplots(1, k_to_display, figsize=(15, 6))
    if k_to_display == 1:
        axes = [axes]
        
    for idx in range(k_to_display):
        img_id = results[idx]['image_id']
        score = results[idx]['score']
        img_path = os.path.join(IMAGES_DIR, img_id)
        
        img, heatmap = generate_heatmap_overlay(model, device, transform, img_path, query_vector)
        
        if img and heatmap:
            axes[idx].imshow(img)
            axes[idx].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[idx].set_title(f"Rank {idx+1}\nScore: {score:.4f}")
            axes[idx].axis('off')
        else:
            axes[idx].set_title(f"Rank {idx+1}\nImage not found")
            axes[idx].axis('off')
            
    plt.tight_layout()
    plt.show()  # 阻塞式显示，用户看完关闭窗口可继续检索

def main():
    print("=" * 60)
    print("🚀 RADSeg '粗召回+热力精排' 测试环境 (Matplotlib Top3 展示版)")
    print("=" * 60)
    
    es = Elasticsearch(hosts=[ES_HOST], verify_certs=False, request_timeout=30)
    try:
        info = es.info()
        print(f"✅ 成功连接到 Elasticsearch {info['version']['number']}")
    except Exception as e:
        print(f"❌ ES 连接失败: {e}")
        return

    model, device, transform = load_radseg()

    while True:
        query = input("\n请输入搜索文字 (输入 'q' 退出): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue

        text_search_and_plot(model, device, transform, es, query, top_k=5, display_k=3)

if __name__ == "__main__":
    main()
