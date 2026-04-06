# import cv2
# from matplotlib import patches
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as T
# from sklearn.decomposition import PCA
# import torchvision.ops as ops
# 
# class RADSegSCGASearcher:
#     def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
#         self.device = device
# 
#         # Load official model via torch.hub. predict=False returns feature maps directly.
#         self.radseg = torch.hub.load(
#             'RADSeg-OVSS/RADSeg', 'radseg_encoder',
#             model_version=model_version, lang_model=lang_model, device=self.device, predict=False
#         )
#         self.radseg.model.eval()
# 
#     def compute_pca(self, feature_map):
#         B, C, H, W = feature_map.shape
#         feat_flat = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
#         pca = PCA(n_components=3)
#         pca_feat = pca.fit_transform(feat_flat)
#         pca_feat = (pca_feat - pca_feat.min(0)) / (pca_feat.max(0) - pca_feat.min(0) + 1e-8)
#         return pca_feat.reshape(H, W, 3)
# 
#     def spherical_kmeans(self, features, num_clusters=8, num_iters=100, tol=1e-4):
#         """
#         Spherical K-Means clustering using cosine similarity.
#         Args:
#             features: Tensor of shape (N, D), N is pixel count, D is feature dim.
#             num_clusters: Number of cluster centers (K).
#             num_iters: Max iterations.
#             tol: Convergence tolerance.
#         Returns:
#             centers: (K, D) cluster center features (L2 normalized).
#             labels: (N,) labels assigning each pixel to a center.
#         """
#         # L2 normalize inputs (project to hypersphere)
#         x = F.normalize(features, p=2, dim=-1)
#         N, D = x.shape
# 
#         # 1. Randomly init K centers from existing features
#         indices = torch.randperm(N, device=x.device)[:num_clusters]
#         centers = x[indices]
# 
#         for i in range(num_iters):
#             # 2. Compute cosine similarity between all pixels and centers
#             # x: (N, D), centers.T: (D, K) -> sim: (N, K)
#             sim = torch.matmul(x, centers.transpose(0, 1))
# 
#             # 3. Assign pixels to the most similar center
#             labels = torch.argmax(sim, dim=-1)
# 
#             # 4. Update centers: sum features of pixels in the same cluster
#             new_centers = torch.zeros_like(centers)
#             new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
# 
#             # 5. Re-apply L2 normalization
#             new_centers = F.normalize(new_centers, p=2, dim=-1)
# 
#             # Check convergence: cosine similarity between old and new centers
#             center_shift = (centers * new_centers).sum(dim=-1).mean()
#             centers = new_centers
#             if 1.0 - center_shift < tol:
#                 print(f"[Spherical K-Means] Converged at iteration {i + 1}")
#                 break
# 
#         return centers, labels
# 
#     @torch.no_grad()
#     def get_clustered_image_representation(self, image_path, num_clusters=8):
#         """
#         Helper: Extracts dense features and clusters them into core representations for image search.
#         """
#         img = Image.open(image_path).convert('RGB')
#         transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         img_tensor = transform(img).unsqueeze(0).to(self.device)
# 
#         # 1. Extract dense SCGA features
#         scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
# 
#         # 2. Map to SigLIP2 semantic space via alignment adapter
#         visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)
# 
#         # 3. Flatten features
#         B, C, H_f, W_f = visual_aligned.shape
#         flat_features = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
#         flat_features = F.normalize(flat_features, dim=-1)  # (H*W, C)
# 
#         print(f"Original dense features shape: {flat_features.shape}")
# 
#         # 4. Run spherical K-Means clustering
#         centers, labels = self.spherical_kmeans(flat_features, num_clusters=num_clusters)
#         print(f"Clustered centers shape: {centers.shape}")
# 
#         return centers, labels, (H_f, W_f)
# 
#     @torch.no_grad()
#     def run_search(self, image_path, query_text, negative_text="background", top_k=10, temperature=50.0):
#         """
#         Core update: Introduces negative_text and Softmax contrastive logic.
#         """
#         img = Image.open(image_path).convert('RGB')
# 
#         transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         img_tensor = transform(img).unsqueeze(0).to(self.device)
# 
#         scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
#         pca_rgb = self.compute_pca(scga_feat)
# 
#         visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)
# 
#         B, C, H_f, W_f = visual_aligned.shape
#         flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
#         flat = F.normalize(flat, dim=-1)  # (H*W, C)
# 
#         # 1. Encode both query and negative text
#         text_vecs = self.radseg.encode_prompts([query_text, negative_text], onehot=False)  # (2, C)
# 
#         # 2. Compute cosine similarity between image features and texts
#         logits = flat @ text_vecs.T  # Shape: (H*W, 2)
# 
#         # 3. Scale by temperature and apply Softmax to get relative probabilities
#         probs = F.softmax(logits * temperature, dim=-1)  # (H*W, 2)
# 
#         # 4. Extract probability of the target class (column 0) for the heatmap
#         target_prob = probs[:, 0]
#         heatmap = target_prob.reshape(H_f, W_f).cpu().numpy()
# 
#         print(f"Target Probability Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
#         self._visualize(img, heatmap, pca_rgb, query_text, top_k)
# 
#     def _visualize(self, img, heatmap, pca_rgb, query, top_k):
#         h_orig, w_orig = img.size[1], img.size[0]
# 
#         heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         heatmap_res = F.interpolate(heatmap_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().numpy()
# 
#         pca_tensor = torch.tensor(pca_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
#         pca_res = F.interpolate(pca_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().permute(1, 2, 0).numpy()
# 
#         fig, axes = plt.subplots(1, 4, figsize=(32, 8), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})
# 
#         axes[0].imshow(img)
#         axes[0].set_title("1. Original Image", fontsize=15, fontweight='bold')
#         axes[0].axis('off')
# 
#         axes[1].imshow(pca_res)
#         axes[1].set_title("2. SCGA PCA Features", fontsize=15)
#         axes[1].axis('off')
# 
#         axes[2].imshow(img.convert('L'), cmap='gray')
#         im_heat = axes[2].imshow(heatmap_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
#         axes[2].set_title(f"3. Prob Heatmap: '{query}'", fontsize=15, fontweight='bold')
#         axes[2].axis('off')
# 
#         cbar = fig.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
#         cbar.set_label('Target Probability (Softmax)', fontsize=12)
# 
#         img_dark = np.array(img).astype(np.float32) * 0.7 / 255.0
#         axes[3].imshow(img_dark)
#         axes[3].set_title(f"4. NMS Top {top_k} BBox", fontsize=15, fontweight='bold', color='darkgreen')
#         axes[3].axis('off')
# 
#         min_area = (h_orig * w_orig) * 0.005
#         min_val, max_val = heatmap_res.min(), heatmap_res.max()
# 
#         boxes = []
#         scores = []
# 
#         for thresh_ratio in np.linspace(0.2, 0.9, 8):
#             thresh = min_val + thresh_ratio * (max_val - min_val)
#             binary_mask = (heatmap_res > thresh).astype(np.uint8) * 255
#             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
#             for cnt in contours:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if w * h > min_area:
#                     score = heatmap_res[y:y + h, x:x + w].max()
#                     boxes.append([x, y, x + w, y + h])
#                     scores.append(score)
# 
#         if len(boxes) > 0:
#             boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#             scores_tensor = torch.tensor(scores, dtype=torch.float32)
# 
#             keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.2)
# 
#             top_k_indices = keep_indices[:top_k]
# 
#             import matplotlib.cm as cm
#             cmap = cm.get_cmap('autumn')
# 
#             for render_idx, idx in enumerate(reversed(top_k_indices)):
#                 rank = len(top_k_indices) - 1 - render_idx
# 
#                 x1, y1, x2, y2 = boxes[idx]
#                 w, h = x2 - x1, y2 - y1
#                 score = scores[idx]
# 
#                 color_ratio = rank / max(1, top_k - 1)
#                 color = cmap(color_ratio)
#                 line_w = max(1.5, 4.0 - rank * 0.3)
# 
#                 rect = patches.Rectangle((x1, y1), w, h, linewidth=line_w, edgecolor=color, facecolor='none')
#                 axes[3].add_patch(rect)
# 
#                 label_text = f"#{rank + 1}: {score:.2f}"
#                 y_offset = max(5, y1 - 8 - (rank % 3) * 12)
# 
#                 axes[3].text(x1, y_offset, label_text, color='white', fontsize=10, fontweight='bold',
#                              bbox=dict(facecolor=color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))
#         else:
#             print("No matching regions found.")
# 
#         plt.tight_layout()
#         plt.show()
# 
# 
# # ==========================================
# # Example Usage
# # ==========================================
# if __name__ == "__main__":
#     searcher = RADSegSCGASearcher()
# 
# 
#     print("\n--- Running Text-to-Image Search ---")
# 
#     # searcher.run_search("football.png", query_text="soccer", negative_text="background", top_k=10,
#     #                     temperature=50.0)
#     searcher.run_search("football.png", query_text="kids", negative_text="background", top_k=10,
#                         temperature=50.0)
import time

import cv2
from matplotlib import patches
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
import torchvision.ops as ops


class RADSegSCGASearcher:
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
        self.device = device

        # 加载模型
        self.radseg = torch.hub.load(
            'RADSeg-OVSS/RADSeg', 'radseg_encoder',
            model_version=model_version, lang_model=lang_model, device=self.device, predict=False
        )
        if hasattr(self.radseg, 'model'):
            self.radseg.model.eval()
        else:
            self.radseg.eval()

    def spherical_kmeans(self, features, num_clusters=8, num_iters=100, tol=1e-4):
        x = F.normalize(features, p=2, dim=-1)
        N, D = x.shape
        indices = torch.randperm(N, device=x.device)[:num_clusters]
        centers = x[indices]

        for i in range(num_iters):
            sim = torch.matmul(x, centers.transpose(0, 1))
            labels = torch.argmax(sim, dim=-1)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
            new_centers = F.normalize(new_centers, p=2, dim=-1)

            center_shift = (centers * new_centers).sum(dim=-1).mean()
            centers = new_centers
            if 1.0 - center_shift < tol:
                print(f"[Spherical K-Means] Converged at iteration {i + 1}")
                break

        return centers, labels

    @torch.no_grad()
    def run_search_comparison(self, image_path, query_text, negative_text=None, top_k=10, temperature=50.0,
                              num_clusters=100, decouple_alpha=0.5):
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # 1. 提取 Dense 特征
        # scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        # visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        # 1. 提取特征（使用 feat_mlp）
        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        visual_aligned = self.radseg.align_spatial_features_with_language(
            scga_feat, onehot=False
        )

        B, C, H_f, W_f = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        dense_flat = F.normalize(dense_flat, dim=-1)

        # # ★ 新增：全局方向解耦
        # if decouple_alpha > 0:
        #     global_dir = F.normalize(dense_flat.mean(0, keepdim=True), dim=-1)  # (1, C)
        #     proj = (dense_flat @ global_dir.T) * global_dir  # 全局分量
        #     dense_flat = F.normalize(dense_flat - decouple_alpha * proj, dim=-1)  # 减去污染

        # 2. 提取 Cluster 特征 (聚类中心和每个像素的标签)
        centers, labels = self.spherical_kmeans(dense_flat, num_clusters=num_clusters)
        clustered_flat = centers[labels]  # 仅用于最后的 PCA 可视化还原

        # 3. 文本编码
        # text_vecs = self.radseg.encode_prompts([query_text, negative_text], onehot=False)
        # 3. 动态文本编码 (处理可选的 negative_text)
        prompts = [query_text]
        # if negative_text:
        #     prompts.append(negative_text)
        #
        #
        #
        # # 3. 动态文本编码 (完美支持多个负面词、单负面词或无负面词)
        # prompts = [query_text]

        if negative_text:
            # 如果传入的是字符串 (例如 "sky, grass, road")，按逗号切分成列表
            if isinstance(negative_text, str):
                neg_list = [n.strip() for n in negative_text.split(',') if n.strip()]
            # 如果传入的已经是列表 (例如 ["sky", "grass"])，直接使用
            elif isinstance(negative_text, list):
                neg_list = negative_text
            else:
                neg_list = ["background"]

            prompts.extend(neg_list)
        else:
            # 兜底机制：什么都没传，强制补一个假想敌，防止 Softmax 算出全图 1.0
            prompts.append("background")

        text_vecs = self.radseg.encode_prompts(prompts, onehot=False)

        # text_vecs = F.normalize(text_vecs, dim=-1)

        # ==================================================
        # 性能对决：分别测量 稠密搜索 vs 聚类搜索 的匹配耗时
        # ==================================================

        # --- A. 稠密搜索 (Dense Search) ---
        if self.device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()

        dense_logits = dense_flat @ text_vecs.T  # 矩阵乘法：(H*W, C) @ (C, 2)
        dense_probs = F.softmax(dense_logits * temperature, dim=-1)
        dense_heatmap = dense_probs[:, 0].reshape(H_f, W_f).cpu().numpy()

        if self.device == "cuda": torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) * 1000  # 转换为毫秒 (ms)

        # --- B. 聚类搜索 (Clustered Search) 性能优化版 ---
        if self.device == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 核心优化：只用 K 个聚类中心去跟文本算相似度！(K, C) @ (C, 2)
        center_logits = centers @ text_vecs.T
        center_probs = F.softmax(center_logits * temperature, dim=-1)
        # 根据 label 将 K 个得分映射回整张图的像素
        clustered_probs = center_probs[labels]
        clustered_heatmap = clustered_probs[:, 0].reshape(H_f, W_f).cpu().numpy()

        if self.device == "cuda": torch.cuda.synchronize()
        cluster_time = (time.perf_counter() - t1) * 1000  # 转换为毫秒 (ms)

        print("-" * 50)
        print(f"[Dense]   Search Time: {dense_time:.3f} ms | Max Prob: {dense_heatmap.max():.4f}")
        print(f"[Cluster] Search Time: {cluster_time:.3f} ms | Max Prob: {clustered_heatmap.max():.4f}")
        print(f"-> Speedup: {dense_time / cluster_time:.2f}x faster!")
        print("-" * 50)

        # 4. 计算 PCA 用于可视化
        pca = PCA(n_components=3)
        dense_pca_flat = pca.fit_transform(dense_flat.cpu().numpy())
        dense_pca_flat = (dense_pca_flat - dense_pca_flat.min(0)) / (
                dense_pca_flat.max(0) - dense_pca_flat.min(0) + 1e-8)
        dense_pca = dense_pca_flat.reshape(H_f, W_f, 3)

        clustered_pca_flat = pca.transform(clustered_flat.cpu().numpy())
        clustered_pca_flat = (clustered_pca_flat - clustered_pca_flat.min(0)) / (
                clustered_pca_flat.max(0) - clustered_pca_flat.min(0) + 1e-8)
        clustered_pca = clustered_pca_flat.reshape(H_f, W_f, 3)

        self._visualize_comparison(img, dense_heatmap, clustered_heatmap, dense_pca, clustered_pca, query_text, top_k,
                                   dense_time, cluster_time)

    def _draw_bboxes_on_ax(self, ax, heatmap_res, top_k, h_orig, w_orig):
        """
        在指定的 matplotlib 轴 (ax) 上提取并绘制 Top-K Bounding Boxes
        """
        min_area = (h_orig * w_orig) * 0.005
        min_val, max_val = heatmap_res.min(), heatmap_res.max()

        boxes = []
        scores = []

        # 如果整张图的得分完全一致（聚类时可能发生全背景的情况），直接跳过
        if max_val - min_val < 1e-4:
            return

        for thresh_ratio in np.linspace(0.2, 0.9, 8):
            thresh = min_val + thresh_ratio * (max_val - min_val)
            binary_mask = (heatmap_res > thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > min_area:
                    score = heatmap_res[y:y + h, x:x + w].max()
                    boxes.append([x, y, x + w, y + h])
                    scores.append(score)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.2)
            top_k_indices = keep_indices[:top_k]

            import matplotlib.cm as cm
            cmap = cm.get_cmap('autumn')

            for render_idx, idx in enumerate(reversed(top_k_indices)):
                rank = len(top_k_indices) - 1 - render_idx
                x1, y1, x2, y2 = boxes[idx]
                w, h = x2 - x1, y2 - y1
                score = scores[idx]

                color_ratio = rank / max(1, top_k - 1)
                color = cmap(color_ratio)
                line_w = max(1.5, 4.0 - rank * 0.3)

                rect = patches.Rectangle((x1, y1), w, h, linewidth=line_w, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                label_text = f"#{rank + 1}: {score:.2f}"
                y_offset = max(5, y1 - 8 - (rank % 3) * 12)

                ax.text(x1, y_offset, label_text, color='white', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

    def _visualize_comparison(self, img, dense_heatmap, cluster_heatmap, dense_pca, cluster_pca, query, top_k,
                              dense_time, cluster_time):
        h_orig, w_orig = img.size[1], img.size[0]

        dense_res = F.interpolate(torch.tensor(dense_heatmap).unsqueeze(0).unsqueeze(0), size=(h_orig, w_orig),
                                  mode='bilinear').squeeze().numpy()
        cluster_res = F.interpolate(torch.tensor(cluster_heatmap).unsqueeze(0).unsqueeze(0), size=(h_orig, w_orig),
                                    mode='nearest').squeeze().numpy()

        dense_pca_res = F.interpolate(torch.tensor(dense_pca).permute(2, 0, 1).unsqueeze(0), size=(h_orig, w_orig),
                                      mode='bilinear').squeeze().permute(1, 2, 0).numpy()
        cluster_pca_res = F.interpolate(torch.tensor(cluster_pca).permute(2, 0, 1).unsqueeze(0), size=(h_orig, w_orig),
                                        mode='nearest').squeeze().permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(2, 4, figsize=(32, 16), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})
        img_dark = np.array(img).astype(np.float32) * 0.7 / 255.0

        # ================= ROW 0: DENSE =================
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Dense: Original Image", fontsize=16, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dense_pca_res)
        axes[0, 1].set_title("Dense: PCA Features", fontsize=16)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img.convert('L'), cmap='gray')
        im1 = axes[0, 2].imshow(dense_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
        # 加上耗时标签
        axes[0, 2].set_title(f"Dense Search: '{query}' ({dense_time:.2f} ms)", fontsize=16, fontweight='bold',
                             color='darkred')
        axes[0, 2].axis('off')
        fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04).set_label('Prob', fontsize=12)

        axes[0, 3].imshow(img_dark)
        self._draw_bboxes_on_ax(axes[0, 3], dense_res, top_k, h_orig, w_orig)
        axes[0, 3].set_title("Dense: Top BBoxes", fontsize=16, fontweight='bold')
        axes[0, 3].axis('off')

        # ================= ROW 1: CLUSTERED =================
        axes[1, 0].imshow(img)
        axes[1, 0].set_title(
            f"Clustered (K={cluster_heatmap.shape[0] if len(cluster_heatmap.shape) == 1 else 'Fixed'}): Original Image",
            fontsize=16, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cluster_pca_res)
        axes[1, 1].set_title("Clustered: PCA Features", fontsize=16)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(img.convert('L'), cmap='gray')
        im2 = axes[1, 2].imshow(cluster_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
        # 加上耗时标签，并标红凸显速度
        axes[1, 2].set_title(f"Cluster Search: '{query}' ({cluster_time:.2f} ms)", fontsize=16, fontweight='bold',
                             color='darkgreen')
        axes[1, 2].axis('off')
        fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04).set_label('Prob', fontsize=12)

        axes[1, 3].imshow(img_dark)
        self._draw_bboxes_on_ax(axes[1, 3], cluster_res, top_k, h_orig, w_orig)
        axes[1, 3].set_title("Clustered: Top BBoxes", fontsize=16, fontweight='bold')
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.show()

    def compute_pca(self, feature_map):
        B, C, H, W = feature_map.shape
        feat_flat = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        pca = PCA(n_components=3)
        pca_feat = pca.fit_transform(feat_flat)
        pca_feat = (pca_feat - pca_feat.min(0)) / (pca_feat.max(0) - pca_feat.min(0) + 1e-8)
        return pca_feat.reshape(H, W, 3)

    @torch.no_grad()
    def run_search(self, image_path, query_text, top_k=10):
        img = Image.open(image_path).convert('RGB')

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        pca_rgb = self.compute_pca(scga_feat)

        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        B, C, H_f, W_f = visual_aligned.shape
        flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        flat = F.normalize(flat, dim=-1)

        text_vec = self.radseg.encode_prompts([query_text], onehot=False)

        similarity = (flat @ text_vec.T).squeeze()
        heatmap = similarity.reshape(H_f, W_f).cpu().numpy()

        self._visualize(img, heatmap, pca_rgb, query_text, top_k)

    def _visualize(self, img, heatmap, pca_rgb, query, top_k):
        h_orig, w_orig = img.size[1], img.size[0]

        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_res = F.interpolate(heatmap_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().numpy()

        pca_tensor = torch.tensor(pca_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        pca_res = F.interpolate(pca_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(32, 8), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})

        axes[0].imshow(img)
        axes[0].set_title("1. Original Image", fontsize=15, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pca_res)
        axes[1].set_title("2. SCGA PCA Features", fontsize=15)
        axes[1].axis('off')

        axes[2].imshow(img.convert('L'), cmap='gray')
        im_heat = axes[2].imshow(heatmap_res, cmap='Reds', alpha=0.6)
        axes[2].set_title(f"3. Search Result: '{query}'", fontsize=15, fontweight='bold')
        axes[2].axis('off')

        cbar = fig.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity Score', fontsize=12)

        axes[3].imshow(img)
        axes[3].set_title(f"4. Raw Top {top_k} BBox", fontsize=15, fontweight='bold', color='darkgreen')
        axes[3].axis('off')

        # ==========================================
        # 核心逻辑：多级切片 + 纯暴力得分降序提取 Top-K
        # ==========================================
        min_area = (h_orig * w_orig) * 0.005
        min_val, max_val = heatmap_res.min(), heatmap_res.max()

        boxes = []
        scores = []

        # 在不同的相似度水位线上切片提取轮廓
        for thresh_ratio in np.linspace(0.2, 0.9, 8):
            thresh = min_val + thresh_ratio * (max_val - min_val)
            binary_mask = (heatmap_res > thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > min_area:
                    score = heatmap_res[y:y + h, x:x + w].max()
                    boxes.append([x, y, x + w, y + h])
                    scores.append(score)

        if len(boxes) > 0:
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # 暴力降序排列，完全不考虑重叠
            _, sorted_indices = torch.sort(scores_tensor, descending=True)
            top_k_indices = sorted_indices[:top_k]

            # 绘制最终结果
            for rank, idx in enumerate(top_k_indices):
                x1, y1, x2, y2 = boxes[idx]
                w, h = x2 - x1, y2 - y1
                score = scores[idx]

                color = 'lime' if rank < 3 else 'yellow'

                rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor=color, facecolor='none')
                axes[3].add_patch(rect)

                label_text = f"Top {rank + 1} ({score:.2f})"
                axes[3].text(x1, y1 - 5, label_text, color=color, fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
        else:
            print("No matching regions found.")

        plt.tight_layout()
        plt.show()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    searcher = RADSegSCGASearcher()

    print("\n--- Running Dense vs Clustered Comparison ---")

    # searcher.run_search_comparison("football.png", query_text="kids", negative_text="background",
    #                                top_k=10, temperature=80, num_clusters=100)

    # searcher.run_search("football.png", "kids")

    # Baseline（你现在的）
    # searcher.run_search_comparison("football.png", "kids", negative_text=None,
    #                                top_k=10, temperature=80, num_clusters=100,
    #                                decouple_alpha=0.0)
    searcher.run_search_comparison("football.png", "football", negative_text="background, environment, surroundings, people, context, other unrelated objects",
                                   top_k=10, temperature=80, num_clusters=100,
                                   decouple_alpha=0.0)

    searcher.run_search_comparison("football.png", "football", negative_text="background",
                                   top_k=10, temperature=80, num_clusters=100,
                                   decouple_alpha=0.0)

    # # 实验1：只加解耦
    # searcher.run_search_comparison("football.png", "football", negative_text="background",
    #                                decouple_alpha=0.5, use_feat_mlp=False)
    #
    # # 实验2：只换 feat_mlp
    # searcher.run_search_comparison("football.png", "football", negative_text="background",
    #                                decouple_alpha=0.0, use_feat_mlp=True)
    #
    # searcher.run_search_comparison("football.png", "football",
    #                                decouple_alpha=0.5, use_feat_mlp=True)
