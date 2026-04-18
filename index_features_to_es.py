import json
import logging
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Elasticsearch 连接配置
ES_HOST = "http://localhost:9200"
INDEX_NAME = "radseg_features"
DIMENSIONS = 1536 # RADSeg (siglip2-g) 提取的特征维度
BATCH_SIZE = 500  # 每次批量写入文档数量，可根据内存调整

def setup_elasticsearch():
    """初始化 Elasticsearch 连接并创建索引"""
    logger.info(f"正在尝试连接 Elasticsearch: {ES_HOST}...")

    # 针对 ES 8.x 的兼容性配置：
    # 1. verify_certs=False 忽略证书校验（解决很多环境下的 400 报错）
    es = Elasticsearch(
        hosts=[ES_HOST],
        verify_certs=False,
        request_timeout=30
    )
    
    try:
        # 获取基本信息来确认连接是否真的通了
        info = es.info()
        logger.info(f"成功连接到 Elasticsearch! 服务版本: {info['version']['number']}")
    except Exception as e:
        logger.error(f"连接失败: {e}")
        logger.info("提示: 请确认 Docker 容器状态为 'Up'，且 9200 端口未被占用。")
        raise ConnectionError(f"无法建立有效连接")

    # 定义索引的 mapping，告诉 ES 怎么存储我们的向量数据
    mapping = {
        "mappings": {
            "properties": {
                "image_id": {
                    "type": "keyword"
                },
                "cluster_vector": {
                    "type": "dense_vector",
                    "dims": DIMENSIONS,
                    "index": True,
                    "similarity": "cosine"
                },
                "cx": {"type": "float"},           # 聚类质心 X (0~1)
                "cy": {"type": "float"},           # 聚类质心 Y (0~1)
                "bbox": {"type": "float"}          # 边界框 [x1,y1,x2,y2] (0~1)
            }
        }
    }

    # 如果索引存在，先删除（谨慎操作，这里为了方便反复测试）
    if es.indices.exists(index=INDEX_NAME):
        logger.warning(f"索引 '{INDEX_NAME}' 已存在，正在删除并重建...")
        es.indices.delete(index=INDEX_NAME)
        
    logger.info(f"正在创建索引: {INDEX_NAME}...")
    es.indices.create(index=INDEX_NAME, body=mapping)
    logger.info("索引创建完成。")
    return es

def generate_actions(file_path):
    """
    流式读取 JSONL 文件，生成 ES 批量导入的 action。
    支持两种格式：
      - 新格式: clusters 是 [{"v": [...], "cx": 0.3, "cy": 0.5, "bbox": [...]}, ...]
      - 旧格式: clusters 是 [[0.1, 0.2, ...], ...]  (无位置信息)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                img_id = data.get("image_id")
                clusters = data.get("clusters", [])
                
                for cluster_idx, cluster in enumerate(clusters):
                    # 判断是新格式(dict)还是旧格式(list)
                    if isinstance(cluster, dict):
                        vector = cluster["v"]
                        cx = cluster.get("cx", 0.5)
                        cy = cluster.get("cy", 0.5)
                        bbox = cluster.get("bbox", [0, 0, 1, 1])
                    else:
                        vector = cluster
                        cx = 0.5
                        cy = 0.5
                        bbox = [0, 0, 1, 1]
                    
                    action = {
                        "_index": INDEX_NAME,
                        "_id": f"{img_id}_{cluster_idx}",
                        "_source": {
                            "image_id": img_id,
                            "cluster_vector": vector,
                            "cx": cx,
                            "cy": cy,
                            "bbox": bbox
                        }
                    }
                    yield action
            except json.JSONDecodeError:
                logger.error(f"跳过无效的 JSON 行")
                continue

def count_total_clusters(file_path):
    """仅仅是为了给进度条预估一个总量，如果文件太大嫌慢可以跳过这一步"""
    logger.info("计算总共需要导入的向量数量 (扫描文件行数)...")
    total_vectors = 0
    num_images = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                total_vectors += len(data.get("clusters", []))
                num_images += 1
            except json.JSONDecodeError:
                pass
    logger.info(f"发现 {num_images} 张图片，预计总计 {total_vectors} 个向量点。")
    return total_vectors

def bulk_index_features(file_path):
    """执行批量导入"""
    if not os.path.exists(file_path):
        logger.error(f"找不到特征文件: {file_path}")
        logger.error("请确认是否已经使用 FileZilla 将 my_features.jsonl 下载到本地 D:\\RADSeg 目录下！")
        return

    es = setup_elasticsearch()
    
    # 估算总量（主要用于显示好看的进度条）
    total_actions = count_total_clusters(file_path)
    
    logger.info("开始向 Elasticsearch 流式导入数据...")
    
    # 使用 streaming_bulk 能够处理任意大文件，它会边读边传，内存占用很低
    success_count = 0
    failed_count = 0
    
    # tqdm 用于显示漂亮的进度条
    progress_bar = tqdm(total=total_actions, desc="导入进度")
    
    try:
        # chunk_size 控制每次打包多少条发给 ES，适当调大可以提高速度
        for ok, action_result in helpers.streaming_bulk(
                client=es, 
                actions=generate_actions(file_path),
                chunk_size=BATCH_SIZE,
                max_retries=3,          # 遇到错误重试几次
                raise_on_error=False    # 这里设为 False，避免一条数据坏了导致全盘崩溃
            ):
            
            progress_bar.update(1)
            
            if ok:
                success_count += 1
            else:
                failed_count += 1
                logger.debug(f"导入失败的一个项目: {action_result}")
                
    except Exception as e:
        logger.error(f"导入过程中发生严重错误: {e}")
        
    finally:
        progress_bar.close()
        
    logger.info(f"导入完成！成功: {success_count} 条，失败: {failed_count} 条。")

if __name__ == "__main__":
    # 指向你在 D 盘根目录下的全量特征文件
    FEATURE_FILE = r"D:\test.jsonl" 
    bulk_index_features(FEATURE_FILE)
