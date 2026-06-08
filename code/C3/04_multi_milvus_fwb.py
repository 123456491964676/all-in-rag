import json
import os
from duckdb import query
from sklearn.feature_extraction import img_to_graph
import torch
import numpy as np
from pymilvus import MilvusClient, DataType
from visual_bge.visual_bge.modeling import Visualized_BGE
import cv2
from PIL import Image

# 我没有在hf下载这个模型，我在魔塔下载
# pip install modelscope
# modelscope download --model AI-ModelScope/bge-base-en-v1.5
# https://www.modelscope.cn/models/AI-ModelScope/bge-base-en-v1.5
# Visualized_base_en_v1.5.pth 文件也是在魔塔下载
MODEL_NAME = "/workspaces/all-in-rag/models/BAAI/bge-base-en-v1.5"
MODEL_PATH = "/workspaces/all-in-rag/models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "/workspaces/all-in-rag/data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URL = "http://localhost:19530"

# 将Visualized-BGE模型的加载和编码逻辑封装在一个Encoder类中，
# 并定义一个 visualize_results 函数用于后续的结果可视化
class Encoder:
    """编码器类，用于将图像和文本编码为向量"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval() # 把模型切换到评估模式。

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]

# 对 查询图像 进行检索，并把输出的 检索图像 创建成一个全景图用于可视化
def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,0,0))

    query_display_area[img_height * (row_count - 1):, : ] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:,:,::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_height - 4, img_width - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
     
    return np.hstack([query_display_area, panoramic_image])
