"""
构建 ChromaDB 数据库脚本
将 GossipCop、FEVER 和 train.csv 数据转换为可用于 RAG pipeline 的向量数据库

输出路径与 Retrieval-Augmented-Generation-for-news-main 项目一致，
生成的数据库可直接被 chroma_retriever.py 加载使用。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ==========================
# 1. 配置文件路径
# ==========================
GOSSIPCOP_SPLITS = {
    'MF': 'data/MF-00000-of-00001-2d256f82f8c8e2dd.parquet',
    'HF': 'data/HF-00000-of-00001-b7ad0013efd98ff4.parquet',
    'MR': 'data/MR-00000-of-00001-c9324d9fd00efb16.parquet',
    'HR': 'data/HR-00000-of-00001-043a35ac2a425b62.parquet',
}
GOSSIPCOP_HF_REPO = "hf://datasets/Jinyan1/GossipCop/"
FEVER_CLAIM_JSONL = "G:/rag/data/shared_task_dev.jsonl"  # FEVER Shared Task Development Dataset
TRAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

# ChromaDB 存储路径 —— 与 RAG 项目 config.py 中 CHROMA_DB_PATH 一致
CHROMA_DB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Retrieval-Augmented-Generation-for-news-main",
    "ChromaDB_data_populate",
    "DataBase",
    "data",
)

# ==========================
# 2. 初始化 Embedding 函数
# ==========================
# 使用 chromadb 内置的 DefaultEmbeddingFunction（底层 all-MiniLM-L6-v2），
# 与 chroma_retriever.py 中加载时使用的 EF 完全一致，确保向量兼容。
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# ==========================
# 3. 初始化 ChromaDB
# ==========================
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# 创建集合（先删除已有的，确保数据完整）
for name in ["gossipcop", "fever_claims", "train"]:
    try:
        client.delete_collection(name=name)
    except Exception:
        pass

gossipcop_collection = client.create_collection(
    name="gossipcop", embedding_function=embedding_fn
)
fever_claims_collection = client.create_collection(
    name="fever_claims", embedding_function=embedding_fn
)
train_collection = client.create_collection(
    name="train", embedding_function=embedding_fn
)

chroma_batch_size = 5000

# ==========================
# 4. 处理 GossipCop 数据
# ==========================
print("Processing GossipCop dataset...")
# 只保留 HF(人类虚假信息) 和 HR(人类真实信息)，删除 MF 和 MR
label_map = {'HF': 'fake', 'HR': 'real'}
dfs = []
for split_name, split_path in GOSSIPCOP_SPLITS.items():
    if split_name not in label_map:
        continue
    df_split = pd.read_parquet(GOSSIPCOP_HF_REPO + split_path)
    df_split['label'] = label_map[split_name]
    dfs.append(df_split)
df = pd.concat(dfs, ignore_index=True)

# 拼接 title + text
df['content'] = df['title'].fillna('') + ". " + df['text'].fillna('')
df = df[['content', 'label']]

# 添加到 ChromaDB（让 chromadb 内部自动调用 embedding_fn 生成向量）
for i in tqdm(range(0, len(df), chroma_batch_size), desc="Adding gossipcop to ChromaDB"):
    gossipcop_collection.add(
        documents=df['content'].tolist()[i:i+chroma_batch_size],
        metadatas=[{'label': l} for l in df['label'].tolist()[i:i+chroma_batch_size]],
        ids=[f"gossipcop_{j}" for j in range(i, min(i+chroma_batch_size, len(df)))]
    )

# ==========================
# 5. 处理 FEVER Claims 数据
# ==========================
print("Processing FEVER claims dataset...")
fever_docs = []
with open(FEVER_CLAIM_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        claim = item['claim']
        label = item['label']
        # 二分类映射，与 config.py NORMALIZE_RULES 一致
        if label == 'SUPPORTED':
            label = 'real'
        elif label == 'REFUTED':
            label = 'fake'
        else:
            label = 'nei'
        fever_docs.append({'content': claim, 'label': label})

# 添加到 ChromaDB（让 chromadb 内部自动调用 embedding_fn 生成向量）
for i in tqdm(range(0, len(fever_docs), chroma_batch_size), desc="Adding fever_claims to ChromaDB"):
    fever_claims_collection.add(
        documents=[d['content'] for d in fever_docs[i:i+chroma_batch_size]],
        metadatas=[{'label': d['label']} for d in fever_docs[i:i+chroma_batch_size]],
        ids=[f"fever_{j}" for j in range(i, min(i+chroma_batch_size, len(fever_docs)))]
    )

# ==========================
# 6. 处理 train.csv 数据
# ==========================
print("Processing train.csv dataset...")
df_train = pd.read_csv(TRAIN_CSV)

# text 列作为文档内容
df_train['content'] = df_train['text'].fillna('')

# label: 1 -> real, 0 -> fake
df_train['label_str'] = df_train['label'].map({1: 'real', 0: 'fake'})

# 添加到 ChromaDB（让 chromadb 内部自动调用 embedding_fn 生成向量）
for i in tqdm(range(0, len(df_train), chroma_batch_size), desc="Adding train to ChromaDB"):
    batch = df_train.iloc[i:i+chroma_batch_size]
    train_collection.add(
        documents=batch['content'].tolist(),
        metadatas=[{'label': l, 'event': str(e)} for l, e in zip(batch['label_str'], batch['event'])],
        ids=[f"train_{j}" for j in range(i, min(i+chroma_batch_size, len(df_train)))]
    )

# ==========================
# 7. 持久化数据库
# ==========================
print("ChromaDB database built successfully at:", CHROMA_DB_DIR)
