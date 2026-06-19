"""
chroma_retriever.py
===================
封装 ChromaDB 检索逻辑：
  - 自动从 CHROMA_DB_PATH 加载 PersistentClient
  - 列出所有 collection，按 COLLECTION_KEYWORDS 过滤
  - 使用与建库一致的 embedding function
  - 返回 top-k 证据（document / label / source / distance）

兼容当前发布的 ChromaDB 数据库 schema（1.x）。
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

from src.rag import config

# ----------------------------------------------------------------------
# 设置离线模式，避免连接 huggingface.co 下载模型
# 模型应已缓存在本地：~/.cache/huggingface/ 或 ~/.cache/torch/
# ----------------------------------------------------------------------
if config.HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
if config.HF_HUB_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
if config.TRANSFORMERS_OFFLINE:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _normalize_label(raw: Any) -> Optional[str]:
    """
    把数据库里五花八门的 label 归一化成 "real" / "fake" / "nei" / None。
    规则参考 config.NORMALIZE_RULES。
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    for target, kws in config.NORMALIZE_RULES.items():
        if s in kws:
            return target
    return None


def _truncate(text: str, limit: int = 400) -> str:
    """证据文本过长时截断，方便阅读。"""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ----------------------------------------------------------------------
# 检索器
# ----------------------------------------------------------------------
class ChromaRetriever:
    """
    加载 ChromaDB 并提供 top-k 检索能力。
    兼容当前发布的 ChromaDB 数据库 schema（1.x）。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ):
        self.db_path = db_path or config.ensure_chroma_database()
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL_NAME
        # 关键字统一小写
        self.keywords = [k.lower() for k in (keywords if keywords is not None else config.COLLECTION_KEYWORDS)]

        if not os.path.isdir(self.db_path):
            raise FileNotFoundError(
                f"ChromaDB 路径不存在: {self.db_path}\n"
                f"请检查 config.CHROMA_DB_PATH 是否正确，或者先把数据库建好。"
            )

        print(f"[debug] chromadb 版本: {chromadb.__version__}")

        # 使用 PersistentClient 创建客户端
        # chromadb 0.4.x 和 1.x 都支持 PersistentClient
        # 关键：建库和查询必须用同一个版本，否则 HNSW 索引格式不兼容
        self.client = chromadb.PersistentClient(path=self.db_path)

        # 统一使用 DefaultEmbeddingFunction（与建库时一致，底层是 all-MiniLM-L6-v2）
        # chromadb 1.x 虽然可以不传 EF，但传了更可靠
        self.embedding_fn: Any = None
        try:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 无法初始化 DefaultEmbeddingFunction ({e})，"
                  f"将不传 EF，依赖数据库中保存的配置。")
            self.embedding_fn = None

        # 缓存 collection
        self._collection_cache: Dict[str, Collection] = {}
        self._target_collections: List[str] = self._discover_collections()

    # ------------------------------------------------------------------
    # 内部：发现 collection
    # ------------------------------------------------------------------
    def _discover_collections(self) -> List[str]:
        """
        列出数据库中所有 collection，按关键字过滤。
        找不到匹配 collection 时给出清晰报错。
        """
        try:
            all_cols = self.client.list_collections()
        except Exception as e:
            if "collections.topic" in str(e):
                raise RuntimeError(
                    "ChromaDB schema/client mismatch: this database no longer has "
                    "collections.topic. Install the unified environment with "
                    "`chromadb>=1.0,<2`, then retry."
                ) from e
            raise RuntimeError(f"列出 ChromaDB collection 失败: {e}") from e

        names: List[str] = []
        for c in all_cols:
            # chromadb 0.4 / 1.x list_collections() 返回对象有 .name
            name = getattr(c, "name", None)
            if not name and isinstance(c, dict):
                name = c.get("name")
            if name:
                names.append(name)

        if not names:
            raise RuntimeError(
                f"ChromaDB ({self.db_path}) 中没有任何 collection，"
                f"请先运行 populate 脚本把数据写入。"
            )

        print(f"[debug] 发现的 collections: {names}")

        if not self.keywords:
            return names

        picked: List[str] = []
        for n in names:
            lname = n.lower()
            for kw in self.keywords:
                if kw in lname:
                    picked.append(n)
                    break
        if not picked:
            raise RuntimeError(
                f"ChromaDB 中没有名字匹配关键字 {self.keywords} 的 collection；\n"
                f"实际存在的 collection: {names}\n"
                f"请调整 config.COLLECTION_KEYWORDS 或者确认数据库。"
            )
        return picked

    def list_collections(self) -> List[str]:
        """返回当前检索器最终使用的 collection 列表。"""
        return list(self._target_collections)

    # ------------------------------------------------------------------
    # 加载 / 缓存
    # ------------------------------------------------------------------
    def _get_collection(self, name: str) -> Collection:
        if name in self._collection_cache:
            return self._collection_cache[name]

        # 统一传 embedding_function（与建库时一致）
        try:
            if self.embedding_fn is not None:
                col = self.client.get_collection(name=name, embedding_function=self.embedding_fn)
            else:
                col = self.client.get_collection(name=name)
        except Exception as e:
            raise RuntimeError(f"获取 collection '{name}' 失败: {e}") from e

        # 检查是否为空
        try:
            count = col.count()
            print(f"[debug] collection '{name}' 包含 {count} 条记录")
        except Exception:
            count = None
            
        self._collection_cache[name] = col
        return col

    # ------------------------------------------------------------------
    # 主检索
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        在所有目标 collection 中检索 query，合并后按距离排序，
        取前 top_k 条作为证据。

        返回每条证据的字典结构：
            {
                "id":         str,    # 整条证据的全局编号，从 1 开始
                "source":     str,    # collection 名称
                "doc_id":     str,    # 数据库内部 id
                "text":       str,    # 文档内容
                "raw_label":  Any,    # 原始 label 元数据
                "label":      str,    # 归一化后的 real/fake/nei
                "distance":   float,  # 距离，越小越相似
                "metadata":   dict,   # 全部元数据
            }
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空。")

        k = top_k if top_k is not None else config.TOP_K

        all_hits: List[Dict[str, Any]] = []
        per_col_k = k  # 每个 collection 取 k 条，合并后再裁剪

        for col_name in self._target_collections:
            col = self._get_collection(col_name)
            try:
                result = col.query(
                    query_texts=[query],
                    n_results=per_col_k,
                )
            except Exception as e:
                # 单个 collection 检索失败不应该让整体崩溃
                print(f"[warn] collection '{col_name}' 检索失败: {e}")
                continue

            # result 形如:
            #   {"ids": [[...]], "documents": [[...]], "metadatas": [[...]],
            #    "distances": [[...]]}
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]

            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else None
                doc_id = ids[i] if i < len(ids) else ""

                # 优先从 metadata 读 label，没有就用空
                raw_label = meta.get("label") if isinstance(meta, dict) else None
                label = _normalize_label(raw_label)

                all_hits.append(
                    {
                        "source": col_name,
                        "doc_id": doc_id,
                        "text": doc or "",
                        "raw_label": raw_label,
                        "label": label,
                        "distance": dist,
                        "metadata": meta or {},
                    }
                )

        # 按距离升序排序，距离小的更相似
        all_hits.sort(
            key=lambda x: (x["distance"] if x["distance"] is not None else float("inf"))
        )
        top = all_hits[:k]

        # 加上全局编号
        for idx, hit in enumerate(top, start=1):
            hit["id"] = idx
            hit["text"] = _truncate(hit["text"])
        return top


# ----------------------------------------------------------------------
# 命令行调试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("正在加载 ChromaDB 检索器 ...")
    r = ChromaRetriever()
    print("使用的 collection:", r.list_collections())

    q = input("\n请输入一段要鉴别的文本：\n> ").strip()
    if not q:
        print("输入为空，退出。")
    else:
        hits = r.retrieve(q, top_k=config.TOP_K)
        print(f"\n共检索到 {len(hits)} 条证据：\n")
        for h in hits:
            print(f"[{h['id']}] source={h['source']}  "
                  f"label={h['label']} (raw={h['raw_label']})  "
                  f"distance={h['distance']}")
            print(f"    text: {h['text']}\n")
