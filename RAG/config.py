"""
config.py
=========
集中管理项目的所有可配置参数，方便后续调整。

OpenAI 兼容接口配置：
- API_BASE
- API_KEY
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# 1) ChromaDB 相关
# ----------------------------------------------------------------------
# ChromaDB 持久化目录的绝对路径。
# 路径是相对于本文件所在的项目根目录解析的，避免因 cwd 不一致出错。
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(PROJECT_ROOT / "ChromaDB_data_populate" / "DataBase" / "data")

# 关键字过滤：自动列出 ChromaDB 中所有 collection 时，
# 只保留名字中含以下任一关键字的 collection。
# 留空列表表示不过滤，使用全部 collection。
COLLECTION_KEYWORDS = ["gossipcop", "fever", "claim"]

# ----------------------------------------------------------------------
# 2) Embedding 模型
# ----------------------------------------------------------------------
# 必须与建库时使用的 embedding 模型保持一致。
# 当前数据库向量维度 = 384，建库时使用 sentence-transformers/all-MiniLM-L6-v2。
# 这里用 Hugging Face 上的完整仓库名（带 sentence-transformers/ 前缀），
# 与 sentence-transformers 库官方一致。
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------
# 3) 检索相关
# ----------------------------------------------------------------------
TOP_K = 5  # 每个 collection 检索的 top-k，最终合并后取 TOP_K

# ----------------------------------------------------------------------
# 4) OpenAI 兼容接口配置
# ----------------------------------------------------------------------

OPENAI_API_KEY = ""  # 请在这里填写你的 API Key

OPENAI_API_BASE = "https://models.sjtu.edu.cn/api/v1/"

OPENAI_MODEL_NAME = "deepseek-chat"

# LLM 调用参数
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 512
LLM_TIMEOUT = 60  # 秒

# ----------------------------------------------------------------------
# 5) 标签规范化（label normalization）
# ----------------------------------------------------------------------
# 把数据库中五花八门的 label 字符串 / 数字统一归并为 real / fake / nei。
# 用户规则：
#   - GossipCop: 0/fake/false -> fake ; 1/real/true -> real
#   - FEVER:     SUPPORTS/SUPPORTED -> real ; REFUTES/REFUTED -> fake
#                NOT ENOUGH INFO / NEI / unverified -> nei
# 规则匹配在 NORMALIZE_RULES 中以 (keywords, target) 的形式列出。
NORMALIZE_RULES = {
    "real": {"1", "real", "true", "supported", "supports", "verifiable", "verifies"},
    "fake": {"0", "fake", "false", "refuted", "refutes", "unverifiable"},
    "nei":  {"nei", "not enough info", "not_enough_info", "unverified"},
}

# ----------------------------------------------------------------------
# 6) 调试 / 兜底
# ----------------------------------------------------------------------
# 如果没有配置 API_KEY，是否允许用「证据多数投票」做兜底分类。
# 设为 True 时，无 LLM 也能返回一个 label，但 confidence 会偏低。
FALLBACK_MAJORITY_VOTE = False
