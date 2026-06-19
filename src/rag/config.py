"""
config.py
=========
集中管理项目的所有可配置参数，方便后续调整。

OpenAI 兼容接口配置：
- API_BASE
- API_KEY
"""

from pathlib import Path
import os
import shutil
import tempfile
import zipfile

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# ----------------------------------------------------------------------
# 1) ChromaDB 相关
# ----------------------------------------------------------------------
# ChromaDB 持久化目录的绝对路径。
# 默认从仓库根目录下的 datasets/ 目录解析，避免因 cwd 不一致出错。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DATA_ROOT = _project_path(os.getenv("RAG_DATA_ROOT", "datasets"))
CHROMA_DB_PATH = os.getenv(
    "RAG_CHROMA_DB_PATH",
    str(RAG_DATA_ROOT / "ChromaDB_data_populate" / "DataBase" / "data"),
)
CHROMA_DB_PATH = str(_project_path(CHROMA_DB_PATH))
CHROMA_HF_REPO_ID = os.getenv(
    "RAG_CHROMA_HF_REPO_ID",
    "MingchenDai/NIS4307-ChromaDB_data_populate",
)
CHROMA_HF_REPO_TYPE = os.getenv("RAG_CHROMA_HF_REPO_TYPE", "dataset")
CHROMA_HF_REVISION = os.getenv("RAG_CHROMA_HF_REVISION", "main")
CHROMA_HF_ARCHIVE = os.getenv(
    "RAG_CHROMA_HF_ARCHIVE",
    "ChromaDB_data_populate.zip",
)


def is_chroma_database(path: str | Path) -> bool:
    return (Path(path) / "chroma.sqlite3").is_file()


def _safe_extract_zip(archive_path: str | Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"Unsafe path in ChromaDB archive: {member.filename}")
        archive.extractall(destination)


def ensure_chroma_database(db_path: str | Path | None = None) -> str:
    """Return a local ChromaDB path, downloading and extracting it when absent."""
    resolved_path = Path(db_path or CHROMA_DB_PATH).expanduser()
    if is_chroma_database(resolved_path):
        return str(resolved_path)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    download_args = {
        "repo_id": CHROMA_HF_REPO_ID,
        "filename": CHROMA_HF_ARCHIVE,
        "repo_type": CHROMA_HF_REPO_TYPE,
        "revision": CHROMA_HF_REVISION,
    }
    try:
        archive_path = hf_hub_download(**download_args, local_files_only=True)
    except LocalEntryNotFoundError:
        archive_path = hf_hub_download(**download_args)

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="chromadb-extract-",
        dir=resolved_path.parent,
    ) as temporary_dir:
        extraction_root = Path(temporary_dir)
        _safe_extract_zip(archive_path, extraction_root)
        candidates = list(extraction_root.rglob("chroma.sqlite3"))
        if len(candidates) != 1:
            raise ValueError(
                "ChromaDB archive must contain exactly one chroma.sqlite3 file; "
                f"found {len(candidates)}"
            )
        extracted_db = candidates[0].parent
        if resolved_path.exists():
            if is_chroma_database(resolved_path):
                return str(resolved_path)
            raise FileExistsError(
                f"Cannot install ChromaDB over incomplete directory: {resolved_path}"
            )
        shutil.move(str(extracted_db), str(resolved_path))

    if not is_chroma_database(resolved_path):
        raise FileNotFoundError(f"Extracted ChromaDB is incomplete at {resolved_path}")
    return str(resolved_path)

# 关键字过滤：自动列出 ChromaDB 中所有 collection 时，
# 只保留名字中含以下任一关键字的 collection。
# 留空列表表示不过滤，使用全部 collection。
COLLECTION_KEYWORDS = [
    item.strip().lower()
    for item in os.getenv("RAG_COLLECTION_KEYWORDS", "gossipcop,fever,claim").split(",")
    if item.strip()
]

# ----------------------------------------------------------------------
# 2) Embedding 模型
# ----------------------------------------------------------------------
# 必须与建库时使用的 embedding 模型保持一致。
# 当前数据库向量维度 = 384，建库时使用 sentence-transformers/all-MiniLM-L6-v2。
# 这里用 Hugging Face 上的完整仓库名（带 sentence-transformers/ 前缀），
# 与 sentence-transformers 库官方一致。
EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
HF_HUB_OFFLINE = _bool_env("RAG_HF_HUB_OFFLINE", True)
TRANSFORMERS_OFFLINE = _bool_env("RAG_TRANSFORMERS_OFFLINE", True)

# ----------------------------------------------------------------------
# 3) 检索相关
# ----------------------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # 每个 collection 检索的 top-k，最终合并后取 TOP_K

# ----------------------------------------------------------------------
# 4) OpenAI 兼容接口配置
# ----------------------------------------------------------------------

# RAG standalone judging intentionally reuses the main LLM settings.
OPENAI_API_KEY = os.getenv("API_SECRET", "")

OPENAI_API_BASE = os.getenv("API", "https://models.sjtu.edu.cn/api/v1/")

OPENAI_MODEL_NAME = os.getenv("API_MODEL", "deepseek-reasoner")

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
FALLBACK_MAJORITY_VOTE = _bool_env("RAG_FALLBACK_MAJORITY_VOTE", False)
