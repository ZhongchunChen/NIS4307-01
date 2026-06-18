"""
main.py
=======
谣言 / 假新闻鉴别 CLI 入口。

流程：
  1) 从命令行 / stdin 读取一段待鉴别文本
  2) 调用 ChromaRetriever 在 GossipCop / FEVER 中检索 top-k 证据
  3) 调用 OpenAI 兼容接口生成真假二分类 + 中文判断依据
  4) 输出固定 JSON

用法：
  python -m src.rag.cli "Breaking news: the queen is dead."

  python -m src.rag.cli          # 进入交互式输入
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List

from src.rag import config
from src.rag.llm_judge import judge
from src.rag.retriever import ChromaRetriever


# ----------------------------------------------------------------------
# 输出拼装
# ----------------------------------------------------------------------
def _evidence_to_schema(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把内部 evidence 转换成对外的 schema 格式。"""
    out = []
    for ev in evidence:
        out.append(
            {
                "id": ev.get("id"),
                "source": ev.get("source"),
                "label": ev.get("raw_label"),  # 保留原始标签，方便排查
                "normalized_label": ev.get("label"),  # real/fake/nei
                "distance": (
                    round(ev["distance"], 4)
                    if isinstance(ev.get("distance"), (int, float))
                    else None
                ),
                "text": ev.get("text", ""),
            }
        )
    return out


def build_output(query: str, evidence: List[Dict[str, Any]], judge_result: Dict[str, Any]) -> Dict[str, Any]:
    """组装最终对外的固定 JSON 结构。"""
    return {
        "label": judge_result.get("label", "fake"),
        "confidence": judge_result.get("confidence", 0.0),
        "reason": judge_result.get("reason", ""),
        "evidence": _evidence_to_schema(evidence),
    }


# ----------------------------------------------------------------------
# 交互入口
# ----------------------------------------------------------------------
def run_once(query: str) -> Dict[str, Any]:
    """对单条 query 执行完整流程。"""
    print(f"\n[1/3] 加载 ChromaDB 检索器 (path={config.CHROMA_DB_PATH}) ...")
    try:
        retriever = ChromaRetriever()
    except Exception as e:
        msg = f"无法加载 ChromaDB: {e}"
        print(f"[error] {msg}")
        return {
            "label": "fake",
            "confidence": 0.0,
            "reason": msg + "；无法判断。",
            "evidence": [],
        }
    print(f"      使用的 collection: {retriever.list_collections()}")

    print("\n[2/3] 检索 top-k 证据 ...")
    try:
        evidence = retriever.retrieve(query, top_k=config.TOP_K)
    except Exception as e:
        traceback.print_exc()
        return {
            "label": "fake",
            "confidence": 0.0,
            "reason": f"检索过程出错: {e}；无法判断。",
            "evidence": [],
        }

    print(f"      检索到 {len(evidence)} 条证据：")
    for ev in evidence:
        print(
            f"        - 证据 {ev['id']} | source={ev['source']} | "
            f"label={ev['label']} (raw={ev['raw_label']}) | "
            f"distance={ev['distance']}"
        )

    print("\n[3/3] 调用 LLM 进行真假二分类 ...")
    judge_result = judge(query, evidence)
    return build_output(query, evidence, judge_result)


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) >= 2:
        # 命令行参数优先
        query = " ".join(argv[1:]).strip()
    else:
        try:
            query = input("请输入待鉴别的新闻 / 声明文本（输入 q 退出）：\n> ").strip()
        except EOFError:
            print("没有可读取的输入，退出。")
            return 0
    if not query:
        print("输入为空，退出。")
        return 0
    if query.lower() in ("q", "quit", "exit"):
        print("Bye.")
        return 0

    result = run_once(query)

    print("\n========== 最终判断结果（JSON） ==========")
    # ensure_ascii=False 让中文正常显示
    # 同时单独 json.dumps 一次以验证可被 json.loads 解析
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    try:
        json.loads(text)
        print("\n[ok] 输出可被 json.loads 正常解析。")
    except Exception as e:
        print(f"\n[error] 输出 JSON 无法被解析: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
