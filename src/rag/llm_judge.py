"""
llm_judge.py
============
调用 OpenAI 兼容接口做真假二分类 + 中文解释。

输入：用户输入文本 + 检索证据列表
输出：dict  { "label": "real"/"fake", "confidence": float, "reason": str }

支持的后端：
- 官方 OpenAI API
- 智谱 GLM（通过 OpenAI 兼容接口）
- 本地模型（如 llama.cpp、text-generation-webui）
- 其他 OpenAI 兼容服务

如果 LLM 输出不是合法 JSON，会尝试用正则修复；依然失败时，
根据 config.FALLBACK_MAJORITY_VOTE 决定是否使用多数投票兜底。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from src.rag import config


# ----------------------------------------------------------------------
# Prompt 构造
# ----------------------------------------------------------------------
SYSTEM_PROMPT_ZH = (
    "你是一名严谨的虚假信息鉴别助手。"
    "我会给你一段待鉴别的文本，以及从 GossipCop / FEVER 知识库中检索到的若干条证据。"
    "你需要结合证据内容判断该文本属于 \"real\"（真实信息）还是 \"fake\"（虚假/误导信息）。\n"
    "要求：\n"
    "1) 必须输出严格的 JSON，不要包含任何额外文字、代码块标记或解释。\n"
    "2) JSON 字段：\n"
    "   - \"label\": 仅取 \"real\" 或 \"fake\"；\n"
    "   - \"confidence\": 0 到 1 之间的小数，越大表示越确定；\n"
    "   - \"reason\": 简短中文判断依据，**必须显式引用至少 1 条证据的编号**（例如 \"证据 2\"）。\n"
    "3) 当证据不足或证据互相矛盾时，confidence 应低于 0.6，并在 reason 中说明 \"证据不足\"。\n"
    "4) 不要复述整段原文，只给结论和依据。"
)


def _build_user_prompt(query: str, evidence: List[Dict[str, Any]]) -> str:
    """把 query 和 evidence 拼成 user prompt。"""
    lines: List[str] = []
    lines.append("【待鉴别文本】")
    lines.append(query.strip())
    lines.append("")
    lines.append("【检索证据】")
    if not evidence:
        lines.append("（未检索到任何证据，证据不足。）")
    else:
        for ev in evidence:
            label_str = ev.get("label") or "未知"
            raw = ev.get("raw_label")
            raw_str = f" (原始标签={raw})" if raw is not None else ""
            src = ev.get("source", "")
            dist = ev.get("distance")
            dist_str = f"，距离={dist:.4f}" if isinstance(dist, (int, float)) else ""
            lines.append(
                f"证据 {ev.get('id', '?')} (来源: {src}{raw_str}；"
                f"归一化标签={label_str}{dist_str})："
            )
            lines.append(ev.get("text", "").strip() or "（无文本）")
            lines.append("")
    lines.append("请基于以上证据输出 JSON。")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# OpenAI 兼容接口调用
# ----------------------------------------------------------------------
def _call_openai(messages: List[Dict[str, str]]) -> str:
    """
    通过 OpenAI 兼容协议调用 LLM。
    优先用 openai>=1.0 的客户端；失败时用 requests 兜底。
    """
    last_err: Optional[Exception] = None

    # 方案 1：openai 官方 SDK
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE,
        )
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        last_err = e

    # 方案 2：requests 兜底（手动 HTTP）
    try:
        import requests  # type: ignore

        url = config.OPENAI_API_BASE.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.OPENAI_MODEL_NAME,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        last_err = e

    raise RuntimeError(
        "调用 OpenAI 兼容接口失败：openai SDK 与 requests 兜底均报错。\n"
        f"最后一次错误: {last_err}\n"
        "请检查：1) 是否在 .env 中设置了 API_SECRET；"
        "2) API / API_MODEL 是否正确；"
        "3) 网络是否可达。"
    )


# ----------------------------------------------------------------------
# JSON 解析 / 兜底
# ----------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """容忍 ```json ... ``` 包裹、夹杂前后说明文字的情况。"""
    if not text:
        return None
    text = text.strip()

    # 去掉 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 抽取第一个 { ... } 块再解析
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _majority_vote(evidence: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    """
    兜底：当 LLM 不可用 / 输出解析失败时，
    按 evidence 的 label 做加权投票（距离越小权重越大）。
    """
    weights: Counter = Counter()
    for ev in evidence:
        lab = ev.get("label")
        if lab not in ("real", "fake"):
            continue
        dist = ev.get("distance")
        if isinstance(dist, (int, float)) and dist > 0:
            w = 1.0 / (1.0 + dist)
        else:
            w = 1.0
        weights[lab] += w

    if not weights:
        # 全是 nei / 未知
        return "unavailable", 0.0, "证据不足，检索结果没有明确真假标签，无法判断。"

    label = weights.most_common(1)[0][0]
    total = sum(weights.values())
    conf = round(weights[label] / total, 3)
    reason = (
        f"未使用 LLM，采用证据多数投票（权重 ∝ 1/(1+distance)）兜底。"
        f"共 {len(evidence)} 条证据，其中 real 权重 {weights['real']:.2f}，"
        f"fake 权重 {weights['fake']:.2f}。证据不足时 confidence 自动降低。"
    )
    return label, conf, reason


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def judge(query: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    调用 OpenAI 兼容接口判断真假，返回固定 schema 的 dict。
    """
    user_prompt = _build_user_prompt(query, evidence)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ZH},
        {"role": "user", "content": user_prompt},
    ]

    raw_text = ""
    parsed: Optional[Dict[str, Any]] = None

    if config.OPENAI_API_KEY:
        try:
            raw_text = _call_openai(messages)
            parsed = _safe_parse_json(raw_text)
        except Exception as e:
            print(f"[warn] 调用 OpenAI 兼容接口出错，将走兜底: {e}")
            parsed = None
    else:
        print("[info] 未配置 API_SECRET，使用证据多数投票兜底。")

    if parsed:
        label = str(parsed.get("label", "")).strip().lower()
        if label not in ("real", "fake"):
            invalid_label = label
            label, conf, reason = _majority_vote(evidence)
            return {
                "label": label,
                "confidence": conf,
                "reason": reason + f"（LLM 输出 label 非法: {invalid_label!r}）",
                "_raw_llm": raw_text,
            }
        try:
            conf = float(parsed.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        reason = str(parsed.get("reason", "")).strip() or "（LLM 未给出 reason）"
        return {
            "label": label,
            "confidence": round(conf, 3),
            "reason": reason,
            "_raw_llm": raw_text,
        }

    # 解析失败 / 未配置 LLM
    if config.FALLBACK_MAJORITY_VOTE:
        label, conf, reason = _majority_vote(evidence)
        if config.OPENAI_API_KEY:
            reason = "LLM 输出无法解析为 JSON，" + reason
        return {
            "label": label,
            "confidence": conf,
            "reason": reason,
            "_raw_llm": raw_text,
        }

    # 完全没有任何兜底
    return {
        "label": "unavailable",
        "confidence": 0.0,
        "reason": "LLM 不可用且未开启兜底，无法判断。",
        "_raw_llm": raw_text,
    }


# ----------------------------------------------------------------------
# 命令行调试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from src.rag.retriever import ChromaRetriever

    print("加载 ChromaDB ...")
    r = ChromaRetriever()
    print("collections:", r.list_collections())

    q = input("\n请输入待鉴别文本：\n> ").strip()
    if not q:
        print("输入为空，退出。")
    else:
        evs = r.retrieve(q)
        print(f"\n检索到 {len(evs)} 条证据：")
        for ev in evs:
            print(f"  [{ev['id']}] {ev['source']} label={ev['label']} dist={ev['distance']}")

        result = judge(q, evs)
        print("\n判断结果：")
        print(json.dumps({k: v for k, v in result.items() if k != "_raw_llm"},
                         ensure_ascii=False, indent=2))
