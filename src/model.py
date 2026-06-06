"""
本模块目前为 MOCK 实现，用于前端测试。
classify_statement() 随机返回 0 或 1，附带一个虚假的置信度。

TO DO:
    用真实模型替换 classify_statement() 的函数体。

    classify_statement(statement: str) -> dict:
        必须精确返回:
        {
            "is_rumor": int,      # 0 = 非谣言, 1 = 谣言
            "confidence": float,  # 0.0 到 1.0
        }
    实现完成后，前端 (src/app.py) 会自动调用此函数，
    并将其输出传递给 api_service 进行文本充实。
"""

import random
import time


def classify_statement(statement: str) -> dict[str, int | float]:
    """Mock classifier — randomly returns 0/1 with a fake confidence.

    Replace this with the real model. Keep the return shape identical.
    """
    time.sleep(0.5 + random.random() * 1.0)  # simulate inference delay

    is_rumor = random.randint(0, 1)
    confidence = round(random.uniform(0.55, 0.95), 2)

    return {
        "is_rumor": is_rumor,
        "confidence": confidence,
    }
