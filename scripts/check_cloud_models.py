from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def required(name: str) -> str:
    """读取必填环境变量；缺失时给出明确提示。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"请先在 .env 中填写 {name}")
    return value


def main() -> None:
    """依次验证文本、视觉和 Embedding 三类云端模型能力。"""
    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=required("OPENAI_API_KEY"),
        base_url=required("OPENAI_BASE_URL"),
    )
    text_model = required("TEXT_MODEL")
    vision_model = required("VISION_MODEL")
    embedding_model = required("EMBEDDING_MODEL")

    # 第一次调用验证文本模型和 OpenAI 兼容接口。
    text_result = client.chat.completions.create(
        model=text_model,
        messages=[{"role": "user", "content": "只回复 CLOUD_TEXT_OK"}],
        temperature=0,
    )
    print("text:", text_result.choices[0].message.content)

    # 第二次调用使用公开报告截图，验证视觉模型能读取图表。
    image_path = ROOT / "data" / "qa" / "amd_p3.png"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    vision_result = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "说出图中柱状图的三个年份，只输出年份。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            }
        ],
        temperature=0,
    )
    print("vision:", vision_result.choices[0].message.content)

    # 第三次调用检查向量条数和维度，维度会决定 Milvus Schema。
    embedding_result = client.embeddings.create(
        model=embedding_model,
        input=["AI data center infrastructure", "光模块与高速互连"],
    )
    vectors = embedding_result.data
    print("embedding_count:", len(vectors))
    print("embedding_dimension:", len(vectors[0].embedding))


if __name__ == "__main__":
    main()
