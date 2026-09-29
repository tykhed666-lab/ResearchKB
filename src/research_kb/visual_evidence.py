"""把视觉模型描述转换成可以写入Milvus的图像证据。"""

import json
from hashlib import sha256
from pathlib import Path

from research_kb.chunker import EvidenceChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"

DEFAULT_DESCRIPTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visual_descriptions.json"
)

# Milvus中text字段的最大长度为4096字节。
MAX_TEXT_BYTES = 4096


def build_visual_evidence_id(
    source: str,
    page_number: int,
) -> str:
    """根据PDF文件名和页码生成稳定的图像证据ID。

    同一份PDF的同一页始终得到相同ID，
    因此重复运行时不会重复入库。
    """
    raw_value = (
        f"{source}|{page_number}|image"
    )
    digest = sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"image_{digest}"


def build_visual_evidence_chunk(
    source: str,
    page_number: int,
    description: str,
) -> EvidenceChunk:
    """把一页视觉描述转换成图像证据。

    Args:
        source:
            PDF在data/raw中的保存文件名。
        page_number:
            用户看到的PDF物理页码，从1开始。
        description:
            视觉模型生成的页面描述。

    Returns:
        可以交给EvidenceIndexer入库的图像证据。

    Raises:
        ValueError:
            来源、页码、描述为空，或者描述超过
            Milvus text字段允许的长度。
    """
    cleaned_source = source.strip()
    cleaned_description = (
        description.strip()
    )

    if not cleaned_source:
        raise ValueError(
            "图像证据source不能为空"
        )

    if page_number <= 0:
        raise ValueError(
            "图像证据页码必须大于0"
        )

    if not cleaned_description:
        raise ValueError(
            "图像证据描述不能为空"
        )

    description_bytes = len(
        cleaned_description.encode(
            "utf-8"
        )
    )

    if (
        description_bytes
        > MAX_TEXT_BYTES
    ):
        raise ValueError(
            "图像证据描述超过"
            f"{MAX_TEXT_BYTES}字节"
        )

    return EvidenceChunk(
        evidence_id=(
            build_visual_evidence_id(
                source=cleaned_source,
                page_number=page_number,
            )
        ),
        source=cleaned_source,
        page_number=page_number,

        # 一页只生成一条综合视觉描述，
        # 因此页内序号固定为1。
        chunk_index=1,
        content_type="image",
        text=cleaned_description,
    )


def load_visual_evidence_chunks(
    json_path: str | Path = DEFAULT_DESCRIPTION_PATH,
) -> list[EvidenceChunk]:
    """读取视觉描述JSON并转换成EvidenceChunk。

    Args:
        json_path: 视觉描述JSON文件路径。

    Returns:
        content_type为image的证据列表。

    Raises:
        FileNotFoundError: JSON或页面图片不存在。
        ValueError: JSON结构或字段内容不正确。
    """
    path = Path(json_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到视觉描述文件：{path}"
        )

    try:
        records = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "视觉描述JSON格式不正确"
        ) from error

    if not isinstance(records, list):
        raise ValueError(
            "视觉描述JSON顶层必须是列表"
        )

    chunks: list[EvidenceChunk] = []
    resolved_image_root = IMAGE_ROOT.resolve()

    for record_index, record in enumerate(
        records,
        start=1,
    ):
        if not isinstance(record, dict):
            raise ValueError(
                f"第{record_index}条记录不是对象"
            )

        source = str(
            record.get("source", "")
        ).strip()

        description = str(
            record.get("description", "")
        ).strip()

        try:
            page_number = int(
                record.get("page_number")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"第{record_index}条记录的页码无效"
            ) from error

        image_value = str(
            record.get("image_path", "")
        ).strip()

        try:
            chunk = build_visual_evidence_chunk(
                source=source,
                page_number=page_number,
                description=description,
            )
        except ValueError as error:
            raise ValueError(
                f"第{record_index}条记录无效："
                f"{error}"
            ) from error

        if not image_value:
            raise ValueError(
                f"第{record_index}条记录缺少image_path"
            )

        image_path = (
            PROJECT_ROOT / image_value
        ).resolve()

        # 只允许引用data/images目录内的图片。
        try:
            image_path.relative_to(
                resolved_image_root
            )
        except ValueError as error:
            raise ValueError(
                f"图片路径不在data/images中："
                f"{image_path}"
            ) from error

        if not image_path.is_file():
            raise FileNotFoundError(
                f"找不到视觉证据图片：{image_path}"
            )

        chunks.append(chunk)

    return chunks
