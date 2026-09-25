# ResearchKB

面向个人的多模态 RAG 行研知识库，研究美股 AI 基础设施产业链，包括算力芯片、光模块、内存和存储。

## 当前状态

第二天文本解析和证据切分已经完成：首轮 100 个物理页生成 276 条可追溯证据。尚未开始向量入库。

## 环境检查

在 PowerShell 中运行：

```powershell
cd E:\AI-Agent-Projects\ResearchKB
uv run python scripts\check_day1.py
```

填入 `.env` 的 API Key 后，再运行：

```powershell
uv run python scripts\check_cloud_models.py
```

如果虚拟机 IP 改变，将 `.env.example` 复制为 `.env`，只修改 `MILVUS_URI`。真实 API Key 只放在 `.env`，不要提交到 Git。

## 目录

- `data/raw`：原始 PDF。
- `data/qa`：PDF 视觉检查图片，不提交 Git。
- `docs/样本资料清单.md`：来源、页数和首轮范围。
- `docs/第一天记录.md`：第一天检查结果、代码关系和复盘笔记。
- `docs/第二天复盘.md`：PDF 加载、清理、切分和证据 JSON 的复盘笔记。
- `docs/学习协作约定.md`：从第二天开始的手写代码、中文注释与 Git 协作方式。
- `docs/每日复盘模板.md`：每日代码关系、数据流和面试复习模板。
- `scripts/check_day1.py`：第一天环境检查。
- `src/research_kb`：后续业务代码。

原始 PDF 只保存在本机，不上传 GitHub；仓库通过 `docs/样本资料清单.md` 记录资料来源。
