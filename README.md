# ResearchKB

面向个人的多模态 RAG 行研知识库，研究美股 AI 基础设施产业链，包括算力芯片、光模块、内存和存储。

## 当前状态

第五天 Streamlit 演示闭环已经完成：页面可以查看知识库状态、按 PDF 限定检索、进行带引用的问答，并上传新 PDF 完成解析、切分、向量化和 Milvus 幂等入库。

当前演示库包含 5 份 AI 基础设施报告、276 条文本证据。来源检索验收通过 5/5，单文件过滤、重复上传去重和 Streamlit 页面检查均已通过。

## 启动网页

在 PowerShell 中运行：

```powershell
cd E:\AI-Agent-Projects\ResearchKB
uv run streamlit run app.py
```

浏览器打开 `http://localhost:8501`。停止服务时，在运行终端按 `Ctrl + C`。

在 PyCharm 中也可以运行：

```text
scripts/run_streamlit.py
```

不要把 `app.py` 直接当作普通 Python 文件运行，否则 Streamlit 没有 `ScriptRunContext`。

## 页面能力

- 查看 Milvus 当前有效实体数和已入库 PDF。
- 在全部资料或单份 PDF 范围内提问。
- 展示回答、证据不足状态、引用页码和证据原文。
- 保存当前会话的聊天记录并支持一键清空。
- 上传不超过 20 MB 的 PDF，并限制单次最多处理 30 页。
- 通过证据 ID 跳过重复片段，避免重复 Embedding 和重复入库。

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
- `docs/第三天复盘.md`：Embedding、Milvus Schema、幂等入库和 Top-5 检索复盘。
- `docs/第四天复盘.md`：结构化问答、引用校验和库内/库外评测复盘。
- `docs/第五天复盘.md`：Streamlit 页面、PDF 上传、幂等入库和第五天验收复盘。
- `docs/第1至4天总体框架与二次学习路线.md`：项目总架构、业务流程、代码关系和推荐复习顺序。
- `docs/学习协作约定.md`：从第二天开始的手写代码、中文注释与 Git 协作方式。
- `docs/每日复盘模板.md`：每日代码关系、数据流和面试复习模板。
- `scripts/check_day1.py`：第一天环境检查。
- `scripts/run_streamlit.py`：供 PyCharm 普通运行按钮使用的 Streamlit 启动入口。
- `src/research_kb`：后续业务代码。

原始 PDF 只保存在本机，不上传 GitHub；仓库通过 `docs/样本资料清单.md` 记录资料来源。
