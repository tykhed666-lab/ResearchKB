# ResearchKB

个人多模态行研资料库：上传不同行业的 PDF，检索文字与图表证据，生成带原文页码的回答，并根据美股 Ticker 查询和保存 SEC 官方披露。美股 AI 基础设施是目前的演示资料，不限制后续研究行业。

项目正在向个人多源行研工作台扩展。新版定位、已完成与计划能力见[新版项目总览](docs/ResearchKB_新版项目总览.md)；后续开发安排见[第 7—14 天开发路线](docs/ResearchKB_第7至14天开发路线.md)。SEC 官方披露目录已经接入，网页正文检索和简报导出仍在后续计划中。

## 当前状态

第十一天 SEC 官方披露接入已经完成：页面根据 Ticker 查询近期 10-K 和 10-Q，显示公司 CIK、报告期、提交日期和官方原文链接，并将用户选择的披露保存到 SQLite 及现有研究项目。SEC 服务与 Milvus 问答服务独立初始化，联网失败不会影响本地知识库。

当前演示库包含 5 份 AI 基础设施报告和 1 份传统零售行业的 Target 年报，共有 308 条文本证据和 4 条图像证据，总计 312 条有效证据。另保存了 NVIDIA 和 Target 各 1 份 SEC 10-K 官方披露。外部披露目前作为带 URL 和日期的研究目录保存，尚未进入 Milvus；第十二天会统一本地页码引用与外部 URL 引用。

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
- 区分文本证据与图表证据，并展开显示 PDF 页面原图和视觉描述。
- 保存当前会话的聊天记录并支持一键清空。
- 上传不超过 20 MB 的 PDF，并限制单次最多处理 30 页。
- 上传时登记标题、公司、Ticker、行业、文档类型和报告日期。
- 展示文档处理状态、文本证据数、图像证据数和失败原因。
- 通过文件哈希和证据 ID 两层去重，避免重复 Embedding 和重复入库。
- 新建研究项目，并调整已有文档或新上传文档的项目归属。
- 按项目、行业、公司和报告日期组合筛选问答资料。
- 在筛选结果中选择多份 PDF，执行带来源隔离的检索问答。
- 重新生成指定文档的文本证据，同时保留已有图像证据。
- 确认并备份后，同步删除向量、登记记录、PDF 和页面图片。
- 为已上传 PDF 选择最多 5 个图表页，生成视觉描述和图像证据。
- 重复处理已有图表页时跳过视觉模型和 Embedding 调用。
- 根据美股 Ticker 查询 SEC 最近的 10-K 和 10-Q。
- 展示 SEC 公司名称、CIK、报告期、提交日期、accession number 和官方原文。
- 将 SEC 披露保存到当前研究项目，并识别重复记录。
- SEC 请求声明 User-Agent、限制请求频率，并处理超时、429 和服务端错误。

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

SEC EDGAR 不需要 API Key，但自动访问必须在 `.env` 中声明应用名称和联系邮箱：

```dotenv
SEC_USER_AGENT=ResearchKB your-email@example.com
```

## 目录

- `data/raw`：原始 PDF。
- `data/research_kb.db`：本机 SQLite 文档登记簿，不提交 Git。
- `data/qa`：PDF 视觉检查图片，不提交 Git。
- `docs/样本资料清单.md`：来源、页数和首轮范围。
- `docs/第一天记录.md`：第一天检查结果、代码关系和复盘笔记。
- `docs/第二天复盘.md`：PDF 加载、清理、切分和证据 JSON 的复盘笔记。
- `docs/第三天复盘.md`：Embedding、Milvus Schema、幂等入库和 Top-5 检索复盘。
- `docs/第四天复盘.md`：结构化问答、引用校验和库内/库外评测复盘。
- `docs/第五天复盘.md`：Streamlit 页面、PDF 上传、幂等入库和第五天验收复盘。
- `docs/第六天复盘.md`：页面渲染、视觉描述、图像证据入库和多模态问答复盘。
- `docs/第七天复盘.md`：跨行业通用化、统一配置和同页证据补全复盘。
- `docs/第八天复盘.md`：SQLite 文档登记、状态管理、迁移和两层去重复盘。
- `docs/第九天复盘.md`：研究项目、组合筛选、多来源检索、备份删除和重新入库复盘。
- `docs/第十天复盘.md`：页码解析、视觉处理、重复跳过、部分失败和图像统计同步复盘。
- `docs/第十一天复盘.md`：SEC Ticker/CIK 映射、申报查询、限流、外部资料登记和页面接入复盘。
- `docs/前八天项目完整复盘与代码导读.md`：完整项目结构、分支流程和推荐读码顺序。
- `docs/ResearchKB_新版项目总览.md`：当前能力、新定位、业务流程和技术取舍。
- `docs/ResearchKB_第7至14天开发路线.md`：第 7—14 天的任务和验收标准。
- `docs/2026-09-29_第七天任务.md`：跨行业通用化的分段任务与传统行业样本。
- `docs/第1至4天总体框架与二次学习路线.md`：项目总架构、业务流程、代码关系和推荐复习顺序。
- `docs/学习协作约定.md`：从第二天开始的手写代码、中文注释与 Git 协作方式。
- `docs/每日复盘模板.md`：每日代码关系、数据流和面试复习模板。
- `scripts/check_day1.py`：第一天环境检查。
- `scripts/run_streamlit.py`：供 PyCharm 普通运行按钮使用的 Streamlit 启动入口。
- `scripts/render_chart_pages.py`：渲染首轮三张 PDF 图表页。
- `scripts/describe_chart_pages.py`：调用视觉模型生成可检索的图表描述。
- `scripts/index_visual_evidence.py`：将图像描述转成向量并写入 Milvus。
- `scripts/migrate_document_registry.py`：把已有 PDF 和 Milvus 证据补登记到 SQLite。
- `scripts/check_upload.py`：验证真实 PDF 重复上传不会增加文档或向量。
- `tests/test_document_registry.py`：文档登记与上传状态的离线自动化测试。
- `tests/test_document_management.py`：项目隔离、组合筛选、备份删除与重新入库测试。
- `tests/test_visual_ingestion.py`：图表页解析、视觉入库、重复跳过和越界拒绝测试。
- `tests/test_sec_edgar.py`：SEC JSON 解析、URL、缓存、超时和限流测试。
- `tests/test_external_source_registry.py`：外部披露保存、去重、项目外键和筛选测试。
- `src/research_kb/sec_edgar.py`：SEC 数据模型、纯函数和官方 API 客户端。
- `src/research_kb/external_source_registry.py`：外部研究资料 SQLite 登记簿。
- `src/research_kb`：后续业务代码。

原始 PDF 只保存在本机，不上传 GitHub；仓库通过 `docs/样本资料清单.md` 记录资料来源。
