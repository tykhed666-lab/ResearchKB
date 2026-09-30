# ResearchKB

个人多模态行研资料库：上传不同行业的 PDF，检索文字与图表证据，生成带原文页码的回答，并根据美股 Ticker 查询和保存 SEC 官方披露。美股 AI 基础设施是目前的演示资料，不限制后续研究行业。

项目正在向个人多源行研工作台扩展。当前架构、真实数据状态、关键取舍和第 14 天后续计划统一记录在[项目交接说明](docs/项目交接说明.md)。SEC 官方披露正文、混合检索和 Markdown 研究简报已经接入。

## 当前状态

第十三天研究简报已经完成：用户沿用当前项目和资料范围，输入研究问题与简报日期，系统按来源平衡召回证据，生成包含核心结论、关键事实、公司与行业对比、催化因素、风险、信息缺口和来源清单的 Markdown 文件。所有事实陈述必须通过证据 ID 校验，无引用内容会转为信息缺口。

当前演示库包含 6 份 PDF 的 312 条证据，以及 Target 10-K 的 427 条 SEC 网页证据和 NVIDIA 10-K 的 500 条 SEC 网页证据，总计 1239 条 Milvus 实体。NVIDIA 原文按单份 500 条的成本上限截断，SQLite 会保留证据数量、索引时间和失败信息。

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
- 下载 SEC 主 HTML，排除脚本、样式和隐藏 XBRL 后切分入库。
- 在“仅资料库”和“综合研究”之间切换，综合模式检索 PDF 与已索引 SEC 正文。
- PDF 引用显示物理页码；SEC 引用显示报告期和官方 URL，不虚构页码。
- 按当前项目和资料范围生成固定结构的 Markdown 研究简报。
- 公司对比按来源平衡召回，防止单个大型 SEC 文档挤占全部结果。
- 简报逐条校验引用 ID，自动保存到 `data/reports` 并支持页面下载。
- 文件名包含微秒时间戳，不覆盖旧简报；无证据内容明确进入信息缺口。
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
- `data/external`：下载的 SEC HTML 原文，不提交 Git。
- `data/reports`：页面生成的 Markdown 研究简报，不提交 Git。
- `data/qa`：PDF 视觉检查图片，不提交 Git。
- `docs/项目交接说明.md`：唯一的项目历史、架构、数据状态、开发约束和后续计划。
- `AGENTS.md`：提示 Codex 在修改项目前先阅读交接说明。
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
- `src/research_kb/sec_html.py`：提取 SEC HTML 可见正文并切分证据。
- `src/research_kb/external_ingestion.py`：SEC 下载、清洗、向量化和状态更新流程。
- `src/research_kb/research_report.py`：平衡检索、简报结构、引用校验和 Markdown 保存。
- `src/research_kb`：后续业务代码。

原始 PDF、SQLite、页面图片和真实 `.env` 只保存在本机，不上传 GitHub。向另一台电脑交接可运行数据时，请按照交接说明的文件清单单独复制。
