# 研究想法系统（开源 DeepSeek 平替）

 ERA 论文复现中 **Gemini Deep Research / AI co-scientist → 人工筛选 → Gemini S16 格式化 → ERA 树搜索** 链路的开源替代实现，模型统一为 **DeepSeek**（复用 `project/secret.txt` 密钥与工作区调用约定）。选型依据见[方案选型报告_20261010.md](方案选型报告_20261010.md)。

> 状态：代码完成，离线测试 8/8 通过，全链路 dry-run 验证通过；**尚未发起真实 API 运行**（由你决定何时消耗预算）。

## 与论文协议的对应

| 论文步骤（Methods + S15/S16/S10） | 本系统 | 命令 |
|---|---|---|
| Gemini Deep Research 按 S15 提示词生成想法 | 递归文献研究（查询→arXiv 检索→提炼→追问）+ S15 收尾请求 → `ideas.json` | `research` |
| AI co-scientist 生成/进化想法 | 多智能体锦标赛：生成→文献反思→六准则评审→排序→Elo 两两裁决→元评审→进化剪枝 → `coscientist_output.json` | `coscientist` |
| 人工筛选（manual filtering） | 生成 `selection.json` 供人工删改 | `filter` |
| Gemini 按 S16 格式化 | S16 提示词原文 → `<description>/<steps>/<notes>` 方法卡 | `format` |
| ERA 树搜索评分（`{method}` 注入） | 不在本系统内：方法卡插入 `project/covid19` 提示词的 `{method}` 两处（S10 位型）后由现有执行器运行 | — |

任务背景文件（论文称 "background information from the main problem description"）：`prompts/任务背景/`，现有 COVID19 住院预测与单细胞批次整合（后者为 S15 原文逐字）。S15/S16/S10 英文原文存档：`prompts/S15_S16_原文存档.md`。

## 快速开始

```powershell
cd D:\workspace\论文复现\研究想法系统
$env:PYTHONIOENCODING = "utf-8"
$PY = "D:\workspace\论文复现\project\venv\Scripts\python.exe"

# 0) 预算（不调 API）
& $PY -m idea_system.pipeline plan

# 1) 深度研究（约 5 次 LLM 调用 + 10 次 arXiv 检索；--dry-run 零成本试跑）
& $PY -m idea_system.pipeline research --task COVID19_住院预测 --dry-run

# 2) co-scientist 锦标赛（继续同一 run 目录，以研究想法为种子）
#    引擎：--engine auto（默认，ocs-venv 存在时用 open-coscientist 框架，
#    否则用内置有界锦标赛）；--engine ocs / --engine self 显式指定；
#    --dry-run 恒用 self 引擎（ocs 框架无 dry 模式）
& $PY -m idea_system.pipeline coscientist --task COVID19_住院预测 --run <run目录> --dry-run

# 3) 人工筛选：编辑 selection.json 的 keep 列表后保存
& $PY -m idea_system.pipeline filter --run <run目录>

# 4) S16 格式化（每个保留想法 1 次调用，+1 次修复上限）
& $PY -m idea_system.pipeline format --run <run目录> --dry-run
```

去掉 `--dry-run` 即真实运行。每次运行在 `runs/<时间戳>_<阶段>_<任务>/` 留下全部中间产物与 `manifest.json`（调用级用量、搜索日志、提示词哈希、dry_run 标记）。

## co-scientist 引擎（2026-10-19 切换）

按用户决定，co-scientist 阶段默认使用 **open-coscientist 框架**（jataware，MIT + Commons Clause）：

- 隔离环境 `ocs-venv/`（open-coscientist 0.2.0 + langgraph/litellm 栈；安装时需钉 `mcp<2`，否则 langchain-mcp-adapters 与 mcp 2.x 的破坏性迁移冲突导致导入失败）。CLI 主链路仍在 project/venv 零新增依赖，经子进程 `ocs_runner.py` 桥接。
- DeepSeek 经 LiteLLM（`deepseek/deepseek-flash`），密钥由 runner 从 `project/secret.txt` 读取并注入环境，不落命令行。参数在 `config.yaml` 的 `ocs:` 节（迭代数、初始/进化假设数、超时）。
- 有 research 种子时注入 research_goal（要求"超越而非复述既有想法"）；框架自带 Supervisor/评审/Elo 锦标赛/元评审/进化/近邻去重，产出的 `elo_rating`/评审/文献接地字段无损映射为本系统 `coscientist_output.json` 格式，filter/format 下游不变。
- 证据链差异（manifest 如实标注）：ocs 引擎的 LiteLLM 调用由框架自管，**无逐调用 token 日志**，记录节点序列与耗时替代；`--dry-run` 恒回退内置 self 引擎。
- 未配置其 MCP 文献服务器时，框架的文献评审节点按无文献模式运行（README 注明质量依赖文献接地时可另起其 `mcp_server/`，或用形态二的 HTTP 模式接入）。

## 三种落地形态（2026-10-19 增补）

| 形态 | 入口 | 证据等级 | 用途 |
|---|---|---|---|
| 一：浏览器收割前置层 | `harvest-scaffold --task` → agent 按 `prompts/收割协议_浏览器研究.md` 填 `harvest/` → `research --corpus <dir>` | 高（URL/时间/摘录/哈希全留痕，C 编号可核查） | 让带浏览器的 agent 把自由检索的真实文献注入有界研究循环 |
| 二：MCP 工具化 | `mcp-venv/` + `mcp_server/serve.py`（stdio 默认，`ERA_MCP_TRANSPORT=streamable-http` 起 HTTP） | 同正式管线（`mcp_runs/` 逐调用日志） | ZCode/Codex/open-coscientist 等任何 MCP 客户端调用 5 个研究工具；默认 `dry_run=true` |
| 三：agent 完全接管 | agent 按 `prompts/研究协议_完全接管.md` 产出 → `takeover --task --file` 校验登记 | 最低（`provenance_tier: agent_takeover`，来源仅 agent 声明） | 检索自由度最大；不得作为正式复现路径引用 |

注册片段（ZCode / Codex / open-coscientist）与 agent 约定见 [AGENTS.md](AGENTS.md)。

## 成本

默认配置：research ~5 次、coscientist ~35 次、format 每想法 1 次（全链路约 40–55 次 DeepSeek 调用）；`runtime.max_llm_calls=200` 硬熔断；`plan` 子命令随时查预算。检索仅 arXiv（免费、限速 3 秒/次）。

## 测试

```powershell
& $PY -m pytest tests -q
mcp-venv\Scripts\python.exe mcp_server\smoke_test.py
```

13 项离线测试（12 项主套件 + 1 项 MCP stdio 冒烟包装，后者需 mcp-venv 已构建）：密钥读取（BOM/空白）、JSON 提取、任务文件与 S15 结构、研究全流程、语料加载（去重/校验）、含语料的研究链路、接管校验（合法/非法）、锦标赛（含 Elo 数学、种子接入、排序）、多阶段 manifest 累计回归、筛选→格式化。零 API、零网络。

## 目录

```text
研究想法系统/
├── README.md / AGENTS.md / 方案选型报告_20261010.md
├── config.yaml                     模型/搜索/各阶段参数
├── requirements.txt                零新增依赖说明（project/venv 直接可用）
├── prompts/
│   ├── S15_S16_原文存档.md          论文提示词英文原文（含哈希与清理说明）
│   ├── 收割协议_浏览器研究.md        形态一协议书
│   ├── 研究协议_完全接管.md          形态三协议书
│   └── 任务背景/                    COVID19_住院预测.md、单细胞批次整合.md
├── idea_system/                    包：config/deepseek/websearch/provenance/
│                                   tasks/deep_research/coscientist/format_s16/
│                                   corpus（形态一）/pipeline
├── mcp_server/                     形态二：serve.py（stdio/HTTP）+ smoke_test.py
├── mcp-venv/                       MCP 独立环境（mcp<2 + openai/requests/PyYAML）
├── mcp_runs/                       MCP 调用日志（jsonl）
├── tests/                          离线单测 + MCP 冒烟包装
└── runs/                           运行产物（含 manifest 证据链、harvest 语料）
```

## 边界与偏差声明

1. **模型替代**：论文用 Gemini Deep Research（2.5 Flash）/ AI co-scientist 生成、Gemini 格式化；本系统为 DeepSeek 同协议替代，输出不等价于论文 Gemini 产出，引用时必须注明。
2. **有界简化**：co-scientist 为论文架构的有界实现（单生成/评审/元评审角色、Elo 裁决 LLM 化），不是 Google 系统的完整复现；文献接地依赖 arXiv 摘要（本网络 PubMed/DDG 不稳，可配置开启）。
3. JSON 解析失败时每调用最多 1 次有界修复重试（记录于 manifest `repair_attempts`），除此无自动重写；**截断响应（finish_reason≠stop）与空响应直接判失败调用并记录，不重试**（2026-10-10 审阅后加入）。
4. S15 收尾请求原文中的 "10 ideas" 按配置 `research.ideas_count` 替换（默认仍为 10），属已披露的参数化适配。
5. 方法卡→树搜索的桥接：`project/covid19/import_method_cards.py` 把人工筛选后的卡片冻结为 `S16_card_idea_NN.txt` 资产并注册为可运行条件（整卡文本注入 S10 双 `{method}` 位点，本地选择已披露）。
6. 与图 1b Kaggle 复现范围的关系见 `project/kaggle/materials/研究想法与图1b范围核对_20261010.md`：本系统服务的是论文跨任务研究想法链路的复现，不改变图 1b 六组对照的边界。
