# 路线图

把 `框架大纲.md` 里的设计一条条落进代码。原则：**每一步都能独立跑通、独立回归**，
不出现"功能加了一半、演示不了"的中间状态。

## ✅ 第 1 步 · 仓库化骨架（已完成）

- `mvp/` 单目录拆成 `app/{api,core,data,web}` 包结构
- 逻辑与 HTTP 分离：判定、状态机、计算层都能脱离 FastAPI 单独测试
- `pyproject.toml` / `.env.example` / `.gitignore` / README
- 82 项测试：数字抽取宽容度、四题闭环、护栏约束、静态资源、接口错误码
- `scripts/run.py` 启动器、`scripts/test.py` 免 pytest 运行器
- **行为零变化**：四道题闭环、自评红灯、变式裁决全部回归通过

顺带修掉的问题：

- `mathcheck.evaluate` 的沙箱挡不住 `().__class__.__bases__` 这类属性遍历 → 加字符串黑名单
- 极限题第三节点第一级提示以句号结尾，违反"每条回复以问句结尾" → 改文案
- 出口统一加 `ensure_question()` 保险：数据或模型漏了也能兜住

## ✅ 第 2 步 · SQLite 五表持久化（已完成）

- `sessions` / `messages` / `steps` / `self_assessments` / `mastery`（SQLAlchemy 2.x）
- `core/store.py`：`MemoryStore` 与 `SqlStore` 同接口，默认走 SQLite，建库失败自动退回内存
- **服务重启后会话能续答**（`test_resume_after_restart` 覆盖）
- `GET /api/reports/{id}/export.csv`：带 UTF-8 BOM，Excel 双击中文不乱码
- 换 PostgreSQL 只改 `DB_URL` 一行

一个设计取舍：**会话的 live 状态用 JSON 列存在 `sessions` 里**（plan / results / variant…），
没有拆成关系表。这些结构的字段随引擎演进，拆表意味着每次改字段都要迁移；
真正值得建表的是分析维度——`messages`、`steps`、`mastery`，它们才是雷达图和答辩数据的来源。
`steps` 与 `sessions.plan` 有冗余，是刻意的：一个是运行时状态，一个是分析用快照。

## ✅ 第 3 步 · 雷达图报告（已完成）

- Chart.js 4.4.1 下载到 `app/web/vendor/chartjs/`（201KB UMD，不走 CDN）
- `GET /api/reports/{id}/radar`：按知识点维度聚合，**只统计真的答过的节点**
  —— 没走到的节点记 0 会把雷达压成一朵花，看不出真实水平
- 维度不足 3 个时前端自动退回条形图（雷达至少要三角形）
- 报告页：雷达图 + 条形明细 + 虚假自信记录，三屏信息不重复

之所以先做它：报告页是答辩上最直观的一屏，而持久化不影响演示效果。


## ✅ 第 4 步 · 拍照 OCR 输入（已完成）

- `POST /api/ocr`：上传图片 → 多模态模型转 LaTeX，`app/core/ocr.py` 独立一层，换 OXR 只改这里
- 前端「拍张照」按钮（`<input capture="environment">`，手机上直接调相机）
- **识别不出来就说识别不出来**：`not_found` / `no_model` / `api_error` 都降级为手动输入，
  绝不把模型瞎猜的式子当成学生的题目送进追问流程
- MIME 以文件头为准（手机常把 content-type 传成 octet-stream），8MB 上限

## ✅ 第 5 步 · PWA + 二维码（已完成）

- `manifest.json` 与 `sw.js` 挂在根路径（放 `/static/` 下只能控制 `/static/`）
- 缓存策略分三类：页面网络优先（改完刷新就是新的）、`/static` 缓存优先（KaTeX 字体断网可用）、
  `/api` 完全不碰（数学答案与会话状态不能缓存）
- 图标用 PIL 现场生成（192/512/maskable + apple-touch-icon）
- `GET /api/qr` 返回 SVG 二维码；启动器还会在终端打印一份

## ✅ 第 6 步 · 流式 + 三模式 + 降维（已完成）

- `GET /ws/chat`：分片下发，`done` 载荷与 POST 完全一致，前端断流自动退回 POST
- `INFERENCE_MODE` 四值：`mock` / `cloud` / `local` / `hybrid`；hybrid 是「云端优先、本地兜底」，
  两条错误都会带出来，方便判断是 Key 问题还是本地没起
- **答错第二次先降维**：退到该节点的前置知识点（5 个关键节点各有前置题），答对再回到主问题；
  第三次才补答。同一节点只降维一次，降维记录写进作答记录和报告（掌握度扣分）
- SymPy 求值补了字符串层黑名单（挡 dunder 遍历）

## ✅ 第 7 步 · 本地模型适配（已完成）

- 默认端点 `http://127.0.0.1:1234/v1` + `qwen3.8`，漏 `/v1` 自动补
- 超时按链路分开（云端 120s / 本地 900s，connect 8s），超时不重试
- 思考型模型 `content` 为空时读 `reasoning_content`
- JSON 里未转义的 LaTeX 反斜杠统一修复；`\f` 这类"合法但会静默损坏"的也处理
- 前端等待层数秒 + 说明 + 可取消；`/api/llm/test` 返回 tokens_per_sec
- `/api/llm/models` 拉端点模型列表，设置面板给下拉候选

## ✅ 第 8 步 · 拍照体验（已完成）

- 拍照 / 从相册选 拆成两个入口（`capture` 属性只能二选一）
- 前端裁剪：可拖动、四角缩放、长边限 1600px，框外不参与识别
- 本地模型不能看图时按钮置灰并说明原因，不发无谓请求

## ⬜ 待办

- 云端判定拿不准时反问澄清，而不是直接判错（语义兜底层）
- SymPy 求值加超时中断（现在只有命名空间隔离 + 黑名单）
- 跨会话知识点画像（mastery 目前按会话聚合，尚未按学生长期聚合）
- 本地模式接 OXR-1.0 做端侧 OCR（现在只有多模态 LLM 一条路）
