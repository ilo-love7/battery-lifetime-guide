# AI 审查报告（第三轮）· GitHub 用户视角

**对象**：`battery-lifetime-guide`（Severson 2019 复现 + 原创四实验 + step6 优化 + 本轮 GitHub 化改造）
**日期**：2026-10-14　**快照**：本轮审查开始时的工作副本（含 13:27~14:30 的全部更新与 17 时前的最新脚本）
**方法**：把自己当成一个第一次点进这个仓库的 GitHub 访客——"首屏 → 想看懂 → 想跑起来 → 浏览目录"——逐段找困惑；同时做发布预演（`git add -A` 全量模拟）、CI 三件套本地复现（标准 Python 3.14 + pytest 9.1.1）、全仓库链接核验、全量 CSV 结构核验、新落盘证据独立复算。

> **一句话结论**：这轮"GitHub 化"改造质量很高——编排器、配置体系、CI、CONTRIBUTING 都到位且本地全绿。**发布前只剩 1 个硬问题（`.gitignore` 会把 `data/README.md` 吞掉，导致数据指引断链）+ 2 个文档小尾巴**；其余都是锦上添花项。

---

## 一、访客旅程：我会在哪里困惑

### 1. 首屏（README）——结构专业，三个小刺

| # | 困惑点 | 证据 | 建议 |
|---|---|---|---|
| F1 | 徽章点了没反应：外链目标为空 `()` | `README.md:8` | 给目标（如指向 `docs/03`）或去掉外层链接 |
| F2 | 除徽章外**全 README 无一张内容图**；GitHub 首屏纯文字 | README 全文 | 选 1 张主图嵌进去（`官方划分_论文主测试_EN散点.png` 或一页报告封面截图），兼顾 social preview |
| F3 | **根目录躺着 5 份"来路不明"的中文 md**：`AI审查任务书.md`、两份 `AI审查报告_*.md`、两份工作记录——README 仓库地图一条都没解释 | 根目录清单 | 收进 `docs/meta/`（或 `docs/review/`）并在地图加一行；或加"元文件说明"小节。这是访客最容易懵的地方 |

### 2. 想跑起来——路线清晰，但有一条**断链**（发布级必修）

| # | 困惑点 | 证据 | 建议 |
|---|---|---|---|
| **F4（必修）** | README/CONTRIBUTING 都让你"按 `data/README.md` 下载数据"——**但 `.gitignore` 的 `data/` 会把 `data/README.md` 一起忽略**：发布预演实测，`git add -A` 后 130 个文件里没有它。发布后这条指引 100% 断链，且 `data/mit`、`data/nasa` 的约定无处可查 | `.gitignore:3`；git 实测 | 把 `data/` 换成 `data/mit/` + `data/nasa/`；或 `data/*` + `!data/README.md`（**两种写法均已实测验证**：data/README.md 入库、大文件仍被忽略） |
| F5 | `docs/01` 的"运行时长参考"表与实际代码不符：`step4_mit_load.py` 一行写"有缓存时 10 s（mit_parse_cache.pkl）"，**但该脚本根本没有缓存读取代码**；`step4_2 / 4.3 系列` 同样直接解析原始数据。实测缓存链是：**step4_3f 首跑创建 `.cache/mit_parse_cache.pkl`，5b/5c/step6 消费**；`step4_mit_load/4.2/4.3/4.3b/4.3c` 每次都全量解析 | `docs/01:43-44`；对 5 个脚本 grep 证实 | 表改真：把缓存说明挪到"4.3f 起共享 `.cache`"；并在 README/CONTRIBUTING 提示 `--stage full` 会多次解析 8.3GB（每个自包含教学脚本各解析一次）——访客按表预估时间会被带偏 |
| F6 | `docs/01` 目录结构树过时：根目录 `*.pkl`（实际在 `.cache/`）；树里没有 `.cache/`、`results/generated/`、`config.toml`、`.github/` | `docs/01:69` | 顺手更新 |
| ✔ | **做得好**：`run_project.py --stage quick` 无数据可跑；`config.example.toml → config.toml → --show-paths` 流程干净；`data/README.md` 里"上传前用 `git status --ignored` 检查"是加分项（讽刺的是它自己会被 ignore 吞掉，见 F4） | 本地实测均通过 | 保持 |

### 3. 浏览目录——两处文档级不一致

| # | 困惑点 | 证据 | 建议 |
|---|---|---|---|
| F7 | `docs/07` 概念地图**两处仍是旧统计措辞**：第 28 行 mermaid `CLAIM[措辞: 与论文9.1%一致 / 差异<联合噪声]`、第 53 行速查表——与 README/docs/03 已统一的"点估计接近、统计差异未定"（论文侧无可比 CI）**互相矛盾**。概念地图是"只看方法论"路线的入口文档，必须同步 | `docs/07:28,53` | 1 分钟改掉 |
| F8 | 两份优化记录副本**已分叉**：`docs/工作记录_代码优化提案` 有"第二轮独立复核后补充"章节 + 更新段落；根目录版有独立的"GitHub 发布准备"段和另一套运行命令——读者无法判断哪份最新 | 两文件 diff | 定 canonical 一份（建议 docs 版），另一份同步或改为一句指向 |
| F9 | `docs/buggy_注释版/step5a v3_v3.py` 文件名含**空格** + 双后缀；错误清单里引用名（`step5a v3（体检分类版）.py`）与实际名也不一致 | 文件名；`docs/archive/项目错误清单.md:41` | 改 `step5a_parse_nasa_v3_buggy.py` 并同步引用 |
| 注 | `figures/_旧版本/` 是 docs/05 引用的教学材料（"v1 空白图现场"），不是残留——建议地图里补一句说明，免得访客误判 | docs/05 有引用 | 可选 |
| 注 | 被忽略的两个 PDF 用**日期文件名做 ignore 规则**（`*_修订前备份_2026-10-14.pdf`），以后换日期重生成就漏了 | `.gitignore` 尾部 | 用模式如 `项目一页报告_*备份*.pdf`（可选） |

### 4. 检查过的、没问题的部分（访客可以放心）

- `scripts/`：22 个脚本全部接入 `project_config`，零盘符硬编码（全仓库 grep 空）；`run_project` 五条路线 + `--show-paths/--dry-run` + 输入前置校验，设计干净。
- `results/`：26 个 CSV **全部**结构合法（含修复后的 `NASA34颗体检分类.csv`，pandas 直读 OK）；新落盘的 `阶段43f_论文主测试_RF逐颗误差.csv`（43 颗）独立复算 中位 8.20 / 均值 13.90 / CI [4.60, 12.20]——与 README 的 `8.2% [4.6~12.2] / 13.9%` **逐位一致**；`MIT特征分位对照.csv` 的 18/18 越界声明复核通过。
- **全仓库 32 个本地链接 0 断链、0 大小写不匹配**（GitHub 是 Linux 大小写敏感，这点通过了）。
- 措辞统一：README/任务书/docs 00/02/03/09/10/11 + HTML + 重生成的 PDF 全部改为"点估计接近，统计差异未定"；**PDF 与 HTML 逐字对拍一致**（上轮 P7 关闭）。

---

## 二、发布预演与 CI 复现（我实际跑过的）

**发布预演（git add -A 全量模拟）**：130 个文件将入库；正确忽略：`data/`、`results/generated/`、两个备份 PDF；唯一误伤 = `data/README.md`（F4）。

**CI 三件套本地复现**（标准 Python 3.14，等价 GitHub Actions 环境）：

| 命令 | 结果 |
|---|---|
| `python -m compileall -q scripts tests` | ✓ exit 0 |
| `python -m pytest tests -q` | ✓ **6 passed** |
| `python scripts/run_project.py --stage quick` | ✓ 跑通并产出 `results/generated/toy_forest.png` |
| `--show-paths` / `--dry-run --stage paper` | ✓ 输出正确 |

> 方法注记：沙箱自带解释器是 safe-path 构建（不注入 cwd/脚本目录），故全部用系统标准 Python 3.14 复现——与 GitHub Actions 行为一致。

**上两轮遗留关闭确认**：P1（措辞）✓、P3（124→84）✓、P4（q2[1] 有效圈）✓、P5（RF 逐颗 + 分位表落盘）✓、P6（CSV 引号）✓、P7（PDF 同步）✓；N1~N6（bootstrap 43 颗、文件名、138 行、11 颗井、17 个现场、27.0/24.1 术语）✓。

---

## 三、"更像一个 GitHub 项目"清单

**A. 发布前文件级（按优先级）**
1. 【必】`.gitignore` 修 `data/README.md` 断链（F4，两种写法已验证）；
2. 【必】`docs/07` 两处措辞同步（F7）；
3. 【建议】`docs/01` 运行时长表 + 目录树校正（F5/F6）；
4. 【建议】两份记录副本统一（F8）；buggy 文件名去空格（F9）。

**B. 发布操作（仓库设置层面，上线时顺手做）**
- `git init` + 首次提交（建议至少一个干净 initial commit；把"事故现场"做成有意义分批提交更佳——这仓库体量 6.9 MB，无需 LFS）；
- GitHub 仓库 description + topics（`battery`、`lithium-ion`、`machine-learning`、`reproducibility`、`data-audit`、`severson2019` 等）；上传 social preview 图；
- （可选）Release 附上 `项目一页报告.pdf` 作为下载物；CITATION.cff 一行致谢论文。

**C. 可选工程增强**
- `tests/conftest.py`（或 pytest ini 的 `pythonpath=.`）：当前 `tests/test_project_config.py` 的 `from scripts.project_config import ...` 依赖 `python -m pytest` 注入 cwd——CI 和文档命令都没问题，加一行可让裸 `pytest tests` 也稳；
- `pyarrow` 未入 requirements（作者环境读取旧缓存需要它；本机实测**新用户无影响**，pickle 往返正常）——若想"作者环境=用户环境"完全对齐可补；
- `.gitattributes`（`* text=auto`）统一行尾（Windows 仓库常见）；
- CI 可选加一步 `run_project.py --dry-run --stage full`，防编排清单漂移。

---

## 四、总体评价

这是把这个项目从"个人笔记本"升级成"可对外协作仓库"的一轮，且完成质量高：`run_project` 编排 + TOML/环境变量配置 + CI + CONTRIBUTING（数据保护、口径纪律、AI 使用披露）+ 数据获取指引，覆盖了 GitHub 访客最关心的"怎么跑、能不能信、别把大数据传上来"三件事；本地 CI 三件套全绿，链接与 CSV 全量干净。**发布前需要动的只有：1 个硬问题（F4）+ 2 个一分钟文档项（F7、docs/01 表）**。作为访客，我会认为这是一个少见的"既真诚又讲究"的复现项目——把事故和修复过程都留着，还顺手把工程边角磨平了。

---

*本报告为第三轮独立审查（GitHub 用户视角）：发布预演 + CI 本地复现 + 链接/CSV/证据全量核验。与前两轮互为独立，不采信任何自述数字。*
