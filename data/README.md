# 数据下载与配置

原始数据不随 GitHub 仓库分发。代码仓库的 MIT License 不改变原始数据各自的许可和使用条款；下载、使用和再分发前请查看来源页面要求，并在报告中引用原论文/数据集。

## MIT-Stanford

- 大小约 8.3 GB，3 个 MATLAB v7.3/HDF5 文件。
- 来源：Severson et al. 2019 数据发布页：<https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation>
- 文件名：`2017-05-12_batchdata_updated_struct_errorcorrect.mat`、`2017-06-30_batchdata_updated_struct_errorcorrect.mat`、`2018-04-12_batchdata_updated_struct_errorcorrect.mat`。
- 默认放置：`data/mit/`。

## NASA PCoE

- 34 个平铺的 `.mat` 文件（`B0005` 到 `B0056`，按官方发布包中的文件为准）。
- 来源：NASA Prognostics Data Repository 的 Battery Data Set。
- 默认放置：`data/nasa/`。

## 路径配置

1. 复制 `config.example.toml` 为项目根目录的 `config.toml`；该本机配置已被 `.gitignore` 忽略。
2. 如数据不在默认目录，修改 `mit_dir` 和 `nasa_dir`；缓存与新生成文件默认位于 `.cache/` 和 `results/generated/`。
3. 检查解析结果：`python scripts/run_project.py --show-paths`。

也可用环境变量覆盖路径：`BATTERY_MIT_DIR`、`BATTERY_NASA_DIR`、`BATTERY_CACHE_DIR`、`BATTERY_OUTPUT_DIR`，或运行脚本时设置 `BATTERY_CONFIG` 指向另一份 TOML。

缓存和生成图表不应提交到 GitHub。上传前用 `git status --ignored` 检查，确认原始 `.mat`、解析 `.pkl` 和本机配置均未被暂存。
