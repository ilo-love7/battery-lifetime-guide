# Contributing

感谢关注这个复现与学习项目。欢迎提交文档勘误、测试、可复现的实验或教学改进。

## First run

```powershell
python -m pip install -r requirements.txt
python scripts/run_project.py --stage quick
python -m pytest tests -q
```

完整复现需要 MIT/NASA 原始数据。请按 `data/README.md` 获取数据、配置 `config.toml`，不要把原始数据、缓存或本机配置提交到仓库。`results/generated/` 是本机生成目录；`results/` 中已有 CSV 是项目核验快照。

README 社交预览图由 `scripts/render_github_banner.py` 从核验快照 `results/阶段43f_论文主测试逐颗误差.csv` 生成；更新逐颗基准数据后可运行 `python scripts/render_github_banner.py` 重绘 `项目宣传图.png`，不要手工改图上的指标。

`--stage full` 会依次运行多个独立教学脚本，可能重复读取约 8.3 GB MIT 数据；一般复现请选 `paper`，只跑 NASA 扩展选 `extensions`，step6 研究选 `optimization`。

## Experiment proposals

提交实验时请说明：

- 假设、数据版本/来源和标签定义；
- 官方划分是否保持不变，调参与特征选择是否只在训练池完成；
- 随机种子、运行命令、逐样本结果、指标和不确定性；
- 失败结果、样本量限制和哪些结论不能外推。

不要根据最终测试集反复选模型或改特征。若结果与已记录数字不同，先提供可复现命令和差异范围，不要静默替换基准结果。

## Attribution and AI assistance

请引用 Severson et al. 的原论文及所用数据来源，并遵守各数据集自己的条款。代码仓库的 MIT License 不自动覆盖原始数据。

AI 工具可以协助编码、整理或审阅，但提交者仍需理解、运行并核验关键代码和数字。PR/报告中请简要说明 AI 参与了哪些工作、你本人做了哪些判断和验证；不要把上游仓库、他人的分析或 AI 未经验证的结论描述为个人原创成果。

## Scope

本仓库当前提供电池寿命复现与相关教学材料。通用数据集审计工具不在当前范围；若提交这类想法，请先作为 proposal 讨论，不要把未来计划写成已交付功能。