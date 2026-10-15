# 生存分析升级 B3

入组为 3 个事件 + 11 个右删失（共 14 颗），其余异常/边界电池按既有审计规则排除。
Weibull AFT MLE：shape=2.872，scale=179.3 圈，中位数=157.8 圈。
KM 中位数=125.0 圈；Weibull 与 KM 的 cell bootstrap 均运行 10,000 次。
Weibull 中位数 bootstrap 95% CI=[102.4, inf]；KM=[97.0, inf]。
Weibull bootstrap 可拟合比例=96.6%；KM 中位数有限比例=75.5%。

11 颗删失电池的 Weibull 区间是基于总体 AFT 分布、并条件于寿命超过各自删失时间的模型化预测区间；它不是对 11 颗各自真实寿命的直接观测。
若 bootstrap 上界无穷或大量重采样不可估，说明 3 个事件不足以支持窄区间；Weibull 的形式假设不能替代信息量。
Nelson-Aalen 累积风险与 KM/Weibull 曲线表见 `survival_curve_comparison.csv`。
