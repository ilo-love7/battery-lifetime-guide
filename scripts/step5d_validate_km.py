# -*- coding: utf-8 -*-
"""
step5d_validate_km.py —— 用 lifelines 交叉验证 step5d 手写 KM 实现
================================================================
背景:step5d 的 Kaplan-Meier 是手写的(风险集递减/删失时点约定/连乘),
审查要求验证"KM 数学实现正确性"。本脚本把
results/5_跨数据集迁移/KM生存表.csv 还原成个体级 (时长, 事件) 记录,
用 lifelines 的 KaplanMeierFitter(参考实现)重算 S(t),逐时间点对比。

用法:python scripts/step5d_validate_km.py
依赖:lifelines(会带 numpy/pandas)。不需要原始 .mat 数据。
"""
import pathlib

import pandas as pd
from lifelines import KaplanMeierFitter

from project_config import enable_utf8_stdout

# Windows 控制台默认 cp1252：本脚本会打印中文对比表，必须先把 stdout 切成 UTF-8。
enable_utf8_stdout()

ROOT = pathlib.Path(__file__).resolve().parents[1]
km_csv = ROOT / "results" / "5_跨数据集迁移" / "KM生存表.csv"

km = pd.read_csv(km_csv)

# 从聚合表还原个体记录:每行 (时间, 事件数, 删失数) 展开为对应个数的记录
durations, events = [], []
for _, r in km.iterrows():
    t = int(r["时间(圈)"])
    for _ in range(int(r["事件(到达EOL)"])):
        durations.append(t)
        events.append(1)
    for _ in range(int(r["删失"])):
        durations.append(t)
        events.append(0)

print(f"还原个体数:{len(durations)}(应 = 3 事件 + 11 删失 = 14)")

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events)

print("\n时间(圈)   手写S(n)   lifelinesS(t)   |差|")
max_diff = 0.0
for _, r in km.iterrows():
    t = int(r["时间(圈)"])
    s_ref = float(kmf.predict(t))
    s_own = float(r["S(n)"])
    diff = abs(s_ref - s_own)
    max_diff = max(max_diff, diff)
    print(f"{t:>7}    {s_own:.4f}      {s_ref:.4f}      {diff:.2e}")

if max_diff < 1e-9:
    print("\nOK:手写 KM 与 lifelines 参考实现完全一致(风险集/删失时点约定正确)")
else:
    raise SystemExit(f"\nFAIL:手写 KM 与 lifelines 最大偏差 {max_diff:.2e},"
                     f"检查风险集递减与删失时点约定")

# 中位数交叉验证:手写口径(首个 S<=0.5 的时间点) vs lifelines 自带估计
med_own = int(km.loc[km["S(n)"] <= 0.5, "时间(圈)"].min())
if float(kmf.median_survival_time_) == float(med_own):
    print(f"OK:中位寿命口径一致 = {med_own} 圈")
else:
    print(f"注意:中位口径差异——手写 {med_own} vs lifelines "
          f"{kmf.median_survival_time_}(S 压线时两者实现细节可能不同)")
