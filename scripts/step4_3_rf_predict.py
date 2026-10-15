# -*- coding: utf-8 -*-
"""
电池寿命数据分析 —— 阶段 4.3:随机森林寿命预测(对标论文)
================================================================
= 4.1 解析 + 4.2 特征(原样内置) + 随机森林训练/测试(新增)。
自包含,一次从头跑到尾。

论文口径(Severson et al. 2019):
  训练集 = batch1(41 颗) / 主测试集 = batch2 / 二级测试集 = batch3
  误差指标 = percentage error = |预测 - 真实| / 真实 × 100%,报中位数
  论文用弹性网络做到 ~9%(多特征);本脚本用随机森林,目标:主测试集中位误差 <15%

输出:
  mit_rf_predictions.csv(逐颗预测结果)
  rf_pred_vs_actual_{batch2,batch3}.png(预测-真实散点)
  rf_feature_importance.png(特征重要性)
  训练集 5 折交叉验证(检查过拟合)
"""

# %% ---------- [0] 导入与配置 ----------
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from project_config import PATHS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_DIR = PATHS.mit_dir
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
OUT_PRED = PATHS.output_dir / "mit_rf_predictions.csv"
PATHS.output_dir.mkdir(parents=True, exist_ok=True)

FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]


# %% ---------- [1] HDF5 解引用工具(4.1 原样) ----------
def deref(f, x):
    if isinstance(x, h5py.Reference):
        return f[x]
    return x


def field_refs(f, field_ds):
    return field_ds[()].ravel()


def read_cell_summary(f, sum_grp):
    out = {}
    for k in sum_grp.keys():
        out[k] = np.ravel(np.array(deref(f, sum_grp[k][()])))
    return out


def read_qdlin(f, cell_cycles_grp, matlab_cycle):
    qd_refs = cell_cycles_grp["Qdlin"][()].ravel()
    idx = matlab_cycle - 1
    if idx < 0 or idx >= len(qd_refs):
        return None
    return np.ravel(np.array(deref(f, qd_refs[idx])))


# %% ---------- [2] 解析三个批次(4.1 原样) ----------
rows = []
for batch_name, fname in FILES.items():
    with h5py.File(rf"{MIT_DIR}\{fname}", "r") as f:
        b = f["batch"]
        life_refs = field_refs(f, b["cycle_life"])
        sum_refs = field_refs(f, b["summary"])
        cyc_refs = field_refs(f, b["cycles"])
        n_cells = len(life_refs)
        print(f"{batch_name}: {n_cells} 颗电芯")

        for i in range(n_cells):
            try:
                cycle_life = deref(f, life_refs[i])[()].item()
            except Exception:
                cycle_life = np.nan
            s = read_cell_summary(f, deref(f, sum_refs[i]))
            q_dis = s.get("QDischarge", np.array([]))
            qd10 = read_qdlin(f, deref(f, cyc_refs[i]), 10)
            qd100 = read_qdlin(f, deref(f, cyc_refs[i]), 100)
            rows.append({
                "batch": batch_name, "cell_idx": i, "cycle_life": cycle_life,
                "q_c100": q_dis[99] if len(q_dis) > 99 else np.nan,
                "qd10_ok": qd10 is not None and np.isfinite(qd10).all(),
                "qd100_ok": qd100 is not None and np.isfinite(qd100).all(),
                "_q_dis": q_dis, "_qd10": qd10, "_qd100": qd100,
            })

df = pd.DataFrame(rows)
print("解析完成:", len(df), "颗")


# %% ---------- [3] 合理性过滤(4.1 逻辑) ----------
def is_sane(r):
    if not np.isfinite(r["cycle_life"]) or r["cycle_life"] <= 0:
        return False
    if not (np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6):
        return False
    if not (r["qd10_ok"] and r["qd100_ok"]):
        return False
    return True


df["keep"] = df.apply(is_sane, axis=1)
print(df.groupby("batch")["keep"].agg(["sum", "count"]))


# %% ---------- [4] 特征计算(4.2 原样) ----------
def calc_features(r) -> dict:
    feats = {}
    qd10, qd100, q_dis = r["_qd10"], r["_qd100"], r["_q_dis"]
    if qd10 is not None and qd100 is not None and len(qd10) == len(qd100):
        dq = qd100 - qd10
        feats["log_var"] = np.log10(max(np.var(dq), 1e-12))
        feats["log_min"] = np.log10(max(-dq.min(), 1e-12))
        feats["log_mean"] = np.log10(max(abs(dq.mean()), 1e-12))
    if len(q_dis) >= 100:
        y = q_dis[1:100]
        x = np.arange(2, 101)
        slope, intercept = np.polyfit(x, y, 1)
        feats["slope"] = slope
        feats["intercept"] = intercept
        feats["discrepancy"] = np.polyval((slope, intercept), 100) - q_dis[99]
    return feats


feat_rows = []
for _, r in df[df["keep"]].iterrows():
    feats = calc_features(r)
    if not feats:
        continue
    feat_rows.append({"batch": r["batch"], "cell_idx": r["cell_idx"],
                      "cycle_life": r["cycle_life"], **feats})

feat_df = pd.DataFrame(feat_rows).dropna(subset=FEATURES)
print(f"\n特征完整可用的电芯: {len(feat_df)} 颗")
print(feat_df.groupby("batch")["cycle_life"].agg(["count", "median"]))


# %% ---------- [5] 训练/测试划分 + 随机森林(本阶段核心) ----------
# 论文口径:batch1 训练,batch2 主测试,batch3 二级测试 —— 划分按批次,不随机
train = feat_df[feat_df["batch"] == "batch1"]
test2 = feat_df[feat_df["batch"] == "batch2"]
test3 = feat_df[feat_df["batch"] == "batch3"]
print(f"\n划分: 训练 {len(train)} 颗(batch1) / 主测试 {len(test2)}(batch2) / "
      f"二级测试 {len(test3)}(batch3)")

X_train = train[FEATURES].values
y_train = np.log10(train["cycle_life"].values)   # 目标取 log,和图的口径一致

rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

# 训练集 5 折交叉验证:R² 应该高但不至于 0.99+(太高=过拟合警报)
cv_r2 = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
print(f"训练集 5 折 CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print("(若均值 >0.99 需警惕过拟合;batch1 只有 41 颗,方差大属正常)")

rf.fit(X_train, y_train)
print(f"训练集拟合 R²: {rf.score(X_train, y_train):.3f}(应明显高于 CV,差太多=过拟合)")

for name, sub in [("batch2", test2), ("batch3", test3)]:
    pred_log = rf.predict(sub[FEATURES].values)
    sub = sub.assign(pred_life=10 ** pred_log)
    if name == "batch2":
        pred2 = sub
    else:
        pred3 = sub

pred_all = pd.concat([pred2, pred3])
pred_all.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
print(f"\n预测结果已保存: {OUT_PRED}")


# %% ---------- [6] 误差评估(论文指标) ----------
def pct_error(pred, actual):
    return np.abs(pred - actual) / actual * 100

print("\n===== 误差汇总(percentage error,论文指标) =====")
for name, sub in [("batch2(主测试)", pred2), ("batch3(二级测试)", pred3)]:
    err = pct_error(sub["pred_life"], sub["cycle_life"])
    print(f"{name}: 中位 {np.median(err):.1f}% | 平均 {err.mean():.1f}% | "
          f"最差 {err.max():.1f}% (n={len(sub)})")

tr_pred = 10 ** rf.predict(X_train)
tr_err = pct_error(tr_pred, train["cycle_life"])
print(f"batch1(训练,仅参考): 中位 {np.median(tr_err):.1f}%")

print("""
对照:论文弹性网络 主测试中位误差 ~9.1%。
我们 41 颗训练 + 默认随机森林,中位 <15% 即算复现成功;
之后可以调参(n_estimators/max_depth/min_samples_leaf)或换弹性网络再压。
""")


# %% ---------- [7] 预测-真实散点(每个测试批次一张图,不堆叠) ----------
for name, sub, color in [("batch2", pred2, "tab:orange"), ("batch3", pred3, "tab:green")]:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [400, 2400]   # MIT 数据寿命大致范围
    ax.plot(lims, lims, "k--", lw=1, label="理想:预测=真实")
    ax.scatter(sub["cycle_life"], sub["pred_life"], s=32, alpha=0.75, color=color)
    err_med = np.median(pct_error(sub["pred_life"], sub["cycle_life"]))
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("真实寿命 (圈)")
    ax.set_ylabel("预测寿命 (圈)")
    ax.set_title(f"{name}:预测 vs 真实(中位误差 {err_med:.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS.output_dir / f"rf_pred_vs_actual_{name}.png", dpi=150)
    plt.show()
    print(f"已出图: rf_pred_vs_actual_{name}.png")


# %% ---------- [8] 特征重要性 ----------
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(imp.index, imp.values, color="tab:blue", alpha=0.8)
ax.set_xlabel("重要性(不纯度下降占比)")
ax.set_title("随机森林特征重要性")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(PATHS.output_dir / "rf_feature_importance.png", dpi=150)
plt.show()
print(f"已出图: rf_feature_importance.png\n特征重要性排序:\n{imp.sort_values(ascending=False)}")
