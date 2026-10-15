# -*- coding: utf-8 -*-
"""
电池寿命数据分析 —— 阶段 4.3b:过拟合修复 + 模型对比 + 误差分析
================================================================
4.3 首轮结果:batch2 中位误差 33.6%,训练集 1.8% —— 过拟合;
    batch3 反而 9.1%(分布运气,不是模型好)。
本脚本做三件事:
  1. 调参随机森林:限深度/限叶子,压住"背题";
  2. 弹性网络(论文同款模型):线性模型可外推,小样本更稳;
  3. 误差分析:把 batch2 最差的 10 颗拉出来看错在哪个方向。
自包含,从头跑到尾。
"""

# %% ---------- [0] 导入与配置 ----------
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from project_config import PATHS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_DIR = PATHS.mit_dir
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
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


# %% ---------- [2] 解析 + 过滤 + 特征(4.1/4.2 原样合并) ----------
rows = []
for batch_name, fname in FILES.items():
    with h5py.File(rf"{MIT_DIR}\{fname}", "r") as f:
        b = f["batch"]
        life_refs = field_refs(f, b["cycle_life"])
        sum_refs = field_refs(f, b["summary"])
        cyc_refs = field_refs(f, b["cycles"])
        print(f"{batch_name}: {len(life_refs)} 颗电芯")
        for i in range(len(life_refs)):
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

def is_sane(r):
    return (np.isfinite(r["cycle_life"]) and r["cycle_life"] > 0
            and np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6
            and r["qd10_ok"] and r["qd100_ok"])

df["keep"] = df.apply(is_sane, axis=1)

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
    if feats:
        feat_rows.append({"batch": r["batch"], "cell_idx": r["cell_idx"],
                          "cycle_life": r["cycle_life"], **feats})
feat_df = pd.DataFrame(feat_rows).dropna(subset=FEATURES)
print(f"\n可用电芯: {len(feat_df)} 颗")

train = feat_df[feat_df["batch"] == "batch1"]
test2 = feat_df[feat_df["batch"] == "batch2"]
test3 = feat_df[feat_df["batch"] == "batch3"]

X_train = train[FEATURES].values
y_train = np.log10(train["cycle_life"].values)
y_train_raw = train["cycle_life"].values
print(f"训练寿命范围: {y_train_raw.min():.0f} ~ {y_train_raw.max():.0f} 圈")
print(f"batch2 寿命范围: {test2['cycle_life'].min():.0f} ~ {test2['cycle_life'].max():.0f} 圈"
      f"  ← 超出训练范围的电池,随机森林外推不了")


# %% ---------- [3] 模型 A:调参随机森林(压过拟合) ----------
def pct_error(pred, actual):
    return np.abs(pred - actual) / actual * 100


cv = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n===== 模型 A:调参随机森林 =====")
best = None
for leaf in [2, 3, 5, 8]:
    for depth in [3, 5, 8, None]:
        m = RandomForestRegressor(n_estimators=500, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        # 用负 MAE 选参(和百分比误差同方向),在 log 空间算
        scores = -cross_val_score(m, X_train, y_train, cv=cv,
                                  scoring="neg_mean_absolute_error")
        mae = scores.mean()
        tag = f"leaf={leaf}, depth={depth}"
        print(f"  {tag:<22} CV MAE(log) = {mae:.4f}")
        if best is None or mae < best[1]:
            best = (tag, mae, leaf, depth)

print(f"  → 最优: {best[0]}")
rf = RandomForestRegressor(n_estimators=500, max_depth=best[3],
                           min_samples_leaf=best[2], random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred2_rf = 10 ** rf.predict(test2[FEATURES].values)
pred3_rf = 10 ** rf.predict(test3[FEATURES].values)
err2_rf = pct_error(pred2_rf, test2["cycle_life"].values)
err3_rf = pct_error(pred3_rf, test3["cycle_life"].values)
tr_err_rf = pct_error(10 ** rf.predict(X_train), y_train_raw)
print(f"  batch2: 中位 {np.median(err2_rf):.1f}% | 平均 {err2_rf.mean():.1f}%")
print(f"  batch3: 中位 {np.median(err3_rf):.1f}% | 平均 {err3_rf.mean():.1f}%")
print(f"  batch1(训练): 中位 {np.median(tr_err_rf):.1f}%(调参后应上升,正常)")


# %% ---------- [4] 模型 B:弹性网络(论文同款,可外推) ----------
print("\n===== 模型 B:弹性网络 =====")
enet = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=100,
                 cv=5, random_state=42, max_iter=100000)
)
enet.fit(X_train, y_train)

pred2_en = 10 ** enet.predict(test2[FEATURES].values)
pred3_en = 10 ** enet.predict(test3[FEATURES].values)
err2_en = pct_error(pred2_en, test2["cycle_life"].values)
err3_en = pct_error(pred3_en, test3["cycle_life"].values)
tr_err_en = pct_error(10 ** enet.predict(X_train), y_train_raw)
print(f"  batch2: 中位 {np.median(err2_en):.1f}% | 平均 {err2_en.mean():.1f}%")
print(f"  batch3: 中位 {np.median(err3_en):.1f}% | 平均 {err3_en.mean():.1f}%")
print(f"  batch1(训练): 中位 {np.median(tr_err_en):.1f}%")


# %% ---------- [5] 两模型对比汇总 ----------
print("\n===== 对比汇总(percentage error 中位数) =====")
print(f"{'':<14}{'batch2 主测试':>14}{'batch3 二级测试':>16}")
print(f"{'RF(4.3 默认)':<14}{'33.6%':>14}{'9.1%':>16}")
print(f"{'RF(调参)':<14}{f'{np.median(err2_rf):.1f}%':>14}{f'{np.median(err3_rf):.1f}%':>16}")
print(f"{'弹性网络':<14}{f'{np.median(err2_en):.1f}%':>14}{f'{np.median(err3_en):.1f}%':>16}")
print("论文弹性网络:      ~9.1%")

# 存两个模型的逐颗预测
out = test2[["batch", "cell_idx", "cycle_life"]].copy()
out["pred_rf"] = pred2_rf; out["pred_enet"] = pred2_en
out2 = test3[["batch", "cell_idx", "cycle_life"]].copy()
out2["pred_rf"] = pred3_rf; out2["pred_enet"] = pred3_en
pd.concat([out, out2]).to_csv(PATHS.output_dir / "mit_model_compare.csv",
                              index=False, encoding="utf-8-sig")
print("逐颗对比已存: mit_model_compare.csv")


# %% ---------- [6] 误差分析:batch2 最差 10 颗错在哪 ----------
best_pred2 = pred2_en if np.median(err2_en) <= np.median(err2_rf) else pred2_rf
model_name = "弹性网络" if np.median(err2_en) <= np.median(err2_rf) else "RF(调参)"
ana = test2.copy()
ana["pred"] = best_pred2
ana["pct_err"] = pct_error(ana["pred"], ana["cycle_life"])
worst = ana.sort_values("pct_err", ascending=False).head(10)
print(f"\n===== batch2 最差 10 颗(模型: {model_name}) =====")
print(worst[["cell_idx", "cycle_life", "pred", "pct_err", "log_var"]]
      .to_string(index=False,
                 formatters={"pct_err": "{:.1f}%".format,
                             "log_var": "{:.3f}".format}))
print("""
判读方法:
  - 若最差样本集中在"真实寿命 > 训练集上限"的电池 → 随机森林外推问题,
    弹性网络应明显更好(线性模型能外推);
  - 若最差样本的 pred > 真实(把短命的预测成长命)→ 看它的 log_var 是否偏小,
    即"早期看起来健康但后来跳水"的拐点型死亡 —— 这是数据本身的极限,
    论文也没完全解决,面试可以当 limitation 讲。
""")


# %% ---------- [7] 两模型 batch2 预测-真实散点(两张图,不堆叠) ----------
for mname, p2, e2 in [("RF_tuned", pred2_rf, err2_rf), ("ElasticNet", pred2_en, err2_en)]:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [400, 2400]
    ax.plot(lims, lims, "k--", lw=1, label="理想:预测=真实")
    ax.scatter(test2["cycle_life"], p2, s=32, alpha=0.75,
               c=e2, cmap="viridis")   # 颜色深浅=误差大小
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("真实寿命 (圈)")
    ax.set_ylabel("预测寿命 (圈)")
    ax.set_title(f"batch2 预测 vs 真实 — {mname}(中位 {np.median(e2):.1f}%)")
    fig.colorbar(ax.collections[1], ax=ax, label="误差 %")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS.output_dir / f"compare_{mname}_batch2.png", dpi=150)
    plt.show()
    print(f"已出图: compare_{mname}_batch2.png")
