# -*- coding: utf-8 -*-
"""
电池寿命数据分析 —— 阶段 4.3c:官方剔除名单 + 留一批次轮换
================================================================
来源:论文官方加载脚本 LoadData.m(github.com/rdbraatz/data-driven-prediction-...)
此前 4.3b 的考卷和论文不同:batch2 混着 5 颗"续跑"电池,batch3 混着
故障/删失电池,batch1 带着 5 颗测试中止电池在训练。本脚本换官方考卷。

官方规则(MATLAB 1-based,已换算成 python 0-based):
  batch1: 剔 {8,10,12,13,22}     ← MATLAB [9,11,13,14,23] 测试中止(optional,论文实际剔了)
  batch2: 剔 {7,8,9,15,16}       ← MATLAB [8:10,16:17] 续跑到 batch1 的 5 颗
  batch3: 剔 {2,37}              ← MATLAB idx3(noisy) + idx38(channel46 采集故障)
           剔末圈 QDischarge > 0.885 Ah 的(删失:没测到 EOL)
           MATLAB [40:41] 因前面剔除导致索引位移,只在数组够长时按原始序号 {39,40} 近似
注:论文总数 124 = batch1 41(含 5 颗续跑、剔 5 颗中止) + batch2 43 + batch3 40。
    我们的 batch1 不做"续跑合并"(那 5 颗的数据在 batch2 文件里被截断,合并逻辑复杂),
    所以总数会与 124 有出入,在 [5] 格核对并说明。

另含:留一批次轮换(Leave-One-Batch-Out),回答"换批训练是否更准"。
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

OFFICIAL_EXCLUDE = {
    "batch1": {8, 10, 12, 13, 22},
    "batch2": {7, 8, 9, 15, 16},
    "batch3": {2, 37},
}


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


# %% ---------- [2] 解析 + 合理性过滤 + 特征(4.3b 原样) ----------
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
                "endcap": q_dis[-1] if len(q_dis) else np.nan,
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
                          "cycle_life": r["cycle_life"], "endcap": r["endcap"],
                          **feats})
feat_df = pd.DataFrame(feat_rows).dropna(subset=FEATURES)
print(f"合理性过滤后可用: {len(feat_df)} 颗")


# %% ---------- [3] 官方剔除(逐批打印,核对总数) ----------
def apply_official(g, batch_name):
    n0 = len(g)
    excl = set(OFFICIAL_EXCLUDE.get(batch_name, set()))
    # batch3 的 MATLAB [40:41](noisy):原始序号近似 python {39,40},越界自动跳过
    if batch_name == "batch3":
        max_idx = g["cell_idx"].max()
        for j in (39, 40):
            if j <= max_idx:
                excl.add(j)
    m_idx = g["cell_idx"].isin(excl)
    # batch3 删失规则:末圈容量 > 0.885 Ah,没测到 EOL
    if batch_name == "batch3":
        m_cens = g["endcap"] > 0.885
    else:
        m_cens = pd.Series(False, index=g.index)
    out = g[~(m_idx | m_cens)].copy()
    print(f"{batch_name}: {n0} → 剔名单 {int(m_idx.sum())} 颗, "
          f"剔删失 {int(m_cens.sum())} 颗 → 剩 {len(out)} 颗")
    return out

print("\n===== 官方剔除(LoadData.m 规则) =====")
off = pd.concat([apply_official(feat_df[feat_df["batch"] == bn], bn)
                 for bn in FILES])
print(f"官方口径总数: {len(off)} 颗(论文 124;差异来自 batch1 不做续跑合并 + "
      f"batch3 的 [40:41] 序号近似,在报告里说明即可)")

train = off[off["batch"] == "batch1"]
test2 = off[off["batch"] == "batch2"]
test3 = off[off["batch"] == "batch3"]
print(f"训练集 {len(train)} 颗,寿命 {train['cycle_life'].min():.0f}~"
      f"{train['cycle_life'].max():.0f} 圈 | batch2 {len(test2)} 颗 | batch3 {len(test3)} 颗")


# %% ---------- [4] 官方考卷上重跑两模型 ----------
def pct_error(pred, actual):
    return np.abs(pred - actual) / actual * 100


cv = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n===== 模型 A:随机森林(小网格) =====")
best = None
for leaf in [2, 3, 5]:
    for depth in [3, 5, None]:
        m = RandomForestRegressor(n_estimators=500, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        mae = -cross_val_score(m, train[FEATURES].values,
                               np.log10(train["cycle_life"].values),
                               cv=cv, scoring="neg_mean_absolute_error").mean()
        if best is None or mae < best[1]:
            best = ((leaf, depth), mae)
print(f"  最优 leaf,depth = {best[0]}  (CV MAE {best[1]:.4f})")
rf = RandomForestRegressor(n_estimators=500, max_depth=best[0][1],
                           min_samples_leaf=best[0][0], random_state=42, n_jobs=-1)
rf.fit(train[FEATURES].values, np.log10(train["cycle_life"].values))

print("===== 模型 B:弹性网络 =====")
enet = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=100,
                 cv=5, random_state=42, max_iter=100000)
)
enet.fit(train[FEATURES].values, np.log10(train["cycle_life"].values))

def eval_model(model, name):
    res = {}
    for bn, test in [("batch2", test2), ("batch3", test3)]:
        pred = 10 ** model.predict(test[FEATURES].values)
        err = pct_error(pred, test["cycle_life"].values)
        res[bn] = (pred, err)
        print(f"  {name} {bn}: 中位 {np.median(err):.1f}% | 平均 {err.mean():.1f}%")
    return res

print("\n===== 官方考卷结果(percentage error 中位数) =====")
res_rf = eval_model(rf, "RF")
res_en = eval_model(enet, "弹性网络")
print(f"参考: 4.3b(旧考卷) RF 32.7% / 弹性网络 40.9%; 论文 ~9.1%")

# 存逐颗预测
out_rows = []
for (bn, test), (pr, er) in zip([("batch2", test2), ("batch3", test3)],
                                [res_rf.values(), res_en.values()]):
    pass
out = test2[["batch", "cell_idx", "cycle_life"]].copy()
out["pred_rf"], out["err_rf"] = res_rf["batch2"]
out["pred_en"], out["err_en"] = res_en["batch2"]
out.to_csv(PATHS.output_dir / "mit_model_compare_official.csv",
           index=False, encoding="utf-8-sig")
print("逐颗预测已存: mit_model_compare_official.csv")


# %% ---------- [5] 留一批次轮换(Leave-One-Batch-Out) ----------
print("\n===== 留一批次轮换:换批训练是否更准? =====")
batches = ["batch1", "batch2", "batch3"]
rotation_rows = []
for held_out in batches:
    tr = off[off["batch"] != held_out]
    te = off[off["batch"] == held_out]
    m = RandomForestRegressor(n_estimators=500, max_depth=5,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
    m.fit(tr[FEATURES].values, np.log10(tr["cycle_life"].values))
    pred = 10 ** m.predict(te[FEATURES].values)
    err = pct_error(pred, te["cycle_life"].values)
    rotation_rows.append({"训练": "+".join(b for b in batches if b != held_out),
                          "测试": held_out, "n_test": len(te),
                          "中位误差%": round(np.median(err), 1),
                          "平均误差%": round(err.mean(), 1)})
rot = pd.DataFrame(rotation_rows)
print(rot.to_string(index=False))
print("""
判读:
  - 三折中位误差若接近(如都在 10~20%)→ 特征跨批次稳健,单次划分的 33.6% 确实掺了运气;
  - 若某折明显更差 → 那个批次有系统性差异(如 batch3 的删失/规程差),单独讨论;
  - 注意:轮换不是论文口径,主结果仍以 [4] 的官方划分为准,轮换是稳健性分析。
""")
rot.to_csv(PATHS.output_dir / "mit_rotation.csv", index=False, encoding="utf-8-sig")

# 出图:轮换误差条形图
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(range(len(rot)), rot["中位误差%"], color="tab:blue", alpha=0.8)
ax.set_xticks(range(len(rot)))
ax.set_xticklabels(rot["训练"] + " → " + rot["测试"], fontsize=9)
ax.set_ylabel("中位百分比误差 %")
ax.set_title("留一批次轮换:三折泛化误差对比")
for i, v in enumerate(rot["中位误差%"]):
    ax.text(i, v + 0.3, f"{v}%", ha="center", fontsize=10)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(PATHS.output_dir / "rotation_bars.png", dpi=150)
plt.show()
print("已出图: rotation_bars.png")
