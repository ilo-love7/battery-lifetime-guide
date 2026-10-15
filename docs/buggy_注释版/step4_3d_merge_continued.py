# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
# 【事故现场注释版】保留此 buggy 版本仅作教学用途,请勿运行
# 项目: Severson 2019 复现 (阶段4.3)
# 错误编号: 清单 #3
# 续跑合并:正确写回示范 + 一个隐患
#
# 【现象】无(此版本跑通:标签 1851/2159/2236/1433/1708 正确)
# 【根因】两个隐患:①merged[1:] 只跳过第1圈占位,若占位不止1圈或中途有单圈假跌就会错(定稿版改为'连续3圈低于阈值');②查找表 qdis_lookup 由 iterrows() 构建——Series 副本!本文件只'读'它所以侥幸没事,一旦用它'写'标签就是 4.3e 那次事故(见清单#3)
# 【修复】另一 AI 版用 feat_df.loc[m, ...] 掩码写回(step4_3d_merge_continued.py 第151行),这是正确姿势 —— 定稿版见 step4_3f_official_split.py
# 【教学点】iterrows 返回副本:读可以,写必炸;查表存索引不存 Series
# ═══════════════════════════════════════════════════════════════
"""
电池寿命预测 —— 复现论文划分并比较跨批次泛化
================================================
按官方 LoadData.m 合并续跑电池、剔除样本，并使用 batch1+batch2 的官方索引划分
训练集和主测试集；以对数寿命 Elastic Net 作为论文方法基线，同时保留随机森林。
另报告 batch1 训练、整批 batch2 测试的跨批次结果，避免与论文划分混为一谈。
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_DIR = r"data/mit"
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
EOL_CAP = 0.88   # Ah = 1.1Ah × 80%,论文标签定义(LoadData.m)

OFFICIAL_EXCLUDE = {
    "batch1": {8, 10, 12, 13, 22},   # MATLAB [9,11,13,14,23] 测试中止
    "batch2": {7, 8, 9, 15, 16},     # MATLAB [8:10,16:17] 续跑(将被合并进 batch1)
}

# 续跑配对:batch1 接收方文件序号(0-based) ← batch2 捐献方文件序号(0-based)
# MATLAB 原文:batch1(i) 依次拼接 batch2_idx=[8:10,16:17](1-based)
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]


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


# %% ---------- [2] 解析 + 合理性过滤 + 特征(4.3c 原样) ----------
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
                          "_q_dis": r["_q_dis"], **feats})
feat_df = pd.DataFrame(feat_rows).dropna(subset=FEATURES)
print(f"合理性过滤后可用: {len(feat_df)} 颗")


# %% ---------- [3] 续跑合并 + 官方剔除 ----------
# 3a. 合并:batch1 前 5 颗的寿命标签用"两段 QDischarge 拼接"重算
#     (早期特征 qd10/qd100 来自 batch1 段,不变;只有寿命标签变长)
print("\n===== 续跑合并(batch2 文件后段 → batch1 前 5 颗) =====")
qdis_lookup = {(r["batch"], r["cell_idx"]): r["_q_dis"]
               for _, r in feat_df.iterrows()}
for recv, donor in MERGE_PAIRS:
    k1, k2 = ("batch1", recv), ("batch2", donor)
    if k1 not in qdis_lookup or k2 not in qdis_lookup:
        print(f"  跳过 pair {k1} ← {k2}(合理过滤中缺席,人工核对)")
        continue
    merged = np.concatenate([qdis_lookup[k1], qdis_lookup[k2]])
    # batch1 的 QDischarge 第一个值是 0 Ah 占位项，不是实际循环容量。
    below = np.flatnonzero(merged[1:] < EOL_CAP)
    new_life = int(below[0] + 2) if len(below) else len(merged)
    old_life = feat_df[(feat_df.batch == "batch1") &
                       (feat_df.cell_idx == recv)]["cycle_life"].iloc[0]
    m = (feat_df.batch == "batch1") & (feat_df.cell_idx == recv)
    feat_df.loc[m, "cycle_life"] = new_life
    print(f"  batch1#{recv} ← batch2#{donor}: 寿命 {old_life:.0f} → {new_life} 圈"
          f"  (拼接后共 {len(merged)} 圈记录)")

# 3b. 官方剔除(与 4.3c 相同;batch2 的捐献方 5 颗此时剔除)
def apply_official(g, batch_name):
    n0 = len(g)
    if batch_name == "batch3":
        # LoadData.m:先删除原始电芯 38，再剔除删失电芯，最后按剩余列表位置
        # 删除 MATLAB 索引 [3, 40:41]。这些位置不能直接当作原始 cell_idx。
        ordered = g.sort_values("cell_idx").copy()
        without_channel = ordered[~ordered["cell_idx"].isin({37})].copy()
        censored = without_channel["endcap"] > 0.885
        uncensored = without_channel[~censored].copy()
        noisy_positions = [i for i in (2, 39, 40) if i < len(uncensored)]
        noisy_indices = uncensored.iloc[noisy_positions].index
        out = uncensored.drop(index=noisy_indices).copy()
        print(f"{batch_name}: {n0} → 删除原始 idx38 {int((ordered['cell_idx'] == 37).sum())} 颗, "
              f"删失 {int(censored.sum())} 颗,按官方剩余列表位置剔除 "
              f"{len(noisy_indices)} 颗 → 剩 {len(out)} 颗")
        return out

    excl = set(OFFICIAL_EXCLUDE.get(batch_name, set()))
    m_idx = g["cell_idx"].isin(excl)
    out = g[~m_idx].copy()
    print(f"{batch_name}: {n0} → 剔名单 {int(m_idx.sum())} 颗 → 剩 {len(out)} 颗")
    return out

print("\n===== 官方剔除(LoadData.m 规则) =====")
off = pd.concat([apply_official(feat_df[feat_df["batch"] == bn], bn)
                 for bn in FILES])
print(f"官方口径总数: {len(off)} 颗(论文 124)")

batch1 = off[off["batch"] == "batch1"].sort_values("cell_idx")
test2 = off[off["batch"] == "batch2"].sort_values("cell_idx")
test3 = off[off["batch"] == "batch3"]
development = pd.concat([batch1, test2], ignore_index=True)
if len(development) != 84:
    print(f"警告:官方 batch1+batch2 开发集应为 84 颗,当前为 {len(development)}; "
          "后续按可用样本应用官方索引模式,结果不应直接与论文数值比较。")

# LoadData.m: test_ind = [1:2:(numBat1+numBat2), 84] (MATLAB 1-based).
# 对应 Python 0-based:0,2,...,82,83；其余为训练集。
paper_test_positions = list(range(0, len(development), 2))
if len(development) >= 84 and 83 not in paper_test_positions:
    paper_test_positions.append(83)
paper_test_positions = sorted(set(paper_test_positions))
paper_test = development.iloc[paper_test_positions].copy()
train_positions = [i for i in range(len(development)) if i not in set(paper_test_positions)]
train = development.iloc[train_positions].copy()

print(f"论文式划分:训练 {len(train)} 颗,主测试 {len(paper_test)} 颗,"
      f"训练寿命 {train['cycle_life'].min():.0f}~{train['cycle_life'].max():.0f} 圈")
print(f"整批次外推对照:batch1 {len(batch1)} 颗 → batch2 {len(test2)} 颗 "
      f"(寿命 {test2['cycle_life'].min():.0f}~{test2['cycle_life'].max():.0f}); "
      f"batch3 {len(test3)} 颗")


# %% ---------- [4] 训练与评估(论文式划分 + 跨批次对照) ----------
def pct_error(pred, actual):
    return np.abs(pred - actual) / actual * 100

cv = KFold(n_splits=4, shuffle=True, random_state=42)
print("\n===== 模型 A:随机森林(仅作对照) =====")
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

print("===== 模型 B:Elastic Net(论文基线,4 折 CV) =====")
enet = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                 cv=cv, random_state=42, max_iter=100000)
)
enet.fit(train[FEATURES].values, np.log10(train["cycle_life"].values))

print("\n===== 论文式主测试 + batch3 二级测试 =====")
summary = []
preds = {}
for name, model in [("RF", rf), ("弹性网络", enet)]:
    for bn, test in [("论文主测试", paper_test), ("batch3二级测试", test3)]:
        pred = 10 ** model.predict(test[FEATURES].values)
        err = pct_error(pred, test["cycle_life"].values)
        m_ok = test["cycle_life"].values >= 500
        err_ok = err[m_ok]
        summary.append({"模型": name, "测试集": bn,
                        "中位误差%": round(np.median(err), 1),
                        "平均误差%": round(err.mean(), 1),
                        "剔夭折后中位%": round(np.median(err_ok), 1),
                        "n_剔夭折后": int(m_ok.sum())})
        preds[(name, bn)] = (test["cell_idx"].values, pred, err)

# 保留原脚本的整批 batch2 外推测试，明确标成不同任务，不用于论文式误差比较。
print("\n===== 整批次外推诊断(batch1 全量训练 → batch2 全量测试) =====")
batch_preds = {}
for name, model in [("RF", rf), ("弹性网络", enet)]:
    if name == "RF":
        diagnostic_model = RandomForestRegressor(
            n_estimators=500, max_depth=best[0][1], min_samples_leaf=best[0][0],
            random_state=42, n_jobs=-1)
    else:
        diagnostic_model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                         cv=KFold(n_splits=4, shuffle=True, random_state=42),
                         random_state=42, max_iter=100000))
    diagnostic_model.fit(batch1[FEATURES].values, np.log10(batch1["cycle_life"].values))
    pred = 10 ** diagnostic_model.predict(test2[FEATURES].values)
    err = pct_error(pred, test2["cycle_life"].values)
    batch_preds[name] = (pred, err)
    m_ok = test2["cycle_life"].values >= 500
    summary.append({"模型": name, "测试集": "整批batch2外推",
                    "中位误差%": round(np.median(err), 1),
                    "平均误差%": round(err.mean(), 1),
                    "剔夭折后中位%": round(np.median(err[m_ok]), 1),
                    "n_剔夭折后": int(m_ok.sum())})
    print(f"  {name}: batch2 中位误差 {np.median(err):.1f}%,"
          f"平均误差 {err.mean():.1f}%")

comp = pd.DataFrame(summary)
print(comp.to_string(index=False))
print("""
判读:
  - 论文主测试采用官方 batch1+batch2 索引切分；它才是与论文主结果相对接近的测试口径。
  - 整批 batch1→batch2 是更严格的跨批次外推任务，不能直接与论文 9.1% 比较。
  - 本脚本当前只有论文公开的 6 个特征；论文完整建模代码未公开，因此不保证复现 9.1%。
""")
comp.to_csv(r"results/generated/mit_43d_summary.csv", index=False, encoding="utf-8-sig")

out = test2[["cell_idx", "cycle_life"]].copy()
out["pred_rf"] = batch_preds["RF"][0]
out["err_rf"] = batch_preds["RF"][1]
out["pred_en"] = batch_preds["弹性网络"][0]
out["err_en"] = batch_preds["弹性网络"][1]
out.to_csv(r"results/generated/mit_43d_batch2_predictions.csv",
           index=False, encoding="utf-8-sig")
print("逐颗预测已存: mit_43d_batch2_predictions.csv")


# %% ---------- [5] batch2 整批次外推预测-真实散点 ----------
for tag, errs in [("RF", batch_preds["RF"]),
                  ("ElasticNet", batch_preds["弹性网络"])]:
    pred, err = errs
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [100, 2400]
    ax.plot(lims, lims, "k--", lw=1, label="理想:预测=真实")
    sc = ax.scatter(test2["cycle_life"], pred, s=32, alpha=0.8,
                    c=err, cmap="viridis")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("真实寿命 (圈)")
    ax.set_ylabel("预测寿命 (圈)")
    ax.set_title(f"batch1→batch2 — {tag}(中位 {np.median(err):.1f}%)")
    fig.colorbar(sc, ax=ax, label="误差 %")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rf"results/generated/merged_{tag}_batch2.png", dpi=150)
    plt.show()
    print(f"已出图: merged_{tag}_batch2.png")
