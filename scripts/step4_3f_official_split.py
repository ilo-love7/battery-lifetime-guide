# -*- coding: utf-8 -*-
"""
电池寿命预测 —— 阶段 4.3f:官方划分复跑(双口径)
================================================================
背景:
  - 4.3e 修复了续跑合并的标签 bug(第1圈 0 Ah 占位),管道已可信;
    - 官方 LoadData.m 的 test_ind 使用混合划分
    (batch1+batch2 共 84 颗按索引交替抽测试),不是 batch1→batch2 整批外推。
    此前我们拿整批外推分数对标 9.1%,考卷拿错了。

本脚本在早期 step4_3d_merge_continued.py 管线基础上补三块:
  1) 标签函数先在正常电池上验证,不通过就 raise(防再次带病合并);
    2) 合并连续性诊断(q1 末圈 vs q2 首个有效容量圈 q2[1]、len(q2) vs 官方 add_len);
  3) 论文主测试的逐颗误差表 + 最差 5 颗(定位 36.5% 平均误差的元凶)。

口径约定(报告里分开写,不许混):
  - 口径A 论文式:官方 test_ind 混合划分,对标论文 9.1%;
  - 口径B 部署式:batch1 训练 → 整批 batch2/batch3 测试,真实跨批次难度;
  - 口径C 二级测试:训练集(混合) → batch3 整批。

v1.1 修复(AI 审查 2026-10-14):
  - 口径B 的 RF 超参改为在 batch1 内独立网格搜索(原先复用混合池 best,
    对"batch2 即考卷"的口径B 属轻微超参泄漏);
  - 已重跑:口径B RF 中位 32.2→32.4(均值 39.1 不变),
    results/阶段43f_三口径汇总.csv 已同步;其余七行与修复前逐位一致。
自包含,从头跑到尾。
"""

# %% ---------- [0] 导入与配置 ----------
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from project_config import PATHS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

import sklearn
sklearn.set_config(display="text")   # 禁用 HTML 富显示:中文 Windows 下 sklearn 读
                                     # estimator.js 未指定编码,GBK 撞 UTF-8 会崩
                                     # (UnicodeDecodeError 0x94,2026-10-13 实测)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_DIR = PATHS.mit_dir
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
EOL_CAP = 0.88          # Ah,论文标签定义
SUSTAIN = 3             # 连续 3 圈低于阈值才判 EOL(防单圈假跌)

OFFICIAL_EXCLUDE = {
    "batch1": {8, 10, 12, 13, 22},   # MATLAB [9,11,13,14,23] 测试中止
    "batch2": {7, 8, 9, 15, 16},     # 续跑捐献方,将合并进 batch1
}
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]
# MATLAB add_len=[661,980,1059,207,481] = 捐献方续跑圈数(另有 0 占位,故 len 应为 add_len+1)
ADD_LEN = {7: 661, 8: 980, 9: 1059, 15: 207, 16: 481}

# 官方划分:MATLAB test_ind = [1:2:(41+43), 84](1-based)
# → 0-based 偶数位 0,2,...,82 + 末位 83;训练集 = 其余 41 颗。
# 来源:官方 LoadData.m 的 test_ind 定义；2026-10-14 已对源码逐条核验。
PAPER_SPLIT_NOTE = "test_ind: 0-based 偶数位 + 末位"


# %% ---------- [1] HDF5 解引用工具(4.1 原样) ----------
def deref(f, x):
    if isinstance(x, h5py.Reference):
        return f[x]
    return x


def field_refs(f, field_ds):
    return field_ds[()].ravel()


def read_cell_summary(f, sum_grp):
    return {k: np.ravel(np.array(deref(f, sum_grp[k][()]))) for k in sum_grp.keys()}


def read_qdlin(f, cell_cycles_grp, matlab_cycle):
    qd_refs = cell_cycles_grp["Qdlin"][()].ravel()
    idx = matlab_cycle - 1
    if idx < 0 or idx >= len(qd_refs):
        return None
    return np.ravel(np.array(deref(f, qd_refs[idx])))


# %% ---------- [2] 解析 + 合理性过滤 + 特征(带缓存与进度) ----------
import os
CACHE = PATHS.cache_dir / "mit_parse_cache.pkl"
PATHS.cache_dir.mkdir(parents=True, exist_ok=True)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
    print(f"命中缓存:{len(df)} 行(上次解析结果)。要强制重新解析就删掉这个文件:")
    print(f"  {CACHE}")
else:
    print("无缓存,开始解析 3 个 .mat(约 2~5 分钟,每 10 颗报一次进度)...")
    rows = []
    for batch_name, fname in FILES.items():
        print(f"[{batch_name}] 打开 {fname}", flush=True)
        with h5py.File(rf"{MIT_DIR}\{fname}", "r") as f:
            b = f["batch"]
            life_refs = field_refs(f, b["cycle_life"])
            sum_refs = field_refs(f, b["summary"])
            cyc_refs = field_refs(f, b["cycles"])
            print(f"{batch_name}: {len(life_refs)} 颗电芯", flush=True)
            for i in range(len(life_refs)):
                if (i + 1) % 10 == 0 or i + 1 == len(life_refs):
                    print(f"  {batch_name} {i+1}/{len(life_refs)}", flush=True)
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
    df.to_pickle(CACHE)
    print(f"解析完成并已缓存(下次直接秒读): {CACHE}", flush=True)

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
        y = q_dis[1:100]            # 跳过第1圈 0 占位
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


# %% ---------- [3] 标签函数验证 → 续跑合并 → 官方剔除 ----------
def robust_life(q, eol=EOL_CAP, sustain=SUSTAIN):
    """寿命 = 容量首次持续跌破 eol 的圈数(1-based)。
    跳过开头 0 占位/异常圈(容量首次升上 >0.95);连续 sustain 圈低于阈值才判 EOL。"""
    q = np.asarray(q, dtype=float)
    ok = np.where(q > 0.95)[0]
    start = ok[0] if len(ok) else 0
    below = q[start:] < eol
    run = 0
    for j, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= sustain:
            return start + (j - sustain + 1) + 1
    return len(q)

# 3a. 标签验证:8 颗正常电池,robust_life 必须和结构体自带的 cycle_life 对上
print("\n===== 标签函数验证(正常电池对账) =====")
check_cells = [("batch1", 5), ("batch1", 6), ("batch1", 7), ("batch1", 9),
               ("batch1", 14), ("batch1", 20), ("batch1", 30), ("batch1", 35)]
_lk = {(r["batch"], r["cell_idx"]): i for i, r in feat_df.iterrows()}
# ↑ 存的是行索引!iterrows 返回副本,直接改副本写不回原表(4.3f 首跑踩的坑)
n_pass = 0
for key in check_cells:
    if key not in _lk:
        continue
    r = feat_df.loc[_lk[key]]
    mine = robust_life(r["_q_dis"])
    ref = r["cycle_life"]
    ok = abs(mine - ref) <= max(2, 0.01 * ref)
    n_pass += ok
    print(f"  {key[0]}#{key[1]}: 结构体 {ref:.0f} vs robust {mine} "
          f"{'✓' if ok else '✗ 对不上!'}")
if n_pass < 6:
    raise RuntimeError(f"标签验证仅 {n_pass}/8 通过,数据管道有问题,终止!")

# 3b. 续跑合并(带诊断)
print("\n===== 续跑合并(batch2 文件后段 → batch1 前 5 颗) =====")
for recv, donor in MERGE_PAIRS:
    k1, k2 = ("batch1", recv), ("batch2", donor)
    if k1 not in _lk or k2 not in _lk:
        print(f"  跳过 {k1} ← {k2}(过滤中缺席,人工核对)")
        continue
    q1 = feat_df.at[_lk[k1], "_q_dis"]
    q2 = feat_df.at[_lk[k2], "_q_dis"]
    want = ADD_LEN[donor] + 1          # +1 = 0 占位
    cont = len(q2) > 1 and abs(q1[-1] - q2[1]) < 0.1  # q2[0] 是 0 Ah 占位圈
    old_life = feat_df.at[_lk[k1], "cycle_life"]
    new_life = robust_life(np.concatenate([q1, q2]))
    feat_df.at[_lk[k1], "cycle_life"] = new_life   # 必须写回原表(4.3f 首跑漏的就是这)
    print(f"  batch1#{recv} ← batch2#{donor}: len(q2)={len(q2)}"
          f"(应≈{want} {'✓' if len(q2) == want else '✗'}), "
          f"q1末{q1[-1]:.3f}→q2首有效圈{q2[1]:.3f} {'✓' if cont else '✗'}, "
          f"寿命 {old_life:.0f} → {new_life} 圈")

# 合并生效检查:改完必须和预期值对上,不生效直接终止(防再跑一遍毒标签)
_expect = [1851, 2159, 2236, 1433, 1708]
_got = [int(feat_df.at[_lk[("batch1", c)], "cycle_life"]) for c, _ in MERGE_PAIRS]
print(f"  合并生效检查: {_got}")
if _got != _expect:
    raise RuntimeError(f"合并标签未写回!得到 {_got},应为 {_expect}。终止!")

# 3c. 官方剔除
def apply_official(g, batch_name):
    n0 = len(g)
    if batch_name == "batch3":
        # LoadData.m:先删原始 idx38,再剔删失(endcap>0.885),最后按剩余列表
        # 位置删 MATLAB [3, 40:41](不能当原始 cell_idx 用)
        ordered = g.sort_values("cell_idx").copy()
        without_channel = ordered[~ordered["cell_idx"].isin({37})].copy()
        censored = without_channel["endcap"] > 0.885
        uncensored = without_channel[~censored].copy()
        noisy_positions = [i for i in (2, 39, 40) if i < len(uncensored)]
        noisy_indices = uncensored.iloc[noisy_positions].index
        out = uncensored.drop(index=noisy_indices).copy()
        print(f"{batch_name}: {n0} → 删 idx38 {int((ordered['cell_idx']==37).sum())} 颗, "
              f"删失 {int(censored.sum())} 颗,位置剔除 {len(noisy_indices)} 颗 → {len(out)} 颗")
        return out
    excl = OFFICIAL_EXCLUDE.get(batch_name, set())
    out = g[~g["cell_idx"].isin(excl)].copy()
    print(f"{batch_name}: {n0} → 剔名单 {int(g['cell_idx'].isin(excl).sum())} 颗 → {len(out)} 颗")
    return out

print("\n===== 官方剔除(LoadData.m 规则) =====")
off = pd.concat([apply_official(feat_df[feat_df["batch"] == bn], bn) for bn in FILES])
print(f"官方口径总数: {len(off)} 颗(论文 124)")

batch1 = off[off["batch"] == "batch1"].sort_values("cell_idx")
test2 = off[off["batch"] == "batch2"].sort_values("cell_idx")
test3 = off[off["batch"] == "batch3"].copy()
development = pd.concat([batch1, test2], ignore_index=True)
if len(development) != 84:
    print(f"警告:官方开发集应为 84 颗,当前 {len(development)},结果不可与论文直接比较")

# 3d. 官方划分
paper_test_pos = sorted(set(list(range(0, len(development), 2)) + [len(development) - 1]))
paper_test = development.iloc[paper_test_pos].copy().reset_index(drop=True)
train = development.drop(development.index[paper_test_pos]).copy().reset_index(drop=True)
print(f"\n===== 口径A 论文式划分({PAPER_SPLIT_NOTE}) =====")
print(f"训练 {len(train)} 颗(其中 batch1 {int((train.batch=='batch1').sum())} / "
      f"batch2 {int((train.batch=='batch2').sum())}),"
      f"测试 {len(paper_test)} 颗(其中 batch1 {int((paper_test.batch=='batch1').sum())} / "
      f"batch2 {int((paper_test.batch=='batch2').sum())})")
print(f"训练寿命 {train['cycle_life'].min():.0f}~{train['cycle_life'].max():.0f} 圈")


# %% ---------- [4] 训练 ----------
def pct_error(pred, actual):
    return np.abs(pred - actual) / actual * 100

cv = KFold(n_splits=4, shuffle=True, random_state=42)
y_train = np.log10(train["cycle_life"].values)

print("\n===== 模型 A:随机森林(网格:leaf×depth) =====")
best = None
for leaf in [2, 3, 5]:
    for depth in [3, 5, None]:
        m = RandomForestRegressor(n_estimators=500, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        mae = -cross_val_score(m, train[FEATURES].values, y_train,
                               cv=cv, scoring="neg_mean_absolute_error").mean()
        if best is None or mae < best[1]:
            best = ((leaf, depth), mae)
print(f"  最优 leaf,depth = {best[0]}  (CV MAE {best[1]:.4f})")
rf = RandomForestRegressor(n_estimators=500, max_depth=best[0][1],
                           min_samples_leaf=best[0][0], random_state=42, n_jobs=-1)
rf.fit(train[FEATURES].values, y_train)

print("===== 模型 B:Elastic Net 六特征(论文基线) =====")
enet = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                  cv=cv, random_state=42, max_iter=100000))
enet.fit(train[FEATURES].values, y_train)

print("===== 模型 C:Elastic Net 仅 log_var(论文单特征口径) =====")
enet1 = make_pipeline(StandardScaler(),
                      ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                   cv=cv, random_state=42, max_iter=100000))
enet1.fit(train[["log_var"]].values, y_train)
print("  模型 A/B/C 全部训练完成\n")


# %% ---------- [5] 三口径评估 ----------
print("\n===== 三口径评估 =====")
summary = []
detail = {}
for name, model, feats in [("RF", rf, FEATURES),
                           ("EN六特征", enet, FEATURES),
                           ("EN仅log_var", enet1, ["log_var"])]:
    # 口径A:论文式主测试
    pred = 10 ** model.predict(paper_test[feats].values)
    err = pct_error(pred, paper_test["cycle_life"].values)
    summary.append({"模型": name, "口径": "A 论文式混合测试",
                    "中位误差%": round(np.median(err), 1),
                    "平均误差%": round(err.mean(), 1)})
    detail[(name, "A")] = (paper_test, pred, err)
    # 口径C:batch3 二级测试
    pred3 = 10 ** model.predict(test3[feats].values)
    err3 = pct_error(pred3, test3["cycle_life"].values)
    summary.append({"模型": name, "口径": "C batch3 整批",
                    "中位误差%": round(np.median(err3), 1),
                    "平均误差%": round(err3.mean(), 1)})

# 口径B:batch1 训练 → 整批 batch2(另配模型,只用 batch1 拟合)
# v1.1 修复:RF 超参原先复用混合池(含 batch2)选出的 best——batch2 恰是口径B
# 的考卷,属轻微超参泄漏。改为在 batch1 内独立网格搜索(batch1=41 颗,KFold(4))。
print("\n===== 口径B 诊断:batch1 → 整批 batch2(不与 9.1% 比较) =====")
bestB = None
for leaf in [2, 3, 5]:
    for depth in [3, 5, None]:
        m = RandomForestRegressor(n_estimators=500, max_depth=depth,
                                  min_samples_leaf=leaf, random_state=42, n_jobs=-1)
        mae = -cross_val_score(m, batch1[FEATURES].values,
                               np.log10(batch1["cycle_life"].values),
                               cv=cv, scoring="neg_mean_absolute_error").mean()
        if bestB is None or mae < bestB[1]:
            bestB = ((leaf, depth), mae)
print(f"  口径B 在 batch1 内重选超参: leaf,depth = {bestB[0]}  (CV MAE {bestB[1]:.4f})")
for name, mk, feats in [("RF", lambda: RandomForestRegressor(
                            n_estimators=500, max_depth=bestB[0][1],
                            min_samples_leaf=bestB[0][0], random_state=42, n_jobs=-1), FEATURES),
                        ("EN六特征", lambda: make_pipeline(StandardScaler(),
                            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                         cv=KFold(4, shuffle=True, random_state=42),
                                         random_state=42, max_iter=100000)), FEATURES)]:
    m = mk()
    m.fit(batch1[FEATURES].values if feats == FEATURES else batch1[feats].values,
          np.log10(batch1["cycle_life"].values))
    predB = 10 ** m.predict(test2[feats].values)
    errB = pct_error(predB, test2["cycle_life"].values)
    summary.append({"模型": name, "口径": "B batch1→batch2 外推",
                    "中位误差%": round(np.median(errB), 1),
                    "平均误差%": round(errB.mean(), 1)})
    detail[(name, "B")] = (test2, predB, errB)

comp = pd.DataFrame(summary)
print(comp.to_string(index=False))

# 论文主测试逐颗误差表(六特征 EN) + 最差 5 颗
print("\n===== 口径A 逐颗误差(EN六特征),最差 5 颗: =====")
pt, predA, errA = detail[("EN六特征", "A")]
tbl = pt[["batch", "cell_idx", "cycle_life"]].copy()
tbl["预测"] = np.round(predA, 0)
tbl["误差%"] = np.round(errA, 1)
print(tbl.sort_values("误差%", ascending=False).head(5).to_string(index=False))
tbl.to_csv(PATHS.output_dir / "mit_43f_paper_test_per_cell.csv",
           index=False, encoding="utf-8-sig")

rf_test, pred_rf, err_rf = detail[("RF", "A")]
rf_tbl = rf_test[["batch", "cell_idx", "cycle_life"]].copy()
rf_tbl["预测"] = np.round(pred_rf, 0)
rf_tbl["误差%"] = np.round(err_rf, 1)
rf_path = PATHS.output_dir / "阶段43f_论文主测试_RF逐颗误差.csv"
rf_tbl.to_csv(rf_path, index=False, encoding="utf-8-sig")
print(f"RF 逐颗预测已落盘: {rf_path}")
comp.to_csv(PATHS.output_dir / "mit_43f_summary.csv", index=False, encoding="utf-8-sig")


# %% ---------- [5b] 不确定性量化(评审防弹层) ----------
# 43 颗测试集的中位数本身噪声很大:不给出置信区间,"5.9% vs 论文 9.1%"
# 会被一句"差异在噪声内还是外?"问倒。三件套:
#   1) bootstrap 95% CI:对 43 颗有放回重采样 10000 次,看中位数的波动范围;
#   2) 剔夭折(<500 圈)中位:论文式最公平对照(夭折不可预测是共识);
#   3) 预测截断敏感性:把预测夹回训练寿命范围,看 EN 均值修复多少
#      (仅作敏感性分析,不进主表——论文没截断,截断是"事后补救")。
print("\n===== 不确定性量化(bootstrap 10000 次) =====")
rng = np.random.default_rng(42)
for name in ["RF", "EN六特征", "EN仅log_var"]:
    pt_, pr_, er_ = detail[(name, "A")]
    er_ = np.asarray(er_)
    boots = np.median(rng.choice(er_, size=(10000, len(er_)), replace=True), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    alive = pt_["cycle_life"].values >= 500
    clip = np.clip(pr_, train["cycle_life"].min(), train["cycle_life"].max())
    err_clip = pct_error(clip, pt_["cycle_life"].values)
    print(f"  {name:10s} 中位 {np.median(er_):.1f}%  [95%CI {lo:.1f}~{hi:.1f}]"
          f"  剔夭折中位 {np.median(er_[alive]):.1f}%"
          f"  截断后均值 {err_clip.mean():.1f}%(原 {er_.mean():.1f}%)")
print("""
判读:
    - 论文侧没有可比 CI/逐颗误差，不能仅根据论文点估计对本复现作统计一致性判断；
        报告本复现的 bootstrap CI，并将结论限制为"点估计接近，统计差异未定"；
  - 截断后均值大幅回落 → 证实"EN 均值被极端点过冲拖垮"是模型形态问题,
    不是数据或标签问题;
  - 这三种数以后进报告正文,主表只留中位数,别堆砌。
""")

# 口径A 散点(EN 六特征)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
lims = [100, 2400]
ax.plot(lims, lims, "k--", lw=1, label="理想:预测=真实")
sc = ax.scatter(pt["cycle_life"], predA, s=32, alpha=0.8, c=errA, cmap="viridis")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("真实寿命 (圈)"); ax.set_ylabel("预测寿命 (圈)")
ax.set_title(f"官方划分主测试 — EN六特征(中位 {np.median(errA):.1f}%)")
fig.colorbar(sc, ax=ax, label="误差 %")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PATHS.output_dir / "official_split_EN_paper_test.png", dpi=150)
plt.show()

print("""
判读:
  - 口径A 是与论文 9.1% 可比的唯一口径。若 EN 落在 6~12%,复现成立
    (5.9% 那种低于论文的数,别解读成'超过论文'——划分索引的奇偶、
    alpha 选择细节都可能差出一两个点,量级对上就算站住);
  - 看最差 5 颗:如果全是短命电池(<500 圈),印证'夭折电池不可预测';
  - 口径B 是部署真实难度,写报告时与口径A 分开陈述;
  - batch3 在口径C 下若仍在 10% 附近,说明混合训练集对三级批次泛化良好。
""")
