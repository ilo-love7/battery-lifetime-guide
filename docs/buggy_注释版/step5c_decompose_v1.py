# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
# 【事故现场注释版】保留此 buggy 版本仅作教学用途,请勿运行
# 项目: Severson 2019 复现 (阶段5c)
# 错误编号: 清单 #10
# 六特征漏算三个 → KeyError
#
# 【现象】KeyError: ['slope','intercept','discrepancy'] not in index
# 【根因】从 5b 改造 5c 时,按实验窗口重算了 ΔQ 三特征(log_var/log_min/log_mean),把不随窗口变化的容量拟合三特征(slope/intercept/discrepancy)忘了加回 D——第158行的循环只写了三个特征名
# 【修复】v2: feats_list 一次算全六特征,再按列填入(见第157行附近的 feats_list 写法) —— 定稿版见 step5c_decompose_v2.py
# 【教学点】改造代码时,列清单(如 FEATURES)本身就是断言:D[FEATURES] 报 KeyError 是好事,它拦住了毒数据
# ═══════════════════════════════════════════════════════════════
"""
step5c v1 (2026-10-13 22:35):域间隙拆账——三个实验把误差分解归因
================================================================
背景(5b 终版):零样本迁移 EN ×75~125 失控、RF ×2.2~3.1 系统偏高;
NASA 特征 5/6 落在 MIT P0/P100;已知嫌疑三件:①电压窗口陷阱
(NASA 极拐点 ~3.55V 落在 3.5-3.9V 窗内,插值尖峰毒化 ΔQ 特征)
②寿命定义差(NASA EOL=70% 衰减 vs MIT 80%)③化学体系/倍率本征差异。
三个实验各切一刀:
  E0 基线   :全窗 3.5-3.9V + EOL@1.4Ah        (= 5b v2 复现)
  E1 窗口匹配:3.6-3.8V 双侧重算特征重训 + EOL@1.4   → 看①的账
  E2 标签匹配:全窗 + NASA EOL 改 1.6Ah(80%,对齐 MIT) → 看②的账
  E3 双匹配  :3.6-3.8V + EOL@1.6Ah            → 剩下的是③
输出:四实验 × 两模型 的逐颗倍率误差 + 几何平均,一张分解柱状图。
MIT 特征在缓存网格上切片即可,无需重新解析;NASA 侧只重算 3 颗。
自包含。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import h5py, os

print("== step5c v1:域间隙拆账(E0/E1/E2/E3) ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_CACHE = r".cache/mit_parse_cache.pkl"
NASA_DIR = r"data/nasa"  # ← NASA 数据目录(34 颗平铺 .mat)
OUT = r"results/generated"
VD = np.linspace(3.5, 3.9, 1000)          # MIT 官方 ΔQ 网格
SL_FULL = np.arange(1000)                  # 全窗
SL_38 = np.arange(100, 900)                # 3.6-3.8V 切片
FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
EOL_CAP = 0.88
SUSTAIN = 3
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]
B1_EXCL = {8, 10, 12, 13, 22}
NASA_CELLS = ["B0005", "B0006", "B0018"]
RATED = 2.0
CV = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(4, shuffle=True, random_state=42)

# %% ---------- [1] MIT 侧(缓存→合并→官方剔除→dev 84 断言) ----------
def robust_life(q, eol, sustain=SUSTAIN):
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

df = pd.read_pickle(MIT_CACHE)
def is_sane(r):
    return (np.isfinite(r["cycle_life"]) and r["cycle_life"] > 0
            and np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6
            and r["qd10_ok"] and r["qd100_ok"])
df["keep"] = df.apply(is_sane, axis=1)
feat = df[df["keep"]].copy().reset_index(drop=True)

def calc(r, sl):
    qd10, qd100 = r["_qd10"], r["_qd100"]
    f = {}
    if qd10 is not None and qd100 is not None and len(qd10) == len(qd100):
        dq = (qd100 - qd10)[sl]
        f["log_var"] = np.log10(max(np.var(dq), 1e-12))
        f["log_min"] = np.log10(max(-dq.min(), 1e-12))
        f["log_mean"] = np.log10(max(abs(dq.mean()), 1e-12))
    q_dis = r["_q_dis"]
    if len(q_dis) >= 100:
        y = q_dis[1:100]
        x = np.arange(2, 101)
        slope, intercept = np.polyfit(x, y, 1)
        f["slope"] = slope; f["intercept"] = intercept
        f["discrepancy"] = np.polyval((slope, intercept), 100) - q_dis[99]
    return f

# 合并标签(生效断言,4.3f 同款)
idx_of = {(r["batch"], r["cell_idx"]): i for i, r in feat.iterrows()}
for recv, donor in MERGE_PAIRS:
    q1 = feat.at[idx_of[("batch1", recv)], "_q_dis"]
    q2 = feat.at[idx_of[("batch2", donor)], "_q_dis"]
    feat.at[idx_of[("batch1", recv)], "cycle_life"] = robust_life(np.concatenate([q1, q2]), EOL_CAP)
got = [int(feat.at[idx_of[("batch1", c)], "cycle_life"]) for c, _ in MERGE_PAIRS]
assert got == [1851, 2159, 2236, 1433, 1708], f"合并标签异常: {got}"
dev = feat[((feat.batch == "batch1") & (~feat.cell_idx.isin(B1_EXCL)))
           | ((feat.batch == "batch2") & (~feat.cell_idx.isin({7, 8, 9, 15, 16})))].copy()
assert len(dev) == 84, f"dev 应为 84,实际 {len(dev)}"
print(f"[MIT] dev {len(dev)} 颗 ✓  合并标签 {got} ✓")

# %% ---------- [2] NASA 3 颗:容量序列 + 双窗口特征 ----------
def cycles_container(c0):
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont

def qv_interp(d):
    v = np.ravel(d["Voltage_measured"]); i = np.ravel(d["Current_measured"])
    t = np.ravel(d["Time"])
    q = -np.concatenate([[0], np.cumsum((i[:-1] + i[1:]) / 2 * np.diff(t))]) / 3600.0
    o = np.argsort(v)
    return np.interp(VD, v[o], q[o])

nasa = {}
for name in NASA_CELLS:
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    caps, ordn, qd = [], 0, {}
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        if "discharge" not in str(np.squeeze(c["type"])).lower():
            continue
        ordn += 1
        d = c["data"][0, 0]
        names = d.dtype.names or ()
        if "Capacity" in names:
            cap = np.ravel(d["Capacity"])
            if len(cap) and np.isfinite(cap[0]) and cap[0] > 0.1:
                caps.append(cap[0])
        if ordn in (10, 100):
            qd[ordn] = qv_interp(d)
    nasa[name] = {"caps": np.array(caps), "qd": qd}
    print(f"[NASA {name}] {len(caps)} 个放电周期, 前三容量 {np.round(np.array(caps[:3]),3)}")

# %% ---------- [3] 四实验 × 两模型 ----------
def nasa_feats(cell, sl):
    r = nasa[cell]
    dq = (r["qd"][100] - r["qd"][10])[sl]
    y = r["caps"][1:100]; x = np.arange(2, 101)
    slope, intercept = np.polyfit(x, y, 1)
    return {"log_var": np.log10(max(np.var(dq), 1e-12)),
            "log_min": np.log10(max(-dq.min(), 1e-12)),
            "log_mean": np.log10(max(abs(dq.mean()), 1e-12)),
            "slope": slope, "intercept": intercept,
            "discrepancy": np.polyval((slope, intercept), 100) - r["caps"][99]}

rows = []
for exp_name, eol, sl in [("E0 全窗+70%", 1.4, SL_FULL),
                          ("E1 窗口3.6-3.8+70%", 1.4, SL_38),
                          ("E2 全窗+80%", 1.6, SL_FULL),
                          ("E3 窗口3.6-3.8+80%", 1.6, SL_38)]:
    for _, r in dev.iterrows():
        pass
    # dev 特征按当前窗口计算
    D = dev.copy()
    for f in ["log_var", "log_min", "log_mean"]:
        D[f] = [calc(r, sl)[f] for _, r in dev.iterrows()]
    y = np.log10(D["cycle_life"].values)
    enet = make_pipeline(StandardScaler(),
                         ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                      cv=CV, random_state=42, max_iter=100000))
    enet.fit(D[FEATURES].values, y)
    rf = RandomForestRegressor(n_estimators=500, max_depth=3,
                               min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(D[FEATURES].values, y)
    for cell in NASA_CELLS:
        truth = robust_life(nasa[cell]["caps"], eol)
        nf = nasa_feats(cell, sl)
        for mname, model in [("EN", enet), ("RF", rf)]:
            pred = 10 ** model.predict(np.array([[nf[f] for f in FEATURES]]))[0]
            rows.append({"实验": exp_name, "模型": mname, "电池": cell,
                         "真实寿命": truth, "预测": round(pred),
                         "倍率误差": round(pred / truth, 2)})
res = pd.DataFrame(rows)
print("\n===== 拆账结果(倍率误差 = 预测/真实,>1 高估) =====")
for exp_name, g in res.groupby("实验", sort=False):
    print(f"\n-- {exp_name} (真实寿命 "
          f"{[int(g[g.电池==c].真实寿命.iloc[0]) for c in NASA_CELLS]} 圈)")
    for mname in ["EN", "RF"]:
        gg = g[g.模型 == mname]
        facs = gg["倍率误差"].values
        gmean = np.exp(np.mean(np.log(facs)))
        print(f"  {mname}: " + "  ".join(
            f"{r.电池}×{r.倍率误差}" for _, r in gg.iterrows())
            + f"  | 几何平均 ×{gmean:.2f}")
res.to_csv(OUT + r"\nasa_5c_decompose.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 4.5))
exps = list(dict.fromkeys(res["实验"]))
w = 0.35
for k, mname in enumerate(["EN", "RF"]):
    gm = [np.exp(np.mean(np.log(res[(res.实验 == e) & (res.模型 == mname)]["倍率误差"])))
          for e in exps]
    ax.bar(np.arange(len(exps)) + (k - 0.5) * w, gm, w, label=mname)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(np.arange(len(exps))); ax.set_xticklabels(exps, fontsize=9)
ax.set_ylabel("几何平均倍率误差(预测/真实,对数轴)")
ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_title("域间隙拆账:E0→E3,越接近 ×1 越好")
fig.tight_layout()
fig.savefig(OUT + r"\nasa_5c_decompose.png", dpi=140)
plt.show()

print("""
判读(发我:四段实验打印+柱状图):
  - E0→E1 大幅回落 → 窗口伪影是主犯(特征可救);
  - E0→E2 大幅回落 → 标签定义是主犯(口径对齐即可救);
  - E3 仍剩的倍率 → 体系/倍率本征差异,这部分迁移救不了,只能标定;
  - 若 E1 的 EN 仍失控 → ΔQ 特征对 NASA 根本失效,5c 结论升级:
    "跨数据集迁移必须换特征族",这也是可写的强结论。
""")
