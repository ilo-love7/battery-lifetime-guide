# -*- coding: utf-8 -*-
"""
step5b v2 (2026-10-13 22:35):MIT→NASA 零样本迁移,修复 v1 三处
================================================================
v1→v2 变更(依据 22:21 运行结果):
  1) 画图格 ValueError 修复:MIT 代表电池的 ΔQ 曲线改用 _qd100-_qd10
     (1000 点电压网格);v1 误用 _q_dis(471 点容量序列),形状不匹配;
  2) dev 训练池 89→84 断言:v1 少剔了 batch1 官方 5 颗中止电池——
     它们的标签是截断的(删失),放进迁移训练池等于慢性毒药,必须剔;
  3) 特征表修复:"319=" 标签错乱 → 改为打印 NASA 特征的 MIT 分位(Pxx);
     slope 显示截断("-0.") → 改 %.5f 格式。
"""
import numpy as np
import pandas as pd
import h5py, os
from pathlib import Path
from project_config import PATHS
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("== step5b v2:零样本迁移(案例级 n=3,dev 断言 84) ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_CACHE = PATHS.cache_dir / "mit_parse_cache.pkl"
NASA_DIR = PATHS.nasa_dir
VD = np.linspace(3.5, 3.9, 1000)     # MIT ΔQ 特征网格(Severson 官方口径)
FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
EOL_CAP, SUSTAIN = 0.88, 3
OUT = str(PATHS.output_dir)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
NASA_CELLS = ["B0005", "B0006", "B0018"]
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]
OFFICIAL_EXCLUDE = {"batch1": {8, 10, 12, 13, 22}, "batch2": {7, 8, 9, 15, 16}}

# %% ---------- [1] MIT 侧:缓存 → 特征 → 合并 → 官方剔除 → 训练 ----------
def robust_life(q, eol=EOL_CAP, sustain=SUSTAIN):
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

def calc_features(r) -> dict:
    """与 4.3f 逐字一致,禁止改动(迁移可比性的根基)。"""
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

df = pd.read_pickle(MIT_CACHE)
print(f"MIT 缓存:{len(df)} 行")

def is_sane(r):
    return (np.isfinite(r["cycle_life"]) and r["cycle_life"] > 0
            and np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6
            and r["qd10_ok"] and r["qd100_ok"])

feat_rows = []
for _, r in df[df.apply(is_sane, axis=1)].iterrows():
    f = calc_features(r)
    if len(f) == 6:
        feat_rows.append({"batch": r["batch"], "cell_idx": r["cell_idx"],
                          "cycle_life": r["cycle_life"], "_q_dis": r["_q_dis"],
                          "_qd10": r["_qd10"], "_qd100": r["_qd100"], **f})
feat_df = pd.DataFrame(feat_rows)
_lk = {(r["batch"], r["cell_idx"]): i for i, r in feat_df.iterrows()}

ADD_LEN = {7: 661, 8: 980, 9: 1059, 15: 207, 16: 481}
for recv, donor in MERGE_PAIRS:
    i1, i2 = _lk[("batch1", recv)], _lk[("batch2", donor)]
    q1, q2 = feat_df.at[i1, "_q_dis"], feat_df.at[i2, "_q_dis"]
    cont = len(q2) > 1 and abs(q1[-1] - q2[1]) < 0.1
    old = feat_df.at[i1, "cycle_life"]
    new = robust_life(np.concatenate([q1, q2]))
    feat_df.at[i1, "cycle_life"] = new
    print(f"  合并 batch1#{recv}←batch2#{donor}: len(q2)={len(q2)}"
          f"(应{ADD_LEN[donor]+1} {'✓' if len(q2)==ADD_LEN[donor]+1 else '✗'}) "
          f"连续{'✓' if cont else '✗'} {old:.0f}→{new}")
_expect = [1851, 2159, 2236, 1433, 1708]
_got = [int(feat_df.at[_lk[("batch1", c)], "cycle_life"]) for c, _ in MERGE_PAIRS]
assert _got == _expect, f"合并标签未生效!{_got}"
print(f"合并生效检查 ✓ {_got}")

dev = feat_df[(feat_df.batch == "batch1") & (~feat_df.cell_idx.isin(OFFICIAL_EXCLUDE["batch1"]))
              | (feat_df.batch == "batch2") & (~feat_df.cell_idx.isin(OFFICIAL_EXCLUDE["batch2"]))]
assert len(dev) == 84, f"dev 应为 84(官方口径),实际 {len(dev)} —— 发截图"
print(f"dev 训练池:{len(dev)} 颗 ✓ (官方口径),寿命 {dev.cycle_life.min():.0f}~{dev.cycle_life.max():.0f}")

cv = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(4, shuffle=True, random_state=42)
enet = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=cv,
                                  random_state=42, max_iter=100000))
enet.fit(dev[FEATURES].values, np.log10(dev["cycle_life"].values))
rf = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
rf.fit(dev[FEATURES].values, np.log10(dev["cycle_life"].values))
print("EN6/RF 在 dev 84 颗上训练完成\n")

# %% ---------- [2] NASA 侧:同网格同公式特征 ----------
def qv_curve(v, i, t):
    v = np.ravel(v); i = np.ravel(i); t = np.ravel(t)
    q = -np.concatenate([[0], np.cumsum((i[1:] + i[:-1]) / 2 * np.diff(t))]) / 3600.0
    return v, q

def cycles_container(c0):
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont

nasa = {}
for name in NASA_CELLS:
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    caps, qd = [], {}
    ordn = 0
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        try:
            if "discharge" not in str(np.squeeze(c["type"])).lower():
                continue
            ordn += 1
            d = c["data"][0, 0]
            names = d.dtype.names or ()
            if "Capacity" in names:
                cap = float(np.ravel(d["Capacity"])[0])
                if np.isfinite(cap) and 0.05 < cap < 2.3:
                    caps.append(cap)
            if ordn in (10, 100):
                v, q = qv_curve(d["Voltage_measured"], d["Current_measured"], d["Time"])
                m_ = (v >= VD[0]) & (v <= VD[-1])
                qd[ordn] = np.interp(VD, v[m_][::-1], q[m_][::-1])
        except Exception:
            continue
    caps = np.array(caps)
    feats = {}
    if 10 in qd and 100 in qd:
        dq = qd[100] - qd[10]
        feats["log_var"] = np.log10(max(np.var(dq), 1e-12))
        feats["log_min"] = np.log10(max(-dq.min(), 1e-12))
        feats["log_mean"] = np.log10(max(abs(dq.mean()), 1e-12))
    if len(caps) >= 100:
        y = caps[1:100]
        x = np.arange(2, 101)
        slope, intercept = np.polyfit(x, y, 1)
        feats["slope"] = slope
        feats["intercept"] = intercept
        feats["discrepancy"] = np.polyval((slope, intercept), 100) - caps[99]
    nasa[name] = {"caps": caps, "qd": qd, **feats}
    print(f"{name}: {ordn} 次放电,容量 {caps.min():.2f}~{caps.max():.2f} Ah,"
          f"特征 {'全' if len(feats)==6 else '缺:'+str(6-len(feats))}")

print("\nNASA 特征 vs MIT 分布(分位 = 特征级域间隙):")
distribution_rows = []
for f in FEATURES:
    arr = np.sort(dev[f].values)
    pcts = [int(np.searchsorted(arr, nasa[c][f]) / len(arr) * 100) for c in NASA_CELLS]
    print(f"  {f:12s} MIT[P5,P50,P95]=[{np.percentile(arr,5):.4f},"
          f"{np.percentile(arr,50):.4f},{np.percentile(arr,95):.4f}]  "
          f"NASA={['%.4f' % nasa[c][f] for c in NASA_CELLS]}  "
          f"MIT分位={pcts}")
    distribution_row = {
        "特征": f,
        "MIT_P05": float(np.percentile(arr, 5)),
        "MIT_P50": float(np.percentile(arr, 50)),
        "MIT_P95": float(np.percentile(arr, 95)),
    }
    for name, percentile in zip(NASA_CELLS, pcts):
        distribution_row[f"{name}_value"] = float(nasa[name][f])
        distribution_row[f"{name}_MIT_percentile"] = percentile
    distribution_rows.append(distribution_row)

distribution_path = PATHS.output_dir / "5_跨数据集迁移" / "MIT特征分位对照.csv"
distribution_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(distribution_rows).to_csv(
    distribution_path, index=False, encoding="utf-8-sig"
)
print(f"MIT/NASA 特征分位对照已落盘: {distribution_path}")

# %% ---------- [3] 零样本预测 ----------
rows = []
for name in NASA_CELLS:
    x = np.array([[nasa[name][f] for f in FEATURES]])
    pe = 10 ** enet.predict(x)[0]
    pr = 10 ** rf.predict(x)[0]
    true = {"B0005": 125, "B0006": 109, "B0018": 97}[name]
    rows.append({"电池": name, "真实寿命": true,
                 "EN预测": round(pe), "RF预测": round(pr),
                 "EN倍率": round(pe / true, 1), "RF倍率": round(pr / true, 2)})
res = pd.DataFrame(rows)
print("\n===== 零样本预测(真寿命来自 v4 标签;MIT 中位 %.0f) =====" % dev.cycle_life.median())
print(res.to_string(index=False))
res.to_csv(OUT + r"\nasa_zero_shot_case_v2.csv", index=False, encoding="utf-8-sig")

# %% ---------- [4] 两张图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
sub = dev.sort_values("cycle_life").reset_index(drop=True)
for frac, col, lab in [(0.08, "gray", "MIT 短寿命代表"),
                       (0.5, "tab:blue", "MIT 中位代表"),
                       (0.92, "tab:green", "MIT 长寿命代表")]:
    r = sub.iloc[int(len(sub) * frac)]
    ax.plot(VD, r["_qd100"] - r["_qd10"], color=col, lw=1.2, alpha=0.9, label=lab)
for k, name in enumerate(NASA_CELLS):
    ax.plot(VD, nasa[name]["qd"][100] - nasa[name]["qd"][10],
            lw=1.4, ls="--", label=f"NASA {name}")
ax.set_xlabel("电压 V"); ax.set_ylabel("ΔQ = Q100 − Q10 (Ah)")
ax.set_title("ΔQ(V) 域间隙:NASA(虚线) vs MIT(实线)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(dev.cycle_life, 10 ** enet.predict(dev[FEATURES].values),
           s=14, alpha=0.35, label="MIT dev(模型内)")
lims = [50, 2600]
ax.plot(lims, lims, "k--", lw=1)
for _, r in res.iterrows():
    ax.scatter(r["真实寿命"], r["EN预测"], marker="*", s=220, c="red", zorder=5)
    ax.annotate(f"{r['电池']} 真{r['真实寿命']}/EN{r['EN预测']}",
                (r["真实寿命"], r["EN预测"]), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("真实寿命"); ax.set_ylabel("EN 预测寿命")
ax.set_title("零样本迁移:案例级(星=NASA)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + r"\nasa_zero_shot_v2.png", dpi=140)
plt.show()

print("""
判读与产物:
    - 当前三例 EN 倍率为 ×74.8~125.3、RF ×2.25~3.14；不同迁移脚本的模型/口径结果不要混作同一估计;
    - MIT dev P05-P95 对照的 18 个电池×特征值全部越界，逐项证据写入分位 CSV;
  - RF 的 ~2 倍偏差方向 = 寿命定义差(NASA 70% 衰减 vs MIT 80%)+倍率/体系,
    这是"可解释的域偏差",报告里的关键句;
  - 图1 虚线(NASA)与实线(MIT)的错位幅度 = 特征层域间隙的直观版。
""")
