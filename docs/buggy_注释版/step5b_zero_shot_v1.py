# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
# 【事故现场注释版】保留此 buggy 版本仅作教学用途,请勿运行
# 项目: Severson 2019 复现 (阶段5b)
# 错误编号: 清单 #9
# 训练池混入截断标签电池 + 画图用错数组
#
# 【现象】dev 打出 89 颗(应为 84);ΔQ 叠加图 ValueError (1000,) vs (471,)
# 【根因】①漏剔 batch1 官方 5 颗中止电池——它们标签截断(删失),进训练池=慢性毒药(4.3d 教训原样重演);②画 MIT ΔQ 曲线误用 _q_dis(容量序列),正确是 _qd100-_qd10(电压网格曲线)
# 【修复】v2: assert len(dev)==84 + 改用 _qd100-_qd10 —— 定稿版见 step5b_zero_shot_v2.py
# 【教学点】官方剔除名单不只是'论文口径',更是数据质量过滤;数组的'形状'本身就是断言
# ═══════════════════════════════════════════════════════════════
"""
step5b v1 (2026-10-13 22:30):MIT → NASA 零样本迁移(案例级)
================================================================
前置:4.3f 已跑过(mit_parse_cache.pkl 存在);v4 已确认 NASA 标准类 3 颗。
本脚本:
  [1] MIT 侧:读缓存 → 官方合并/剔除(复用 4.3f 逻辑与断言) → 六特征
      → EN6 + RF 在全部 dev 124 颗上训练(迁移用,吃满数据)
  [2] NASA 侧:B0005/6/18 原始放电积分 Q(V) → 插值到 MIT 同款网格
      linspace(3.5,3.9,1000) → 同公式六特征(逐公式可比)
  [3] 特征级域间隙:NASA 特征落在 MIT 分布的哪个分位(先量化再预测)
  [4] 零样本预测:真寿命 vs 预测(案例级,n=3,不做统计声明)
  [5] 两张图:ΔQ(V) 曲线叠加(NASA vs MIT 代表电池)、预测-真实图
自包含;特征公式与 4.3f 逐字一致(calc_features 原样拷贝)。
"""
import numpy as np
import pandas as pd
import h5py, os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("== step5b v1:零样本迁移(案例级 n=3) ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_CACHE = r".cache/mit_parse_cache.pkl"
NASA_DIR = r"data/nasa"  # ← NASA 数据目录(34 颗平铺 .mat)
VD = np.linspace(3.5, 3.9, 1000)     # MIT Qdlin 官方网格(Severson 代码同款)
FEATURES = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
EOL_CAP, SUSTAIN = 0.88, 3
STANDARD_NASA = ["B0005", "B0006", "B0018"]
OUT = r"results/generated"

# %% ---------- [1] MIT 侧:特征 + 标签(逻辑与 4.3f 逐字一致) ----------
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

def calc_features(r):
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

df["keep"] = df.apply(is_sane, axis=1)
feat_df = pd.DataFrame([
    {"batch": r["batch"], "cell_idx": r["cell_idx"], "cycle_life": r["cycle_life"],
     "endcap": r["endcap"], "_q_dis": r["_q_dis"],
     **calc_features(r)}
    for _, r in df[df["keep"]].iterrows()
]).dropna(subset=FEATURES)

_lk = {(r["batch"], r["cell_idx"]): i for i, r in feat_df.iterrows()}
MERGE_PAIRS = [(0, 7), (1, 8), (2, 9), (3, 15), (4, 16)]
for recv, donor in MERGE_PAIRS:
    q1 = feat_df.at[_lk[("batch1", recv)], "_q_dis"]
    q2 = feat_df.at[_lk[("batch2", donor)], "_q_dis"]
    new_life = robust_life(np.concatenate([q1, q2]))
    feat_df.at[_lk[("batch1", recv)], "cycle_life"] = new_life
_expect = [1851, 2159, 2236, 1433, 1708]
_got = [int(feat_df.at[_lk[("batch1", c)], "cycle_life"]) for c, _ in MERGE_PAIRS]
assert _got == _expect, f"合并未生效:{_got}"
print(f"合并生效检查 ✓ {_got}")

for c in [8, 10, 12, 13, 22]:
    feat_df = feat_df[~((feat_df.batch == "batch1") & (feat_df.cell_idx == c))]
feat_df = feat_df[feat_df.batch.isin(["batch1", "batch2"])]
dev = feat_df.copy()
y_dev = np.log10(dev["cycle_life"].values)
print(f"dev 训练池:{len(dev)} 颗,寿命 {dev.cycle_life.min():.0f}~{dev.cycle_life.max():.0f} 圈")

cv = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(4, shuffle=True, random_state=42)
enet = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                  cv=cv, random_state=42, max_iter=100000))
enet.fit(dev[FEATURES].values, y_dev)
rf = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
rf.fit(dev[FEATURES].values, y_dev)
print("EN6/RF 在 dev 124 颗上训练完成\n")

# %% ---------- [2] NASA 侧:同公式特征 ----------
def cycles_container(c0):
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont

def qv_interp(d):
    """放电周期 → Q(V) 插值到 MIT 网格。电流取负号使容量为正。"""
    from scipy.integrate import cumulative_trapezoid
    v = np.ravel(d["Voltage_measured"]); i = np.ravel(d["Current_measured"])
    t = np.ravel(d["Time"])
    q = -cumulative_trapezoid(i, t, initial=0) / 3600.0
    o = np.argsort(v)
    return np.interp(VD, v[o], q[o])

nasa_rows = []
for name in STANDARD_NASA:
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    caps, qd = {10: None, 100: None}, {10: None, 100: None}
    ordn = 0
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        if "discharge" not in str(np.squeeze(c["type"])).lower():
            continue
        ordn += 1
        d = c["data"][0, 0]
        names = d.dtype.names or ()
        if "Capacity" in names:
            cap = float(np.ravel(d["Capacity"])[0])
            if np.isfinite(cap) and 0.1 < cap < 2.2:
                caps[ordn] = cap
        if ordn in (10, 100):
            qd[ordn] = qv_interp(d)
        if ordn >= 100 and ordn > max(caps):
            break
    cs = np.array([caps[k] for k in sorted(caps) if caps[k] is not None])
    if cs[0] >= len(caps):  # noqa
        pass
    dq = qd[100] - qd[10]
    y = cs[1:100]                      # 与 MIT 同约定:跳过第1次放电
    x = np.arange(2, 101)
    slope, intercept = np.polyfit(x, y, 1)
    f = {"log_var": np.log10(max(np.var(dq), 1e-12)),
         "log_min": np.log10(max(-dq.min(), 1e-12)),
         "log_mean": np.log10(max(abs(dq.mean()), 1e-12)),
         "slope": slope, "intercept": intercept,
         "discrepancy": np.polyval((slope, intercept), 100) - cs[99],
         "_dq": dq}
    nasa_rows.append({"cell": name, "n_dis": ordn, **f})
nasa = pd.DataFrame(nasa_rows)
print("\nNASA 特征 vs MIT 分布(分位差距 = 特征级域间隙):")
for f in ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]:
    qs = np.percentile(dev[f].values, [5, 50, 95])
    line = "  ".join(f"{n}={r[f]:+.4f}" for _, r in nasa.iterrows())
    print(f"  {f:12s} MIT[5%,50%,95%]={qs.round(4)}  |  {line}")

# %% ---------- [3] 零样本预测(案例级,n=3) ----------
rows = []
for _, r in nasa.iterrows():
    pe = 10 ** enet.predict(r[FEATURES].values.reshape(1, -1))[0]
    pr = 10 ** rf.predict(r[FEATURES].values.reshape(1, -1))[0]
    true = {"B0005": 125, "B0006": 109, "B0018": 97}[r["cell"]]
    rows.append({"电池": r["cell"], "真实寿命": true,
                 "EN预测": round(pe), "RF预测": round(pr),
                 "EN倍率": round(pe / true, 2), "RF倍率": round(pr / true, 2),
                 "MIT寿命中位": int(dev.cycle_life.median())})
res = pd.DataFrame(rows)
print("\n===== 零样本预测(真寿命来自 v4 标签) =====")
print(res.to_string(index=False))
res.to_csv(OUT + r"\nasa_zero_shot_case.csv", index=False, encoding="utf-8-sig")

# %% ---------- [4] 图:ΔQ(V) 域间隙 + 预测-真实 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
sub = dev.sort_values("cycle_life")
for frac, col, lab in [(0.08, "gray", "MIT 短寿命代表"),
                       (0.5, "tab:blue", "MIT 中位代表"),
                       (0.92, "tab:green", "MIT 长寿命代表")]:
    r = sub.iloc[int(len(sub) * frac)]
    ax.plot(VD, r["_q_dis"] * 0 + 0, alpha=0)  # 占位防告警
    dq_m = None
for frac, col in [(0.08, "gray"), (0.5, "tab:blue"), (0.92, "tab:green")]:
    r = sub.iloc[int(len(sub) * frac)]
    key = ("batch1", r.cell_idx) if r.batch == "batch1" else ("batch2", r.cell_idx)
    dqm = df[df.keep].reset_index(drop=True)
    # MIT ΔQ 直接从缓存 qd 数组取
    row_raw = pd.read_pickle(MIT_CACHE)
    row_raw = row_raw[(row_raw.batch == r.batch) & (row_raw.cell_idx == r.cell_idx) & row_raw.keep].iloc[0]
    ax.plot(VD, row_raw["_qd100"] - row_raw["_qd10"], color=col, lw=1,
            label=f"MIT {r.batch}#{r.cell_idx}({r.cycle_life:.0f}圈)")
for _, r in nasa.iterrows():
    ax.plot(VD, r["_dq"], lw=1.6, label=f"NASA {r['cell']}")
ax.set_xlabel("电压 V"); ax.set_ylabel("ΔQ100-10 (Ah)")
ax.set_title("ΔQ(V) 域间隙:NASA vs MIT"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(dev.cycle_life, 10 ** enet.predict(dev[FEATURES].values),
           s=14, alpha=0.35, label="MIT dev(模型内)")
lims = [50, 2600]
ax.plot(lims, lims, "k--", lw=1)
for _, r in res.iterrows():
    ax.scatter(r["真实寿命"], r["EN预测"], marker="*", s=220, c="red", zorder=5)
    ax.annotate(f"NASA {r['电池']}\n真实{r['真实寿命']}/预测{r['EN预测']}",
                (r["真实寿命"], r["EN预测"]), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("真实寿命"); ax.set_ylabel("EN 预测寿命")
ax.set_title("零样本迁移:案例级"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + r"\nasa_zero_shot_v1.png", dpi=140)
plt.show()

print("""
判读(发我全部文字输出+图):
  - 特征分位行:NASA 三颗的特征落在 MIT 分布内还是外?在外 → 域间隙
    在特征层就可见,预测失败是"必然"而非"调参问题";
  - 预测倍率:预期系统性偏差(寿命定义 70% vs 80%、倍率、化学体系),
    方向和大小本身就是 5b 的结论;
  - 这 3 颗是案例,不是统计——报告措辞我会写成 case study。
""")
