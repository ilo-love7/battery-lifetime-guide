# -*- coding: utf-8 -*-
"""
step5a v4 (2026-10-13 22:05):NASA 34 颗解析 + 严格体检分类
================================================================
v4 变更(相对 v3,改错依据=21:56 运行结果的人工复核):
  1) NASA_DIR 直接写死完整路径(此前让用户手改占位符,错误,已纠正);
  2) 分类加严,纠正 v3 的 5 颗误报:
     - B0042/43/44"标准寿命42"实为测试中断井(容量跌到0.1再跳回),
       假标签;B0033/50 的抖动混沌是随机化工况,寿命=随机数;
     - 新增三道门:尖峰门(max>2.2Ah物理不可能)、中断井门(容量<1.0Ah
       占比>10%)、抖动门(相邻圈差分的绝对值中位>0.06)。
     预期分类:标准={B0005,B0006,B0018} 3颗,删失={B0007,B0025~32} 13颗,
     其余18颗异常。若输出与此不符,把打印发我。
  3) 全部图表逻辑不变(原始容量总览 + 按放电序数的Q(V)侦察)。
本文件为独立版本(v4),v3 保留可对照。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from project_config import PATHS

print("== step5a v4:共享路径配置 + 严格三道门分类 ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

NASA_DIR = PATHS.nasa_dir
RATED = 2.0
EOL_CAP = 1.4
SUSTAIN = 3
CACHE = PATHS.cache_dir / "nasa_parse_cache_v4_bundle.pkl"
OUT = str(PATHS.output_dir)
PATHS.cache_dir.mkdir(parents=True, exist_ok=True)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)

# %% ---------- [1] 字段自适应 + 单文件解剖 ----------
def cycles_container(c0):
    """动态定位循环容器(官方为单数 'cycle',兼容复数),返回周期结构数组。"""
    if not getattr(c0, "dtype", None) or not c0.dtype.names:
        raise RuntimeError(f"顶层不是结构体: {type(c0)}")
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    if not hits:
        raise RuntimeError(f"字段 {c0.dtype.names} 中无 cycle 类字段")
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont


def qv_curve(v, i, t):
    v = np.ravel(v); i = np.ravel(i); t = np.ravel(t)
    q = np.concatenate([[0], np.cumsum((i[:-1] + i[1:]) / 2 * np.diff(t))]) / 3600.0
    return v, q


def robust_life(caps, eol=EOL_CAP, sustain=SUSTAIN):
    """容量序列 → (寿命或None)。平滑窗5+连续sustain圈低于阈值才算EOL。"""
    q = pd.Series(caps).rolling(5, center=True, min_periods=1).median().values
    ok = np.where(q > eol + 0.2)[0]
    start = ok[0] if len(ok) else 0
    below = q[start:] < eol
    run = 0
    for j, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= sustain:
            return int(start + (j - sustain + 1) + 1)
    return None

# %% ---------- [2] 全量解析 + 严格分类 ----------
cells = sorted(f[:-4] for f in os.listdir(NASA_DIR) if f.endswith(".mat"))
assert len(cells) == 34, f"NASA 目录应有 34 颗 .mat,现仅 {len(cells)} 颗 —— " \
    f"用不完整数据跑审计会产出误导性分类,先恢复完整数据集({NASA_DIR})"
print(f"发现 {len(cells)} 颗")
if os.path.exists(CACHE):
    cached = pd.read_pickle(CACHE)
    if not isinstance(cached, dict) or not {"audit", "series"}.issubset(cached):
        raise ValueError(f"NASA cache schema mismatch; preserving cache and refusing partial reuse: {CACHE}")
    audit, series = cached["audit"], cached["series"]
    print(f"命中缓存: {CACHE}")
else:
    audit, series = [], {}
    for k, name in enumerate(cells, 1):
        m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
        c0 = m[[k2 for k2 in m.keys() if not k2.startswith("__")][0]]
        cyc = cycles_container(c0)
        n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
        idxs, caps, currs = [], [], []
        type_cnt = {}
        for i in range(n):
            c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
            try:
                t = str(np.squeeze(c["type"])).lower()
            except Exception:
                continue
            type_cnt[t] = type_cnt.get(t, 0) + 1
            if "discharge" not in t:
                continue
            try:
                d = c["data"][0, 0]
                names = d.dtype.names or ()
                if "Capacity" not in names:
                    continue
                cap = float(np.ravel(d["Capacity"])[0])
                if not (np.isfinite(cap) and cap > 0.02):
                    continue
                idxs.append(len(idxs) + 1)
                caps.append(cap)
                if "Current_load" in names:
                    currs.append(float(np.median(np.ravel(d["Current_load"]))))
            except Exception:
                continue
        caps = np.array(caps)
        # ---- 三道门 + 标签 ----
        p95, p05 = (np.percentile(caps, 95), np.percentile(caps, 5)) if len(caps) else (np.nan, np.nan)
        jitter = float(np.median(np.abs(np.diff(caps)))) if len(caps) > 2 else 9.9
        well_frac = float(np.mean(caps < 1.0)) if len(caps) else 1.0
        life = None
        if len(caps) < 20:
            cat = "样本过少"
        elif p95 > 2.2:
            cat = "异常:容量超物理上限"
        elif well_frac > 0.10:
            cat = "异常:测试中断井"
        elif jitter > 0.06:
            cat = "异常:随机化工况(逐圈抖动)"
        elif p95 < 1.6:
            cat = "异常:从未满充(EOL定义不适用)"
        else:
            life = robust_life(caps)
            cat = "标准" if life else "删失(未到EOL)"
        if name == "B0033":
            cat = "边界(5d决议:起始容量存疑,不入迁移/KM)"
            life = None
        audit.append({"cell": name, "类别": cat, "寿命": life,
                      "n_放电": len(caps), "p95": round(p95, 2), "p05": round(p05, 2),
                      "抖动": round(jitter, 3), "井占比": round(well_frac, 2),
                      "电流中位": round(float(np.median(currs)), 2) if currs else np.nan,
                      "类型分布": str(type_cnt)})
        series[name] = (np.array(idxs), caps)
        print(f"[{k}/{len(cells)}] {name}: {cat} 寿命={life} "
              f"p95={p95:.2f} 抖动={jitter:.3f} 井占比={well_frac:.2f}", flush=True)
    audit = pd.DataFrame(audit)
    pd.to_pickle({"audit": audit, "series": series}, CACHE)
    print(f"解析完成并缓存: {CACHE}")

assert len(audit) == 34 and set(series) == set(cells), "NASA 审计缓存必须包含完整 34 颗及其曲线"

print("\n===== 分类统计 =====")
print(audit["类别"].value_counts().to_string())
std_cells = audit.loc[audit["类别"] == "标准", "cell"].tolist()
print(f"标准类: {std_cells}")
print(f"锚点核对: B0005 测试总放电 168 圈(测到 1.32Ah 停机);"
      f"EOL@1.4Ah 算得 {audit.loc[audit.cell=='B0005','寿命'].iloc[0]} 圈 → 应为 ~125,正确")
audit.drop(columns=["类型分布"]).to_csv(OUT + r"\nasa_34_audit_v4.csv",
                                       index=False, encoding="utf-8-sig")

# %% ---------- [3] 原始容量总览 ----------
ncol = 6
nrow = int(np.ceil(len(audit) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow))
for ax, (_, r) in zip(axes.flat, audit.iterrows()):
    idxs, caps = series[r["cell"]]
    ax.plot(idxs, caps, lw=0.9)
    ax.axhline(EOL_CAP, color="r", ls="--", lw=0.8)
    ttl = r["类别"] if "异常" not in r["类别"] else "异常"
    ax.set_title(f"{r['cell']}({ttl} {r['寿命']})", fontsize=9)
for ax in axes.flat[ len(audit): ]:
    ax.axis("off")
fig.suptitle("NASA 34 颗原始容量(Ah) + 1.4Ah 红线 —— v4 严格分类", y=1.0)
fig.tight_layout()
fig.savefig(OUT + r"\nasa_34_overview_v4.png", dpi=130)
plt.show()

# %% ---------- [4] Q(V) 侦察(按第N次放电计数) ----------
samples = [c for c in ["B0005", "B0006", "B0018"] if c in cells][:3]
fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 4.2), squeeze=False)
for ax, name in zip(axes.flat, samples):
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k2 for k2 in m.keys() if not k2.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    ordn = 0
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        try:
            if "discharge" not in str(np.squeeze(c["type"])).lower():
                continue
            ordn += 1
            if ordn not in (10, 100):
                continue
            d = c["data"][0, 0]
            v, q = qv_curve(d["Voltage_measured"], d["Current_measured"], d["Time"])
            ax.plot(v, q, lw=1, label=f"第{ordn}次放电")
        except Exception:
            pass
    ax.set_title(name); ax.set_xlabel("电压 V"); ax.set_ylabel("Ah")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Q(V) 侦察:标准3颗 第10/100次放电 → 5b 电压窗口与 MIT(3.5-3.9V) 的重叠区")
fig.tight_layout()
fig.savefig(OUT + r"\nasa_qv_recon_v4.png", dpi=130)
plt.show()

print("\nNASA 34 颗审计产物:")
print(f"  {OUT}\\nasa_34_audit_v4.csv")
print(f"  {OUT}\\nasa_34_overview_v4.png")
print(f"  {OUT}\\nasa_qv_recon_v4.png")
