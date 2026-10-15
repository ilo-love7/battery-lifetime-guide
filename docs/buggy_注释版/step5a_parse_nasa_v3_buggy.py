# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
# 【事故现场注释版】保留此 buggy 版本仅作教学用途,请勿运行
# 项目: Severson 2019 复现 (阶段5a)
# 错误编号: 清单 #8
# 体检分类缺两道门:5 颗异常电池混进'标准'类
#
# 【现象】B0042/43/44 被标为'标准 寿命42/55/50',实为测试中断井(容量跌到0.1再跳回)
# 【根因】v3 只有'从未满充 p95<1.6'一道门,缺:①中断井门(容量<1.0Ah 占比>10%)②抖动门(相邻圈差分中位>0.06);更早 v1 还有解析过滤器 v[0]>0.1 把井周期整个滤掉的叠加 bug
# 【修复】v4 三道门+从未满充门,共四道(见 step5a_parse_nasa_v4.py) —— 定稿版见 step5a_parse_nasa_v4.py
# 【教学点】数据分类的门是防错最后一道闸;每道门背后都必须有一个真实的坏例子
# ═══════════════════════════════════════════════════════════════
"""
step5a v3 (2026-10-13 21:55):NASA 34 颗解析 + 逐颗体检 + 分类
================================================================
v3 变更(相对 v2):
  1) Q(V) 侦察图改按"第 N 次放电"计数(v2 错按容器序号,616 条目里混着
     charge/impedance,第 10/100 个条目多半不是放电 → 图空白);
  2) 新增 [2b] 逐颗体检表:周期类型分布、原始容量分位、超物理上限占比
     → 自动分类:标准协议 / 删失 / 随机化异常;
  3) 锚点更正:B0005 的 168 是"测试总放电数"(测到 1.32Ah 停机),
     EOL@1.4Ah 的寿命本来就应是 ~125,v2 的锚点文案错了,管线没错;
  4) 总览图改画原始容量 Ah(不再按首圈归一)——"从未满充"的病态一眼可见。
字段适配:cycles_container() 动态探测(官方字段为单数 'cycle',已实物确认)。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import os

print("== step5a v3:字段自适应 + 体检分类 ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ↓↓↓ 你的 extracted 目录(v2 里已填好的那份路径) ↓↓↓
NASA_DIR = r"data/nasa/extracted  # ← 改成你的 NASA 数据目录"
RATED = 2.0       # 额定容量 Ah(2Ah LCO)
EOL_CAP = 1.4     # 30% 衰减(同阶段2口径)
SUSTAIN = 3
CACHE = r"results/generated/nasa_parse_cache_v3.pkl"
OUT = r"results/generated"

cells = sorted(f[:-4] for f in os.listdir(NASA_DIR) if f.endswith(".mat"))
print(f"发现 {len(cells)} 颗: {cells}")

# %% ---------- [1] 解剖 + 通用工具 ----------
def cycles_container(c0):
    if not getattr(c0, "dtype", None) or not c0.dtype.names:
        raise RuntimeError(f"顶层不是结构体: {type(c0)}")
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    if not hits:
        raise RuntimeError(f"字段 {c0.dtype.names} 中无 cycle 类")
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont

m = sio.loadmat(os.path.join(NASA_DIR, cells[0] + ".mat"))
c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
print(f"[{cells[0]}] 顶层字段: {c0.dtype.names}")
cyc = cycles_container(c0)
print(f"循环容器形状: {cyc.shape}")

def robust_life(caps, eol=EOL_CAP, sustain=SUSTAIN, smooth=5):
    """平滑窗 median(smooth) 后,连续 sustain 圈低于 eol 才判 EOL。
    跳过开头未满充的圈。返回 (life 或 None=删失)。caps: 原始 Ah 序列。"""
    q = pd.Series(caps).rolling(smooth, center=True, min_periods=1).median().values
    ok = np.where(q > eol)[0]
    if len(ok) == 0:
        return None                      # 从未满充到 eol 以上 → 无法判
    start = ok[0]
    below = q[start:] < eol
    run = 0
    for j, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= sustain:
            return start + (j - sustain + 1) + 1
    return None                          # 到测试结束都没跌破 → 删失

# %% ---------- [2] 全量解析 + 逐颗体检 ----------
audit, series = [], {}
for k, name in enumerate(cells, 1):
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    types, caps, amps = {}, [], []
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        try:
            t = str(np.squeeze(c["type"])).lower()
        except Exception:
            continue
        types[t] = types.get(t, 0) + 1
        if "discharge" not in t:
            continue
        try:
            d = c["data"][0, 0]
            names = d.dtype.names or ()
            if "Capacity" in names:
                v = float(np.ravel(d["Capacity"])[0])
                if np.isfinite(v):
                    caps.append(v)
            if "Current_load" in names:
                amps.append(float(np.median(np.ravel(d["Current_load"]))))
        except Exception:
            pass
    caps = np.array(caps)
    p95, p05 = (np.percentile(caps, 95), np.percentile(caps, 5)) if len(caps) else (np.nan, np.nan)
    frac_over = float(np.mean(caps > 2.2)) if len(caps) else np.nan   # 超 2Ah 物理上限(留 10% 余量)
    if len(caps) == 0:
        cls, life = "无放电数据", None
    elif frac_over > 0.05:
        cls, life = "异常:容量超物理上限", None
    elif p95 < 1.6:
        cls, life = "异常:从未满充(EOL定义不适用)", None
    else:
        life = robust_life(caps)
        cls = "标准" if life is not None else "删失(未到EOL)"
    series[name] = caps
    audit.append({"cell": name, "总周期": n, **{f"n_{t}": c for t, c in sorted(types.items())},
                  "n_放电": int(types.get("discharge", 0)),
                  "容量p95": round(p95, 3) if np.isfinite(p95) else None,
                  "容量p05": round(p05, 3) if np.isfinite(p05) else None,
                  "超限占比%": round(100 * frac_over, 1) if np.isfinite(frac_over) else None,
                  "电流中位A": round(float(np.median(amps)), 2) if amps else None,
                  "分类": cls, "寿命圈": life})
    print(f"[{k}/{len(cells)}] {name}: {cls} 寿命={life} p95={p95:.2f} p05={p05:.2f}", flush=True)

au = pd.DataFrame(audit)
print("\n===== 逐颗体检表 =====")
print(au.drop(columns=[c for c in au.columns if c.startswith("n_") and c != "n_放电"]).to_string(index=False))
print(f"\n分类统计: {au['分类'].value_counts().to_dict()}")
print("锚点核对:B0005 测试总放电 168 圈,测到 1.32Ah 停机;EOL@1.4Ah 算得 ~125 圈 = 正确。")
au.to_csv(OUT + r"\nasa_34_audit.csv", index=False, encoding="utf-8-sig")
if not os.path.exists(CACHE):
    pd.to_pickle({"audit": au, "series": series}, CACHE)
    print(f"已缓存: {CACHE}")

# %% ---------- [3] 总览图(原始 Ah) ----------
std = au[au["分类"] == "标准"]["cell"].tolist()
fig, axes = plt.subplots(6, 6, figsize=(20, 16))
for ax, name in zip(axes.flat, au["cell"]):
    caps = series.get(name, np.array([]))
    cls = au.loc[au.cell == name, "分类"].iloc[0]
    life = au.loc[au.cell == name, "寿命圈"].iloc[0]
    if len(caps):
        ax.plot(caps, lw=1)
    ax.axhline(EOL_CAP, color="r", ls="--", lw=0.8)
    ax.set_title(f"{name}({cls.split(':')[0]} {life if life else ''})", fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("NASA 34 颗原始容量(Ah) + 1.4Ah 红线 —— 注意异常类的量纲/形状")
fig.tight_layout()
fig.savefig(OUT + r"\nasa_34_overview_v3.png", dpi=120)
plt.show()

# %% ---------- [4] Q(V) 侦察:按"第 N 次放电"计数(v3 修复) ----------
def qv_curve(volt, curr, t):
    v = np.ravel(volt); i = np.ravel(curr); t = np.ravel(t)
    q = np.concatenate([[0], np.cumsum(-i[:-1] * np.diff(t))]) / 3600.0
    return v, q

samples = ["B0005", "B0042", "B0050"]
fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 4.2), squeeze=False)
for ax, name in zip(axes.flat, samples):
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
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
fig.suptitle("Q(V) 侦察:第10/100次放电 → 5b 电压窗口从这里选(看曲线共同覆盖区)")
fig.tight_layout()
fig.savefig(OUT + r"\nasa_qv_recon_v3.png", dpi=130)
plt.show()

print("""
发我四样:
  1) 体检表截图(重点:三类的划分是否与总览图直觉一致)
  2) 总览图 v3(原始容量版,异常类应当一眼假)
  3) Q(V) 侦察图 v3(这次该有曲线了;看三条曲线的共同电压覆盖区)
  4) 若'标准'类不足 10 颗,我们把 5b 的目标改成'标准类零样本迁移+全类生存分析'
""")
