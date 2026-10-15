# -*- coding: utf-8 -*-
"""
step5d v1 (2026-10-13 22:50):NASA 删失电池生存分析(阶段2伏笔回收)
================================================================
背景:5a 审计后,标准类仅 3 颗(有寿命标签),11 颗删失(未测到 EOL 就停机)。
删失电池不能硬算寿命(阶段2 的教训),正确工具=Kaplan-Meier 生存分析:
  - 事件 = 容量首次持续跌破 1.4Ah(仅标准 3 颗发生);
  - 删失 = 停测时仍未跌到(11 颗),进入风险集直到停测圈数;
  - KM 曲线给出 S(n)=P(寿命>n) 的无偏估计;若 S 始终 >0.5,
    中位寿命不可估,只能给"下界=最长删失时间"——这本身是结论。
复用 v4 的三道门分类逻辑,自包含。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from project_config import PATHS

print("== step5d v2:删失生存分析 [修复:0.1过滤吃井/life残留/B0033边界] ==")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

NASA_DIR = PATHS.nasa_dir
EOL_CAP = 1.4
SUSTAIN = 3
OUT = str(PATHS.output_dir)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)

# %% ---------- [1] 解析 + 复用 v4 三道门分类 ----------
def cycles_container(c0):
    hits = [f for f in c0.dtype.names if "cycle" in f.lower()]
    cont = c0[hits[0]]
    while cont.ndim >= 2 and cont.shape[0] == 1 and cont.size == 1:
        cont = cont[0, 0]
    return cont

def robust_life(caps, eol=EOL_CAP, sustain=SUSTAIN):
    q = np.asarray(caps, float)
    sm = pd.Series(q).rolling(5, center=True, min_periods=1).median().values
    ok = np.where(sm > eol + 0.05)[0]
    start = ok[0] if len(ok) else 0
    below = sm[start:] < eol
    run, hit = 0, None
    for j, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= sustain:
            hit = j - sustain + 1
            break
    return (start + hit + 1) if hit is not None else None   # None=删失

cells = sorted(f[:-4] for f in os.listdir(NASA_DIR) if f.endswith(".mat"))
assert len(cells) == 34, f"NASA 目录应有 34 颗 .mat,现仅 {len(cells)} 颗 —— " \
    f"KM 入组/删失结构依赖完整 34 颗,先恢复完整数据集({NASA_DIR})"
rows = []
for name in cells:
    m = sio.loadmat(os.path.join(NASA_DIR, name + ".mat"))
    c0 = m[[k for k in m.keys() if not k.startswith("__")][0]]
    cyc = cycles_container(c0)
    n = cyc.shape[1] if cyc.ndim == 2 else len(cyc)
    caps, ordn = [], 0
    for i in range(n):
        c = cyc[0, i] if cyc.ndim == 2 else cyc[i]
        if "discharge" not in str(np.squeeze(c["type"])).lower():
            continue
        ordn += 1
        d = c["data"][0, 0]
        if "Capacity" in (d.dtype.names or ()):
            v = np.ravel(d["Capacity"])
            # v2 修复:阈值 0.1→0.02。原 0.1 会把中断井周期(容量≈0.1)整个
            # 滤掉 → 井占比恒为 0 → 中断井门失效,B0042/43/44 误入标准类
            if len(v) and np.isfinite(v[0]) and v[0] > 0.02:
                caps.append(float(v[0]))
    caps = np.array(caps)
    q = np.median(caps) if len(caps) else np.nan
    life = None  # v2 修复:每颗重置(v1 残留上一颗的 life,污染异常类)
    # v4 三道门(与 v4 逐字对齐:尖峰用 p95 而非 max)
    p95 = float(np.percentile(caps, 95)) if len(caps) else 9.9
    if len(caps) < 10:
        cls = "样本过少"
    elif p95 > 2.2:
        cls = "异常:尖峰"
    elif np.mean(caps < 1.0) > 0.10:
        cls = "异常:中断井"
    elif np.median(np.abs(np.diff(caps))) > 0.06:
        cls = "异常:抖动"
    elif p95 < 1.6:
        cls = "异常:从未满充"
    else:
        life = robust_life(caps)
        cls = "标准" if life is not None else "删失"
    if name == "B0033":  # 5a 决议:起始容量 1.55Ah 存疑,单列边界,不入 KM
        cls = "边界(起始容量存疑)"
        life = None
    rows.append({"cell": name, "n_dis": len(caps), "cap_med": round(q, 3),
                 "class": cls,
                 "time": (life if cls == "标准" else len(caps)),
                 "event": 1 if cls == "标准" else 0,
                 "life": life})
audit = pd.DataFrame(rows)
print(audit.to_string(index=False))
n_std = (audit["class"] == "标准").sum(); n_cen = (audit["class"] == "删失").sum()
print(f"\n标准 {n_std} / 删失 {n_cen} / 异常 {len(audit)-n_std-n_cen}")
assert (audit.cell[audit["class"] == "标准"].tolist()
        == ["B0005", "B0006", "B0018"]), "标准类与 v4 不符,发打印!"

# %% ---------- [2] Kaplan-Meier ----------
sub = audit[audit["class"].isin(["标准", "删失"])].copy()
t = sub["time"].values.astype(float)
e = sub["event"].values.astype(int)
order = np.argsort(t)
t, e = t[order], e[order]
S, at_risk, km_rows = 1.0, len(t), []
i = 0
while i < len(t):
    tt = t[i]; d = c = 0
    while i < len(t) and t[i] == tt:
        d += e[i]; c += 1 - e[i]; i += 1
    if d > 0:
        S *= 1 - d / at_risk
    km_rows.append({"时间(圈)": int(tt), "风险集": at_risk,
                    "事件(到达EOL)": d, "删失": c, "S(n)": round(S, 4)})
    at_risk -= d + c
km = pd.DataFrame(km_rows)
print("\n===== Kaplan-Meier 表 =====")
print(km.to_string(index=False))
S_min = km["S(n)"].iloc[-1]
max_cen = int(t[e == 0].max())
if S_min > 0.5:
    print(f"\n中位寿命:不可估(S 始终 {S_min:.2f} > 0.5)。"
          f"可下结论:中位寿命 > {max_cen} 圈(最长删失时间)——"
          f"这是删失数据的正确说法,'平均寿命多少圈'是错的问法。")
else:
    med = int(km.loc[km["S(n)"] <= 0.5, "时间(圈)"].min())
    print(f"\n中位寿命:{med} 圈(S 首次降到 0.5 以下的时间点;S 末值 {S_min:.2f})。"
          f"事件 {int(e.sum())} 个 / 删失 {int((1 - e).sum())} 个 → 不确定性大,"
          f"引用时务必带'估计'二字;'平均寿命多少圈'仍是错的问法。")
km.to_csv(OUT + r"\nasa_km_table.csv", index=False, encoding="utf-8-sig")

# %% ---------- [3] KM 曲线 ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
xs, ss, step_t = [0], [1.0], 0
for _, r in km.iterrows():
    if r["事件(到达EOL)"] > 0:
        xs += [r["时间(圈)"], r["时间(圈)"]]
        ss += [ss[-1], r["S(n)"]]
cen = np.sort(t[e == 0])
ax.plot(xs, ss, drawstyle="steps-post", lw=2, label="KM 生存曲线 S(n)")
ax.plot(cen, [1.03] * len(cen), ls="", marker="|", ms=12, mew=2,
        color="gray", label="删失电池(停测圈数)")
ax.axvline(547, color="r", ls=":", lw=1.5,
           label="MIT dev 中位 547 圈(仅示意:协议不同)")
ax.set_xlabel("循环数 n"); ax.set_ylabel("S(n) = P(寿命 > n)")
ax.set_ylim(0, 1.1)
ax.set_title("NASA 标准+删失 14 颗:Kaplan-Meier 生存分析")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + r"\nasa_km_curve.png", dpi=140)
plt.show()

print("""
判读与产物:
  - 事件只有 3 个、删失 11 个 → S 段稀疏。中位寿命取首个 S≤0.5 的时间点,
    只是点估计、不确定性大;若 S 始终 >0.5,则只能给"中位寿命 > 最长删失
    时间"的下界——'平均寿命多少圈'是错的问法,这个纪律要原样写进报告;
  - 三颗标准电池的 S 跳降位置(125/109/97) vs MIT 中位 547 的相对位置;
  - 灰色刻度线(删失)散布在哪一段 → 数据集的测试截止习惯。
""")
