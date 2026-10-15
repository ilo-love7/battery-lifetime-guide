# -*- coding: utf-8 -*-
"""
电池循环寿命数据分析 —— 第 3 步:dQ/dV 老化指纹与 ΔQ(V) 特征
=============================================================
核心思想(对应巩固过的概念):
  - 容量曲线只告诉你"死得多快";
  - dQ/dV 曲线的峰(电极相变指纹)告诉你"怎么死的":
      * 峰相对错位  → LII(锂库存损失)
      * 峰变矮/变宽 → LAM(活性材料损失)
      * 所有峰整体平移 → 阻抗上升
  - ΔQ(V) = 第100圈和第10圈放电曲线之差,方差是 Severson 2019 预测寿命的最强特征。

数据处理的三道工序(每道都在治一种真实数据的病):
  1. 剔除暂停段:放电数据里混着电流≈0 的暂停点,暂停时电压回升松弛,
     会把 dQ/dV 搅成锯齿 —— 只保留 |I|>1A 的真正放电点;
  2. 统一电压网格:每圈采样时刻不同,两圈相减必须"同一电压比同一容量";
  3. Savitzky-Golay 平滑:每圈只有 ~300 个点,直接微分会放大噪声,
     SG 局部多项式拟合比滑动平均更保峰形。
"""

# %% ---------- 导入库 ----------
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter
from project_config import PATHS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = PATHS.nasa_dir
PATHS.output_dir.mkdir(parents=True, exist_ok=True)


# %% ---------- 数据加载:每一圈放电的 (电压V, 累计容量Q) ----------
def load_discharge_vq(mat_path):
    """
    读取一颗电池所有放电圈的电压曲线。
    返回 dict: 圈数 -> (V 数组, Q 数组)
      V: 该圈放电过程中的电压
      Q: 累计放出的容量(Ah),由 Q = ∫|I|·dt 数值积分得到
         (数据里没有现成 Q,用电流对时间自己积分,和官方标定误差 <0.5%)

    注意 mask:只保留 |I| > 1A 的点 —— 放电是 2A 恒流,
    电流≈0 的暂停点电压会回升,属于"非放电状态",必须剔除。
    """
    raw = loadmat(mat_path)
    key = [k for k in raw.keys() if not k.startswith("__")][0]
    battery = raw[key][0, 0]["cycle"][0]

    out = {}
    n = 0
    for entry in battery:
        if entry["type"][0] != "discharge":
            continue
        n += 1
        d = entry["data"][0, 0]
        V = d["Voltage_measured"].ravel()
        I = d["Current_measured"].ravel()
        t = d["Time"].ravel()

        mask = np.abs(I) > 1.0
        V, I, t = V[mask], I[mask], t[mask]
        dt = np.diff(t, prepend=t[0])
        Q = np.cumsum(np.abs(I) * dt) / 3600.0    # As -> Ah
        out[n] = (V, Q)
    return out


def q_on_v_grid(V, Q, v_min=3.4, v_max=4.15, step=0.005):
    """
    把一圈的 (V,Q) 插值到统一电压网格(5mV 步长)。
    """
    order = np.argsort(V)
    v_grid = np.arange(v_min, v_max, step)
    q_grid = np.interp(v_grid, V[order], Q[order])
    return v_grid, q_grid


def dqdv(v_grid, q_grid, window=31):
    """
    计算 dQ/dV。放电时 Q 随 V 升高而减少,导数为负,
    取负号 -dQ/dV 让峰朝上(教科书惯例)。
    """
    q_smooth = savgol_filter(q_grid, window, 3)
    return -np.gradient(q_smooth, v_grid)


# %% ---------- 加载四颗电池 ----------
cells = {b: load_discharge_vq(DATA_DIR / f"{b}.mat")
         for b in ["B0005", "B0006", "B0007", "B0018"]}
print("加载完成:", {b: len(d) for b, d in cells.items()}, "圈")


# %% ---------- 图1:每颗电池的 dQ/dV 随老化的演化(单独出图) ----------
def plot_dqdv_evolution(cell_name, cycle_list=(10, 60, 120, 160), window=31):
    data = cells[cell_name]
    n_total = max(data.keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cycle_list)))

    for c, color in zip(cycle_list, colors):
        c = min(c, n_total)
        V, Q = data[c]
        v_grid, q_grid = q_on_v_grid(V, Q)
        ax.plot(v_grid, dqdv(v_grid, q_grid, window),
                lw=1.8, color=color, label=f"第 {c} 圈")

    ax.set_xlabel("电压 (V)")
    ax.set_ylabel("-dQ/dV (Ah/V)")
    ax.set_title(f"{cell_name} dQ/dV 随循环的演化(蓝=早期,黄=晚期)")
    ax.legend()
    ax.grid(alpha=0.3)
    output_path = PATHS.output_dir / f"{cell_name}_dqdv_evolution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {output_path}")
    plt.show()


for b in ["B0005", "B0006","B0007","B0018"]:
    plot_dqdv_evolution(b)


# %% ---------- 图2:ΔQ(V) —— Severson 的最强寿命特征 ----------
def delta_q(cell_name, c_late=50, c_early=10):
    """ΔQ(V) = 第 c_late 圈与第 c_early 圈在同一电压网格上的容量之差。"""
    data = cells[cell_name]
    n_total = max(data.keys())
    V1, Q1 = data[c_early]
    V2, Q2 = data[min(c_late, n_total)]
    v_grid, q1 = q_on_v_grid(V1, Q1)
    _, q2 = q_on_v_grid(V2, Q2)
    return v_grid, q2 - q1


fig, ax = plt.subplots(figsize=(9, 6))
print("=" * 60)
for name, color in [("B0005", "tab:blue"), ("B0006", "tab:red"),("B0007", "tab:green"),("B0018", "tab:orange")]:
    v_grid, dQ = delta_q(name)
    ax.plot(v_grid, dQ, lw=1.8, color=color, label=f"{name}: 第50圈 − 第10圈")
    print(f"{name}  ΔQ(V) 方差 = {np.var(dQ):.5f} Ah²")
print("=" * 60)

ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("电压 (V)")
ax.set_ylabel("ΔQ (Ah)")
ax.set_title("ΔQ(V):第100圈与第10圈放电容量曲线之差")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(PATHS.output_dir / "deltaQ_comparison.png", dpi=150, bbox_inches="tight")
print("已保存: deltaQ_comparison.png")
plt.show()
