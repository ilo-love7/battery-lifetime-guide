# -*- coding: utf-8 -*-
"""
电池循环寿命数据分析 —— 第 1 步:容量衰减曲线
=================================================
目标:跑通"数据 → 计算 → 画图"的完整流程,画出电池最核心的一张图:
     放电容量 vs 循环次数(容量衰减曲线)

现在用的是模拟数据(因为 NASA 真实数据集要单独下载)。
模拟数据完全模仿真实电池的行为,分析方法一模一样。
等你下载好 NASA 数据,把最下面的 main() 里改成 load_nasa_data() 即可。

运行方式:双击本文件,或在终端执行  python step1_capacity_fade.py
"""

# %% ---------- 导入库 ----------
import numpy as np
import matplotlib.pyplot as plt
from project_config import PATHS

# 让 matplotlib 正常显示中文(Windows 自带微软雅黑)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# %% ---------- 第一部分:生成模拟的电池循环数据 ----------
def simulate_battery(seed, n_cycles=800, start_capacity=1.85, fade_rate=0.0003,
                     knee_cycle=650, knee_severity=0.004):
    """
    模拟一颗 18650 电池循环老化的放电容量序列。

    真实电池的容量衰减分三个阶段(你以后在任何论文里都会看到):
      1. 前几十圈:SEI 膜形成,容量掉得稍快
      2. 中间几百圈:近似线性的缓慢衰减
      3. "拐点(knee)"之后:容量断崖式下跌(隔膜变形/析锂导致的恶性循环)

    参数:
        seed          随机种子(固定它,每次运行结果一致,便于复现)
        n_cycles      总循环圈数
        start_capacity 初始放电容量(Ah)
        fade_rate     线性阶段的每圈衰减率
        knee_cycle    拐点出现的圈数
        knee_severity 拐点后每圈的额外衰减率
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)
    capacity = np.full(n_cycles, start_capacity, dtype=float)

    for i, c in enumerate(cycles):
        # 阶段1:前 30 圈,SEI 形成,快速损失约 2%(第 30 圈后损失永久保留)
        capacity[i] -= 0.02 * start_capacity * min(c, 30) / 30
        # 阶段2:线性缓慢衰减
        capacity[i] -= fade_rate * start_capacity * c
        # 阶段3:超过拐点圈数后,额外加速衰减
        if c > knee_cycle:
            capacity[i] -= knee_severity * start_capacity * (c - knee_cycle) ** 1.6
        # 加一点测量噪声(真实测试仪也有噪声)
        capacity[i] += rng.normal(0, 0.002 * start_capacity)

    return cycles, capacity


# %% ---------- 第二部分:读取 NASA 真实数据(下载好数据后使用) ----------
def load_nasa_data(mat_path, charge_mode="charge"):
    r"""
    读取 NASA PCoE 电池数据集中的 .mat 文件。

    下载地址:搜索 "NASA PCoE Battery Data Set",
    进入 NASA 官网页面下载 B0005/B0006/B0007/B0018 等文件。
    每个 .mat 文件里包含 cycles 结构体,其中 'data' 字段是逐圈数据。

    用法(下载好之后):
        cycles, capacity = load_nasa_data(PATHS.nasa_dir / "B0005.mat")
    """
    from scipy.io import loadmat  # 读取 .mat 文件需要 scipy

    raw = loadmat(mat_path)
    # NASA 文件的键名类似 "B0005",去掉 matlab 自带的两个元数据键后剩下的就是
    keys = [k for k in raw.keys() if not k.startswith("__")]
    battery = raw[keys[0]][0, 0]["cycle"][0]

    cycles, capacity = [], []
    cycle_no = 0
    for entry in battery:
        if entry["type"][0] == "discharge":          # 只统计放电圈
            cycle_no += 1
            data = entry["data"][0, 0]
            cap = data["Capacity"][0][0] if "Capacity" in data.dtype.names else \
                  data["Cap"][:, 0].sum()
            cycles.append(cycle_no)
            capacity.append(float(cap))

    return np.array(cycles), np.array(capacity)


# %% ---------- 第三部分:分析 + 画图 ----------
def plot_capacity_fade(results, eol_ratio=0.7, save_path=None):
    """
    每颗电池单独画一张容量衰减图(不堆叠),并标出寿命终点(EOL)。

    results: dict,名字 -> (cycles 数组, capacity 数组)
    eol_ratio: 寿命终点定义 —— 容量跌到初始值的 70% 时视为报废
               (行业里 18650 通用标准就是 70%~80%)
    save_path: 兼容旧参数,现在被忽略(文件名自动按电池名生成)
    """
    for name, (cycles, capacity) in results.items():
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(cycles, capacity, lw=1.8, label=name)

        # ---- 计算循环寿命:EOL = 容量首次跌破 初始容量×eol_ratio 的圈数 ----
        eol_threshold = capacity[0] * eol_ratio
        below = np.where(capacity < eol_threshold)[0]
        if len(below) > 0:
            life = cycles[below[0]]
            ax.axvline(life, color="gray", ls=":", lw=1, alpha=0.6)
            ax.annotate(f"寿命 ≈ {life} 圈",
                        xy=(life, eol_threshold),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=9, color="dimgray")

        ax.axhline(eol_threshold, color="red", ls="--", lw=1, alpha=0.5,
                   label=f"EOL 阈值(初始容量的 {eol_ratio:.0%})")

        ax.set_xlabel("循环次数 (cycle)")
        ax.set_ylabel("放电容量 (Ah)")
        ax.set_title(f"{name} 容量衰减曲线")
        ax.legend()
        ax.grid(alpha=0.3)

        # 文件名只保留字母数字,如 "B0005 (真实)" -> "fade_B0005.png"
        safe = "".join(ch for ch in name if ch.isalnum())
        filename = f"fade_{safe}.png"
        PATHS.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = PATHS.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存到: {output_path}")

    plt.show()  # 把所有画好的图显示出来(在 VS Code 交互窗口里内嵌显示,双击运行时弹出窗口)


# %% ---------- 主程序:加载四颗电池,每颗单独出图 ----------
if __name__ == "__main__":
    # ---- 模拟数据(已毕业,注释保留作参考;想回去玩就取消注释) ----
    # b1 = simulate_battery(seed=42, fade_rate=0.00030, knee_cycle=650)
    # b2 = simulate_battery(seed=7,  fade_rate=0.00025, knee_cycle=720)
    # b3 = simulate_battery(seed=99, fade_rate=0.00038, knee_cycle=560)
    # results = {
    #     "Cell-A (正常)": b1,
    #     "Cell-B (较优)": b2,
    #     "Cell-C (较差)": b3,
    # }

    # ---- NASA 真实数据:同批四颗电池,每颗单独输出一张图 ----
    results = {b: load_nasa_data(PATHS.nasa_dir / f"{b}.mat")
               for b in ["B0005", "B0006", "B0007", "B0018"]}

    # 打印每颗电池的循环寿命
    print("=" * 50)
    for name, (cycles, capacity) in results.items():
        eol = capacity[0] * 0.7
        below = np.where(capacity < eol)[0]
        life = cycles[below[0]] if len(below) else "未达到(数据截止)"
        print(f"{name:<14} 初始容量 {capacity[0]:.3f} Ah | 循环寿命: {life}")
    print("=" * 50)

    plot_capacity_fade(results, save_path="capacity_fade.png")

    # ★ 下载好 NASA 真实数据后,把上面 simulate 的部分注释掉,改用:
    #   results = {"B0005": load_nasa_data(PATHS.nasa_dir / "B0005.mat")}
