# -*- coding: utf-8 -*-
"""
电池循环寿命数据分析 —— 第 2 步:四颗真实电池的单独对比图
==========================================================
把 NASA 同批的 4 颗电池(B0005/B0006/B0007/B0018)各自画在独立子图里,
每张子图包含:原始容量(细线)+ 10 圈滑动平均(粗线)+ EOL 阈值线。

运行方式:python step2_individual_plots.py
输出:每颗电池一张单独的 PNG(单图版)+ 一张 2x2 汇总图
"""

import numpy as np
import matplotlib.pyplot as plt

from step1_capacity_fade import load_nasa_data
from project_config import PATHS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = PATHS.nasa_dir
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
EOL_RATIO = 0.7
SMOOTH_WINDOW = 10  # 平滑窗口:10 圈滑动平均


def smooth(y, window=SMOOTH_WINDOW):
    """滑动平均。返回的数组比原数组短 window-1 个点(右端对齐损失)。"""
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def eol_of(cycles, capacity):
    """用平滑数组判定 EOL。返回 (寿命圈数 或 None, 平滑曲线)。"""
    sm = smooth(capacity)
    threshold = capacity[0] * EOL_RATIO
    below = np.where(sm < threshold)[0]
    if len(below) == 0:
        return None, sm
    # 平滑数组短 window-1 个点,且滑动平均是"中心趋势",
    # 这里把寿命对齐回原始圈数轴:平滑数组的第 i 个点对应第 i+window-1 圈
    return cycles[below[0] + SMOOTH_WINDOW - 1], sm


def plot_one_cell(ax, name, cycles, capacity):
    """在给定坐标系上画一颗电池:原始 + 平滑 + EOL 标注。"""
    life, sm = eol_of(cycles, capacity)
    threshold = capacity[0] * EOL_RATIO

    ax.plot(cycles, capacity, lw=0.9, alpha=0.55, color="steelblue",
            label="原始数据")
    if sm is not None:
        ax.plot(cycles[SMOOTH_WINDOW - 1:], sm, lw=2.2, color="black",
                label=f"{SMOOTH_WINDOW} 圈滑动平均")
    ax.axhline(threshold, color="red", ls="--", lw=1, alpha=0.6,
               label=f"EOL 阈值({EOL_RATIO:.0%})")

    if life is not None:
        ax.axvline(life, color="gray", ls=":", lw=1, alpha=0.7)
        ax.annotate(f"EOL ≈ {life} 圈", xy=(life, threshold),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=9, color="dimgray")
    else:
        ax.annotate("未达 EOL\n(测试提前截止)", xy=(cycles[-1], capacity[-1]),
                    xytext=(-70, -30), textcoords="offset points",
                    fontsize=9, color="dimgray")

    ax.set_title(f"{name}  初始容量 {capacity[0]:.3f} Ah")
    ax.set_xlabel("循环次数 (cycle)")
    ax.set_ylabel("放电容量 (Ah)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


if __name__ == "__main__":
    data = {b: load_nasa_data(DATA_DIR / f"{b}.mat") for b in BATTERIES}
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 先打印寿命总表 ----
    print("=" * 60)
    for name, (cycles, capacity) in data.items():
        life, _ = eol_of(cycles, capacity)
        life_str = f"{life} 圈" if life else "未达到(数据截止)"
        fade_pct = (1 - capacity[-1] / capacity[0]) * 100
        print(f"{name}  {len(cycles):>3} 圈 | 平滑判定寿命: {life_str:<14} "
              f"| 测试结束时已衰减 {fade_pct:.1f}%")
    print("=" * 60)

    # ---- 2x2 汇总图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, name in zip(axes.flat, BATTERIES):
        plot_one_cell(ax, name, *data[name])
    fig.suptitle("NASA PCoE 四颗 18650 电池容量衰减对比", fontsize=14)
    fig.tight_layout()
    grid_path = PATHS.output_dir / "four_cells_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"汇总图已保存: {grid_path}")

    # ---- 每颗电池单独一张图(单图版,放大细节看拐点用) ----
    for name, (cycles, capacity) in data.items():
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        plot_one_cell(ax1, name, cycles, capacity)
        fig1.tight_layout()
        path = PATHS.output_dir / f"{name}_fade.png"
        fig1.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig1)          # 关掉画布,循环生成下一张
        print(f"单图已保存: {path}")
