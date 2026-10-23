"""Render the GitHub social preview (项目宣传图.png) from the checked-in EN test predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

from project_config import enable_utf8_stdout

# Windows 控制台默认 cp1252：本脚本会打印中文进度，必须先把 stdout 切成 UTF-8。
enable_utf8_stdout()


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "阶段43f_论文主测试逐颗误差.csv"
PLOT_TEMP = ROOT / "results" / "generated" / "banner_en_scatter.png"
WIDTH, HEIGHT = 1280, 640

THEMES = {
    "light": dict(
        bg="#FFFFFF", text="#10203A", muted="#5C6C80", blue="#2E6BE6",
        green="#0E9F6E", amber="#C77E14", red="#E04755",
        divider="#DCE4EE", subtle="#8A99AC", footer="#7A8899", vline="#E2E8F0",
        pt_edge="#FFFFFF", dashed="#9FB0C2", grid="#334155", grid_alpha=0.15,
        spine="#C3CEDB", out="项目宣传图.png",
    ),
}


def find_font(*names: str) -> str:
    candidates = [
        Path("C:/Windows/Fonts") / name for name in names
    ] + [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("需要微软雅黑、Noto Sans CJK 或 PingFang 字体来渲染中文 banner")


def draw_scatter(data: pd.DataFrame, t: dict) -> Image.Image:
    regular_path = find_font("msyh.ttc", "arial.ttf")
    font_prop = FontProperties(fname=regular_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(7.15, 5.15), dpi=130)
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    actual = data["cycle_life"].to_numpy(dtype=float)
    predicted = data["预测"].to_numpy(dtype=float)
    errors = data["误差%"].to_numpy(dtype=float)
    axis.scatter(actual, predicted, c=errors, cmap="viridis", s=34,
                 alpha=0.92, edgecolors=t["pt_edge"], linewidths=0.45, zorder=3)
    limits = [100, 2400]
    axis.plot(limits, limits, linestyle="--", color=t["dashed"], linewidth=1.25,
              alpha=0.85, zorder=2)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("真实寿命（圈）", color=t["text"], fontsize=11, labelpad=8)
    axis.set_ylabel("预测寿命（圈）", color=t["text"], fontsize=11, labelpad=8)
    axis.tick_params(colors=t["muted"], labelsize=9)
    axis.grid(color=t["grid"], alpha=t["grid_alpha"], linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color(t["spine"])
        spine.set_alpha(0.9)
    colorbar = figure.colorbar(axis.collections[0], ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label("绝对百分比误差", color=t["muted"], fontsize=9)
    colorbar.ax.tick_params(colors=t["muted"], labelsize=8)
    colorbar.outline.set_edgecolor(t["spine"])
    figure.subplots_adjust(left=0.12, right=0.91, top=0.96, bottom=0.14)

    PLOT_TEMP.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_TEMP, transparent=True, dpi=130)
    plt.close(figure)
    return Image.open(PLOT_TEMP).convert("RGBA")


def render(predictions: pd.DataFrame, name: str) -> Path:
    t = THEMES[name]

    canvas = Image.new("RGB", (WIDTH, HEIGHT), t["bg"])
    draw = ImageDraw.Draw(canvas)

    regular = find_font("msyh.ttc", "arial.ttf")
    bold = find_font("msyhbd.ttc", "arialbd.ttf", "msyh.ttc")
    eyebrow = ImageFont.truetype(regular, 17)
    title_cn = ImageFont.truetype(bold, 52)
    title_en = ImageFont.truetype(bold, 55)
    subtitle = ImageFont.truetype(regular, 18)
    hero = ImageFont.truetype(bold, 102)
    label = ImageFont.truetype(regular, 16)
    metric = ImageFont.truetype(bold, 32)
    metric_label = ImageFont.truetype(regular, 14)
    footer = ImageFont.truetype(regular, 15)
    chart_title = ImageFont.truetype(bold, 17)

    for x in range(WIDTH):
        t_grad = x / max(WIDTH - 1, 1)
        left = np.array([39, 164, 255], dtype=float)
        right = np.array([56, 201, 141], dtype=float)
        color = tuple(np.round(left * (1 - t_grad) + right * t_grad).astype(int))
        draw.line((x, 0, x, 5), fill=color)

    draw.text((58, 38), "SEVERSON ET AL. · NATURE ENERGY 2019", font=eyebrow, fill=t["blue"])
    draw.text((56, 74), "从零复现", font=title_cn, fill=t["text"])
    draw.text((55, 133), "Severson 2019", font=title_en, fill=t["text"])
    draw.text((60, 202), "电池寿命预测 · 复现、审计与学习攻略", font=subtitle, fill=t["muted"])

    draw.text((55, 244), "5.9%", font=hero, fill=t["green"])
    draw.text((350, 282), "官方 test_ind · EN6 中位误差", font=label, fill=t["text"])
    draw.text((350, 312), "95% bootstrap CI  [5.1, 8.9]", font=label, fill=t["muted"])
    draw.text((350, 342), "论文点估计 9.1% · 统计差异未定", font=label, fill=t["subtle"])

    draw.line((60, 405, 580, 405), fill=t["divider"], width=1)
    metrics = [
        ("3/34", "NASA 标签成立", t["amber"]),
        ("18/18", "目标特征越界", t["red"]),
        ("125", "KM 中位寿命（圈）", t["blue"]),
    ]
    starts = [62, 242, 424]
    for (value, caption, color), x in zip(metrics, starts):
        draw.text((x, 424), value, font=metric, fill=color)
        draw.text((x, 466), caption, font=metric_label, fill=t["muted"])
    draw.line((60, 507, 580, 507), fill=t["divider"], width=1)
    draw.text((62, 527), "误差归因链", font=metric_label, fill=t["muted"])
    draw.text((164, 520), "33.6%  →  5.9%", font=ImageFont.truetype(bold, 25), fill=t["text"])
    draw.text((62, 567), "每个结论都有数据、代码与复核记录", font=footer, fill=t["footer"])

    draw.line((613, 35, 613, 605), fill=t["vline"], width=1)
    draw.text((650, 42), "OFFICIAL TEST  ·  n = 43", font=eyebrow, fill=t["blue"])
    draw.text((650, 72), "预测寿命 vs 真实寿命", font=chart_title, fill=t["text"])
    scatter = draw_scatter(predictions, t)
    chart_x, chart_y = 625, 105
    max_width, max_height = 635, 485
    scale = min(max_width / scatter.width, max_height / scatter.height)
    scatter = scatter.resize((int(scatter.width * scale), int(scatter.height * scale)), Image.Resampling.LANCZOS)
    canvas.paste(scatter, (chart_x, chart_y), scatter)

    output = ROOT / t["out"]
    canvas.save(output, optimize=True)
    print(f"Banner 已生成: {output.name} ({WIDTH}x{HEIGHT})")
    return output


def main() -> None:
    if not PREDICTIONS.is_file():
        raise FileNotFoundError(f"逐颗测试结果不存在: {PREDICTIONS}")
    predictions = pd.read_csv(PREDICTIONS)
    if len(predictions) != 43:
        raise ValueError(f"官方测试逐颗表应有 43 行，实际 {len(predictions)} 行")
    render(predictions, "light")


if __name__ == "__main__":
    main()
