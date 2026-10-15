# -*- coding: utf-8 -*-
"""
电池寿命数据分析 —— 阶段 4.2:ΔQ(V) 特征移植(自包含版)
================================================================
= 4.1 的解析代码(原样保留,已验证)+ 4.2 的特征计算(新增)。
一次从头跑到尾,不依赖任何旧会话/旧窗口。

流程:
  [0] 导入与配置
  [1] HDF5 解引用工具(复用 4.1)
  [2] 解析三个批次 → df(含每颗的 qd10/qd100/q_dis)
  [3] 数据合理性过滤(4.1 逻辑)
  [4] 特征计算:ΔQ(V) 三件套 + 容量衰减三件套
  [5] 保存 mit_features.csv
  [6] 散点图(每特征一张)
  [7] Pearson r 诊断

注意:8.3GB 是惰性读取(只解引用需要的字段),内存不会爆;
     全程约几分钟,主要耗在逐颗解引用 Qdlin。
"""

# %% ---------- [0] 导入与配置 ----------
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from project_config import PATHS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MIT_DIR = PATHS.mit_dir
FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
OUT_CSV = PATHS.output_dir / "mit_cells_table.csv"
OUT_FEAT = PATHS.output_dir / "mit_features.csv"


# %% ---------- [1] HDF5 解引用工具(4.1 原样) ----------
def deref(f, x):
    """把 HDF5 引用解成真实对象;不是引用就原样返回。"""
    if isinstance(x, h5py.Reference):
        return f[x]
    return x


def field_refs(f, field_ds):
    """batch 结构体某字段的全部引用(每颗电芯一个),按电芯顺序返回 1D 数组。"""
    return field_ds[()].ravel()


def read_cell_summary(f, sum_grp):
    """解出一颗电芯的 summary 组,返回各逐圈数组的 1D numpy。"""
    out = {}
    for k in sum_grp.keys():
        out[k] = np.ravel(np.array(deref(f, sum_grp[k][()])))
    return out


def read_qdlin(f, cell_cycles_grp, matlab_cycle):
    """取某圈 Qdlin;MATLAB 1-based:h5py 索引 = 圈数 - 1。"""
    qd_refs = cell_cycles_grp["Qdlin"][()].ravel()
    idx = matlab_cycle - 1
    if idx < 0 or idx >= len(qd_refs):
        return None
    return np.ravel(np.array(deref(f, qd_refs[idx])))


# %% ---------- [2] 解析三个批次(4.1 原样,含 _qd10/_qd100/_q_dis) ----------
rows = []
for batch_name, fname in FILES.items():
    with h5py.File(rf"{MIT_DIR}\{fname}", "r") as f:
        b = f["batch"]
        life_refs = field_refs(f, b["cycle_life"])
        bar_refs = field_refs(f, b["barcode"])
        sum_refs = field_refs(f, b["summary"])
        cyc_refs = field_refs(f, b["cycles"])
        vd_refs = field_refs(f, b["Vdlin"])
        n_cells = len(life_refs)
        print(f"{batch_name}: {n_cells} 颗电芯")

        for i in range(n_cells):
            try:
                cycle_life = deref(f, life_refs[i])[()].item()
            except Exception:
                cycle_life = np.nan

            s = read_cell_summary(f, deref(f, sum_refs[i]))
            q_dis = s.get("QDischarge", np.array([]))

            qd10 = read_qdlin(f, deref(f, cyc_refs[i]), 10)
            qd100 = read_qdlin(f, deref(f, cyc_refs[i]), 100)

            rows.append({
                "batch": batch_name,
                "cell_idx": i,
                "cycle_life": cycle_life,
                "n_summary": len(q_dis),
                "q_c100": q_dis[99] if len(q_dis) > 99 else np.nan,
                "qd10_ok": qd10 is not None and np.isfinite(qd10).all(),
                "qd100_ok": qd100 is not None and np.isfinite(qd100).all(),
                "_q_dis": q_dis,
                "_qd10": qd10,
                "_qd100": qd100,
            })

df = pd.DataFrame(rows)
print("\n解析完成:", len(df), "颗电芯")
print(df.groupby("batch")[["cycle_life"]].agg(["count", "min", "median", "max"]))


# %% ---------- [3] 合理性过滤(4.1 逻辑) ----------
def is_sane(r):
    if not np.isfinite(r["cycle_life"]) or r["cycle_life"] <= 0:
        return False
    if not (np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6):
        return False
    if not (r["qd10_ok"] and r["qd100_ok"]):
        return False
    return True


df["keep"] = df.apply(is_sane, axis=1)
print("\n过滤统计:")
print(df.groupby("batch")["keep"].agg(["sum", "count"]))


# %% ---------- [4] 特征计算(本阶段核心) ----------
def calc_features(r) -> dict:
    """对一颗电芯算 Severson 前 100 圈特征;缺数据则少几项。"""
    feats = {}
    qd10, qd100, q_dis = r["_qd10"], r["_qd100"], r["_q_dis"]

    # --- ΔQ_{100-10}(V):两条曲线同在 Vdlin 网格上,直接作差 ---
    if qd10 is not None and qd100 is not None and len(qd10) == len(qd100):
        dq = qd100 - qd10
        feats["log_var"] = np.log10(max(np.var(dq), 1e-12))
        feats["log_min"] = np.log10(max(-dq.min(), 1e-12))
        feats["log_mean"] = np.log10(max(abs(dq.mean()), 1e-12))

    # --- 容量衰减:第 2~100 圈线性拟合(1-based → 索引 = 圈数-1) ---
    if len(q_dis) >= 100:
        y = q_dis[1:100]                      # 第 2..100 圈
        x = np.arange(2, 101)
        slope, intercept = np.polyfit(x, y, 1)
        feats["slope"] = slope
        feats["intercept"] = intercept
        # discrepancy:拟合线外推到第 100 圈与实际容量的偏差
        feats["discrepancy"] = np.polyval((slope, intercept), 100) - q_dis[99]
    return feats


feat_rows = []
n_skip = 0
for _, r in df[df["keep"]].iterrows():
    feats = calc_features(r)
    if not feats:
        n_skip += 1
        continue
    feat_rows.append({
        "batch": r["batch"],
        "cell_idx": r["cell_idx"],
        "cycle_life": r["cycle_life"],
        **feats,
    })

feat_df = pd.DataFrame(feat_rows)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n特征提取完成: {len(feat_df)} 颗(跳过 {n_skip} 颗)")
print(f"各特征缺失数:\n{feat_df.isna().sum()}")


# %% ---------- [5] 保存特征表 ----------
feat_df.to_csv(OUT_FEAT, index=False, encoding="utf-8-sig")
print(f"特征表已保存: {OUT_FEAT}")


# %% ---------- [6] 散点图:每个特征一张,不堆叠 ----------
features = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
batch_colors = {"batch1": "tab:blue", "batch2": "tab:orange", "batch3": "tab:green"}
titles = {
    "log_var": "ΔQ(V) 方差 vs 寿命 —— Severson Fig.1c 复现",
    "log_min": "ΔQ(V) 最小值 vs 寿命",
    "log_mean": "ΔQ(V) 均值 vs 寿命",
    "slope": "容量衰减斜率 vs 寿命",
    "intercept": "容量拟合截距 vs 寿命",
    "discrepancy": "差异度 discrepancy vs 寿命",
}

for feat in features:
    sub = feat_df.dropna(subset=[feat])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for bname, g in sub.groupby("batch"):
        ax.scatter(g[feat], np.log10(g["cycle_life"].clip(lower=1)),
                   s=28, alpha=0.75, label=bname,
                   color=batch_colors.get(bname, "gray"))
    ax.set_xlabel(feat)
    ax.set_ylabel("log10(循环寿命)")
    ax.set_title(titles[feat])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = PATHS.output_dir / f"scatter_{feat}.png"
    fig.savefig(out_png, dpi=150)
    plt.show()          # 在交互窗口里直接显示;同时保存 PNG 到统一输出目录。
    print(f"已存图: {out_png}")


# %% ---------- [7] 诊断:各特征与 log(寿命) 的 Pearson r ----------
print("各特征与 log10(寿命) 的 Pearson r(预期:log_var 强负相关,|r| 最大):")
for feat in features:
    sub = feat_df.dropna(subset=[feat])
    if len(sub) < 3:
        print(f"  {feat}: 数据不足")
        continue
    r = np.corrcoef(sub[feat], np.log10(sub["cycle_life"].clip(lower=1)))[0, 1]
    print(f"  {feat:>12}: r = {r:+.3f}  (n={len(sub)})")

print("""
验收口径(对照论文):
  - log_var 应为明显负相关(方差大 → 死得快),且 |r| 在六个特征里居首;
  - scatter_log_var.png 里 batch1(训练)与 batch2(测试)应落在同一条趋势带上;
  - 若 log_var 的 r 异常(弱/为正)→ 说明 1-based 对齐有问题,报错给我。
""")
