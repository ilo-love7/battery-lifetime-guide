# -*- coding: utf-8 -*-
"""
电池寿命数据分析 —— 阶段 4.1:MIT-Severson 数据集解析与电芯总表
================================================================
    数据:data/mit/ 下三个 .mat(MATLAB v7.3 / HDF5 格式)
目标:把 8.3GB 的二进制文件提炼成一张"一行一颗电芯"的总表(CSV)

与 NASA 数据的三个不同(都已在准备阶段踩明):
  1. 格式是 HDF5 不是普通 .mat —— 用 h5py,MATLAB 结构体在 HDF5 里
     是"引用(reference)"的层层嵌套,需要 deref 解引用;
  2. MATLAB 是 1-based:第 10 圈 = h5py 里的索引 9(差 1 对齐);
  3. 数据自带 Vdlin(统一电压网格)和每圈 Qdlin(Q-V 曲线)——
     作者预计算好的 ΔQ(V) 快车道,不需要我们像 step3 那样自己插值。

论文口径(Severson et al. 2019):140 颗 → 过滤 → 124 颗
  batch1(2017-05-12, 41 颗)= 训练集
  batch2(2017-06-30, 43 颗)= 主测试集
  batch3(2018-04-12, 40 颗)= 二级测试集
本脚本先用"cycle_life 有限 + 容量数据合理"过滤,官方逐颗名单待补,
管线做成参数化,名单到位后改一个函数即可。
"""

# %% ---------- 导入与配置 ----------
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


# %% ---------- HDF5 解引用工具 ----------
# 真实层级(实测):batch 的字段是 (N,1) 的引用数据集(每颗电芯一个引用);
# 解引用后,cycles 是"按字段名分组"的 Group,每个字段下挂 (n_cycles,) 的引用数组。
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
    """
    取某圈的 Qdlin(Q-V 曲线,画在 Vdlin 网格上)。
    MATLAB cycles 结构体是 1-based:h5py 索引 = MATLAB 圈数 - 1。
    """
    qd_refs = cell_cycles_grp["Qdlin"][()].ravel()
    idx = matlab_cycle - 1
    if idx < 0 or idx >= len(qd_refs):
        return None
    return np.ravel(np.array(deref(f, qd_refs[idx])))


# %% ---------- 解析三个批次 ----------
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
            # --- 标量字段 ---
            try:
                cycle_life = deref(f, life_refs[i])[()].item()
            except Exception:
                cycle_life = np.nan
            try:
                bc_val = deref(f, bar_refs[i])[()]
                barcode = bc_val.decode() if isinstance(bc_val, bytes) else np.ravel(np.array(bc_val))[0]
            except Exception:
                barcode = -1

            # --- summary:逐圈容量 / 充电时间 ---
            s = read_cell_summary(f, deref(f, sum_refs[i]))
            q_dis = s.get("QDischarge", np.array([]))
            chg_t = s.get("chargetime", np.array([]))

            # --- Vdlin 与第 10 / 100 圈的 Q-V 曲线 ---
            try:
                vdlin = np.ravel(np.array(deref(f, vd_refs[i])))
            except Exception:
                vdlin = np.array([])
            qd10 = read_qdlin(f, deref(f, cyc_refs[i]), 10)
            qd100 = read_qdlin(f, deref(f, cyc_refs[i]), 100)

            rows.append({
                "batch": batch_name,
                "cell_idx": i,
                "barcode": barcode,
                "cycle_life": cycle_life,
                "n_summary": len(q_dis),
                "q_c2": q_dis[1] if len(q_dis) > 1 else np.nan,
                "q_c100": q_dis[99] if len(q_dis) > 99 else np.nan,
                "chg_c100": chg_t[99] if len(chg_t) > 99 else np.nan,
                "has_vdlin": len(vdlin) > 0,
                "qd10_ok": qd10 is not None and np.isfinite(qd10).all(),
                "qd100_ok": qd100 is not None and np.isfinite(qd100).all(),
                "_q_dis": q_dis,
                "_vdlin": vdlin,
                "_qd10": qd10,
                "_qd100": qd100,
            })

df = pd.DataFrame(rows)
print("\n解析完成:", len(df), "颗电芯")
print(df.groupby("batch")[["cycle_life"]].agg(["count", "min", "median", "max"]))


# %% ---------- 数据合理性过滤(官方白名单的占位逻辑) ----------
# 论文口径:batch3 有 6 颗没循环到 80% 阈值(cycle_life = NaN),必须剔除;
# batch1/batch2 各有少量因稳定性原因被排除的颗(官方名单待补),先用数据合理性占位。
def is_sane(r):
    if not np.isfinite(r["cycle_life"]) or r["cycle_life"] <= 0:
        return False
    if not (np.isfinite(r["q_c100"]) and 0.4 < r["q_c100"] < 1.6):
        return False                       # LFP 标称 1.1Ah,超出即异常
    if not (r["qd10_ok"] and r["qd100_ok"]):
        return False
    return True


df["keep"] = df.apply(is_sane, axis=1)
print("\n过滤统计:")
print(df.groupby("batch")["keep"].agg(["sum", "count"]))
df[df["keep"]].to_csv(OUT_CSV.replace(".csv", "_raw.csv"), index=False)


# %% ---------- 对齐校验:Qdlin 的峰和 summary 容量是否对应同一圈 ----------
# max(Qdlin@c100) 应 ≈ QDischarge 第 100 圈;若系统性偏差则说明差 1 对齐错了
errs = []
for _, r in df[df["keep"]].iterrows():
    if r["_qd100"] is None or len(r["_q_dis"]) < 100:
        continue
    errs.append(abs(np.nanmax(r["_qd100"]) - r["_q_dis"][99]) / r["_q_dis"][99])
errs = np.array(errs)
print(f"\n对齐校验:max(Qdlin@cycle100) vs QDischarge[100] 的相对误差中位数 "
      f"= {np.median(errs):.2%}(应 <2%)")
print("若误差大,说明 1-based 对齐或 summary 偏移需要修正 —— 报错给我")


# %% ---------- 产出:总表 CSV + 寿命分布图 ----------
df_out = df[df["keep"]].drop(columns=["_q_dis", "_vdlin", "_qd10", "_qd100"])
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"\n电芯总表已保存: {OUT_CSV}({len(df_out)} 颗)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for bname, color in [("batch1", "tab:blue"), ("batch2", "tab:orange"),
                     ("batch3", "tab:green")]:
    sub = df[(df["batch"] == bname) & df["keep"]]
    if len(sub) == 0:
        continue
    axes[0].hist(sub["cycle_life"], bins=20, alpha=0.55, color=color, label=bname)
    axes[1].scatter(sub["q_c100"], sub["cycle_life"], s=25, alpha=0.7,
                    color=color, label=bname)
axes[0].set_xlabel("循环寿命 cycle_life(到 80% 容量的圈数)")
axes[0].set_ylabel("电芯数量")
axes[0].set_title("124 颗电芯的寿命分布(按批次)")
axes[0].legend()
axes[1].set_xlabel("第 100 圈放电容量 (Ah)")
axes[1].set_ylabel("循环寿命")
axes[1].set_title("容量 vs 寿命:单特征的相关性")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PATHS.output_dir / "mit_dataset_overview.png", dpi=150)
print("分布图已保存: mit_dataset_overview.png")
plt.show()
