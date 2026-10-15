# -*- coding: utf-8 -*-
"""
NASA 34 颗数据 —— 侦察脚本(阶段 5 第一步)
================================================================
目的:不动数据,先看清 extracted 文件夹的真实结构。
之前的 step2 脚本只处理过 4 颗教学装(B0005/6/7/18,.mat 格式),
34 颗全量集是另一套文件夹结构,不猜,先打印出来看。
把输出整屏截图发回,我按实物写解析器。
"""

# %% ---------- 侦察 ----------
import os
from project_config import PATHS

NASA_DIR = PATHS.nasa_dir

print(f"根目录: {NASA_DIR}")
print(f"是否存在: {os.path.isdir(NASA_DIR)}")
if not os.path.isdir(NASA_DIR):
    raise SystemExit("路径不对,改上面那行再跑")

entries = sorted(os.listdir(NASA_DIR))
print(f"\n根目录共 {len(entries)} 个条目,前 20 个:")
for e in entries[:20]:
    full = os.path.join(NASA_DIR, e)
    tag = "目录" if os.path.isdir(full) else f"文件({os.path.getsize(full)} 字节)"
    print(f"  {e}   ← {tag}")

# 找第一个子目录,再往下钻两层
sub1 = [e for e in entries if os.path.isdir(os.path.join(NASA_DIR, e))]
if sub1:
    d1 = os.path.join(NASA_DIR, sub1[0])
    print(f"\n钻进第一个子目录: {sub1[0]}")
    e1 = sorted(os.listdir(d1))
    print(f"共 {len(e1)} 个条目,前 20 个:")
    for e in e1[:20]:
        full = os.path.join(d1, e)
        tag = "目录" if os.path.isdir(full) else f"文件({os.path.getsize(full)} 字节)"
        print(f"  {e}   ← {tag}")

    sub2 = [e for e in e1 if os.path.isdir(os.path.join(d1, e))]
    if sub2:
        d2 = os.path.join(d1, sub2[0])
        print(f"\n再钻一层: {sub2[0]}")
        e2 = sorted(os.listdir(d2))
        print(f"共 {len(e2)} 个条目,前 10 个:")
        for e in e2[:10]:
            full = os.path.join(d2, e)
            tag = "目录" if os.path.isdir(full) else f"文件({os.path.getsize(full)} 字节)"
            print(f"  {e}   ← {tag}")

# 统计全库规模:多少个 .csv / .mat,分布在多少个文件夹里
n_csv = n_mat = 0
for root, _, files in os.walk(NASA_DIR):
    for fn in files:
        if fn.lower().endswith(".csv"):
            n_csv += 1
        elif fn.lower().endswith(".mat"):
            n_mat += 1
print(f"\n全库统计: .csv 共 {n_csv} 个, .mat 共 {n_mat} 个")

# 打印找到的第一个 csv 的前 5 行(看列名和数据格式)
first_csv = None
for root, _, files in os.walk(NASA_DIR):
    for fn in sorted(files):
        if fn.lower().endswith(".csv"):
            first_csv = os.path.join(root, fn)
            break
    if first_csv:
        break
if first_csv:
    print(f"\n第一个 csv: {first_csv.replace(NASA_DIR, '...')}")
    print("前 5 行内容:")
    with open(first_csv, "r", errors="ignore") as fh:
        for k in range(5):
            line = fh.readline().rstrip()
            print(f"  {line}")
