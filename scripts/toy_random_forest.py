# -*- coding: utf-8 -*-
"""
随机森林玩具模型 —— 亲手造一棵决策树,看穿它的工作原理
========================================================
12 颗玩具电池,1 个特征(log_var),1 个标签(寿命)。
不用 sklearn(sklearn 是黑盒,手写的才看得见齿轮)。

三个格子:
  [1] 手写决策树:训练(找最佳切分)→ 打印树 → 预测
  [2] 地板效应演示:预测一颗"比训练集所有电池都健康"的电池
  [3] 随机森林:5 棵 bootstrap 树取平均,对比单棵树
跑完后做文件末尾的 3 个练习。
"""

# %% ---------- [1] 数据 + 手写决策树 ----------
import numpy as np
import matplotlib.pyplot as plt
from project_config import PATHS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 12 颗玩具电池(对应真实项目:log_var 越小 = 指纹越稳 = 越长寿)
x = np.array([-4.2, -3.9, -3.6, -3.4, -3.1, -2.9, -2.7, -2.5, -2.2, -2.0, -1.8, -1.5])
y = np.array([1800, 1600, 1400, 1250, 1100, 1000, 900, 800, 650, 550, 450, 380], float)


def best_split(x, y):
    """挨个试所有切分点,返回让'两边各自内部方差之和'最小的那个。这就是'训练'的全部。"""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    best_thr, best_cost = None, np.inf
    for i in range(1, len(xs)):
        thr = (xs[i - 1] + xs[i]) / 2
        cost = (np.var(ys[:i]) * i + np.var(ys[i:]) * (len(ys) - i))
        if cost < best_cost:
            best_thr, best_cost = thr, cost
    return best_thr


def build_tree(x, y, depth=2, min_leaf=2):
    """递归建树。depth 是还能切几刀;min_leaf 防止叶子里只剩一两颗(背诵)。"""
    if depth == 0 or len(y) < 2 * min_leaf:
        return {"leaf": True, "pred": y.mean(), "n": len(y), "members": y.copy()}
    thr = best_split(x, y)
    left = x <= thr
    return {"leaf": False, "thr": thr,
            "L": build_tree(x[left], y[left], depth - 1, min_leaf),
            "R": build_tree(x[~left], y[~left], depth - 1, min_leaf)}


def show_tree(t, indent=""):
    if t["leaf"]:
        print(f"{indent}叶子: 预测 {t['pred']:.0f} 圈(装着 {t['n']} 颗: {t['members'].astype(int)})")
        return
    print(f"{indent}问题: log_var ≤ {t['thr']:.2f} ?")
    print(f"{indent}├─ 是 →")
    show_tree(t["L"], indent + "│    ")
    print(f"{indent}└─ 否 →")
    show_tree(t["R"], indent + "     ")


def predict_tree(t, x0):
    """预测 = 拿新电池走一遍问题链,输出叶子里的平均值。"""
    while not t["leaf"]:
        t = t["L"] if x0 <= t["thr"] else t["R"]
    return t["pred"]


print("===== 训练好的决策树(depth=2) =====")
tree = build_tree(x, y, depth=2)
show_tree(tree)
print("\n注意每个叶子的预测值 = 叶子里训练样本的平均 → 台阶线,不是光滑曲线")


# %% ---------- [2] 地板效应:预测一颗'更健康'的电池 ----------
print("===== 预测 3 颗测试电池 =====")
x_test = np.array([-5.0, -2.6, -0.9])
y_true = np.array([2850, 880, 270])   # 按真实规律外推的"标准答案"

print(f"{'log_var':>8} {'真实寿命':>8} {'树的预测':>10} {'误差%':>8}")
for x0, yt in zip(x_test, y_true):
    p = predict_tree(tree, x0)
    print(f"{x0:>8} {yt:>8.0f} {p:>10.0f} {abs(p-yt)/yt*100:>7.1f}%")

# 直线模型(log 空间拟合,弹性网络的灵魂)对同 3 颗电池的预测
coef = np.polyfit(x, np.log10(y), 1)
linear_pred = 10 ** np.polyval(coef, x_test)
print(f"\n直线模型的预测: {np.round(linear_pred).astype(int)}   真实: {y_true.astype(int)}")
print("对比:射程内两者都准;射程外(x=-5.0 和 -0.9)树被地板焊死,直线能外推。")
print("这是合成教学示意，不代表当前官方测试集上的模型指标。")

print("""
看第 1 行:x=-5.0 的电池比训练集里所有电池都健康(训练集最小 log_var=-4.2),
但树只能把它送进最左边的叶子 → 输出叶子平均。
叶子平均不可能超过叶子里最大的标签 → 这就是 4.3a 的'地板效应'。
原理性焊死,调参救不了 —— 换任何随机森林超参数都一样。
""")


# %% ---------- [3] 随机森林:5 棵 bootstrap 树取平均 ----------
print("===== 随机森林:5 棵树,每棵只随机抽 8 颗电池训练 =====")
rng = np.random.default_rng(42)
test_preds = []
for t_i in range(5):
    idx = rng.choice(len(x), size=8, replace=True)   # 有放回抽样 = bootstrap
    t = build_tree(x[idx], y[idx], depth=2)
    preds = [predict_tree(t, x0) for x0 in x_test]
    test_preds.append(preds)
    print(f"树{t_i}(抽到 {np.sort(idx)}): 预测 {np.round(preds).astype(int)}")

forest_pred = np.mean(test_preds, axis=0)
print(f"\n森林平均:  {np.round(forest_pred).astype(int)}")
print(f"真实寿命:  {y_true.astype(int)}")
print("""
对比单棵树和森林:每棵树各自切歪一点,但错误方向不同,
平均后互相抵消 → 森林通常比单棵树稳。这就是'集成'的全部秘密。
(但注意:平均仍然救不了地板 —— 所有树的叶子平均都够不到 2850。)
""")


# %% ---------- [4] 画图:台阶 vs 直线 ----------
grid = np.linspace(-5.2, -1.2, 200)
tree_line = [predict_tree(tree, g) for g in grid]
coef = np.polyfit(x, np.log10(y), 1)              # log 空间直线(弹性网络的灵魂)
linear_line = 10 ** np.polyval(coef, grid)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, s=60, color="black", zorder=3, label="12 颗训练电池")
ax.scatter(x_test, y_true, s=60, marker="^", color="red", zorder=3, label="3 颗测试电池(真实)")
ax.plot(grid, tree_line, lw=2, color="steelblue", label="决策树预测(台阶)")
forest_line = []
for t_i in range(5):
    idx = rng.choice(len(x), size=8, replace=True)
    forest_line.append([predict_tree(build_tree(x[idx], y[idx], 2), g) for g in grid])
ax.plot(grid, np.mean(forest_line, axis=0), lw=2, color="orange",
        label="随机森林(5 棵平均,仍偏台阶)")
ax.plot(grid, linear_line, lw=2, color="green", ls="--",
        label="log 空间直线(弹性网络的灵魂,能外推)")
ax.axvspan(-5.2, min(x), alpha=0.08, color="red")
ax.text(-5.05, 600, "训练射程之外\n(树被地板焊死\n直线能外推)", fontsize=9, color="darkred")
ax.set_xlabel("log_var(ΔQ 方差,越小越健康)")
ax.set_ylabel("寿命(圈)")
ax.set_title("决策树是台阶,森林是碎台阶,直线能外推 —— 4.3 全部谜底的齿轮级解释")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
PATHS.output_dir.mkdir(parents=True, exist_ok=True)
toy_path = PATHS.output_dir / "toy_forest.png"
fig.savefig(toy_path, dpi=150, bbox_inches="tight")
print(f"图已保存: {toy_path}")
plt.show()

# ============================================================
# 练习(改参数 → 预测 → 运行):
# 1. build_tree 的 depth 改成 1 和 3:台阶变粗还是变细?depth=3 时
#    叶子更贴训练数据了吗?这和"过拟合"是什么关系?
# 2. 把 x_test 加一个 -6.0:树、森林、直线各自预测多少?谁最接近真实?
# 3. 把 rng 的种子 42 改成 0 / 1 / 2:森林的预测变不变?单棵树呢?
#    这解释了为什么项目里森林要 500 棵而不是 5 棵。
# ============================================================
