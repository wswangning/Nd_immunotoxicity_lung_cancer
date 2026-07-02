import matplotlib.pyplot as plt
import numpy as np

# ========== 数据（基于灰度分析1.xlsx整理，均值±SD，n=3）==========
groups = ['NC', '1.0 µM Nd', '2.5 µM Nd', '2.5 µM Nd + CX-4945']
proteins = ['E-Cadherin', 'N-Cadherin', 'Vimentin', 'SNAIL', 'SLUG', 'ZO-1']

# 均值矩阵（行: groups, 列: proteins）
means = np.array([
    [0.824, 1.362, 1.738, 0.610, 0.988, 0.669],   # NC
    [1.171, 0.987, 1.728, 0.607, 0.592, 0.599],   # 1.0 µM Nd
    [0.842, 1.348, 1.857, 0.714, 0.919, 0.784],   # 2.5 µM Nd
    [0.892, 0.921, 0.902, 0.503, 0.775, 0.853]    # 2.5 µM Nd + CX-4945
])

# 标准差矩阵（需替换为实际计算值，此处根据原始数据估算）
stds = np.array([
    [0.012, 0.023, 0.012, 0.011, 0.023, 0.013],
    [0.001, 0.012, 0.006, 0.001, 0.012, 0.006],
    [0.005, 0.014, 0.006, 0.004, 0.013, 0.006],
    [0.011, 0.003, 0.018, 0.010, 0.003, 0.018]
])

# 图形设置
x = np.arange(len(groups))
width = 0.12  # 每个蛋白柱的宽度
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 绘制分组柱状图
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, p in enumerate(proteins):
    ax.bar(x + i*width, means[:, i], width, yerr=stds[:, i], capsize=3,
           label=p, color=colors[i], edgecolor='black', linewidth=0.5)

# 坐标轴与标签
ax.set_xticks(x + width*(len(proteins)-1)/2)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel('Relative protein expression (fold of control)', fontsize=12)
ax.set_title('EMT markers in A549 cells treated with conditioned media', fontsize=14)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('EMT_results.tiff', dpi=300, bbox_inches='tight', format='tiff')
plt.show()