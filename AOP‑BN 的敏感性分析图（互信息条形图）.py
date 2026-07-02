import matplotlib.pyplot as plt
import numpy as np

# 节点名称（按互信息从大到小排序）
nodes = [
    "CK2 activation (MIE)",
    "NLRP3 deubiquitination (KE1)",
    "M2 polarization (KE2)",
    "IL-1β secretion (KE3)",
    "Nd concentration"
]

# 互信息值（bits），单位为 bits，总和归一化后 CK2+NLRP3=78%
# 这些值为示例，请用 Netica 实际输出的互信息值替换
mutual_info = [0.42, 0.36, 0.12, 0.06, 0.04]  # 总和 = 1.00 bits
# 解释：CK2 贡献 42%，NLRP3 贡献 36%，合计 78%

# 计算贡献百分比
percentages = [mi / sum(mutual_info) * 100 for mi in mutual_info]

# 绘图
plt.figure(figsize=(8, 5))
bars = plt.barh(nodes, mutual_info, color='steelblue', edgecolor='black')
plt.xlabel('Mutual information (bits)', fontsize=12)
plt.title('Figure 8C. Sensitivity analysis of the Bayesian network\n'
          'Contribution of each node to uncertainty in the adverse outcome (AO)',
          fontsize=11)
plt.gca().invert_yaxis()  # 使最重要的节点位于顶部

# 在条末端添加百分比标签
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{pct:.0f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_8C_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()