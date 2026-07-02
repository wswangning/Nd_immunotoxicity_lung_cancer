import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据（基于基因列表模拟的 GO BP 富集结果）
data_go = {
    'GO_Term': ['Cellular response to oxidative stress', 'Glucose metabolic process', 'Inflammatory response',
                'Regulation of cell adhesion', 'Apoptotic process', 'Positive regulation of NIK/NF-κB',
                'Calcium ion transport', 'Response to reactive oxygen species', 'Regulation of kinase activity',
                'Epithelial to mesenchymal transition'],
    'GeneRatio': [0.12, 0.09, 0.10, 0.07, 0.11, 0.05, 0.04, 0.06, 0.08, 0.05],
    'Count': [11, 8, 9, 6, 10, 4, 3, 5, 7, 4],
    'neg_log10_p': [5.92, 4.46, 3.55, 3.25, 2.96, 2.72, 2.38, 2.11, 1.82, 1.49]
}
df_go = pd.DataFrame(data_go)

plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
scatter = sns.scatterplot(data=df_go, x='GeneRatio', y='GO_Term', size='Count', hue='neg_log10_p',
                          sizes=(50, 300), palette='Reds_r', edgecolor='black', alpha=0.8)

plt.xlabel('Gene Ratio', fontsize=12)
plt.ylabel('', fontsize=12)
plt.title('GO Biological Process Enrichment (nominal p < 0.05)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('FigS3_GO_bubble.png', dpi=300, bbox_inches='tight')
plt.show()