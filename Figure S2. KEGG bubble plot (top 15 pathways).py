import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Pathway': ['Galactose metabolism', 'HIF-1 signaling', 'Fructose/ mannose', 'Pentose/ glucuronate',
                'Glycolysis/ gluconeogenesis', 'Central carbon cancer', 'Cell adhesion molecules',
                'Steroid hormone', 'Retinol metabolism', 'ECM-receptor', 'Hypertrophic cardiomyopathy',
                'Arrhythmogenic RV cardiomyopathy', 'Salivary secretion', 'Dilated cardiomyopathy', 'Amoebiasis'],
    'GeneRatio': [0.0645, 0.03, 0.0606, 0.0588, 0.0294, 0.0286, 0.0207, 0.0328, 0.0299, 0.0244, 0.0235, 0.02597, 0.0215, 0.0208, 0.0196],
    'Count': [2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    'neg_log10_p': [2.125, 2.064, 2.073, 2.048, 1.479, 1.515, 1.634, 1.579, 1.491, 1.332, 1.303, 1.301, 1.234, 1.210, 1.164]
}
df = pd.DataFrame(data)
plt.figure(figsize=(10,7))
sns.scatterplot(data=df, x='GeneRatio', y='Pathway', size='Count', hue='neg_log10_p',
                sizes=(50, 300), palette='viridis_r', edgecolor='black', alpha=0.8)
plt.xlabel('Gene Ratio')
plt.title('KEGG pathways (nominal p < 0.05)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('FigS2_KEGG_bubble.png', dpi=300)
plt.show()