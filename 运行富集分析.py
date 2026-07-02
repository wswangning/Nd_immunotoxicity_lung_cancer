import pandas as pd
from gseapy import prerank

# 读取原始数据（包含所有基因的 log2FC 和 FDR）
df = pd.read_excel("Case1_vs_Ctrl1.log2FC0.585.FDR0.05.xlsx", sheet_name=0)

# 创建排名文件：gene_symbol 和 log2FC（如果 log2FC 缺失则填充 0）
ranking = df[["GeneSymbol", "log2FC"]].dropna()
ranking = ranking.set_index("GeneSymbol")
ranking = ranking[~ranking.index.duplicated(keep='first')]  # 去重
# 保存为 .rnk 文件（可选）
ranking.to_csv("gene_rank.rnk", sep="\t", header=False, float_format="%.6f")

# 运行 prerank 分析（不需要 cls）
prerank_res = prerank(data=ranking,
                      gene_sets='KEGG_2021_Human',
                      outdir='prerank_results',
                      permutation_num=1000,
                      seed=42)
# 查看结果
print(prerank_res.results.head(10))