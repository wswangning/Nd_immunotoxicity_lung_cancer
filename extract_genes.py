import pandas as pd

# 读取 Excel 文件（请确保文件路径正确）
file_path = "Case1_vs_Ctrl1.log2FC0.585.FDR0.05.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# 筛选条件：|log2FC| >= 0.585 且 FDR < 0.05
filtered = df[(abs(df["log2FC"]) >= 0.585) & (df["FDR"] < 0.05)]

# 选择所需三列
result = filtered[["GeneSymbol", "log2FC", "FDR"]].copy()

# 按照 log2FC 绝对值降序排序（可选）
result["abs_log2FC"] = result["log2FC"].abs()
result = result.sort_values("abs_log2FC", ascending=False).drop(columns="abs_log2FC")

# 保存为制表符分隔的文本文件（无索引）
result.to_csv("gene_list.txt", sep="\t", index=False, float_format="%.6e")

print(f"成功提取 {len(result)} 个差异基因，已保存至 gene_list.txt")