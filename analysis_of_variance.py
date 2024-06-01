import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# テキストファイルを読み込む。先頭行をヘッダーとして認識し、utf-8でエンコードを指定
data = pd.read_csv('Analgesics.txt', delimiter=',', encoding='utf-8')

# CSVファイルとして保存
data.to_csv('Analgesics.csv', index=False)

# CSVファイルを読み込む
data = pd.read_csv('Analgesics.csv')

# 一元配置分散分析
model_one_way = ols('pain ~ C(drug)', data=data).fit()
anova_result_one_way = sm.stats.anova_lm(model_one_way, typ=2)
print(anova_result_one_way)

print("")
print("")

# 二元配置分散分析
model_two_way = ols('pain ~ C(drug) + C(Gender) + C(drug):C(Gender)', data=data).fit()
anova_result_two_way = sm.stats.anova_lm(model_two_way, typ=2)
print(anova_result_two_way)

# 各列を異なる精度で四捨五入
anova_result_one_way['df'] = anova_result_one_way['df'].astype(int)
anova_result_one_way['sum_sq'] = anova_result_one_way['sum_sq'].round(3)
anova_result_one_way['F'] = anova_result_one_way['F'].round(3)
anova_result_one_way['PR(>F)'] = anova_result_one_way['PR(>F)'].round(5)

anova_result_two_way['df'] = anova_result_two_way['df'].astype(int)
anova_result_two_way['sum_sq'] = anova_result_two_way['sum_sq'].round(3)
anova_result_two_way['F'] = anova_result_two_way['F'].round(3)
anova_result_two_way['PR(>F)'] = anova_result_two_way['PR(>F)'].round(5)

# 一元配置分散分析表をプロットし、JPEGファイルとして保存
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table_data_one_way = plt.table(cellText=anova_result_one_way.values, colLabels=anova_result_one_way.columns, rowLabels=anova_result_one_way.index, loc='center', cellLoc='center')
plt.savefig('anova_table_one_way.jpg', format='jpg', bbox_inches='tight')
plt.close()

# 二元配置分散分析表をプロットし、JPEGファイルとして保存
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table_data_two_way = plt.table(cellText=anova_result_two_way.values, colLabels=anova_result_two_way.columns, rowLabels=anova_result_two_way.index, loc='center', cellLoc='center')
plt.savefig('anova_table_two_way.jpg', format='jpg', bbox_inches='tight')
plt.close()
