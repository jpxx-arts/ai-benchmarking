import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega o relatório final gerado pelo analysis.py
df_report = pd.read_csv('final_summary_report_with_scenarios.csv')

plt.figure(figsize=(15, 8))
sns.barplot(data=df_report, x='model_id', y='duration_seconds_overhead_%', hue='scenario')
plt.title('Overhead Percentual de Tempo do CC Mode por Custo Computacional', fontsize=16)
plt.ylabel('Overhead de Tempo (%)')
plt.xlabel('Modelo de Machine Learning')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
