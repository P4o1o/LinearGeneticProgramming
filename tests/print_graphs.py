import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Impostazioni ottimizzate per grafici in formato a due colonne
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8  # Ridotto da 12 a 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7

target = "shopping-4"

lgp_data = pd.read_csv(target + "x100.csv")
deap_data = pd.read_csv("deap_" + target + "x100.csv")

ratio_lgp = lgp_data['evaluations'] / lgp_data['exec_time']
ratio_DEAP = (deap_data['select_args'] + (deap_data['lambda'] * deap_data['gen'])) / deap_data['exec_time']

data = pd.DataFrame({
    'Ratio': pd.concat([ratio_lgp, ratio_DEAP], ignore_index=True),
    'Programma': ['lgp'] * len(ratio_lgp) + ['DEAP'] * len(ratio_DEAP)
})

plt.figure(figsize=(3.5, 2.8))
sns.boxplot(x='Programma', hue='Programma', y= 'Ratio',data=data, legend=False)
plt.title("Valutazioni al secondo")
plt.tight_layout()
plt.savefig(target + "esecuzioni_al_sec.png")
plt.close()

plt.figure(figsize=(3.5, 2.8))
sns.scatterplot(
    x='exec_time',
    y='mse',
    data=lgp_data,
    label='LGP',
    alpha=0.7,
    s=30  # Ridotto da 80 a 40
)
sns.scatterplot(
    x='exec_time',
    y='mse',
    data=deap_data,
    label='DEAP',
    alpha=0.7,
    marker='X',
    s=35  # Ridotto da 100 a 50
)

plt.yscale('log')
plt.xlabel('Tempo (s)')
plt.ylabel('MSE')
plt.title('Confronto prestazioni', fontsize=9)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(target + "_performance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()