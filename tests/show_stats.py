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

# Load the data
lgp_data = pd.read_csv(target + "x100.csv")
deap_data = pd.read_csv("deap_" + target + "x100.csv")

ratio_lgp = lgp_data['exec_time'] / lgp_data['evaluations']
ratio_DEAP = deap_data['exec_time'] / (deap_data['select_args'] + (deap_data['lambda'] * deap_data['gen']))

# Crea un DataFrame che contiene tutti i valori con una colonna per il dataset
data = pd.DataFrame({
    'Ratio': pd.concat([ratio_lgp, ratio_DEAP], ignore_index=True),
    'Program': ['lgp'] * len(ratio_lgp) + ['DEAP'] * len(ratio_DEAP)
})

plt.figure(figsize=(8, 6))
# Crea il boxplot per ciascun dataset, mostrando anche la media

plt.figure(figsize=(8,6))
sns.boxplot(x='Program', hue='Program', y='Ratio', data=data, palette=['blue', 'red'])
plt.title("Rapporto tempo di esecuzione / valutazioni")
plt.tight_layout()
plt.savefig(target + "barplot_executiontime.png")
plt.close()

# Load the data
lgp_data = pd.read_csv(target + "x100.csv")
deap_data = pd.read_csv("deap_" + target + "x100.csv")

ratio_lgp = lgp_data['evaluations'] / lgp_data['exec_time']
ratio_DEAP = (deap_data['select_args'] + (deap_data['lambda'] * deap_data['gen'])) / deap_data['exec_time']

# Crea un DataFrame che contiene tutti i valori con una colonna per il dataset
data = pd.DataFrame({
    'Ratio': pd.concat([ratio_lgp, ratio_DEAP], ignore_index=True),
    'Program': ['lgp'] * len(ratio_lgp) + ['DEAP'] * len(ratio_DEAP)
})

plt.figure(figsize=(3.5, 2.8))
sns.boxplot(x='Program', hue='Program', y= 'Ratio',data=data, legend=False)
plt.title("Valutazioni al secondo")
plt.tight_layout()
plt.savefig(target + "esecuzioni_al_sec.png")
plt.close()


# 1. Performance Comparison: Execution Time vs MSE
plt.figure(figsize=(10, 8))
# Plot LGP data
sns.scatterplot(
    x='exec_time',
    y='mse',
    data=lgp_data,
    label='LGP (C)',
    alpha=0.7,
    s=40
)
# Plot DEAP data
sns.scatterplot(
    x='exec_time',
    y='mse',
    data=deap_data,
    label='DEAP (Python)',
    alpha=0.7,
    marker='X',
    s=50
)
plt.yscale('log')
plt.xlabel('Tempo di Esecuzione (secondi)')
plt.ylabel('MSE (scala logaritmica)')
plt.title('Confronto delle Prestazioni: Tempo di Esecuzione vs Precisione')
plt.legend()
plt.tight_layout()
plt.savefig(target + "performance_comparison.png", dpi=300)
plt.close()

# 1. Performance Comparison: Limit
plt.figure(figsize=(10, 8))
# Plot DEAP data
sns.scatterplot(
    x='exec_time',
    y='mse',
    hue='limit',
    data=deap_data,
    alpha=0.7,
    marker='X',
    s=50
)
plt.yscale('log')
plt.xlabel('Tempo di Esecuzione (secondi)')
plt.ylabel('MSE (scala logaritmica)')
plt.title('Confronto delle Prestazioni: Tempo di Esecuzione vs Precisione')
plt.legend()
plt.tight_layout()
plt.savefig(target + "bloat_performance_comparison.png", dpi=300)
plt.close()

# 6. Ratio Analysis: Time per Evaluation
# This shows computational efficiency accounting for problem complexity
lgp_ratio = lgp_data['exec_time'] / lgp_data['evaluations']
deap_ratio = deap_data['exec_time'] / (deap_data['lambda'] * deap_data['gen'])

# Create a DataFrame for the ratio data
ratio_data = pd.DataFrame({
    'Ratio': pd.concat([lgp_ratio, deap_ratio], ignore_index=True),
    'Implementation': ['LGP (C)'] * len(lgp_ratio) + ['DEAP (Python)'] * len(deap_ratio)
})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Implementation', y='Ratio', data=ratio_data)
plt.title('Tempo di Esecuzione per Valutazione')
plt.ylabel('Secondi per Valutazione (scala logaritmica)')
plt.yscale('log')
plt.tight_layout()
plt.savefig(target + "time_per_evaluation.png", dpi=300)
plt.close()