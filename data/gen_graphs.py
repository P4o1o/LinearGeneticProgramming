import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

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

# Funzione di formattazione per numeri grandi
def format_value(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'
    else:
        return f'{x:.1f}'

# Load the data
lgp_data = pd.read_csv(target + "x100.csv")
deap_data = pd.read_csv("deap_" + target + "x100.csv")

# Rapporto tempo/valutazione
ratio_lgp = lgp_data['exec_time'] / lgp_data['evaluations']
ratio_DEAP = deap_data['exec_time'] / (deap_data['select_args'] + (deap_data['lambda'] * deap_data['gen']))

# DataFrame per il boxplot
data = pd.DataFrame({
    'Ratio': pd.concat([ratio_lgp, ratio_DEAP], ignore_index=True),
    'Program': ['LGP'] * len(ratio_lgp) + ['DEAP'] * len(ratio_DEAP)
})

# Grafico 1: Tempo per valutazione (boxplot)
plt.figure(figsize=(3.5, 2.8))  # Ridotto per formato a colonna singola
sns.boxplot(x='Program', y='Ratio', data=data)
plt.yscale('log')
plt.ylabel('Tempo per valutazione (s)')
plt.title("Efficienza computazionale", fontsize=9)

# Aggiunta annotazioni con valori medi
for i, program in enumerate(['LGP', 'DEAP']):
    mean_val = data[data['Program'] == program]['Ratio'].mean()
    plt.annotate(f'{mean_val:.2e}s', 
                xy=(i, mean_val), 
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(target + "_tempo_per_valutazione.pdf", dpi=300, bbox_inches='tight')
plt.savefig(target + "_tempo_per_valutazione.png", dpi=300, bbox_inches='tight')
plt.close()

# Valutazioni al secondo
ratio_lgp = lgp_data['evaluations'] / lgp_data['exec_time']
ratio_DEAP = (deap_data['select_args'] + (deap_data['lambda'] * deap_data['gen'])) / deap_data['exec_time']

data = pd.DataFrame({
    'Ratio': pd.concat([ratio_lgp, ratio_DEAP], ignore_index=True),
    'Program': ['LGP'] * len(ratio_lgp) + ['DEAP'] * len(ratio_DEAP)
})

# Grafico 2: Valutazioni al secondo (barplot)
plt.figure(figsize=(3.5, 2.8))
means = [data[data['Program'] == p]['Ratio'].mean() for p in ['LGP', 'DEAP']]
bars = plt.bar(['LGP', 'DEAP'], means, color=['blue', 'red'])

# Formattazione delle etichette
formatter = FuncFormatter(format_value)
plt.gca().yaxis.set_major_formatter(formatter)

plt.ylabel('Valutazioni al secondo')
plt.title("Valutazioni/secondo", fontsize=9)

# Aggiunta valori sopra le barre
for bar in bars:
    height = bar.get_height()
    formatted = format_value(height, 0)
    plt.annotate(formatted,
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(target + "_valutazioni_al_secondo.pdf", dpi=300, bbox_inches='tight')
plt.savefig(target + "_valutazioni_al_secondo.png", dpi=300, bbox_inches='tight')
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
plt.savefig(target + "_performance_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(target + "_performance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()



plt.figure(figsize=(3.5, 2.8))
sns.scatterplot(
    x='gen',
    y='mse',
    data=lgp_data,
    label='LGP',
    alpha=0.7,
    s=30  # Ridotto da 80 a 40
)
sns.scatterplot(
    x='gen',
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
plt.savefig(target + "_genxmse.pdf", dpi=300, bbox_inches='tight')
plt.savefig(target + "_genxmse.png", dpi=300, bbox_inches='tight')
plt.close()