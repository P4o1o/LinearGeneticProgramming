import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8
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

shopping = pd.read_csv("shopping-4x100.csv")
vector = pd.read_csv("vector-6x100.csv")
balls = pd.read_csv("balls-3x100.csv")
snow = pd.read_csv("snow-4x100.csv")
dice = pd.read_csv("dice-2x100.csv")

def add_label(df, label):
    df["label"] = label
    return df

data = pd.concat([
    add_label(shopping, "Shopping List"),
    add_label(vector, "Vector Distance"),
    add_label(dice, "Dice Game"),
    add_label(snow, "Snow Day"),
    add_label(balls, "Bounching Balls"),
])

data["mse_ratio"] = (data["evaluations"] / data["exec_time"])

# Creazione dello scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="mse_ratio", y="mse", hue="label", palette="Set1")
plt.yscale('log')
plt.xscale('log')
plt.title("Prestazioni su PSB2")
plt.xlabel("Valutazioni al secondo")
plt.ylabel("MSE")
plt.legend(title="Problema")
plt.tight_layout()
plt.savefig("ProblemsMSExEvauationssec.png", dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot MSE vs Evaluations
plt.figure(figsize=(6, 5))
sns.scatterplot(data=data, x="evaluations", y="mse", hue="label", palette="Set1", s=60)
plt.yscale('log')
plt.title("Prestazioni su PSB2")
plt.xlabel("Valutazioni")
plt.ylabel("MSE")
plt.legend(title="Problema", title_fontsize=14, fontsize=12)
plt.tight_layout()
plt.savefig("ProblemsMSExEvauations.png", dpi=300)
plt.close()

# Scatter plot MSE vs Execution Time
plt.figure(figsize=(6, 5))
sns.scatterplot(data=data, x="exec_time", y="mse", hue="label", palette="Set1")
plt.yscale('log')
plt.title("Prestazioni su PSB2")
plt.xlabel("Tempo di esecuzione")
plt.ylabel("MSE")
plt.legend(title="Problema")
plt.tight_layout()
plt.savefig("ProblemsMSExTime.png", dpi=300, bbox_inches='tight')
plt.close()
