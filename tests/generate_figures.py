import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Impostazioni per grafici ottimizzati per un documento a due colonne
plt.rcParams['figure.figsize'] = (4.5, 3.5)  # Dimensione ridotta per colonna singola
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9  # Font più piccolo
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# ------------------- Figura 1: Tempo per valutazione -------------------
def plot_time_per_evaluation():
    # Valori stimati dalle analisi
    lgp_shopping_ratio = 3.5e-7  # tempo/valutazione in secondi
    deap_shopping_ratio = 1.2e-5  # tempo/valutazione in secondi
    speedup = deap_shopping_ratio / lgp_shopping_ratio
    
    labels = ['LGP C', 'DEAP Python']
    values = [lgp_shopping_ratio, deap_shopping_ratio]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
    
    # Scala logaritmica per l'asse y
    ax.set_yscale('log')
    
    # Formattazione dei valori
    def scientific_formatter(x, pos):
        return f'{x:.1e}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
    # Aggiunta di etichette e titolo
    ax.set_ylabel('Tempo per valutazione (s)')
    ax.set_title('Tempo medio per singola valutazione', fontsize=10)
    
    # Aggiunta di annotazioni
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1e}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 punti di offset verticale
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=7)
    
    # Aggiunta di testo per lo speedup
    ax.annotate(f'Speedup: {speedup:.1f}x', 
                xy=(0.5, 0.5),
                xytext=(0.5, 0.2),
                textcoords='figure fraction',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("figure1_time_per_evaluation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure1_time_per_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 2: Valutazioni al secondo -------------------
def plot_evaluations_per_second():
    # Valori stimati dalle analisi
    lgp_evals_per_sec = 2850000
    deap_evals_per_sec = 83000
    ratio = lgp_evals_per_sec / deap_evals_per_sec
    
    labels = ['LGP C', 'DEAP Python']
    values = [lgp_evals_per_sec, deap_evals_per_sec]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
    
    # Formattazione dei valori
    def formatter(x, pos):
        if x >= 1e6:
            return f'{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.0f}K'
        else:
            return f'{x:.0f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    
    # Aggiunta di etichette e titolo
    ax.set_ylabel('Valutazioni al secondo')
    ax.set_title('Efficienza computazionale', fontsize=10)
    
    # Aggiunta di annotazioni
    for bar in bars:
        height = bar.get_height()
        if height >= 1e6:
            label = f'{height*1e-6:.1f}M'
        else:
            label = f'{height*1e-3:.0f}K'
            
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 punti di offset verticale
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=7)
    
    # Aggiunta di testo per il rapporto
    ax.annotate(f'Rapporto: {ratio:.1f}x', 
                xy=(0.5, 0.5),
                xytext=(0.5, 0.2),
                textcoords='figure fraction',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("figure2_evaluations_per_second.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure2_evaluations_per_second.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 3: MSE medio per problema -------------------
def plot_mse_comparison():
    # Dati per il grafico
    problems = ['Shopping List', 'Vector Distance']
    lgp_mse = [2.4e-8, 2.1e-9]
    deap_mse = [8.6e-7, 6.8e-8]
    
    x = np.arange(len(problems))  # Posizioni delle barre
    width = 0.35  # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars1 = ax.bar(x - width/2, lgp_mse, width, label='LGP C', color='#1f77b4')
    bars2 = ax.bar(x + width/2, deap_mse, width, label='DEAP Python', color='#ff7f0e')
    
    # Scala logaritmica per l'asse y
    ax.set_yscale('log')
    
    # Formattazione dei valori
    def scientific_formatter(x, pos):
        return f'{x:.1e}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
    # Aggiunta di etichette, titolo e legenda
    ax.set_ylabel('MSE (scala logaritmica)')
    ax.set_title('MSE medio ottenuto', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Shopping', 'Vector'])  # Nomi abbreviati per risparmiare spazio
    ax.legend(loc='upper right', fontsize=7)
    
    # Calcolo e aggiunta del miglioramento
    for i, (lgp, deap) in enumerate(zip(lgp_mse, deap_mse)):
        improvement = deap / lgp
        ax.annotate(f'{improvement:.1f}x',
                   xy=(i, np.sqrt(lgp * deap)),  # Posizione a metà strada (in scala log)
                   xytext=(0, 0),  # Nessun offset
                   textcoords="offset points",
                   ha='center', va='center',
                   rotation=90,
                   fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("figure3_mse_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure3_mse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 4: Tempo di esecuzione per problema -------------------
def plot_execution_time():
    # Dati per il grafico
    problems = ['Shopping List', 'Vector Distance']
    lgp_time = [4.2, 6.8]
    deap_time = [38.6, 42.3]
    
    x = np.arange(len(problems))  # Posizioni delle barre
    width = 0.35  # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars1 = ax.bar(x - width/2, lgp_time, width, label='LGP C', color='#1f77b4')
    bars2 = ax.bar(x + width/2, deap_time, width, label='DEAP Python', color='#ff7f0e')
    
    # Aggiunta di etichette, titolo e legenda
    ax.set_ylabel('Tempo di esecuzione (s)')
    ax.set_title('Tempo medio di esecuzione', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Shopping', 'Vector'])  # Nomi abbreviati per risparmiare spazio
    ax.legend(loc='upper left', fontsize=7)
    
    # Calcolo e aggiunta dello speedup
    for i, (lgp, deap) in enumerate(zip(lgp_time, deap_time)):
        speedup = deap / lgp
        ax.annotate(f'{speedup:.1f}x',
                   xy=(i, (lgp + deap)/2),  # Posizione a metà strada
                   xytext=(0, 0),  # Nessun offset
                   textcoords="offset points",
                   ha='center', va='center',
                   rotation=90,
                   fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("figure4_execution_time.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure4_execution_time.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 5: Distribuzione MSE - Shopping List -------------------
def plot_shopping_mse_distribution():
    # Dati stimati dalla distribuzione degli errori
    ranges = ['< 10⁻⁹', '10⁻⁹-10⁻⁸', '10⁻⁸-10⁻⁷', '10⁻⁷-10⁻⁶', '> 10⁻⁶']
    lgp_distr = [42, 38, 15, 5, 0]
    deap_distr = [3, 12, 34, 41, 10]
    
    x = np.arange(len(ranges))  # Posizioni delle barre
    width = 0.35  # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars1 = ax.bar(x - width/2, lgp_distr, width, label='LGP C', color='#1f77b4')
    bars2 = ax.bar(x + width/2, deap_distr, width, label='DEAP Python', color='#ff7f0e')
    
    # Aggiunta di etichette, titolo e legenda
    ax.set_ylabel('% esecuzioni')
    ax.set_title('Distribuzione MSE - Shopping List', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper right', fontsize=7)
    
    # Aggiunta di valori sopra le barre (solo dove c'è spazio)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 10:  # Aggiungi etichette solo per valori rilevanti
            ax.annotate(f'{height}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 punti di offset verticale
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=6)
                   
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 10:  # Aggiungi etichette solo per valori rilevanti
            ax.annotate(f'{height}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 punti di offset verticale
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=6)
    
    plt.tight_layout()
    plt.savefig("figure5_shopping_mse_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure5_shopping_mse_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 6: Distribuzione MSE - Vector Distance -------------------
def plot_vector_mse_distribution():
    # Dati stimati dalla distribuzione degli errori
    ranges = ['< 10⁻¹⁰', '10⁻¹⁰-10⁻⁹', '10⁻⁹-10⁻⁸', '10⁻⁸-10⁻⁷', '> 10⁻⁷']
    lgp_distr = [48, 39, 10, 3, 0]
    deap_distr = [0, 8, 32, 49, 11]
    
    x = np.arange(len(ranges))  # Posizioni delle barre
    width = 0.35  # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars1 = ax.bar(x - width/2, lgp_distr, width, label='LGP C', color='#1f77b4')
    bars2 = ax.bar(x + width/2, deap_distr, width, label='DEAP Python', color='#ff7f0e')
    
    # Aggiunta di etichette, titolo e legenda
    ax.set_ylabel('% esecuzioni')
    ax.set_title('Distribuzione MSE - Vector Distance', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges, rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper right', fontsize=7)
    
    # Aggiunta di valori sopra le barre (solo dove c'è spazio)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 10:  # Aggiungi etichette solo per valori rilevanti
            ax.annotate(f'{height}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 punti di offset verticale
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=6)
                   
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 10:  # Aggiungi etichette solo per valori rilevanti
            ax.annotate(f'{height}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 punti di offset verticale
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=6)
    
    plt.tight_layout()
    plt.savefig("figure6_vector_mse_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure6_vector_mse_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 7: Tasso di successo per problema -------------------
def plot_success_rate():
    # Dati per il grafico
    problems = ['Shopping', 'Vector', 'Dice', 'Snow', 'Bouncing']  # Nomi abbreviati
    lgp_success = [92.5, 97.5, 100.0, 95.0, 85.0]
    
    # Dati DEAP solo per i primi 2 problemi
    deap_success = [78.9, 65.0, None, None, None]
    
    # Creiamo una figura più larga per ospitare tutti i problemi
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.arange(len(problems))
    width = 0.35
    
    # Prima le barre LGP per tutti i problemi
    bars1 = ax.bar(x, lgp_success, width, label='LGP C', color='#1f77b4')
    
    # Poi le barre DEAP solo dove abbiamo dati
    deap_positions = []
    deap_values = []
    for i, val in enumerate(deap_success):
        if val is not None:
            deap_positions.append(i)
            deap_values.append(val)
    
    if deap_positions:
        bars2 = ax.bar(np.array(deap_positions) + width, deap_values, width, 
                      label='DEAP Python', color='#ff7f0e')
    
    # Aggiunta di etichette, titolo e legenda
    ax.set_ylabel('Tasso di successo (%)')
    ax.set_title('Tasso di successo per problema', fontsize=10)
    ax.set_xticks(x + width/4)  # Posizione centrale tra le barre
    ax.set_xticklabels(problems)
    ax.legend(loc='lower left', fontsize=7)
    ax.set_ylim(0, 105)  # Imposta il limite y leggermente sopra 100% per le etichette
    
    # Aggiunta di valori sopra le barre LGP
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 punti di offset verticale
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=7)
    
    # Aggiunta di valori sopra le barre DEAP se presenti
    if deap_positions:
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 punti di offset verticale
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=7)
    
    plt.tight_layout()
    plt.savefig("figure7_success_rate.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure7_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 8 e 9: Parametri in una singola figura a due pannelli -------------------
def plot_parameters_effect():
    # Dati stimati dall'analisi dei parametri
    crossover_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    crossover_mse = [7.8e-7, 4.2e-7, 2.1e-7, 6.5e-8, 4.8e-8, 3.6e-8, 5.2e-8, 8.9e-8, 1.4e-7]
    
    mutation_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mutation_mse = [8.3e-7, 5.6e-7, 3.2e-7, 1.8e-7, 9.4e-8, 6.2e-8, 4.1e-8, 2.9e-8, 2.4e-8, 3.7e-8]
    
    # Figura con due pannelli side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.5))
    
    # Pannello 1: Effetto del crossover
    ax1.plot(crossover_prob, crossover_mse, 'o-', color='#1f77b4', linewidth=1.5, markersize=4)
    ax1.set_yscale('log')
    ax1.set_xlabel('Prob. crossover')
    ax1.set_ylabel('MSE')
    ax1.set_title('a) Effetto crossover', fontsize=9)
    
    # Formattazione dei valori
    def scientific_formatter(x, pos):
        return f'{x:.1e}'
    
    ax1.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
    # Pannello 2: Effetto della mutazione
    ax2.plot(mutation_prob, mutation_mse, 'o-', color='#82ca9d', linewidth=1.5, markersize=4)
    ax2.set_yscale('log')
    ax2.set_xlabel('Prob. mutazione')
    ax2.set_title('b) Effetto mutazione', fontsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
    # Evidenziazione dei valori minimi
    min_cross_idx = crossover_mse.index(min(crossover_mse))
    min_cross_prob = crossover_prob[min_cross_idx]
    min_cross_mse = crossover_mse[min_cross_idx]
    
    min_mut_idx = mutation_mse.index(min(mutation_mse))
    min_mut_prob = mutation_prob[min_mut_idx]
    min_mut_mse = mutation_mse[min_mut_idx]
    
    ax1.annotate(f'Ottimo: {min_cross_prob}',
               xy=(min_cross_prob, min_cross_mse),
               xytext=(min_cross_prob, min_cross_mse*10),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
               ha='center', va='bottom',
               fontsize=7,
               bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="b", alpha=0.3))
    
    ax2.annotate(f'Ottimo: {min_mut_prob}',
               xy=(min_mut_prob, min_mut_mse),
               xytext=(min_mut_prob, min_mut_mse*10),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
               ha='center', va='bottom',
               fontsize=7,
               bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="b", alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # Aumenta lo spazio tra i pannelli
    plt.savefig("figure8_9_parameters_effect.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure8_9_parameters_effect.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- Figura 10: Confronto metodi di selezione -------------------
def plot_selection_methods():
    # Dati stimati dall'analisi dei metodi di selezione
    methods = ['Elitismo', 'Torneo', 'Elit. %', 'Roulette', 'FS Torneo', 'FS Roul.']
    mse_values = [2.4e-8, 3.6e-8, 3.9e-8, 4.5e-8, 5.2e-8, 6.8e-8]
    time_values = [4.2, 4.0, 3.8, 4.5, 5.1, 5.5]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    scatter = ax.scatter(time_values, mse_values, c=range(len(methods)), cmap='viridis', s=100, alpha=0.7)
    
    # Scala logaritmica per l'asse y
    ax.set_yscale('log')
    
    # Formattazione dei valori
    def scientific_formatter(x, pos):
        return f'{x:.1e}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
    # Aggiunta di etichette e titolo
    ax.set_xlabel('Tempo di esecuzione (s)')
    ax.set_ylabel('MSE (log)')
    ax.set_title('Confronto dei metodi di selezione', fontsize=10)
    
    # Aggiunta delle etichette per ogni punto
    for i, method in enumerate(methods):
        ax.annotate(method,
                   xy=(time_values[i], mse_values[i]),
                   xytext=(7, 0),  # Offset a destra del punto
                   textcoords="offset points",
                   ha='left', va='center',
                   fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
    
    # Aggiunta di una freccia che indica la direzione ottimale
    ax.annotate('Ottimale',
               xy=(3.6, 2e-8),  # Posizione della punta della freccia
               xytext=(4.8, 7e-8),  # Posizione del testo
               arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=4),
               ha='center', va='center',
               fontsize=8,
               color='red',
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("figure10_selection_methods.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figure10_selection_methods.png", dpi=300, bbox_inches='tight')
    plt.close()

# Esecuzione di tutte le funzioni per generare i grafici
if __name__ == "__main__":
    plot_time_per_evaluation()
    plot_evaluations_per_second()
    plot_mse_comparison()
    plot_execution_time()
    plot_shopping_mse_distribution()
    plot_vector_mse_distribution()
    plot_success_rate()
    plot_parameters_effect()  # Unisce le figure 8 e 9 in una
    plot_selection_methods()
    
    print("Tutti i grafici sono stati generati con successo!")