import pandas as pd
import numpy as np

def analyze_dataset(file_path, implementation, problem, is_deap=False):
    """Analizza un file CSV e calcola le statistiche richieste."""
    try:
        # Caricamento del file CSV
        df = pd.read_csv(file_path)
        
        # Calcolo delle statistiche di base
        mse_mean = df['mse'].mean()
        exec_time_mean = df['exec_time'].mean()
        
        # Calcolo delle valutazioni per DEAP vs LGP (diverso per i due formati)
        if is_deap:
            if 'lambda' in df.columns:
                df['evaluations'] = df['select_args'] + (df['lambda'] * df['gen'])
            else:
                print(f"AVVISO: Formato CSV DEAP non standard per {file_path}")
                df['evaluations'] = 0
        
        # Calcolo valutazioni al secondo
        evals_per_second = df['evaluations'].mean() / exec_time_mean if exec_time_mean > 0 else 0
        
        # Calcolo probabilità di successo (MSE <= 1e-7)
        success_rate = (df['mse'] <= 1e-7).mean() * 100
        
        # Tempo medio per trovare soluzione con MSE <= 1e-7
        successful_runs = df[df['mse'] <= 1e-7]
        time_to_solution = successful_runs['exec_time'].mean() if not successful_runs.empty else "N/A"
        
        # Formattazione dei risultati
        if evals_per_second >= 1e6:
            evals_formatted = f"{evals_per_second/1e6:.2f}M"
        else:
            evals_formatted = f"{evals_per_second/1e3:.2f}K"
        
        # Stampa dei risultati
        print(f"\n{implementation} - {problem}:")
        print(f"  MSE medio: {mse_mean:.2e}")
        print(f"  Tempo medio di esecuzione: {exec_time_mean:.2f} s")
        print(f"  Valutazioni al secondo: {evals_formatted}")
        print(f"  Probabilità di successo (MSE ≤ 1e-7): {success_rate:.2f}%")
        
        if isinstance(time_to_solution, float):
            print(f"  Tempo medio per trovare soluzione (MSE ≤ 1e-7): {time_to_solution:.2f} s")
        else:
            print(f"  Tempo medio per trovare soluzione (MSE ≤ 1e-7): {time_to_solution}")
        
        return {
            'implementation': implementation,
            'problem': problem,
            'mse_mean': mse_mean,
            'exec_time_mean': exec_time_mean,
            'evals_per_second': evals_per_second,
            'evals_formatted': evals_formatted,
            'success_rate': success_rate,
            'time_to_solution': time_to_solution
        }
    
    except Exception as e:
        print(f"Errore nell'analisi di {file_path}: {e}")
        return None

def calculate_improvement(lgp_stats, deap_stats):
    """Calcola i miglioramenti relativi tra LGP e DEAP."""
    if not lgp_stats or not deap_stats:
        return
    
    # Miglioramento in MSE
    mse_improvement = deap_stats['mse_mean'] / lgp_stats['mse_mean'] if lgp_stats['mse_mean'] > 0 else float('inf')
    
    # Speedup in tempo di esecuzione
    speedup = deap_stats['exec_time_mean'] / lgp_stats['exec_time_mean'] if lgp_stats['exec_time_mean'] > 0 else float('inf')
    
    # Rapporto di valutazioni al secondo
    evals_ratio = lgp_stats['evals_per_second'] / deap_stats['evals_per_second'] if deap_stats['evals_per_second'] > 0 else float('inf')
    
    print(f"\nConfronti {lgp_stats['problem']}:")
    print(f"  Miglioramento MSE (DEAP/LGP): {mse_improvement:.1f}x")
    print(f"  Speedup tempo di esecuzione (DEAP/LGP): {speedup:.1f}x")
    print(f"  Rapporto valutazioni/secondo (LGP/DEAP): {evals_ratio:.1f}x")

def calculate_table_data():
    """Calcola e stampa i dati per le tabelle della tesi."""
    # Analisi di Shopping List
    lgp_shopping = analyze_dataset("shopping-4x100.csv", "LGP", "Shopping List")
    deap_shopping = analyze_dataset("deap_shopping-4x100.csv", "DEAP", "Shopping List", is_deap=True)
    
    # Analisi di Vector Distance
    lgp_vector = analyze_dataset("vector-6x100.csv", "LGP", "Vector Distance")
    deap_vector = analyze_dataset("deap_vector-6x100.csv", "DEAP", "Vector Distance", is_deap=True)
    
    # Calcolo miglioramenti relativi
    calculate_improvement(lgp_shopping, deap_shopping)
    calculate_improvement(lgp_vector, deap_vector)
    '''
    # Generazione tabella in formato LaTeX
    print("\n\nTabella LaTeX per il confronto sintetico:")
    print(r"""\begin{table}
   \centering
   \footnotesize
   \caption{Confronto sintetico LGP vs DEAP}
   \label{tab:comparison_compact}
   \setlength{\tabcolsep}{3.5pt}
   \begin{tabular}{@{}llcccc@{}}
      \toprule
      \textbf{Problema} & \textbf{Impl.} & \textbf{MSE} & \textbf{Tempo (s)} & \textbf{Val/s} & \textbf{Migl.} \\
      \midrule""")
    
    # Shopping List
    print(rf"      \multirow{{2}}{{*}}{{Shopping}} & LGP & {lgp_shopping['mse_mean']:.2e} & {lgp_shopping['exec_time_mean']:.1f} & {lgp_shopping['evals_formatted']} & \multirow{{2}}{{*}}{{{deap_shopping['mse_mean']/lgp_shopping['mse_mean']:.1f}$\times$}} \\")
    print(rf"      & DEAP & {deap_shopping['mse_mean']:.2e} & {deap_shopping['exec_time_mean']:.1f} & {deap_shopping['evals_formatted']} & \\")
    
    # Vector Distance
    print(r"      \midrule")
    print(rf"      \multirow{{2}}{{*}}{{Vector}} & LGP & {lgp_vector['mse_mean']:.2e} & {lgp_vector['exec_time_mean']:.1f} & {lgp_vector['evals_formatted']} & \multirow{{2}}{{*}}{{{deap_vector['mse_mean']/lgp_vector['mse_mean']:.1f}$\times$}} \\")
    print(rf"      & DEAP & {deap_vector['mse_mean']:.2e} & {deap_vector['exec_time_mean']:.1f} & {deap_vector['evals_formatted']} & \\")
    
    print(r"""      \bottomrule
   \end{tabular}
\end{table}""")
'''
# Esecuzione dell'analisi
if __name__ == "__main__":
    calculate_table_data()