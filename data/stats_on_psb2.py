import pandas as pd
import numpy as np

def analyze_dataset(file_path, implementation, problem, is_deap=False):
    """Analizza un file CSV e calcola le statistiche richieste."""
    try:
        # Caricamento del file CSV
        df = pd.read_csv(file_path)
        df_succ = df[df['mse'] <= 1e-7]
        # Calcolo delle statistiche di base
        mse_mean = df['mse'].mean()
        exec_time_mean = df_succ['exec_time'].mean()
        
        # Calcolo delle valutazioni per DEAP vs LGP (diverso per i due formati)
        if is_deap:
            if 'lambda' in df.columns:
                df['evaluations'] = df['select_args'] + (df['lambda'] * df['gen'])
            else:
                print(f"AVVISO: Formato CSV DEAP non standard per {file_path}")
                df['evaluations'] = 0
        
        # Calcolo valutazioni al secondo
        evals_per_second = df_succ['evaluations'].mean()
        
        # Calcolo probabilità di successo (MSE <= 1e-9)
        success_rate = (df['mse'] <= 1e-9).mean() * 100
        
        # Tempo medio per trovare soluzione con MSE <= 1e-9
        successful_runs = df[df['mse'] <= 1e-9]
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
        print(f"  Valutazioni : {evals_formatted} ({evals_per_second:.0f})")
        print(f"  Probabilità di successo (MSE ≤ 1e-9): {success_rate:.2f}%")
        
        if isinstance(time_to_solution, float):
            print(f"  Tempo medio per trovare soluzione (MSE ≤ 1e-9): {time_to_solution:.2f} s")
        else:
            print(f"  Tempo medio per trovare soluzione (MSE ≤ 1e-9): {time_to_solution}")
        
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
    
    return {
        'mse_improvement': mse_improvement,
        'speedup': speedup,
        'evals_ratio': evals_ratio
    }

def calculate_table_data():
    """Calcola e stampa i dati per le tabelle della tesi."""
    # Analisi di tutti i problemi LGP
    print("Analisi dei dataset LGP...")
    lgp_shopping = analyze_dataset("shopping-4x100.csv", "LGP", "Shopping List")
    lgp_vector = analyze_dataset("vector-6x100.csv", "LGP", "Vector Distance")
    lgp_dice = analyze_dataset("dice-2x100.csv", "LGP", "Dice Game")
    lgp_snow = analyze_dataset("snow-4x100.csv", "LGP", "Snow Day")
    lgp_balls = analyze_dataset("balls-3x100.csv", "LGP", "Bouncing Balls")
    
    # Analisi dei dataset DEAP disponibili
    print("\nAnalisi dei dataset DEAP...")
    deap_shopping = analyze_dataset("deap_shopping-4x100.csv", "DEAP", "Shopping List", is_deap=True)
    deap_vector = analyze_dataset("deap_vector6-x100.csv", "DEAP", "Vector Distance", is_deap=True)
    
    # Calcolo miglioramenti relativi
    shopping_improvements = calculate_improvement(lgp_shopping, deap_shopping)
    vector_improvements = calculate_improvement(lgp_vector, deap_vector)
    
    # Aggiungo ogni problema alla tabella
    problems = [
        ("Shopping List", lgp_shopping),
        ("Vector Distance", lgp_vector),
        ("Dice Game", lgp_dice),
        ("Snow Day", lgp_snow),
        ("Bouncing Balls", lgp_balls)
    ]
    
    for name, stats in problems:
        if stats:
            print(rf"      {name} & {stats['exec_time_mean']:.1f} & {stats['mse_mean']:.1e} & {stats['evals_per_second']:.2e} & {stats['success_rate']:.1f} \\")

    # Stampa i dati formattati per la pubblicazione scientifica
    print("\n\nDati formattati per pubblicazione:")
    for name, stats in problems:
        if stats:
            print(f"{name}: {stats['exec_time_mean']:.1f}s, {stats['mse_mean']:.2e}, {stats['evals_per_second']:.2e} val/s, {stats['success_rate']:.1f}%")

# Esecuzione dell'analisi
if __name__ == "__main__":
    calculate_table_data()