#!/usr/bin/env python3
"""
Esempi Completi di Linear Genetic Programming (LGP)

Questo file contiene esempi pratici e completi che dimostrano l'uso 
dell'interfaccia Python LGP per vari tipi di problemi con evoluzioni reali.
"""

import lgp
import numpy as np
import pandas as pd
import time
import warnings

def setup_lgp():
    """Setup iniziale del sistema LGP"""
    print("🚀 Inizializzazione sistema LGP...")
    
    # Inizializza generatore numeri casuali
    lgp.random_init(42, 1)
    
    # Test delle funzionalità base
    print(f"✓ LGP versione {lgp.__version__}")
    print(f"✓ {len([op for op in lgp.Operation])} operazioni VM disponibili")
    print(f"✓ Sistema pronto per l'evoluzione")
    print()


def example_polynomial_regression():
    """
    Esempio 1: Regressione Polinomiale Completa
    Obiettivo: Scoprire la formula f(x) = x³ - 2x² + x + 5
    """
    print("=" * 60)
    print("🧮 ESEMPIO 1: REGRESSIONE POLINOMIALE")
    print("=" * 60)
    print("Obiettivo: Scoprire f(x) = x³ - 2x² + x + 5")
    print()
    
    # 1. Generazione dataset
    print("📊 Generazione dataset...")
    n_samples = 300
    np.random.seed(123)
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y_true = X[:, 0]**3 - 2*X[:, 0]**2 + X[:, 0] + 5
    noise = np.random.normal(0, 0.2, n_samples)
    y = y_true + noise
    
    print(f"✓ Dataset: {n_samples} campioni")
    print(f"✓ Input range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"✓ Output range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"✓ Noise std: {noise.std():.3f}")
    print()
    
    # 2. Creazione instruction set ottimizzato
    print("🔧 Configurazione instruction set...")
    operations = [
        # Aritmetica base
        lgp.Operation.ADD_F, lgp.Operation.SUB_F, 
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        # Funzioni avanzate per polinomi
        lgp.Operation.POW, lgp.Operation.SQRT,
        # Accesso memoria
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        # Funzioni matematiche aggiuntive
        lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.EXP
    ]
    instruction_set = lgp.InstructionSet(operations)
    print(f"✓ Instruction set: {instruction_set.size} operazioni")
    print()
    
    # 3. Creazione input LGP
    print("🎯 Creazione input LGP...")
    lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=12)
    print(f"✓ Input: {lgp_input.input_num} campioni")
    print(f"✓ ROM size: {lgp_input.rom_size}")
    print(f"✓ RAM size: {lgp_input.ram_size}")
    print()
    
    # 4. Evoluzione con parametri ottimizzati
    print("🧬 Avvio evoluzione...")
    print("Parametri: pop_size=150, generazioni=80, tournament_size=4")
    print()
    
    start_time = time.time()
    
    try:
        population, evaluations, generations, best_idx = lgp.evolve(
            lgp_input,
            fitness=lgp.MSE(),
            selection=lgp.Tournament(4),
            initialization=lgp.UniquePopulation(),
            init_params=(150, 8, 30),  # pop_size, min_len, max_len
            target=0.05,  # Termina se MSE < 0.05
            mutation_prob=0.8,
            crossover_prob=0.95,
            max_clock=8000,
            max_individ_len=40,
            generations=80,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Errore durante l'evoluzione: {e}")
        return
    
    elapsed_time = time.time() - start_time
    
    # 5. Analisi risultati dettagliata
    print()
    print("📈 RISULTATI EVOLUZIONE:")
    print("-" * 40)
    
    best_individual = population.get(best_idx)
    
    print(f"✓ Evoluzione completata in {elapsed_time:.2f} secondi")
    print(f"✓ Generazioni eseguite: {generations}")
    print(f"✓ Valutazioni totali: {evaluations:,}")
    print(f"✓ Valutazioni/secondo: {evaluations/elapsed_time:.0f}")
    print()
    
    print(f"🏆 MIGLIOR SOLUZIONE:")
    print(f"   MSE: {best_individual.fitness:.6f}")
    print(f"   RMSE: {np.sqrt(best_individual.fitness):.6f}")
    print(f"   Lunghezza programma: {best_individual.size} istruzioni")
    
    # Calcola R² per valutazione aggiuntiva
    predictions = []
    for i in range(min(lgp_input.rows, 100)):  # Test su prima centina
        try:
            output = best_individual.execute(lgp_input, i)
            predictions.append(output[0] if len(output) > 0 else 0.0)
        except:
            predictions.append(0.0)
    
    if len(predictions) > 0:
        predictions = np.array(predictions)
        targets = y[:len(predictions)]
        
        # R² calculation
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"   R²: {r_squared:.6f}")
        print(f"   Correlazione: {np.corrcoef(targets, predictions)[0,1]:.6f}")
    
    print()
    print("📝 PROGRAMMA EVOLUTIVO:")
    print("-" * 40)
    lgp.print_program(best_individual)
    print()
    
    # 6. Statistiche popolazione
    print("📊 STATISTICHE POPOLAZIONE FINALE:")
    print("-" * 40)
    fitnesses = []
    sizes = []
    for i in range(min(population.size, 50)):  # Analizza prima 50
        try:
            ind = population.get(i)
            fitnesses.append(ind.fitness)
            sizes.append(ind.size)
        except:
            continue
    
    if fitnesses:
        fitnesses = np.array(fitnesses)
        sizes = np.array(sizes)
        
        print(f"Fitness - Media: {np.mean(fitnesses):.6f}, Std: {np.std(fitnesses):.6f}")
        print(f"Fitness - Range: [{np.min(fitnesses):.6f}, {np.max(fitnesses):.6f}]")
        print(f"Dimensioni - Media: {np.mean(sizes):.1f}, Std: {np.std(sizes):.1f}")
        print(f"Dimensioni - Range: [{np.min(sizes)}, {np.max(sizes)}]")
    
    print()
    return best_individual
    print("Operazioni nel set:")
    for i, op in enumerate(operations):
        print(f"  {i}: {op.name()} (code: {op.code()})")
    print()


def example_simple_regression():
    """Esempio: regressione simbolica semplice"""
    print("=== Esempio: Regressione Simbolica ===")
    
    # Inizializzazione del sistema
    lgp.random_init(42, 1)
    print("Sistema inizializzato con seed 42")
    
    # Genera dataset sintetico: y = x1^2 + 2*x2 + noise
    np.random.seed(42)
    n_samples = 100
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    y = x1**2 + 2*x2 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })
    
    print(f"Dataset creato: {len(df)} campioni")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: y")
    print(f"Esempio di dati:")
    print(df.head())
    
    # Crea instruction set per regressione
    regression_ops = [
        lgp.Operation.ADD_F, lgp.Operation.SUB_F,
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.MOV_F, lgp.Operation.MOV_I_F,
        lgp.Operation.LOAD_ROM_F
    ]
    instruction_set = lgp.InstructionSet(regression_ops)
    
    # Crea LGPInput usando from_numpy
    X = df[['x']].values
    y = df['y'].values
    lgp_input = lgp.LGPInput.from_numpy(
        X, 
        y, 
        instruction_set,
        ram_size=5
    )
    
    print(f"LGPInput creato:")
    print(f"  Input num: {lgp_input.input_num}")
    print(f"  ROM size: {lgp_input.rom_size}")
    print(f"  RAM size: {lgp_input.ram_size}")
    print(f"  Result size: {lgp_input.res_size}")
    print()


def example_fitness_assessment():
    """Esempio: funzioni di fitness assessment"""
    print("=== Esempio: Fitness Assessment ===")
    
    # Fitness per regressione
    print("Fitness per regressione:")
    mse = lgp.MSE()
    rmse = lgp.RMSE()
    mae = lgp.MAE()
    r2 = lgp.RSquared()
    
    print(f"  MSE: {type(mse).__name__}")
    print(f"  RMSE: {type(rmse).__name__}")
    print(f"  MAE: {type(mae).__name__}")
    print(f"  R²: {type(r2).__name__}")
    
    # Fitness penalizzate
    print("\nFitness penalizzate:")
    length_pen = lgp.LengthPenalizedMSE(alpha=0.01)
    clock_pen = lgp.ClockPenalizedMSE(alpha=0.005)
    
    print(f"  Length Penalized MSE (α=0.01): {type(length_pen).__name__}")
    print(f"  Clock Penalized MSE (α=0.005): {type(clock_pen).__name__}")
    
    # Fitness per classificazione
    print("\nFitness per classificazione:")
    accuracy = lgp.Accuracy()
    f1 = lgp.F1Score()
    balanced_acc = lgp.BalancedAccuracy()
    
    print(f"  Accuracy: {type(accuracy).__name__}")
    print(f"  F1 Score: {type(f1).__name__}")
    print(f"  Balanced Accuracy: {type(balanced_acc).__name__}")
    print()


def example_selection_methods():
    """Esempio: metodi di selezione"""
    print("=== Esempio: Metodi di Selezione ===")
    
    # Selezione base
    print("Metodi di selezione base:")
    tournament = lgp.Tournament(tournament_size=3)
    elitism = lgp.Elitism(elite_size=10)
    percentual = lgp.PercentualElitism(elite_percentage=0.1)
    roulette = lgp.Roulette(sampling_size=50)
    
    print(f"  Tournament (size=3): {type(tournament).__name__}")
    print(f"  Elitism (size=10): {type(elitism).__name__}")
    print(f"  Percentual Elitism (10%): {type(percentual).__name__}")
    print(f"  Roulette (sampling=50): {type(roulette).__name__}")
    
    # Fitness sharing
    print("\nMetodi con Fitness Sharing:")
    fs_tournament = lgp.FitnessSharingTournament(
        tournament_size=3, alpha=1.0, beta=1.0, sigma=1.0
    )
    fs_elitism = lgp.FitnessSharingElitism(
        elite_size=10, alpha=1.0, beta=1.0, sigma=1.0
    )
    
    print(f"  FS Tournament: {type(fs_tournament).__name__}")
    print(f"  FS Elitism: {type(fs_elitism).__name__}")
    print()


def example_initialization():
    """Esempio: metodi di inizializzazione"""
    print("=== Esempio: Inizializzazione ===")
    
    # Metodi disponibili
    unique = lgp.UniquePopulation()
    random = lgp.RandPopulation()
    
    print("Metodi di inizializzazione:")
    print(f"  Unique Population: {type(unique).__name__} (raccomandato)")
    print(f"  Random Population: {type(random).__name__}")
    print()


def example_vector_distance():
    """Esempio: problema VectorDistance"""
    print("=== Esempio: Vector Distance Problem ===")
    
    # Crea instruction set per vector distance
    vector_ops = [
        lgp.Operation.ADD_F, lgp.Operation.SUB_F,
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        lgp.Operation.SQRT, lgp.Operation.POW,
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.MOV_F
    ]
    instruction_set = lgp.InstructionSet(vector_ops)
    
    # Crea problema vector distance
    try:
        vector_problem = lgp.VectorDistance(
            instruction_set=instruction_set,
            vector_len=3,
            instances=50
        )
        
        print("Problema Vector Distance creato:")
        print(f"  Lunghezza vettori: 3")
        print(f"  Numero istanze: 50")
        print(f"  Input num: {vector_problem.input_num}")
        print(f"  ROM size: {vector_problem.rom_size}")
        print(f"  RAM size: {vector_problem.ram_size}")
        
    except Exception as e:
        print(f"Errore nella creazione VectorDistance: {e}")
        print("(Potrebbe essere necessaria la libreria C compilata)")
    print()


def example_complete_evolution():
    """Esempio: evoluzione completa con esecuzione reale"""
    print("=== Esempio: Evoluzione Completa ===")
    
    # Dataset di esempio: y = x1^2 + 2*x2 + noise
    np.random.seed(42)
    n_samples = 100
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    y = x1**2 + 2*x2 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    print(f"Dataset creato: {len(df)} campioni")
    
    # Inizializzazione
    lgp.random_init(seed=42, threadnum=1)
    
    # Instruction set ottimizzato
    operations = [
        lgp.Operation.ADD_F, lgp.Operation.SUB_F,
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        lgp.Operation.POW, lgp.Operation.SQRT
    ]
    instruction_set = lgp.InstructionSet(operations)
    
    # Creazione input
    X = df[['x1', 'x2']].values
    y = df['y'].values
    lgp_input = lgp.LGPInput.from_numpy(
        X, y, instruction_set, ram_size=6
    )
    
    print(f"LGPInput: {lgp_input.input_num} samples, ROM={lgp_input.rom_size}, RAM={lgp_input.ram_size}")
    
    # Configurazione evoluzione
    try:
        print("\\nInizio evoluzione...")
        result = lgp.evolve(
            lgp_input=lgp_input,
            fitness=lgp.MSE(),
            selection=lgp.Tournament(tournament_size=3),
            initialization=lgp.UniquePopulation(),
            init_params=(50, 3, 15),  # pop_size=50, min_len=3, max_len=15
            target=1e-4,              # Termina se MSE < 0.0001
            mutation_prob=0.8,
            crossover_prob=0.9,
            max_clock=3000,
            generations=50,
            verbose=1
        )
        
        # Analisi risultati
        population, evaluations, generations, best_idx = result
        
        print(f"\\n=== RISULTATI EVOLUZIONE ===")
        print(f"Generazioni completate: {generations}")
        print(f"Evaluations totali: {evaluations}")
        print(f"Dimensione popolazione finale: {population.size}")
        
        # Migliore individuo
        best_individual = population.get(best_idx)
        print(f"\\nMigliore individuo (indice {best_idx}):")
        print(f"  Fitness (MSE): {best_individual.fitness:.8f}")
        print(f"  RMSE: {np.sqrt(best_individual.fitness):.8f}")
        
        # Stampa programma
        print(f"\\nProgramma del migliore individuo:")
        lgp.print_program(best_individual)
        
        # Statistiche popolazione
        fitnesses = [population.get(i).fitness for i in range(min(10, population.size))]
        print(f"\\nFitness dei primi 10 individui:")
        for i, fit in enumerate(fitnesses):
            print(f"  #{i}: {fit:.8f}")
        
        print("\\n✓ Evoluzione completata con successo!")
        
    except Exception as e:
        print(f"Errore durante l'evoluzione: {e}")
        print("Nota: Questo può accadere se la libreria C non è compilata correttamente")
        print("Esegui 'make python' per compilare liblgp.so")
    
    print()


def example_classification_evolution():
    """Esempio: evoluzione per classificazione binaria"""
    print("=== Esempio: Classificazione Binaria ===")
    
    # Dataset di classificazione sintetico
    np.random.seed(123)
    n_samples = 200
    
    # Crea dataset linearmente separabile
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    
    # Regola di classificazione: y = 1 se x1 + x2 > 0, altrimenti 0
    y = (x1 + 2*x2 + np.random.normal(0, 0.3, n_samples) > 0).astype(float)
    
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'target': y})
    print(f"Dataset classificazione: {len(df)} campioni")
    print(f"Distribuzione classi: {np.bincount(y.astype(int))}")
    
    # Inizializzazione
    lgp.random_init(seed=123, threadnum=1)
    
    # Instruction set per classificazione
    classification_ops = [
        lgp.Operation.ADD_F, lgp.Operation.SUB_F,
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        lgp.Operation.CMP_F, lgp.Operation.TEST_F,
        # Operazioni per controllo logico
        lgp.Operation.JMP_L, lgp.Operation.JMP_G,
        lgp.Operation.JMP_Z, lgp.Operation.JMP_NZ
    ]
    instruction_set = lgp.InstructionSet(classification_ops)
    
    # Creazione input
    X = df[['x1', 'x2']].values
    y = df['target'].values
    lgp_input = lgp.LGPInput.from_numpy(
        X, y, instruction_set, ram_size=4
    )
    
    print(f"LGPInput: {lgp_input.input_num} samples, ROM={lgp_input.rom_size}, RAM={lgp_input.ram_size}")
    
    try:
        print("\\nInizio evoluzione per classificazione...")
        result = lgp.evolve(
            lgp_input=lgp_input,
            fitness=lgp.Accuracy(),  # Massimizza accuratezza
            selection=lgp.Tournament(tournament_size=4),
            initialization=lgp.UniquePopulation(),
            init_params=(40, 4, 20),  # Popolazione più piccola per classificazione
            target=0.95,              # Termina se accuracy > 95%
            mutation_prob=0.75,
            crossover_prob=0.85,
            max_clock=2000,
            generations=40,
            verbose=1
        )
        
        # Analisi risultati
        population, evaluations, generations, best_idx = result
        
        print(f"\\n=== RISULTATI CLASSIFICAZIONE ===")
        print(f"Generazioni completate: {generations}")
        print(f"Evaluations totali: {evaluations}")
        
        # Migliore classificatore
        best_classifier = population.get(best_idx)
        print(f"\\nMigliore classificatore:")
        print(f"  Accuracy: {best_classifier.fitness:.4f} ({best_classifier.fitness*100:.2f}%)")
        
        # Programma del migliore
        print(f"\\nProgramma del migliore classificatore:")
        lgp.print_program(best_classifier)
        
        # Top 5 accuracies
        top_fitnesses = sorted([population.get(i).fitness for i in range(population.size)], reverse=True)[:5]
        print(f"\\nTop 5 accuracies:")
        for i, acc in enumerate(top_fitnesses):
            print(f"  #{i+1}: {acc:.4f} ({acc*100:.2f}%)")
        
        print("\\n✓ Evoluzione classificazione completata!")
        
    except Exception as e:
        print(f"Errore durante l'evoluzione: {e}")
        print("Assicurati che la libreria C sia compilata (make python)")
    
    print()


def example_advanced_math_evolution():
    """Esempio: evoluzione con funzioni matematiche avanzate"""
    print("=== Esempio: Funzioni Matematiche Avanzate ===")
    
    # Dataset con funzione trigonometrica: y = sin(x1) + cos(x2) + x1*x2
    np.random.seed(789)
    n_samples = 150
    x1 = np.random.uniform(-np.pi, np.pi, n_samples)
    x2 = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.sin(x1) + np.cos(x2) + 0.5*x1*x2 + np.random.normal(0, 0.05, n_samples)
    
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    print(f"Dataset matematico: {len(df)} campioni")
    print(f"Target: y = sin(x1) + cos(x2) + 0.5*x1*x2 + noise")
    
    # Inizializzazione
    lgp.random_init(seed=789, threadnum=1)
    
    # Instruction set con funzioni matematiche avanzate
    math_ops = [
        # Aritmetica base
        lgp.Operation.ADD_F, lgp.Operation.SUB_F,
        lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        # Memoria
        lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
        lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F,
        # Funzioni trigonometriche
        lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN,
        # Funzioni esponenziali
        lgp.Operation.EXP, lgp.Operation.LN,
        # Potenze e radici
        lgp.Operation.POW, lgp.Operation.SQRT
    ]
    instruction_set = lgp.InstructionSet(math_ops)
    
    # Preparazione dati per LGPInput
    X = df[['x1', 'x2']].values
    y = df['y'].values
    lgp_input = lgp.LGPInput.from_numpy(
        X, y, instruction_set, ram_size=8
    )
    
    try:
        print("\\nInizio evoluzione con funzioni matematiche...")
        result = lgp.evolve(
            lgp_input=lgp_input,
            fitness=lgp.RMSE(),  # Root Mean Square Error
            selection=lgp.Elitism(elite_size=8),  # Preserva i migliori
            initialization=lgp.UniquePopulation(),
            init_params=(60, 5, 25),
            target=0.1,   # RMSE target
            mutation_prob=0.85,
            crossover_prob=0.9,
            max_clock=4000,
            generations=80,
            verbose=1
        )
        
        population, evaluations, generations, best_idx = result
        
        print(f"\\n=== RISULTATI FUNZIONI MATEMATICHE ===")
        print(f"Generazioni: {generations}, Evaluations: {evaluations}")
        
        best = population.get(best_idx)
        print(f"\\nMigliore approssimazione:")
        print(f"  RMSE: {best.fitness:.6f}")
        print(f"  R² equivalente: {1 - (best.fitness**2 / np.var(y)):.6f}")
        
        print(f"\\nProgramma matematico:")
        lgp.print_program(best)
        
        # Confronta con funzione target teorica
        y_target_var = np.var(y)
        mse_target = best.fitness**2
        print(f"\\nAnalisi approssimazione:")
        print(f"  Varianza target: {y_target_var:.6f}")
        print(f"  MSE ottenuto: {mse_target:.6f}")
        print(f"  Percentuale spiegata: {(1-mse_target/y_target_var)*100:.2f}%")
        
        print("\\n✓ Evoluzione matematica completata!")
        
    except Exception as e:
        print(f"Errore durante l'evoluzione: {e}")
    
    print()


def main():
    """Esegue tutti gli esempi"""
    print("=" * 60)
    print("ESEMPI INTERFACCIA PYTHON LINEAR GENETIC PROGRAMMING")
    print("=" * 60)
    print()
    
    try:
        setup_lgp()
        example_polynomial_regression()
        example_simple_regression()
        example_fitness_assessment()
        example_selection_methods()
        example_initialization()
        example_vector_distance()
        example_complete_evolution()
        example_classification_evolution()
        example_advanced_math_evolution()
        
        print("=" * 60)
        print("TUTTI GLI ESEMPI COMPLETATI CON SUCCESSO")
        print("=" * 60)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione degli esempi: {e}")
        print("Assicurati che la libreria C sia compilata (make python)")


if __name__ == "__main__":
    main()
