#!/usr/bin/env python3
"""
Demo LGP Python che corrisponde al main.c del progetto
Testa l'evoluzione completa con il problema vector_distance
"""

import sys
import os
import time
import numpy as np

# Aggiungi il percorso della libreria
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lgp

def get_time_sec():
    """Equivalente della funzione get_time_sec() in main.c"""
    return time.time()

def create_instruction_set():
    """
    Crea un instruction set equivalente a quello in main.c:
    {OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F, OP_POW, OP_LOAD_ROM_F, OP_STORE_RAM_F, OP_MOV_F}
    """
    operations = [
        lgp.Operation.ADD_F,        # OP_ADD_F
        lgp.Operation.SUB_F,        # OP_SUB_F  
        lgp.Operation.MUL_F,        # OP_MUL_F
        lgp.Operation.DIV_F,        # OP_DIV_F
        lgp.Operation.POW,          # OP_POW
        lgp.Operation.LOAD_ROM_F,   # OP_LOAD_ROM_F
        lgp.Operation.STORE_RAM_F,  # OP_STORE_RAM_F
        lgp.Operation.MOV_F         # OP_MOV_F
    ]
    
    return lgp.InstructionSet(operations)

def demo_main():
    """Demo principale che replica la logica di main.c"""
    print("=== Demo LGP Python (equivalente main.c) ===\n")
    
    # 1. Inizializzazione random (come in main.c)
    print("1. Inizializzazione random:")
    initial_seed = 0x47afeb91
    lgp.random_init(initial_seed, 0)
    print(f"   Seed iniziale: 0x{initial_seed:x}")
    
    # Simula l'inizializzazione multi-thread (per ora single thread)
    print("   Thread 0 inizializzato")
    
    # 2. Configurazione parametri LGP (identici a main.c)
    print("\n2. Configurazione parametri LGP:")
    
    # Fitness function
    fitness = lgp.MSE()
    print(f"   Fitness: {fitness.name}")
    
    # Selection method  
    selection = lgp.Tournament(tournament_size=3)
    print(f"   Selection: {selection.name} (size: {selection.tournament_size})")
    
    # Initialization method
    initialization = lgp.UniquePopulation()
    print(f"   Initialization: {initialization.name}")
    
    # Parametri di inizializzazione (identici a main.c)
    init_params = (1000, 2, 5)  # pop_size=1000, minsize=2, maxsize=5
    print(f"   Init params: pop_size={init_params[0]}, minsize={init_params[1]}, maxsize={init_params[2]}")
    
    # Altri parametri (identici a main.c)
    target = 1e-27
    mutation_prob = 0.76
    max_mutation_len = 5
    crossover_prob = 0.95
    max_clock = 5000
    max_individ_len = 50
    generations = 40
    verbose = 1
    
    print(f"   Target: {target}")
    print(f"   Mutation prob: {mutation_prob}")
    print(f"   Crossover prob: {crossover_prob}")
    print(f"   Max clock: {max_clock}")
    print(f"   Generations: {generations}")
    
    # 3. Creazione instruction set
    print("\n3. Instruction Set:")
    instruction_set = create_instruction_set()
    print(f"   ✓ InstructionSet creato con {instruction_set.size} operazioni")
    
    # Mostra le operazioni incluse
    operations_used = [
        lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
        lgp.Operation.POW, lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.MOV_F
    ]
    print("   Operazioni incluse:")
    for op in operations_used:
        print(f"     - {op.name()} (codice: {op.code()})")
    
    # 4. Creazione problema vector_distance
    print("\n4. Creazione problema vector_distance:")
    vector_len = 2
    instances = 100
    print(f"   Vector length: {vector_len}")
    print(f"   Instances: {instances}")
    
    try:
        # Crea il problema vector_distance (equivalente a main.c)
        lgp_input = lgp.VectorDistance(instruction_set, vector_len, instances)
        print(f"   ✓ LGPInput creato: {lgp_input.input_num} istanze, ROM size: {lgp_input.rom_size}")
    except Exception as e:
        print(f"   ✗ Errore creazione LGPInput: {e}")
        return
    
    # 5. Esecuzione evoluzione
    print("\n5. Esecuzione evoluzione:")
    print("   Avvio evoluzione...")
    
    start_time = get_time_sec()
    
    try:
        # Chiama evolve con gli stessi parametri di main.c
        result = lgp.evolve(
            lgp_input=lgp_input,
            fitness=fitness,
            fitness_param=None,
            selection=selection,
            select_param=3,  # tournament size
            initialization=initialization,
            init_params=init_params,
            target=target,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            max_clock=max_clock,
            max_individ_len=max_individ_len,
            max_mutation_len=max_mutation_len,
            generations=generations,
            verbose=verbose
        )
        
        end_time = get_time_sec()
        
        # Estrai risultati (come in main.c)
        population, evaluations, generations_done, best_individ_idx = result
        
        print(f"   ✓ Evoluzione completata!")
        
        # 6. Risultati (identici a main.c)
        print("\n6. Risultati:")
        print("Solution:")
        
        # Ottieni il miglior individuo
        best_individual = population.get(best_individ_idx)
        print(f"   Best fitness: {best_individual.fitness}")
        
        # Stampa il programma (come print_program in main.c)
        lgp.print_program(best_individual)
        
        # Statistiche temporali (identiche a main.c)
        elapsed_time = end_time - start_time
        eval_per_sec = evaluations / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nTime: {elapsed_time:.6f}s, evaluations: {evaluations}, eval/sec: {eval_per_sec:.2f}")
        print(f"Generations completed: {generations_done}")
        
        print("\n=== Demo completata con successo! ===")
        
    except Exception as e:
        print(f"   ✗ Errore durante evoluzione: {e}")
        import traceback
        traceback.print_exc()

def demo_vector_distance_simulation():
    """
    Demo del problema vector_distance senza libreria C
    Simula il comportamento per testare l'interfaccia
    """
    print("\n" + "="*60)
    print("DEMO SIMULAZIONE VECTOR DISTANCE")
    print("="*60)
    
    print("\nProblema vector_distance:")
    print("- Genera coppie di vettori casuali")
    print("- Target: calcolare la distanza euclidea tra i vettori")
    print("- ROM: contiene i due vettori concatenati")
    print("- RAM: deve contenere il risultato (distanza)")
    
    # Simula i dati che verrebbero generati da vector_distance()
    vector_len = 2
    instances = 5  # Pochi per la demo
    
    print(f"\nSimulazione con vector_len={vector_len}, instances={instances}:")
    
    np.random.seed(42)  # Per riproducibilità
    
    for i in range(instances):
        # Genera due vettori casuali (come in psb2.c)
        vec1 = np.random.uniform(-100.0, 100.0, vector_len)
        vec2 = np.random.uniform(-100.0, 100.0, vector_len)
        
        # Calcola distanza euclidea (come in psb2.c)
        diff = vec1 - vec2
        distance = np.sqrt(np.sum(diff * diff))
        
        print(f"  Istanza {i+1}:")
        print(f"    Vettore 1: [{vec1[0]:.3f}, {vec1[1]:.3f}]")
        print(f"    Vettore 2: [{vec2[0]:.3f}, {vec2[1]:.3f}]")
        print(f"    Distanza target: {distance:.6f}")
    
    print(f"\nL'LGP dovrebbe imparare a calcolare: sqrt((x1-x3)² + (x2-x4)²)")
    print("dove x1,x2 sono il primo vettore e x3,x4 il secondo vettore")

def show_operations_demo():
    """Mostra le operazioni disponibili per l'instruction set"""
    print("\n" + "="*60) 
    print("OPERAZIONI DISPONIBILI PER INSTRUCTION SET")
    print("="*60)
    
    print("\nOperazioni usate in main.c:")
    main_c_ops = [
        lgp.Operation.ADD_F,      # OP_ADD_F
        lgp.Operation.SUB_F,      # OP_SUB_F
        lgp.Operation.MUL_F,      # OP_MUL_F
        lgp.Operation.DIV_F,      # OP_DIV_F
        lgp.Operation.POW,        # OP_POW
        lgp.Operation.LOAD_ROM_F, # OP_LOAD_ROM_F
        lgp.Operation.STORE_RAM_F,# OP_STORE_RAM_F
        lgp.Operation.MOV_F       # OP_MOV_F
    ]
    
    for i, op in enumerate(main_c_ops, 1):
        print(f"  {i}. {op.name()} (codice: {op.code()})")
    
    print(f"\nTotale operazioni disponibili nell'interfaccia: {len(lgp.Operation)}")
    
    print("\nAltri esempi di operazioni utili:")
    other_ops = [
        lgp.Operation.SQRT,   # Per distanze
        lgp.Operation.CMP_F,  # Per confronti
        lgp.Operation.JMP,    # Per controllo flusso
        lgp.Operation.SIN,    # Funzioni trigonometriche
        lgp.Operation.COS,
        lgp.Operation.LOG
    ]
    
    for op in other_ops:
        print(f"  - {op.name()} (codice: {op.code()})")

if __name__ == "__main__":
    try:
        demo_main()
        demo_vector_distance_simulation()
        show_operations_demo()
        
    except Exception as e:
        print(f"\n✗ Errore nella demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
