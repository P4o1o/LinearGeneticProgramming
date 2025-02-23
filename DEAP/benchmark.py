import numpy as np
import time
import csv
import operator
import math
from deap import algorithms, base, creator, tools, gp
from multiprocessing import Pool

TEST_NUMS = 40
datafile = "/home/p4o1o/Documenti/LinearGeneticProgramming/shopping-4x100.bin"
x_lenght = 4
input_size = 100
params = {
    "tollerance": 1e-10,
    "max_generations": 230,
    "pop_size": 4000,
    "min_depth": 2,
    "max_depth": 5,
    "max_mut_len": 5,
    "elite_size": 4000,
    "lambda": 14000
}

def load_c_data(filename, x_len, in_size):
    """
    Legge i dati binari generati dalla funzione C save_problem_data.
    Restituisce X e y nel formato richiesto da DEAP.
    """
    total_doubles = in_size * (x_len + 1)
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
        if data.size != total_doubles:
            raise ValueError(f"File size mismatch. Expected {total_doubles} doubles, got {data.size}")
        data = data.reshape((in_size, x_len + 1))
        X = data[:, :x_len]
        y = data[:, x_len]
    return X, y

# Caricamento dei dati
X, y = load_c_data(datafile, x_lenght, input_size)

# Definizione globale del toolbox

# 1. Creazione del PrimitiveSet
pset = gp.PrimitiveSet("MAIN", x_lenght)
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(np.divide, 2)
pset.addPrimitive(lambda a, b: a * b / 100, 2, name="perc")
for i in range(x_lenght):
    pset.renameArguments(**{f'ARG{i}': f'x{i}'})

# 2. Creazione dei tipi di fitness e individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 3. Setup del toolbox globale
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                 min_=params["min_depth"], max_=params["max_depth"])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 4. Definizione della funzione di fitness
def eval_func(individual):
    func = toolbox.compile(expr=individual)
    # Decomprime i 4 valori di ogni input usando *x
    squarederr = ((func(*x) - trueval)**2 for x, trueval in zip(X, y))
    mse = math.fsum(squarederr) / len(y)
    return (mse,) 

toolbox.register("evaluate", eval_func)
toolbox.register("select", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=params["max_mut_len"])
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limiti dinamici per la profondità degli alberi
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params["max_depth"]))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params["max_depth"]))

# 5. Setup del pool di multiprocessing e registrazione della mappa
pool = Pool()
toolbox.register("map", pool.map)

def run_single_test(test_num):
    start_time = time.perf_counter()
    pop = toolbox.population(n=params["pop_size"])
    hof = tools.HallOfFame(1)
    
    algorithms.eaMuPlusLambda(
        pop, toolbox, params["elite_size"], params["lambda"],
        0.5, 0.5,
        ngen=params["max_generations"],
        halloffame=hof,
        verbose=False
    )
    exec_time = time.perf_counter() - start_time
    return {
        "test_num": test_num + 1,
        "mse": hof[0].fitness.values[0],
        "exec_time": exec_time,
        "generations": params["max_generations"],
        "found": hof[0].fitness.values[0] < params["tollerance"]
    }

# Esecuzione dei test e salvataggio dei risultati in CSV
with open("deap_shopping-4x100.csv", "w", newline='') as f:
    writer = csv.writer(f)
    header = ["test_num", "select_type", "select_args", "found", "initial_pop", 
              "in_min_len", "in_max_len", "lambda", "max_mut_len", "tests", 
              "mse", "exec_time", "gen"]
    writer.writerow(header)
    
    for test_idx in range(TEST_NUMS):
        result = run_single_test(test_idx)
        row = [
            result["test_num"],
            "elitism",
            params["elite_size"],
            int(result["found"]),
            params["pop_size"],
            params["min_depth"],
            params["max_depth"],
            params["lambda"],
            params["max_mut_len"],
            input_size,
            result["mse"],
            result["exec_time"],
            result["generations"]
        ]
        writer.writerow(row)
        print(f"Completed test {result['test_num']}/{TEST_NUMS}", end='\r')
