# LGP Python Interface Documentation

This directory contains the Python interface for the Linear Genetic Programming (LGP) framework. The Python interface provides a user-friendly wrapper around the high-performance C core, adding input validation, memory management, and integration with popular data science libraries.

## 🏗️ Architecture

The Python interface is organized into several modules:

- **`base.py`**: Core ctypes bindings and constants
- **`genetics.py`**: Population, individuals, and problem definition classes  
- **`fitness.py`**: Fitness functions for evaluation
- **`selection.py`**: Selection methods for evolution
- **`creation.py`**: Population initialization strategies
- **`evolution.py`**: Main evolution loop and configuration
- **`vm.py`**: Virtual machine operations and instruction set
- **`utils.py`**: Utility functions (printing, random initialization)

## 📊 Core Classes

### LGPInput
The main class for defining optimization problems.

```python
import lgp
import numpy as np

# From NumPy arrays (most common)
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([3.0, 7.0, 11.0])  # Target: x1 + 2*x2
instruction_set = lgp.InstructionSet.complete()

lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set)

# From Pandas DataFrame (requires pandas import only when used)
import pandas as pd
df = pd.DataFrame({'x1': [1, 3, 5], 'x2': [2, 4, 6], 'target': [3, 7, 11]})
lgp_input = lgp.LGPInput.from_df(df, y=['target'], instruction_set)

# Custom RAM size for complex programs
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=64)
```

**Key Methods:**
- `from_numpy(X, y, instruction_set, ram_size=None)`: Create from NumPy arrays
- `from_df(df, y, instruction_set, ram_size=None)`: Create from Pandas DataFrame

**Memory Layout:**
- **ROM**: Contains input features (read-only during execution)
- **RAM**: Working memory, initialized with target values at start
- **Block Structure**: Each sample has `rom_size + ram_size` memory locations

### InstructionSet
Defines available operations for evolution.

```python
# Use complete instruction set (87 operations)
instruction_set = lgp.InstructionSet.complete()

# Create custom instruction set
from lgp.vm import Operation
custom_ops = [
    Operation.ADD_F,     # Floating-point addition
    Operation.SUB_F,     # Floating-point subtraction  
    Operation.MUL_F,     # Floating-point multiplication
    Operation.DIV_F,     # Floating-point division
    Operation.STORE_RAM_F, # Store to RAM
    Operation.LOAD_ROM_F,  # Load from ROM
    Operation.JMP_Z,     # Conditional jump if zero
    Operation.CMP        # Compare operation
]
instruction_set = lgp.InstructionSet(custom_ops)
```

### Population and Individual
Containers for evolved programs.

```python
# Access individuals in population
result = lgp.evolve(lgp_input, params)
population = result.population
best_idx = result.best_individual

# Get specific individual
best_individual = population.get(best_idx)

# Examine individual properties
print(f"Program size: {best_individual.size}")
print(f"Fitness: {best_individual.fitness}")
best_individual.print_program()
```

## 🎯 Fitness Functions

Fitness functions evaluate how well programs solve the target problem. The framework provides specialized functions for different output types:

### Regression Fitness (Floating-Point Output)
These functions analyze the **floating-point result** from `vm.ram[0].f64`:

```python
# Basic regression metrics
fitness = lgp.MSE()          # Mean Squared Error
fitness = lgp.RMSE()         # Root Mean Squared Error  
fitness = lgp.MAE()          # Mean Absolute Error
fitness = lgp.RSquared()     # Coefficient of determination

# Percentage-based metrics
fitness = lgp.MAPE()         # Mean Absolute Percentage Error
fitness = lgp.SymmetricMAPE() # Symmetric MAPE

# Robust loss functions
fitness = lgp.HuberLoss(delta=1.0)    # Huber loss with threshold
fitness = lgp.LogCosh()               # Log-cosh loss
fitness = lgp.PinballLoss(quantile=0.5) # Quantile regression

# Statistical measures
fitness = lgp.PearsonCorrelation()    # Correlation coefficient
fitness = lgp.WorstCaseError()        # Maximum absolute error

# Penalized fitness functions
fitness = lgp.LengthPenalizedMSE(alpha=0.01)  # MSE + program length penalty
fitness = lgp.ClockPenalizedMSE(alpha=0.01)   # MSE + execution time penalty
```

### Classification Fitness (Integer/Boolean Output)
These functions interpret the **sign bit** of `vm.ram[0].i64` where:
- **Negative values** = False/Class 0
- **Positive values** = True/Class 1

```python
# Basic classification metrics
fitness = lgp.Accuracy()            # Classification accuracy
fitness = lgp.BalancedAccuracy()    # Accuracy corrected for imbalance

# Advanced classification metrics  
fitness = lgp.F1Score()             # Harmonic mean of precision/recall
fitness = lgp.FBetaScore(beta=2.0)  # F-beta score with custom beta
fitness = lgp.MatthewsCorrelation() # Matthews correlation coefficient
fitness = lgp.CohensKappa()         # Cohen's kappa statistic
fitness = lgp.GMean()               # Geometric mean of sensitivity/specificity

# Loss functions for binary classification
fitness = lgp.HingeLoss()           # SVM-style hinge loss
fitness = lgp.BinaryCrossEntropy(tolerance=1e-15) # Cross-entropy loss
```

### Hybrid and Specialized Fitness
```python
# Threshold-based accuracy (floating-point with tolerance)
fitness = lgp.ThresholdAccuracy(threshold=0.1)

# Probabilistic fitness
fitness = lgp.GaussianLogLikelihood(sigma=1.0)

# Risk measures
fitness = lgp.ConditionalValueAtRisk()

# Robustness testing (requires NumPy for perturbation_vector)
import numpy as np
perturbation = np.array([0.1, 0.05, 0.02])  # Small perturbations per input
fitness = lgp.AdversarialPerturbationSensitivity(perturbation)
```

**Important Note**: Only `AdversarialPerturbationSensitivity.from_numpy()`, `LGPInput.from_df()`, and `LGPInput.from_numpy()` require NumPy/Pandas imports. All other functionality works without these dependencies.

## 🧬 Evolution Configuration

### Selection Methods
```python
# Tournament selection
selection = lgp.Tournament(size=3)
selection = lgp.FitnessSharingTournament(size=3, sigma=0.1)

# Elitism (top N individuals)
selection = lgp.Elitism(size=10)
selection = lgp.FitnessSharingElitism(size=10, sigma=0.1)

# Percentual elitism (top N% of population)  
selection = lgp.PercentualElitism(percentage=0.1)
selection = lgp.FitnessSharingPercentualElitism(percentage=0.1, sigma=0.1)

# Roulette wheel selection
selection = lgp.Roulette()
selection = lgp.FitnessSharingRoulette(sigma=0.1)
```

### Population Initialization
```python
# Unique population (no duplicate programs)
initialization = lgp.UniquePopulation()

# Random population (allows duplicates)
initialization = lgp.RandPopulation()
```

### Evolution Parameters
```python
params = lgp.LGPOptions(
    fitness=lgp.MSE(),                    # Fitness function
    selection=lgp.Tournament(size=3),     # Selection method
    initialization=lgp.UniquePopulation(), # Population initialization
    
    # Population parameters
    population_size=500,                  # Number of individuals
    min_program_size=5,                   # Minimum program length
    max_program_size=50,                  # Maximum program length
    
    # Evolution parameters
    generations=100,                      # Number of generations
    target_fitness=1e-6,                  # Stop when fitness reached
    mutation_rate=0.1,                    # Probability of mutation
    max_mutation_length=5,                # Maximum mutation size
    crossover_rate=0.9,                   # Probability of crossover
    
    # Execution parameters
    max_clock_cycles=5000,                # VM execution limit per program
    verbose=True                          # Print evolution progress
)
```

## 🔧 Advanced Usage

### Custom Fitness Function
```python
# Define custom fitness evaluation
def custom_fitness(lgp_input, individual, max_clock=5000):
    fitness_func = lgp.MSE()
    return fitness_func(lgp_input, individual, max_clock)

# Use in evaluation
fitness_value = custom_fitness(lgp_input, individual)
```

### Memory Management
```python
# The Python interface handles memory management automatically
# LGPInput objects clean up their memory when garbage collected
# No manual memory management required

# For large datasets, consider:
del lgp_input  # Explicit cleanup
import gc; gc.collect()  # Force garbage collection
```

### Thread Safety
```python
# Initialize random number generators for all threads
lgp.random_init_all(seed=42)

# The number of OpenMP threads is determined by:
# 1. OMP_NUM_THREADS environment variable
# 2. System default (usually number of CPU cores)
# 3. Compile-time THREADS variable

import os
os.environ['OMP_NUM_THREADS'] = '8'  # Limit to 8 threads
```

## 🚨 Error Handling

The Python interface provides comprehensive input validation:

```python
try:
    # Invalid input dimensions
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Wrong length
    lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set)
except ValueError as e:
    print(f"Input error: {e}")

try:
    # Invalid fitness parameters
    fitness = lgp.HuberLoss(delta=-1.0)  # Negative delta
except ValueError as e:
    print(f"Parameter error: {e}")

try:
    # Empty instruction set
    instruction_set = lgp.InstructionSet([])
except ValueError as e:
    print(f"Instruction set error: {e}")
```

## 📈 Performance Tips

1. **Use NumPy arrays**: More efficient than Python lists
2. **Appropriate RAM size**: Start with `ram_size = num_outputs`, increase if needed
3. **Instruction set size**: Smaller sets evolve faster, larger sets more flexible
4. **Population size**: Balance between diversity and speed (500-2000 typical)
5. **Early stopping**: Set appropriate `target_fitness` to avoid overtraining

## 🔗 Integration Examples

### Scikit-learn Integration
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train LGP model
lgp_input_train = lgp.LGPInput.from_numpy(X_train, y_train, instruction_set)
result = lgp.evolve(lgp_input_train, params)

# Evaluate on test set (would require implementing prediction function)
# This is a conceptual example - actual implementation would need
# a way to execute the evolved program on new data
```

### Pandas Workflow
```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Prepare features and targets
feature_cols = ['feature1', 'feature2', 'feature3']
target_cols = ['target']

# Create LGP input directly from DataFrame
lgp_input = lgp.LGPInput.from_df(
    df[feature_cols + target_cols], 
    y=target_cols, 
    instruction_set=instruction_set
)

# Evolve solution
result = lgp.evolve(lgp_input, params)
```

---

For more examples and advanced usage patterns, see `../examples.py` in the root directory.
