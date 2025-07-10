# Linear Genetic Programming - Python Interface

High-level Python wrapper for the Linear Genetic Programming C library. Provides unified classes that combine C structures with user-friendly interfaces, comprehensive fitness functions, automatic memory management, and seamless integration with NumPy and Pandas.

## Installation

1. Build the C library with Python support:
   ```bash
   make python
   ```

2. The Python module will be available as `lgp` with the shared library `liblgp.so` in the project root.

## Quick Start

```python
import lgp
import numpy as np

# Create test data: symbolic regression y = x^2 + 2*x + 1
X = np.random.uniform(-3, 3, (100, 1))
y = X[:, 0]**2 + 2*X[:, 0] + 1

# Set up LGP problem - note: y must be reshaped for single output
instruction_set = lgp.InstructionSet.complete()  # All 87 VM operations
lgp_input = lgp.LGPInput.from_numpy(X, y.reshape(-1, 1), instruction_set)

# Configure evolution components
fitness = lgp.MSE()  # Mean squared error (MINIMIZE)
initialization = lgp.UniquePopulation(pop_size=200, minsize=3, maxsize=20)

# Run evolution
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    initialization=initialization,
    target=1e-6,
    generations=50,
    verbose=1
)

# Get best individual and results
best = population.get(best_idx)
print(f"Best fitness: {best.fitness}")
print(f"Program size: {best.size} instructions")
best.print_program()
```

## Core Classes

### LGPInput
Unified structure that represents problem datasets with features, targets, and execution environment. Acts as interface between C structures and Python objects.

#### Creation Methods

**From NumPy arrays:**
```python
# Single output (must be reshaped to 2D)
X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape: (n_samples, n_features) 
y = np.array([3.0, 7.0])                 # Shape: (n_samples,)
y_reshaped = y.reshape(-1, 1)            # Required: (n_samples, 1)

# Multiple outputs
y_multi = np.array([[3.0, 1.0], [7.0, 2.0]])  # Shape: (n_samples, n_outputs)

lgp_input = lgp.LGPInput.from_numpy(X, y_reshaped, instruction_set, ram_size=10)
```

**From Pandas DataFrame:**
```python
import pandas as pd

df = pd.DataFrame({
    'x1': [1, 3, 5], 
    'x2': [2, 4, 6], 
    'target': [3, 7, 11]
})

# y parameter must be a list of column names
lgp_input = lgp.LGPInput.from_df(df, y=['target'], instruction_set)
```

#### PSB2 Benchmark Problems
Pre-defined problems from Program Synthesis Benchmark Suite 2:
```python
# Vector distance calculation - find Euclidean distance
lgp_input = lgp.VectorDistance(instruction_set, vector_len=3, instances=100)

# Physics simulation problems
lgp_input = lgp.BouncingBalls(instruction_set, instances=200)  # Predict ball trajectory
lgp_input = lgp.SnowDay(instruction_set, instances=150)      # Weather prediction

# Game theory and optimization
lgp_input = lgp.DiceGame(instruction_set, instances=300)     # Optimal dice strategy
lgp_input = lgp.ShoppingList(instruction_set, num_of_items=5, instances=100)  # Budget optimization
```

**Memory Layout:**
- `input_num`: Number of training samples
- `rom_size`: Number of input features per sample (read-only memory)
- `res_size`: Number of output values per sample  
- `ram_size`: Working memory size (must be ≥ `res_size`, default: max(1, res_size))
- Memory organization: `[ROM: features][RAM: targets + workspace]` per sample
- Programs read inputs from ROM, write outputs to first `res_size` RAM positions

### InstructionSet
Defines the operations available to evolved programs. Contains 87 VM operations with integer and floating-point variants.

```python
# Complete instruction set (all 87 operations)
instruction_set = lgp.InstructionSet.complete()

# Custom instruction set from Operation enum
from lgp.vm import Operation
custom_ops = [
    # Floating-point arithmetic
    Operation.ADD_F,         # Floating-point addition
    Operation.SUB_F,         # Floating-point subtraction  
    Operation.MUL_F,         # Floating-point multiplication
    Operation.DIV_F,         # Floating-point division
    
    # Mathematical functions
    Operation.SQRT,          # Square root
    Operation.SIN,           # Sine function
    Operation.COS,           # Cosine function
    Operation.EXP,           # Exponential
    Operation.LN,            # Natural logarithm
    Operation.POW,           # Power function
    
    # Memory operations
    Operation.LOAD_ROM_F,    # Load from ROM (input features)
    Operation.STORE_RAM_F,   # Store to RAM (output/working memory)
    Operation.LOAD_RAM_F,    # Load from RAM
    
    # Control flow
    Operation.JMP_Z,         # Jump if zero flag set
    Operation.JMP_NZ,        # Jump if not zero
    Operation.CMP_F,         # Compare floating-point values
    Operation.CMOV_L_F,      # Conditional move if less
    
    # Integer operations (for classification)
    Operation.ADD,           # Integer addition
    Operation.CMP,           # Integer comparison
    Operation.LOAD_ROM,      # Load integer from ROM
    Operation.STORE_RAM      # Store integer to RAM
]
instruction_set = lgp.InstructionSet(custom_ops)
```

**Operation Categories:**
- **Arithmetic**: ADD/SUB/MUL/DIV (integer and float variants)
- **Mathematical**: SQRT, POW, EXP, LN, LOG, trigonometric functions
- **Memory**: LOAD_ROM/RAM, STORE_RAM (address modes 2,4,5)
- **Control Flow**: JMP variants, conditional moves (CMOV)
- **Logic**: AND, OR, XOR, NOT, bit shifts
- **Utility**: CAST, NOP, RAND, ROUND

### Fitness Functions
Evaluate program performance on the given problem. **30+ fitness functions available** with parameters via `FitnessFactor` union.

#### Regression Fitness Functions (Floating-Point Output)

**Basic Error Metrics (MINIMIZE):**
```python
fitness = lgp.MSE()                     # Mean Squared Error
fitness = lgp.RMSE()                    # Root Mean Squared Error  
fitness = lgp.MAE()                     # Mean Absolute Error
fitness = lgp.MAPE()                    # Mean Absolute Percentage Error
fitness = lgp.SymmetricMAPE()          # Symmetric MAPE
fitness = lgp.LogCosh()                # LogCosh loss function
fitness = lgp.WorstCaseError()         # Maximum error across samples
```

**Robust Loss Functions (MINIMIZE):**
```python
fitness = lgp.HuberLoss(delta=1.5)                    # Robust to outliers (FitnessFactor.delta)
fitness = lgp.PinballLoss(quantile=0.9)               # Quantile regression (FitnessFactor.quantile)
fitness = lgp.BinaryCrossEntropy(tolerance=1e-15)     # Cross-entropy (FitnessFactor.tolerance)
fitness = lgp.BrierScore()                            # Probabilistic accuracy
fitness = lgp.HingeLoss()                             # SVM hinge loss
```

**Penalized Functions (MINIMIZE):**
```python
fitness = lgp.LengthPenalizedMSE(alpha=0.01)         # MSE + length penalty (FitnessFactor.alpha)
fitness = lgp.ClockPenalizedMSE(alpha=0.001)         # MSE + execution time penalty (FitnessFactor.alpha)
```

**Statistical Measures (MAXIMIZE):**
```python
fitness = lgp.RSquared()                              # Coefficient of determination
fitness = lgp.PearsonCorrelation()                   # Pearson correlation coefficient
```

**Advanced Functions:**
```python
fitness = lgp.GaussianLogLikelihood(sigma=2.0)       # Maximum likelihood (FitnessFactor.sigma)
fitness = lgp.ConditionalValueAtRisk(alpha=0.05)     # Risk measure - 5% worst cases (FitnessFactor.alpha)

# Robustness analysis - requires perturbation vector
import numpy as np
perturbation = np.array([0.1, -0.05, 0.02])  # Must match input_num
fitness = lgp.AdversarialPerturbationSensitivity(perturbation)  # FitnessFactor.perturbation_vector
```

#### Classification Fitness Functions

Programs output to RAM - classification interprets values based on sign bit or exact match.

**Basic Classification (MAXIMIZE):**
```python
fitness = lgp.Accuracy()                             # Per-label accuracy (sign-bit interpretation)
fitness = lgp.StrictAccuracy()                       # Exact vector match per sample
fitness = lgp.BinaryAccuracy()                       # Binary classification accuracy
fitness = lgp.StrictBinaryAccuracy()                 # Strict binary vector match
```

**Advanced Classification Metrics (MAXIMIZE):**
```python
fitness = lgp.F1Score()                              # F1 score (harmonic mean precision/recall)
fitness = lgp.FBetaScore(beta=2.0)                   # F-beta score (FitnessFactor.beta)
fitness = lgp.MatthewsCorrelation()                  # Matthews correlation coefficient
fitness = lgp.BalancedAccuracy()                     # Average sensitivity/specificity
fitness = lgp.GMean()                                # Geometric mean sensitivity/specificity
fitness = lgp.CohensKappa()                          # Cohen's kappa statistic
```

**Threshold-Based Classification (MAXIMIZE):**
```python
fitness = lgp.ThresholdAccuracy(threshold=0.8)       # Tolerance-based accuracy (FitnessFactor.threshold)
fitness = lgp.StrictThresholdAccuracy(threshold=0.5) # Strict threshold vector match (FitnessFactor.threshold)
```

#### Output Range Selection
Control which RAM positions are evaluated:

```python
# Evaluate only outputs 0, 1, 2 (RAM indices)
fitness = lgp.MSE(start=0, end=3)

# Evaluate only output 1 (single output)
fitness = lgp.Accuracy(start=1, end=2)

# Default: start=0, end=0 (automatically set to res_size during evolution)
```

**Important:** `start` is inclusive, `end` is exclusive. Range must be within `[0, res_size]`.

#### Fitness Parameters and FitnessFactor Union
Fitness functions use `FitnessParams` with `FitnessFactor` union for parameters:
```python
# Access via helper methods
params = lgp.FitnessParams.new_alpha(0.01, start=0, end=2)
params = lgp.FitnessParams.new_threshold(0.7, start=1, end=3)
params = lgp.FitnessParams.new_perturbation_vector(numpy_array, start=0, end=1)

# Direct fitness evaluation
fitness_value = fitness(lgp_input, individual, max_clock=5000)
```

### Selection Methods
Determine which individuals survive to reproduce. All methods use `SelectionParams` union.

```python
# Tournament selection (default) - selects best from random tournaments
selection = lgp.Tournament(tournament_size=3)

# Elitism selection - keep absolute best individuals
selection = lgp.Elitism(elite_size=10)                      # Keep 10 best
selection = lgp.PercentualElitism(elite_percentage=0.1)     # Keep top 10%

# Roulette wheel selection - probability proportional to fitness
selection = lgp.Roulette(sampling_size=100)

# Fitness sharing variants - promotes diversity via `FitnessSharingParams`
selection = lgp.FitnessSharingTournament(
    tournament_size=3, 
    alpha=1.0,      # Sharing function exponent
    beta=1.0,       # Fitness power
    sigma=0.1       # Sharing radius
)

selection = lgp.FitnessSharingElitism(
    elite_size=10, alpha=1.0, beta=1.0, sigma=0.1
)

selection = lgp.FitnessSharingPercentualElitism(
    elite_percentage=0.1, alpha=1.0, beta=1.0, sigma=0.1
)

selection = lgp.FitnessSharingRoulette(
    sampling_size=100, alpha=1.0, beta=1.0, sigma=0.1
)
```

### Population Initialization
Creates the initial population of programs using `InitializationParams`.

```python
# Unique population (recommended - avoids duplicates)
initialization = lgp.UniquePopulation(
    pop_size=500,    # Population size
    minsize=2,       # Minimum program length (instructions)
    maxsize=15       # Maximum program length  
)

# Random population (allows duplicates - faster generation)
initialization = lgp.RandPopulation(
    pop_size=500,
    minsize=2, 
    maxsize=15
)
```

**Note:** Either `initialization` OR `initial_pop` must be provided to `evolve()`, but not both.

## Evolution Function

Main function for running genetic programming evolution:

```python
def evolve(lgp_input: LGPInput, 
          fitness: Fitness = MSE(),
          selection: Selection = Tournament(3),
          initialization: Initialization = None,
          initial_pop: Optional[Population] = None,
          target: float = 1e-27,
          mutation_prob: float = 0.76,
          crossover_prob: float = 0.95,
          max_clock: int = 5000,
          max_individ_len: int = 50,
          max_mutation_len: int = 5,
          generations: int = 40,
          verbose: int = 1) -> Tuple[Population, int, int, int]
```

**Parameters:**

**Required:**
- `lgp_input`: Problem definition (`LGPInput`) with features, targets, and instruction set
- **Either** `initialization` OR `initial_pop` (exactly one must be provided)

**Optional Evolution Configuration:**
- `fitness`: Fitness function (default: `MSE()`)
- `selection`: Selection method (default: `Tournament(3)`)
- `target`: Target fitness for early termination (default: `1e-27`)
- `mutation_prob`: Mutation probability per individual (default: `0.76`) 
- `crossover_prob`: Crossover probability per individual (default: `0.95`)
- `max_clock`: Maximum VM cycles per program execution (default: `5000`)
- `max_individ_len`: Maximum program length in instructions (default: `50`)
- `max_mutation_len`: Maximum length of mutation segments (default: `5`)
- `generations`: Maximum generations to run (default: `40`)
- `verbose`: Print progress (`0`=silent, `1`=per-generation stats)

**Returns:** 
`Tuple[Population, int, int, int]` = `(final_population, total_evaluations, actual_generations, best_individual_index)`

**Probability Values > 1.0:**
- Values > 1.0 apply operations multiple times per individual
- E.g., `mutation_prob=1.5` = 1 guaranteed mutation + 50% chance for a second mutation
- E.g., `crossover_prob=2.3` = 2 guaranteed crossovers + 30% chance for a third

## Advanced Usage Examples

### Multi-Output Regression
```python
# Problem: predict both sin(x) and cos(x) simultaneously
X = np.random.uniform(-np.pi, np.pi, (200, 1))
y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 0])])  # Shape: (200, 2)

# Custom instruction set optimized for trigonometry
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.SQRT, lgp.Operation.POW,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.LOAD_RAM_F,
    lgp.Operation.CMOV_L_F, lgp.Operation.CMP_F
])

lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=4)

# Fitness evaluates both outputs (res_size=2)
fitness = lgp.MSE(start=0, end=2)  # Evaluate RAM positions 0 and 1

population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    initialization=lgp.UniquePopulation(pop_size=300, minsize=5, maxsize=25),
    selection=lgp.Tournament(tournament_size=4),
    target=1e-4,
    generations=100,
    max_clock=10000  # Increased for complex programs
)

print(f"Final fitness: {population.get(best_idx).fitness}")
print(f"Program size: {population.get(best_idx).size} instructions")
```

### Classification with Custom Fitness
```python
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(n_samples=300, n_features=4, n_classes=2, random_state=42)

# For classification: reshape y and convert to expected format
y_reshaped = y.reshape(-1, 1).astype(float)  # Shape: (300, 1), required format

# Classification benefits from mixed integer/float operations
instruction_set = lgp.InstructionSet([
    # Floating-point for feature processing
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.CMP_F,
    # Integer for final classification decisions
    lgp.Operation.ADD, lgp.Operation.SUB, lgp.Operation.CMP, lgp.Operation.CMOV_L,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    # Control flow for decision boundaries
    lgp.Operation.JMP_Z, lgp.Operation.JMP_L, lgp.Operation.CAST, lgp.Operation.CAST_F
])

lgp_input = lgp.LGPInput.from_numpy(X, y_reshaped, instruction_set)

# F-beta score emphasizes recall over precision (beta > 1)
fitness = lgp.FBetaScore(beta=2.0)  

population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    selection=lgp.FitnessSharingTournament(tournament_size=5, alpha=1.0, beta=1.0, sigma=0.2),
    initialization=lgp.UniquePopulation(pop_size=400, minsize=3, maxsize=20),
    mutation_prob=0.3,
    crossover_prob=1.2,  # Multiple crossovers per individual
    max_individ_len=30,
    generations=80
)

print(f"Best F-beta score: {population.get(best_idx).fitness}")
```

### PSB2 Benchmark Problems
```python
# Vector Distance - find Euclidean distance between vectors
instruction_set = lgp.InstructionSet.complete()
lgp_input = lgp.VectorDistance(instruction_set, vector_len=5, instances=250)

# Bouncing Balls - physics simulation
lgp_input = lgp.BouncingBalls(instruction_set, instances=200)

# Use strict accuracy for exact problem solutions
fitness = lgp.StrictAccuracy()  # Must match expected output exactly

population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    initialization=lgp.UniquePopulation(pop_size=1000, minsize=5, maxsize=50),
    max_clock=20000,  # PSB2 problems may need more computation
    generations=200,
    verbose=1
)
```

```python
# Evaluate fitness of specific individuals outside evolution
best_individual = population.get(best_idx)

# Direct evaluation using fitness function callable interface
fitness_value = fitness(lgp_input, best_individual, max_clock=1000)
print(f"Direct fitness evaluation: {fitness_value}")
print(f"Stored fitness: {best_individual.fitness}")

# Program analysis
print(f"Program size: {best_individual.size} instructions")
best_individual.print_program()  # Shows assembly-like representation

# Access program structure
print(f"ROM size: {lgp_input.rom_size} (input features)")
print(f"RAM size: {lgp_input.ram_size} (working memory)")
print(f"Result size: {lgp_input.res_size} (expected outputs)")
print(f"Training samples: {lgp_input.input_num}")

# Population statistics
fitness_values = [population.get(i).fitness for i in range(population.size)]
print(f"Population size: {population.size}")
print(f"Best fitness: {min(fitness_values)}")  # Assuming MINIMIZE fitness
print(f"Average fitness: {sum(fitness_values) / len(fitness_values)}")
```

### Custom Instruction Sets for Different Domains
```python
# For symbolic regression
regression_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.EXP, lgp.Operation.LN,
    lgp.Operation.POW, lgp.Operation.SQRT, lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F,
    lgp.Operation.LOAD_RAM_F, lgp.Operation.MOV_F, lgp.Operation.CMP_F
]

# For digital signal processing
dsp_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN, lgp.Operation.ROUND,
    lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
    lgp.Operation.JMP_Z, lgp.Operation.CMP_F, lgp.Operation.CMOV_L_F
]

# For logical operations and discrete problems
logic_ops = [
    lgp.Operation.AND, lgp.Operation.OR, lgp.Operation.XOR, lgp.Operation.NOT,
    lgp.Operation.SHL, lgp.Operation.SHR, lgp.Operation.CMP, lgp.Operation.TEST,
    lgp.Operation.JMP_Z, lgp.Operation.JMP_NZ, lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    lgp.Operation.CMOV_L, lgp.Operation.CMOV_G, lgp.Operation.MOV
]

# For mixed arithmetic (regression + classification)
mixed_ops = [
    # Floating-point core
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.CMP_F,
    # Integer operations for decisions
    lgp.Operation.ADD, lgp.Operation.SUB, lgp.Operation.CMP, lgp.Operation.CMOV_L,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    # Type conversion
    lgp.Operation.CAST, lgp.Operation.CAST_F,
    # Control flow
    lgp.Operation.JMP_Z, lgp.Operation.JMP_L
]

# Create custom instruction set
custom_instruction_set = lgp.InstructionSet(regression_ops)  # or dsp_ops, logic_ops, mixed_ops
```

## Error Handling and Validation

The Python interface provides comprehensive input validation and error handling:

```python
# Invalid fitness parameters  
try:
    fitness = lgp.HuberLoss(delta=-1.0)  # delta must be positive
except ValueError as e:
    print(f"Error: {e}")

# Invalid threshold values
try:
    fitness = lgp.ThresholdAccuracy(threshold=1.5)  # threshold must be in [0, 1]
except ValueError as e:
    print(f"Error: {e}")

# Invalid evolution configuration
try:
    lgp.evolve(lgp_input)  # Missing initialization AND initial_pop
except ValueError as e:
    print(f"Error: {e}")

# Both initialization and initial_pop provided
try:
    lgp.evolve(lgp_input, initialization=init, initial_pop=pop)  # Only one allowed
except ValueError as e:
    print(f"Error: {e}")

# Invalid output range for fitness function
try:
    fitness = lgp.MSE(start=5, end=3)  # end must be > start
    lgp.evolve(lgp_input, fitness=fitness, initialization=init)
except ValueError as e:
    print(f"Error: {e}")

# Output range exceeds problem size
try:
    fitness = lgp.MSE(start=0, end=10)  # end > res_size
    lgp.evolve(lgp_input, fitness=fitness, initialization=init)  # lgp_input has res_size < 10
except ValueError as e:
    print(f"Error: {e}")

# Mismatched data dimensions
try:
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Wrong number of samples
    lgp.LGPInput.from_numpy(X, y, instruction_set)
except ValueError as e:
    print(f"Error: {e}")

# Invalid ram_size
try:
    lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=0)  # ram_size must be >= res_size
except ValueError as e:
    print(f"Error: {e}")

# Empty instruction set
try:
    empty_set = lgp.InstructionSet([])  # Cannot be empty
except ValueError as e:
    print(f"Error: {e}")

# Invalid initialization parameters
try:
    lgp.UniquePopulation(pop_size=0, minsize=5, maxsize=3)  # maxsize < minsize
except ValueError as e:
    print(f"Error: {e}")

# Perturbation vector size mismatch
try:
    perturbation = np.array([0.1, 0.2])  # Size doesn't match input_num
    fitness = lgp.AdversarialPerturbationSensitivity(perturbation)
    fitness(lgp_input, individual)  # lgp_input.input_num != 2
except ValueError as e:
    print(f"Error: {e}")

# Invalid population access
try:
    population.get(population.size)  # Index out of range
except IndexError as e:
    print(f"Error: {e}")

# Library loading errors
try:
    # If liblgp.so is not found or not built with Python support
    import lgp  # This will fail with OSError and exit
except SystemExit:
    print("Error: Please build the library with 'make python'")
```

## Memory Management and Performance
- **Automatic cleanup**: Python objects handle C memory management automatically via `__del__` methods
- **Thread safety**: All C structures are safe for concurrent read-only access during parallel evaluation
- **NumPy integration**: Arrays are properly copied and aligned to prevent modification during evolution
- **Population management**: Large populations are automatically freed when Python objects are garbage collected
- **Explicit cleanup**: For very large problems, consider explicit cleanup:
  ```python
  del population, lgp_input  # Force garbage collection
  import gc; gc.collect()    # Explicit collection
  ```

### Threading and Parallelization

- **OpenMP parallelization**: Fitness evaluation runs in parallel automatically across available CPU cores
- **Thread control**: Control thread count via environment variable before import:
  ```python
  import os
  os.environ['OMP_NUM_THREADS'] = '8'  # Use 8 threads
  import lgp  # Apply thread setting
  ```
- **Performance constants**: Access parallelization information:
  ```python
  print(f"Available threads: {lgp.NUMBER_OF_OMP_THREADS}")
  print(f"SIMD alignment: {lgp.VECT_ALIGNMENT} bytes")
  print(f"Total VM operations: {lgp.INSTR_NUM}")
  ```

### Performance Tips

- **Instruction set size**: Smaller instruction sets evolve faster but may limit expressiveness
- **Population size**: Balance between diversity (larger) and speed (smaller)
- **Program length**: Shorter programs (`maxsize`) evolve faster but may lack complexity
- **Clock cycles**: Increase `max_clock` for complex problems requiring more computation
- **Fitness evaluation**: Use output range selection (`start`/`end`) to focus on relevant outputs

## Library Dependencies and Requirements

**Required Python packages:**
- `ctypes` (built-in) - C library interface
- `numpy` - numerical arrays and operations
- `typing` (built-in) - type hints
- `enum` (built-in) - enumeration support

**Optional packages:**
- `pandas` - DataFrame support for `LGPInput.from_df()`
- `scikit-learn` - for dataset generation examples

**System requirements:**
- `liblgp.so` - C library must be built with `make python`
- OpenMP support - for parallel fitness evaluation
- x86-64 architecture - for SIMD optimizations

## API Reference Summary

### Core Classes and Structures
- `LGPInput`: Problem definition with unified C structure + Python interface
- `InstructionSet`: VM operations (87 total) with Operation enum integration
- `Individual`: Single evolved program with fitness and program access
- `Population`: Collection of individuals with bounds checking
- `LGPResult`: Evolution results with statistics

### Input Creation
- `LGPInput.from_numpy(X, y, instruction_set, ram_size=None)`: NumPy array input
- `LGPInput.from_df(df, y, instruction_set, ram_size=None)`: Pandas DataFrame input
- `VectorDistance`, `BouncingBalls`, `DiceGame`, `ShoppingList`, `SnowDay`: PSB2 benchmarks

### Fitness Functions (30+ available)
**Regression (MINIMIZE)**: `MSE`, `RMSE`, `MAE`, `MAPE`, `SymmetricMAPE`, `LogCosh`, `WorstCaseError`, `HuberLoss`, `PinballLoss`, `BinaryCrossEntropy`, `GaussianLogLikelihood`, `BrierScore`, `HingeLoss`, `LengthPenalizedMSE`, `ClockPenalizedMSE`, `ConditionalValueAtRisk`, `AdversarialPerturbationSensitivity`

**Statistical (MAXIMIZE)**: `RSquared`, `PearsonCorrelation`

**Classification (MAXIMIZE)**: `Accuracy`, `StrictAccuracy`, `BinaryAccuracy`, `StrictBinaryAccuracy`, `ThresholdAccuracy`, `StrictThresholdAccuracy`, `F1Score`, `FBetaScore`, `MatthewsCorrelation`, `BalancedAccuracy`, `GMean`, `CohensKappa`

### Selection Methods
- **Basic**: `Tournament`, `Elitism`, `PercentualElitism`, `Roulette`
- **Fitness sharing**: `FitnessSharingTournament`, `FitnessSharingElitism`, `FitnessSharingPercentualElitism`, `FitnessSharingRoulette`

### Initialization Methods
- `UniquePopulation`: Ensures diversity, avoids duplicates
- `RandPopulation`: Completely random, allows duplicates

### Utility Functions
- `evolve()`: Main evolution function with comprehensive parameter control
- `print_program()`: Display program instructions in assembly-like format
- `random_init_all()`: Initialize all thread-local random number generators

### Constants and Configuration
- `NUMBER_OF_OMP_THREADS`: Available parallel threads
- `VECT_ALIGNMENT`: SIMD alignment in bytes
- `INSTR_NUM`: Total VM operations (87)
