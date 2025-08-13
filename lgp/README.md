# Linear Genetic Programming - Python Interface

A comprehensive, high-level Python wrapper for the Linear Genetic Programming C library. This interface provides unified classes that combine C structures with user-friendly Python abstractions, comprehensive fitness functions, automatic memory management, and seamless integration with NumPy and Pandas.

## Overview

The Python interface is designed to make LGP accessible while maintaining the performance of the underlying C implementation. Key features include:

- **Type-safe wrapper classes** - Direct mapping to C structures with Python convenience
- **Comprehensive input validation** - Automatic error checking and bounds validation
- **Scientific computing integration** - Native NumPy arrays and Pandas DataFrame support
- **Memory management** - Automatic cleanup and garbage collection
- **Performance optimization** - Efficient data transfer and minimal overhead
- **Complete API coverage** - Access to all C library functionality

## Installation

### Prerequisites
- **Python 3.8+** with NumPy
- **Compiled C library** - `liblgp.so` must be built with `make python`
- **Optional**: Pandas for DataFrame support

### Setup
```bash
# Build the shared library
make python

# Test installation
python3 -c "import lgp; print('LGP imported successfully')"
```

The library automatically loads `liblgp.so` from the current directory and initializes all thread-local random number generators with seed 0.

## Quick Start

```python
import lgp
import numpy as np

# Generate test data for symbolic regression: y = x² + 2x + 1
X = np.random.uniform(-3, 3, (200, 1))
y = X[:, 0]**2 + 2*X[:, 0] + 1

# Create optimized instruction set for symbolic regression
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, 
    lgp.Operation.DIV_F, lgp.Operation.POW, lgp.Operation.SQRT,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, 
    lgp.Operation.MOV_F, lgp.Operation.CMP_F
])

# Create LGP input from NumPy arrays
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=5)

# Configure and execute evolution
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,
    fitness=lgp.MSE(),                              # Mean Squared Error
    selection=lgp.Tournament(tournament_size=4),    # Tournament selection
    initialization=lgp.UniquePopulation(            # Ensure diversity
        pop_size=150, minsize=5, maxsize=25
    ),
    target=1e-6,                                    # Stop when MSE < 1e-6
    mutation_prob=0.8,                              # High mutation rate
    crossover_prob=0.95,                            # High crossover rate
    max_clock=5000,                                 # VM execution limit
    generations=50,                                 # Maximum generations
    verbose=1                                       # Show progress
)

# Analyze results
best = population.get(best_idx)
print(f"Best fitness: {best.fitness:.6e}")
print(f"Program size: {best.size} instructions")
print(f"Total evaluations: {evaluations}")

# Display evolved program
best.print_program()
```

## Core Classes

### LGPInput
The central class representing evolutionary problems with training data, expected outputs, and execution environment. Acts as a bridge between Python data structures and C memory layout.

#### Creation Methods

**From NumPy Arrays:**
```python
# Single output regression (must reshape y to 2D)
X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape: (n_samples, n_features) 
y = np.array([3.0, 7.0])                 # Shape: (n_samples,)
y_reshaped = y.reshape(-1, 1)            # Required: (n_samples, n_outputs)

# Multi-output problems
y_multi = np.array([[3.0, 1.0], [7.0, 2.0]])  # Shape: (n_samples, n_outputs)

lgp_input = lgp.LGPInput.from_numpy(X, y_reshaped, instruction_set, ram_size=10)
```

**From Pandas DataFrame:**
```python
import pandas as pd

df = pd.DataFrame({
    'x1': [1, 3, 5], 
    'x2': [2, 4, 6], 
    'target': [5, 11, 17]
})

# Automatically splits features and target
lgp_input = lgp.LGPInput.from_df(df, 'target', instruction_set, ram_size=8)

# Multi-target support
lgp_input = lgp.LGPInput.from_df(df, ['target1', 'target2'], instruction_set)
```

#### PSB2 Benchmark Problems
Pre-configured problems from the Program Synthesis Benchmark Suite 2:

```python
# Vector distance calculation - Euclidean distance between n-dimensional vectors
lgp_input = lgp.VectorDistance(instruction_set, vector_len=3, instances=200)

# Physics simulation problems
lgp_input = lgp.BouncingBalls(instruction_set, instances=150)  # Ball trajectory with gravity
lgp_input = lgp.SnowDay(instruction_set, instances=100)       # Weather accumulation modeling

# Optimization and game theory
lgp_input = lgp.DiceGame(instruction_set, instances=300)      # Optimal dice strategies
lgp_input = lgp.ShoppingList(instruction_set, num_of_items=5, instances=200)  # Budget optimization
```

#### Memory Layout and Architecture
- **`input_num`**: Number of training samples in the dataset
- **`rom_size`**: Number of input features per sample (read-only memory)
- **`res_size`**: Number of output values per sample (expected results)
- **`ram_size`**: Working memory size (must be ≥ `res_size`, default: max(8, res_size))

**Memory Organization Pattern:**
Each sample follows the layout: `[ROM: input_features][RAM: expected_outputs + workspace]`
- Programs read input data from ROM positions 0 to `rom_size-1`
- Programs write outputs to RAM positions 0 to `res_size-1`
- Remaining RAM positions serve as working memory for intermediate calculations

### InstructionSet
Configures the VM operations available during program evolution. The instruction set significantly impacts both evolution speed and solution quality.

#### Complete Instruction Set
```python
# All 87 VM operations (comprehensive but slow evolution)
instruction_set = lgp.InstructionSet.complete()
```

#### Custom Instruction Sets
```python
# Minimal floating-point set for fast evolution
basic_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.MOV_F
]

# Advanced mathematical functions for symbolic regression
math_ops = basic_ops + [
    lgp.Operation.SQRT, lgp.Operation.POW, lgp.Operation.EXP, lgp.Operation.LN,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN,
    lgp.Operation.CMP_F, lgp.Operation.JMP_L, lgp.Operation.JMP_G
]

# Mixed integer/float operations for classification
mixed_ops = [
    # Floating-point for feature processing
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.CMP_F,
    
    # Integer operations for classification decisions
    lgp.Operation.ADD, lgp.Operation.SUB, lgp.Operation.CMP, lgp.Operation.CMOV_L,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    
    # Control flow for complex logic
    lgp.Operation.JMP_Z, lgp.Operation.JMP_L, lgp.Operation.JMP_G
]

instruction_set = lgp.InstructionSet(math_ops)
```

#### Operation Categories
- **Arithmetic**: ADD/SUB/MUL/DIV (integer and floating-point variants)
- **Mathematical**: SQRT, POW, EXP, LN, LOG, trigonometric and hyperbolic functions
- **Memory**: LOAD_ROM/RAM, STORE_RAM (multiple addressing modes)
- **Control Flow**: Conditional and unconditional jumps, conditional moves
- **Logic**: AND, OR, XOR, NOT, bit shifts
- **Utility**: CAST (type conversion), NOP, RAND, ROUND

### Fitness Functions
Comprehensive collection of 30+ fitness functions for regression and classification problems with different data type expectations and vectorial output support.

#### Data Type Classifications
Each fitness function expects specific data types and interprets program outputs differently:

**[FLOAT] Functions**: Expect **arbitrary floating-point values**
- Program outputs compared directly as `float64` values
- No constraints on output range - can be any real number  
- Used for: regression, continuous prediction, mathematical modeling

**[INT] Functions**: Expect **exact integer matches**
- Program outputs compared directly as `int64` values
- Used for: discrete classification, counting problems, symbolic matching

**[SIGN_BIT] Functions**: **Binary classification via sign bit interpretation**
- Program outputs interpreted as: negative integers → class 0/false, positive integers → class 1/true
- Target data should use same encoding: negative for class 0, positive for class 1
- Used for: binary classification, boolean function learning, decision problems

**[PROB] Functions**: Expect **probability values in [0,1] range**
- Program outputs must be probabilities between 0.0 and 1.0
- Target values typically binary (0.0, 1.0) or probability distributions
- Used for: probabilistic classification, uncertainty quantification

#### Regression Functions (MINIMIZE) [FLOAT]
```python
# Basic error metrics - expect any floating-point outputs
fitness = lgp.MSE(start=0, end=1)                    # Mean Squared Error
fitness = lgp.RMSE(start=0, end=2)                   # Root Mean Squared Error  
fitness = lgp.MAE(start=0, end=1)                    # Mean Absolute Error
fitness = lgp.MAPE(start=0, end=1)                   # Mean Absolute Percentage Error
fitness = lgp.SYMMETRIC_MAPE(start=0, end=1)         # Symmetric MAPE variant
fitness = lgp.LOGCOSH(start=0, end=1)                # Smooth approximation of MAE
fitness = lgp.WORST_CASE_ERROR(start=0, end=1)       # Maximum error (robustness)
fitness = lgp.HINGE_LOSS(start=0, end=1)             # SVM loss function

# Robust regression with parameters
fitness = lgp.HUBER_LOSS(delta=1.5, start=0, end=1)  # Robust loss function
fitness = lgp.PINBALL_LOSS(quantile=0.9, start=0, end=1)  # Quantile regression
fitness = lgp.GAUSSIAN_LOG_LIKELIHOOD(sigma=1.0, start=0, end=1)  # MLE estimation

# Regularized variants for model complexity control
fitness = lgp.LENGTH_PENALIZED_MSE(alpha=0.01, start=0, end=1)  # Penalize long programs
fitness = lgp.CLOCK_PENALIZED_MSE(alpha=0.001, start=0, end=1)   # Penalize slow programs
```

#### Regression Functions (MAXIMIZE) [FLOAT]
```python
# Correlation-based metrics - expect any floating-point outputs
fitness = lgp.R_SQUARED(start=0, end=2)              # Coefficient of determination
fitness = lgp.PEARSON_CORRELATION(start=0, end=1)    # Statistical correlation
```

#### Classification Functions (MAXIMIZE) [INT]
```python
# Integer-based classification with exact matching
y_int = np.array([0, 1, 2, 1, 0], dtype=np.int64)   # Integer class labels

fitness = lgp.ACCURACY(start=0, end=3)               # Per-label accuracy (multi-label)
fitness = lgp.STRICT_ACCURACY(start=0, end=3)        # Exact vector match per sample
fitness = lgp.BINARY_ACCURACY(start=0, end=1)        # Optimized binary classification
fitness = lgp.STRICT_BINARY_ACCURACY(start=0, end=1) # Strict binary with exact matching
```

#### Classification Functions (MAXIMIZE) [SIGN_BIT]
```python
# Binary classification using sign bit interpretation
# Prepare targets: negative integers = class 0, positive integers = class 1
y_bool = np.array([True, False, True, False, True])  # Original boolean data
y_sign = np.where(y_bool, 1, -1).astype(np.int64)   # Convert to sign-bit encoding

fitness = lgp.F1_SCORE(start=0, end=2)               # Harmonic mean of precision/recall
fitness = lgp.F_BETA_SCORE(beta=2.0, start=0, end=1) # Weighted F-score
fitness = lgp.BALANCED_ACCURACY(start=0, end=1)      # Handles class imbalance
fitness = lgp.G_MEAN(start=0, end=1)                 # Geometric mean of sensitivity/specificity
fitness = lgp.MATTHEWS_CORRELATION(start=0, end=1)   # Balanced metric for binary classification
fitness = lgp.COHENS_KAPPA(start=0, end=1)           # Inter-rater agreement
```

#### Classification Functions (MAXIMIZE) [FLOAT]
```python
# Threshold-based classification with floating-point outputs
fitness = lgp.THRESHOLD_ACCURACY(threshold=0.5, start=0, end=1)        # Tolerance-based accuracy
fitness = lgp.STRICT_THRESHOLD_ACCURACY(threshold=0.1, start=0, end=1) # Strict threshold matching
```

#### Probabilistic Functions [PROB]
```python
# Functions expecting probability outputs in [0,1] range
y_prob = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)  # Binary probabilities

fitness = lgp.BINARY_CROSS_ENTROPY(tolerance=1e-10, start=0, end=1)  # MINIMIZE
fitness = lgp.BRIER_SCORE(start=0, end=1)                            # MINIMIZE - probabilistic accuracy
```

#### Specialized Functions
```python
# Robustness and risk measures [FLOAT]
perturbation = np.array([0.1, 0.05, 0.2])  # Must match input_num
fitness = lgp.ADVERSARIAL_PERTURBATION_SENSITIVITY(perturbation, start=0, end=1)  # MINIMIZE

fitness = lgp.CONDITIONAL_VALUE_AT_RISK(alpha=0.05, start=0, end=1)  # MINIMIZE - financial risk
```

#### Data Type Usage Examples
```python
import numpy as np

# FLOAT regression example
X_reg = np.random.randn(1000, 3)
y_reg = X_reg[:, 0]**2 + np.sin(X_reg[:, 1])  # Any real numbers
lgp_input_reg = lgp.LGPInput.from_numpy(X_reg, y_reg.reshape(-1, 1), instruction_set)
fitness_reg = lgp.MSE(start=0, end=1)

# SIGN_BIT binary classification example  
X_bin = np.random.randn(1000, 3)
y_bool = (X_bin[:, 0] + X_bin[:, 1] > 0)      # Boolean outcomes
y_sign = np.where(y_bool, 1, -1).astype(np.int64).reshape(-1, 1)  # Sign-bit encoding
lgp_input_bin = lgp.LGPInput.from_numpy(X_bin, y_sign, instruction_set)
fitness_bin = lgp.F1_SCORE(start=0, end=1)

# INT discrete classification example
X_cat = np.random.randn(1000, 3)  
y_cat = np.random.choice([0, 1, 2], size=1000).astype(np.int64).reshape(-1, 1)  # Integer classes
lgp_input_cat = lgp.LGPInput.from_numpy(X_cat, y_cat, instruction_set)
fitness_cat = lgp.ACCURACY(start=0, end=1)

# PROB probabilistic classification example
X_prob = np.random.randn(1000, 3)
y_prob = np.random.choice([0.0, 1.0], size=1000).reshape(-1, 1)  # Binary probabilities
lgp_input_prob = lgp.LGPInput.from_numpy(X_prob, y_prob, instruction_set)
fitness_prob = lgp.BINARY_CROSS_ENTROPY(tolerance=1e-10, start=0, end=1)
```

#### Fitness Parameters and Output Range Selection
All fitness functions support configurable output ranges for multi-output problems:

```python
# Evaluate only first output (RAM position 0)
fitness = lgp.MSE(start=0, end=1)

# Evaluate outputs at positions 0, 1, and 2
fitness = lgp.MSE(start=0, end=3)

# Evaluate outputs at positions 2 and 3 (useful for multi-stage problems)
fitness = lgp.MSE(start=2, end=4)

# Direct fitness evaluation on specific individual
fitness_value = fitness(lgp_input, individual, max_clock=5000)
```

### Selection Methods
Eight selection algorithms with advanced diversity preservation through fitness sharing:

#### Basic Selection Methods
```python
# Tournament selection - balanced exploration/exploitation
selection = lgp.Tournament(tournament_size=4)        # Most common choice

# Elitism - preserve best individuals
selection = lgp.Elitism(elite_size=20)               # Keep 20 best individuals
selection = lgp.PercentualElitism(elite_percentage=0.1)  # Keep top 10%

# Roulette wheel - probability proportional to fitness
selection = lgp.Roulette(sampling_size=100)          # Sample 100 individuals
```

#### Fitness Sharing Variants
Advanced selection methods that promote diversity by penalizing similar individuals:

```python
# Tournament with diversity preservation
selection = lgp.FitnessSharingTournament(
    tournament_size=4, 
    alpha=1.0,      # Sharing function exponent
    beta=1.0,       # Fitness scaling power
    sigma=0.1       # Sharing radius (problem-dependent)
)

# Elitism with niching behavior
selection = lgp.FitnessSharingElitism(
    elite_size=15, alpha=1.0, beta=1.0, sigma=0.15
)

selection = lgp.FitnessSharingPercentualElitism(
    elite_percentage=0.05, alpha=1.0, beta=1.0, sigma=0.1
)

# Roulette wheel with diversity
selection = lgp.FitnessSharingRoulette(
    sampling_size=80, alpha=1.0, beta=1.0, sigma=0.12
)
```

**Fitness Sharing Parameters:**
- **`alpha`**: Sharing function exponent (typically 1.0, higher values increase diversity pressure)
- **`beta`**: Fitness scaling power (typically 1.0, affects selection pressure)
- **`sigma`**: Sharing radius (problem-specific, typically 0.05-0.2, smaller values reduce sharing effect)

### Population Initialization
Two initialization strategies with different trade-offs between speed and diversity:

```python
# Unique population - ensures genetic diversity (recommended)
initialization = lgp.UniquePopulation(
    pop_size=200,     # Population size
    minsize=5,        # Minimum program length
    maxsize=30        # Maximum program length
)

# Random population - faster generation, may contain duplicates
initialization = lgp.RandPopulation(
    pop_size=150,     # Population size  
    minsize=3,        # Minimum program length
    maxsize=25        # Maximum program length
)
```

**Design Considerations:**
- **Unique Population**: Uses hash-based deduplication to ensure all individuals are genetically different, preventing premature convergence but requiring more computation time
- **Random Population**: Generates individuals without checking for duplicates, faster but may reduce initial diversity

## Evolution Function

The main evolution function provides comprehensive control over the evolutionary process:

```python
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,                                       # Problem definition
    fitness=lgp.MSE(),                              # Fitness function
    selection=lgp.Tournament(tournament_size=3),    # Selection method
    initialization=lgp.UniquePopulation(100, 5, 20), # Population initialization
    target=1e-6,                                    # Target fitness for early stop
    mutation_prob=0.8,                              # Mutation probability
    crossover_prob=0.95,                            # Crossover probability
    max_clock=5000,                                 # VM execution limit
    max_individ_len=50,                             # Maximum program length
    max_mutation_len=8,                             # Maximum mutation segment length
    generations=100,                                # Maximum generations
    verbose=1                                       # Progress reporting
)
```

### Parameters and Advanced Configuration

**Genetic Operator Probabilities:**
- **Values 0.0-1.0**: Standard probability interpretation
- **Values > 1.0**: Enable multiple applications per individual
  - `mutation_prob=1.5` → 1 guaranteed mutation + 50% chance for second mutation
  - `crossover_prob=2.3` → 2 guaranteed crossovers + 30% chance for third crossover

**Resource Control:**
- **`max_clock`**: Prevents infinite loops by limiting VM execution cycles per program
- **`max_individ_len`**: Controls program bloat by limiting maximum program size
- **`max_mutation_len`**: Limits the size of mutation segments to control disruption

**Early Termination:**
- **`target`**: Evolution stops automatically when best fitness reaches this value
- Useful for problems with known optimal solutions or when sufficient accuracy is achieved

### Return Values
```python
population, evaluations, generations, best_idx = lgp.evolve(...)

# Access best individual
best_individual = population.get(best_idx)
print(f"Best fitness: {best_individual.fitness}")
print(f"Program size: {best_individual.size} instructions")

# Population analysis
fitness_values = [population.get(i).fitness for i in range(population.size)]
print(f"Population diversity: {len(set(fitness_values))} unique fitness values")
```

## Advanced Usage Examples

### Multi-Output Regression
```python
# Generate multi-output regression data
X = np.random.uniform(-2, 2, (150, 2))
y = np.column_stack([
    X[:, 0]**2 + X[:, 1],           # Output 1: x1² + x2
    np.sin(X[:, 0]) * X[:, 1]       # Output 2: sin(x1) * x2
])

# Create LGP input with sufficient working memory
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=8)

# Configure fitness to evaluate both outputs
fitness = lgp.MSE(start=0, end=2)  # Evaluate RAM positions 0 and 1

population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    initialization=lgp.UniquePopulation(pop_size=250, minsize=8, maxsize=35),
    selection=lgp.Tournament(tournament_size=5),
    target=1e-4,
    generations=80,
    max_clock=8000,                 # Increased for complex multi-output programs
    verbose=1
)
```

### Classification with Advanced Metrics
```python
from sklearn.datasets import make_classification

# Generate multi-class classification data
X, y = make_classification(n_samples=400, n_features=6, n_classes=3, 
                          n_informative=4, random_state=42)

# Convert to LGP format (reshape for single output)
y_reshaped = y.reshape(-1, 1).astype(float)

# Classification instruction set combining float and integer operations
classification_ops = [
    # Floating-point for feature processing
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.CMP_F,
    
    # Integer operations for final decisions
    lgp.Operation.ADD, lgp.Operation.SUB, lgp.Operation.CMP, 
    lgp.Operation.CMOV_L, lgp.Operation.CMOV_G,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    
    # Control flow for decision logic
    lgp.Operation.JMP_Z, lgp.Operation.JMP_L, lgp.Operation.JMP_G
]

instruction_set = lgp.InstructionSet(classification_ops)
lgp_input = lgp.LGPInput.from_numpy(X, y_reshaped, instruction_set, ram_size=12)

# Use balanced accuracy for class imbalance robustness
fitness = lgp.BalancedAccuracy(start=0, end=1)

population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    selection=lgp.FitnessSharingTournament(
        tournament_size=4, alpha=1.0, beta=1.0, sigma=0.1
    ),
    initialization=lgp.UniquePopulation(pop_size=300, minsize=10, maxsize=40),
    target=0.95,                    # Stop at 95% balanced accuracy
    mutation_prob=1.2,              # Multiple mutations for exploration
    crossover_prob=0.9,
    max_clock=6000,
    generations=120,
    verbose=1
)
```

### PSB2 Benchmark Evaluation
```python
# Comprehensive PSB2 benchmark testing
benchmark_problems = [
    ('Vector Distance', lgp.VectorDistance(instruction_set, vector_len=4, instances=200)),
    ('Bouncing Balls', lgp.BouncingBalls(instruction_set, instances=150)),
    ('Dice Game', lgp.DiceGame(instruction_set, instances=300)),
    ('Shopping List', lgp.ShoppingList(instruction_set, num_of_items=6, instances=180)),
    ('Snow Day', lgp.SnowDay(instruction_set, instances=120))
]

results = {}
for problem_name, lgp_input in benchmark_problems:
    print(f"\n=== {problem_name} ===")
    
    population, evaluations, generations, best_idx = lgp.evolve(
        lgp_input=lgp_input,
        fitness=lgp.MSE(),
        selection=lgp.Tournament(tournament_size=4),
        initialization=lgp.UniquePopulation(pop_size=200, minsize=5, maxsize=30),
        target=1e-5,
        mutation_prob=0.8,
        crossover_prob=0.95,
        max_clock=5000,
        generations=60,
        verbose=1
    )
    
    best_individual = population.get(best_idx)
    results[problem_name] = {
        'fitness': best_individual.fitness,
        'evaluations': evaluations,
        'generations': generations,
        'program_size': best_individual.size
    }
    
    print(f"Final fitness: {best_individual.fitness:.6e}")
    print(f"Program size: {best_individual.size} instructions")

# Performance summary
print("\n=== Benchmark Summary ===")
for problem, result in results.items():
    print(f"{problem}: fitness={result['fitness']:.2e}, "
          f"evals={result['evaluations']}, size={result['program_size']}")
```

### Custom Instruction Sets for Different Domains
```python
# Symbolic regression with advanced mathematics
regression_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.SQRT, lgp.Operation.POW, lgp.Operation.EXP, lgp.Operation.LN,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.STORE_RAM_F, 
    lgp.Operation.LOAD_RAM_F, lgp.Operation.MOV_F, lgp.Operation.CMP_F
]

# Digital signal processing
dsp_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    lgp.Operation.SIN, lgp.Operation.COS, lgp.Operation.TAN,
    lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
    lgp.Operation.ROUND, lgp.Operation.CMP_F,
    lgp.Operation.JMP_L, lgp.Operation.JMP_G
]

# Logic and decision making
logic_ops = [
    lgp.Operation.AND, lgp.Operation.OR, lgp.Operation.XOR, lgp.Operation.NOT,
    lgp.Operation.SHL, lgp.Operation.SHR, lgp.Operation.CMP,
    lgp.Operation.CMOV_L, lgp.Operation.CMOV_G, lgp.Operation.CMOV_Z,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    lgp.Operation.JMP_Z, lgp.Operation.JMP_NZ
]

# Mixed-type operations for complex problems
mixed_ops = regression_ops + [
    # Integer operations for decisions
    lgp.Operation.ADD, lgp.Operation.SUB, lgp.Operation.CMP, lgp.Operation.CMOV_L,
    lgp.Operation.LOAD_ROM, lgp.Operation.STORE_RAM,
    # Type conversion
    lgp.Operation.CAST, lgp.Operation.CAST_F,
    # Additional control flow
    lgp.Operation.JMP_Z, lgp.Operation.JMP_NZ
]

# Create domain-specific instruction set
custom_instruction_set = lgp.InstructionSet(regression_ops)  # or dsp_ops, logic_ops, mixed_ops
```

## Error Handling and Validation

The Python interface provides comprehensive error checking and validation:

### Input Validation Examples
```python
try:
    # Mismatched array dimensions
    X = np.random.rand(100, 3)
    y = np.random.rand(50, 1)  # Wrong number of samples
    lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set)
except ValueError as e:
    print(f"Input validation error: {e}")

try:
    # Invalid RAM size (too small for results)
    lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=0)  # ram_size must be >= res_size
except ValueError as e:
    print(f"Memory configuration error: {e}")

try:
    # Empty instruction set
    empty_set = lgp.InstructionSet([])  # Cannot be empty
except ValueError as e:
    print(f"Instruction set error: {e}")

try:
    # Invalid initialization parameters
    initialization = lgp.UniquePopulation(pop_size=0, minsize=5, maxsize=3)  # maxsize < minsize
except ValueError as e:
    print(f"Initialization error: {e}")

try:
    # Perturbation vector size mismatch
    perturbation = np.array([0.1, 0.2])  # Size doesn't match input_num
    fitness = lgp.AdversarialPerturbationSensitivity(perturbation)
    fitness_value = fitness(lgp_input, individual)  # lgp_input.input_num != 2
except ValueError as e:
    print(f"Fitness parameter error: {e}")
```

### Runtime Error Handling
```python
try:
    # Evolution with problematic parameters
    population, evaluations, generations, best_idx = lgp.evolve(
        lgp_input=lgp_input,
        fitness=lgp.MSE(start=0, end=10),  # end > res_size
        selection=lgp.Tournament(tournament_size=3),
        initialization=lgp.UniquePopulation(pop_size=50, minsize=2, maxsize=15),
        generations=30
    )
except RuntimeError as e:
    print(f"Evolution runtime error: {e}")
except ValueError as e:
    print(f"Evolution parameter error: {e}")
```

## Memory Management and Performance

### Automatic Memory Management
The Python interface handles all memory management automatically:

```python
# Objects are automatically cleaned up when they go out of scope
def run_evolution():
    lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set)
    population, evaluations, generations, best_idx = lgp.evolve(lgp_input, ...)
    return population.get(best_idx)  # Safe to return, memory managed automatically

best_program = run_evolution()  # No manual cleanup required
```

### Performance Optimization Tips

**Instruction Set Size:**
- Smaller instruction sets evolve faster but may limit solution expressiveness
- Start with minimal sets and add complexity as needed

**Population Parameters:**
- Balance population size between diversity (larger) and speed (smaller)
- Typical ranges: 50-500 individuals depending on problem complexity

**Program Length Control:**
- Shorter maximum program lengths (`maxsize`) evolve faster
- Longer programs may find more complex solutions but require more computation

**VM Execution Limits:**
- Increase `max_clock` for complex problems requiring more computation steps
- Monitor execution time vs. solution quality trade-offs

**Output Range Selection:**
- Use precise `start`/`end` ranges in fitness functions to focus evaluation
- Avoid evaluating unnecessary outputs to improve performance

### Threading and Parallelization
```python
# The library automatically uses all available CPU cores
print(f"Available OpenMP threads: {lgp.NUMBER_OF_OMP_THREADS}")

# Random number generation is thread-safe and deterministic
lgp.random_init_all(42)  # Same seed produces identical results
```

## Library Dependencies and Requirements

### Required Python Packages
- **`ctypes`** (built-in) - C library interface and foreign function library
- **`numpy`** - Numerical arrays and mathematical operations
- **`typing`** (built-in) - Type hints and annotations
- **`enum`** (built-in) - Enumeration support for Operation types

### Optional Packages
- **`pandas`** - DataFrame support for `LGPInput.from_df()` method
- **`scikit-learn`** - Dataset generation utilities for examples and testing

### System Requirements
- **`liblgp.so`** - C shared library built with `make python`
- **OpenMP support** - For parallel fitness evaluation (optional but recommended)
- **x86-64 or ARM architecture** - For SIMD optimizations

### Platform Compatibility
- **Linux**: Full support with GCC or Clang
- **macOS**: Full support with Xcode Command Line Tools
- **Windows**: Supported with MinGW-w64 or Visual Studio
- **FreeBSD**: Supported with system compiler

## API Reference Summary

### Core Classes and Structures
- **`LGPInput`**: Problem definition combining C structures with Python interface
- **`InstructionSet`**: VM operations configuration with 87 available operations
- **`Individual`**: Single evolved program with fitness evaluation and program access
- **`Population`**: Collection of individuals with bounds-checked access methods
- **`LGPResult`**: Evolution results containing population and performance statistics

### Input Creation Methods
- **`LGPInput.from_numpy(X, y, instruction_set, ram_size=None)`**: Create from NumPy arrays
- **`LGPInput.from_df(df, y_cols, instruction_set, ram_size=None)`**: Create from Pandas DataFrame
- **PSB2 Benchmarks**: `VectorDistance`, `BouncingBalls`, `DiceGame`, `ShoppingList`, `SnowDay`

### Fitness Functions (30+ Available)
**Regression (MINIMIZE)**: MSE, RMSE, MAE, HuberLoss, PinballLoss, LogCosh, WorstCaseError, BinaryCrossEntropy, GaussianLogLikelihood, BrierScore, HingeLoss, LengthPenalizedMSE, ClockPenalizedMSE

**Regression (MAXIMIZE)**: RSquared, PearsonCorrelation

**Classification (MAXIMIZE)**: Accuracy, StrictAccuracy, BinaryAccuracy, ThresholdAccuracy, BalancedAccuracy, F1Score, FBetaScore, MatthewsCorrelation, CohensKappa, GMean

**Specialized**: AdversarialPerturbationSensitivity, ConditionalValueAtRisk

### Selection Methods
- **Basic**: `Tournament`, `Elitism`, `PercentualElitism`, `Roulette`
- **Fitness Sharing**: `FitnessSharingTournament`, `FitnessSharingElitism`, `FitnessSharingPercentualElitism`, `FitnessSharingRoulette`

### Initialization Methods
- **`UniquePopulation`**: Ensures diversity through hash-based deduplication
- **`RandPopulation`**: Fast random generation allowing duplicates

### Utility Functions
- **`evolve()`**: Main evolution function with comprehensive parameter control
- **`print_program()`**: Display program instructions in human-readable assembly format
- **`random_init_all(seed)`**: Initialize all thread-local random number generators

### Constants and Configuration
- **`NUMBER_OF_OMP_THREADS`**: Available parallel processing threads
- **`VECT_ALIGNMENT`**: SIMD memory alignment requirements (16, 32, or 64 bytes)
- **`INSTR_NUM`**: Total number of available VM operations (87)

## Best Practices and Recommendations

### Problem Setup
1. **Start Simple**: Begin with minimal instruction sets and small populations
2. **Validate Data**: Ensure input arrays have correct shapes and data types
3. **Choose Appropriate Fitness**: Match fitness function to problem type and requirements
4. **Scale Features**: Normalize input features for better evolution performance

### Evolution Configuration
1. **Population Size**: Start with 100-200 individuals, increase for complex problems
2. **Program Length**: Begin with short programs (5-20 instructions), expand as needed
3. **Selection Pressure**: Use tournament sizes 3-5 for balanced exploration/exploitation
4. **Genetic Operators**: High mutation rates (0.7-0.9) often work well for LGP

### Performance Optimization
1. **Instruction Set Design**: Include only necessary operations to speed evolution
2. **Resource Limits**: Set appropriate `max_clock` values based on problem complexity
3. **Early Stopping**: Use realistic target fitness values for automatic termination
4. **Parallel Evaluation**: Ensure OpenMP is available for maximum performance

### Debugging and Analysis
1. **Verbose Output**: Use `verbose=1` to monitor evolution progress
2. **Program Inspection**: Use `print_program()` to understand evolved solutions
3. **Population Analysis**: Examine fitness distributions and program sizes
4. **Validation Testing**: Test evolved programs on held-out datasets
