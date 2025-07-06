# Python Interface Documentation - Linear Genetic Programming (LGP)

This Python interface provides a complete and high-performance binding for the LGP C library, designed to maintain the original C project nomenclature while ensuring an idiomatic and type-safe Python API.

## 📚 Table of Contents
1. [Architecture and Design](#architecture-and-design)
2. [Installation and Setup](#installation-and-setup)
3. [API Reference](#api-reference)
4. [Complete Examples](#complete-examples)
5. [Fitness Functions](#fitness-functions)
6. [Selection Methods](#selection-methods)
7. [Troubleshooting](#troubleshooting)
8. [Performance and Optimization](#performance-and-optimization)
9. [Extensions and Customization](#extensions-and-customization)

## Architecture and Design

### Design Principles
- **1:1 Mapping** with underlying C structures for maximum compatibility
- **Identical Nomenclature** to all C project structs, unions, and enums
- **Object-Oriented Structure** with classes for all components except the 3 main functions
- **Type Safety** with complete type annotations and runtime validation
- **Zero Overhead** - direct access to C structures without unnecessary copies
- **Automatic Memory Management** via ctypes with deterministic cleanup

### Naming Conventions
- **Wrapper Classes**: All classes wrapping C struct/union end with "Wrapper"
  - `IndividualWrapper`, `PopulationWrapper`, `OperationWrapper`, etc.
- **High-Level Classes**: Python classes using wrappers without suffix
  - `Individual`, `Population`, `LGPInput`, etc.
- **`c_wrapper` Property**: All classes with C representation expose this property
- **Enum Operations**: 87 operations VM with identical C names (e.g., `ADD_F`, `SUB_F`, `SQRT`)

### Layered Structure
1. **Base Layer** (`base.py`): C library loading, common types, error handling
2. **Wrapper Layer**: Direct mapping of C structures with validation
3. **High-Level Layer**: User-friendly Python API with helper methods
4. **Public API Layer**: Interface exposed through `__init__.py`

## Installation and Setup

### Prerequisites
- **Python 3.7+** with ctypes support
- **GCC 7+** or **Clang 10+** with OpenMP support
- **Make** for the build system
- **NumPy** (optional but recommended for performance with large arrays)
- **Pandas** (optional, required only for `LGPInput.from_df()`)

### Compilation
```bash
# Compile C library and generate liblgp.so
make clean
make python

# Verify that liblgp.so was created
ls -la liblgp.so

# Basic test
python3 -c "import lgp; print('✓ LGP installed correctly')"
```

### Complete Installation Verification
```python
import lgp
import numpy as np

# Test basic functionalities
print(f"✓ LGP version {lgp.__version__}")
print(f"✓ {len(lgp.Operation)} VM operations available")
print(f"✓ Test operation: {lgp.Operation.ADD_F.name()} (code: {lgp.Operation.ADD_F.code()})")

# Test c_wrapper access
op = lgp.Operation.SQRT
print(f"✓ c_wrapper test: {op.c_wrapper.regs} registers, addr={op.c_wrapper.addr}")
```

## Quick Usage

```python
import lgp
import numpy as np

# Random number generator initialization
lgp.random_init(42, 1)

# Sample data creation (function x^2 + 2*x + 1)
X = np.random.uniform(-5, 5, (200, 1))
y = X[:, 0]**2 + 2*X[:, 0] + 1

# Instruction set creation
instruction_set = lgp.InstructionSet([
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F,
    lgp.Operation.DIV_F, lgp.Operation.LOAD_RAM_F, lgp.Operation.MOV_F
])

# LGP input creation
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=10)

# Evolution
population, evaluations, generations, best_idx = lgp.evolve(
    lgp_input,
    fitness=lgp.MSE(),
    selection=lgp.Tournament(3),
    initialization=lgp.UniquePopulation(),
    init_params=(100, 5, 20),
    generations=50,
    verbose=1
)

# Results
best_individual = population.get(best_idx)
print(f"Best fitness: {best_individual.fitness}")
lgp.print_program(best_individual)
```

## Complete API Reference

### 🔧 Main Functions (3 global functions)

#### `evolve(lgp_input, **kwargs) -> Tuple[Population, int, int, int]`
Executes the evolution of an LGP population.

**Parameters:**
- `lgp_input: LGPInput` - Problem input (data and instruction set)
- `fitness: FitnessAssessment = MSE()` - Fitness function to optimize
- `selection: Selection = Tournament(3)` - Population selection method
- `initialization: Initialization = UniquePopulation()` - Initialization method
- `init_params: Tuple[int, int, int] = (1000, 2, 5)` - (pop_size, min_size, max_size)
- `target: float = 1e-27` - Target fitness value for early termination
- `mutation_prob: float = 0.76` - Mutation probability per individual
- `crossover_prob: float = 0.95` - Crossover probability per pair
- `max_clock: int = 5000` - Maximum VM clock cycles per program
- `max_individ_len: int = 50` - Maximum individual length
- `max_mutation_len: int = 5` - Maximum mutation length
- `generations: int = 40` - Maximum number of generations
- `verbose: int = 1` - Verbosity level (0=silent, 1=normal, 2=detailed)

**Returns:** `Tuple[Population, int, int, int]`
- `Population` - Final population
- `int` - Total fitness evaluations performed
- `int` - Number of generations completed
- `int` - Index of best individual in final population

#### `random_init(seed: int, threads: int = 1)`
Initializes the global random number generator.

**Parameters:**
- `seed: int` - Seed for random number generator
- `threads: int = 1` - Number of threads for parallelization

#### `print_program(individual: Individual)`
Prints the assembly-like representation of an LGP program.

**Parameters:**
- `individual: Individual` - Individual to print

### 🧬 Core Classes

#### `LGPInput`
Represents input for an LGP problem (dataset + instruction set).

**Constructors:**
```python
# From NumPy arrays
LGPInput.from_numpy(X: np.ndarray, y: np.ndarray, instruction_set: InstructionSet, ram_size: int = None)

# From Pandas DataFrame
LGPInput.from_df(df: pd.DataFrame, y: List[str], instruction_set: InstructionSet, 
                 ram_size: int = None)
```

**Properties:**
- `input_num: int` - Number of samples in dataset
- `rom_size: int` - ROM size (number of input features)
- `res_size: int` - Result size (number of target variables)
- `ram_size: int` - RAM size of virtual machine
- `c_wrapper: LGPInputWrapper` - Underlying C wrapper

#### `Individual`
Represents a single individual (program) in the population.

**Properties:**
- `fitness: float` - Current fitness value
- `prog: ProgramWrapper` - Program (instruction sequence)
- `c_wrapper: IndividualWrapper` - Underlying C wrapper

**Methods:**
```python
def print_program() -> None
```

#### `Population`
Represents a population of individuals.

**Properties:**
- `size: int` - Number of individuals in population
- `c_wrapper: PopulationWrapper` - Underlying C wrapper

**Methods:**
```python
def get(index: int) -> Individual
```

#### `InstructionSet`
Represents a set of operations available for evolution.

**Constructor:**
```python
InstructionSet(operations: List[Operation])
```

**Properties:**
- `size: int` - Number of operations in set
- `c_wrapper: InstructionSetWrapper` - Underlying C wrapper

### 🎯 Operation Enum (87 operations)

The `Operation` enum contains all 87 virtual machine operations:

#### Control Operations
- `EXIT` - Terminate execution
- `NOP` - No operation
- `CLC` - Clear flags

#### Memory Operations
- `LOAD_RAM`, `LOAD_RAM_F` - Load from RAM (int/float)
- `STORE_RAM`, `STORE_RAM_F` - Store to RAM (int/float)
- `LOAD_ROM`, `LOAD_ROM_F` - Load from ROM (input)

#### Movement Operations
- `MOV`, `MOV_F` - Move between registers (int/float)
- `MOV_I`, `MOV_I_F` - Immediate move (int/float)
- `CAST`, `CAST_F` - Type conversion

#### Conditional Operations
- `CMOV_Z`, `CMOV_NZ` - Conditional move (zero/non-zero)
- `CMOV_L`, `CMOV_G`, `CMOV_LE`, `CMOV_GE` - Conditional move (comparison)
- `CMOV_EXIST`, `CMOV_NEXIST` - Conditional move (existence)
- `CMOV_ODD`, `CMOV_EVEN` - Conditional move (parity)
- Float versions: `CMOV_*_F`

#### Jump Operations
- `JMP` - Unconditional jump
- `JMP_Z`, `JMP_NZ` - Conditional jump (zero/non-zero)
- `JMP_L`, `JMP_G`, `JMP_LE`, `JMP_GE` - Conditional jump (comparison)
- `JMP_EXIST`, `JMP_NEXIST` - Conditional jump (existence)
- `JMP_ODD`, `JMP_EVEN` - Conditional jump (parity)

#### Arithmetic Operations (Integer)
- `ADD`, `SUB`, `MUL`, `DIV`, `MOD` - Basic arithmetic
- `INC`, `DEC` - Increment/decrement
- `CMP`, `TEST` - Comparison and test

#### Arithmetic Operations (Float)
- `ADD_F`, `SUB_F`, `MUL_F`, `DIV_F` - Basic arithmetic
- `CMP_F`, `TEST_F` - Comparison and test

#### Advanced Mathematical Operations
- `SQRT` - Square root
- `POW` - Power
- `EXP` - Exponential
- `LN`, `LOG`, `LOG10` - Logarithms
- `ROUND` - Rounding

#### Trigonometric Functions
- `SIN`, `COS`, `TAN` - Trigonometric functions
- `ASIN`, `ACOS`, `ATAN` - Inverse trigonometric functions
- `SINH`, `COSH`, `TANH` - Hyperbolic functions
- `ASINH`, `ACOSH`, `ATANH` - Inverse hyperbolic functions

#### Logical Operations
- `AND`, `OR`, `XOR`, `NOT` - Logical operations
- `SHL`, `SHR` - Logical shifts

#### Special Operations
- `RAND` - Random number generation

**Accessing operation properties:**
```python
op = lgp.Operation.ADD_F
print(op.name())  # "ADD_F"
print(op.code())  # 62
print(op.c_wrapper.regs)  # 3 (number of registers used)
print(op.c_wrapper.addr)  # 0 (uses addressing)
print(op.c_wrapper.state_changer)  # 0 (changes flag state)
```

### Fitness Assessment (25+ functions)

#### Regression
```python
# Basic functions
lgp.MSE()                    # Mean Squared Error
lgp.RMSE()                   # Root Mean Squared Error  
lgp.MAE()                    # Mean Absolute Error
lgp.RSquared()               # Coefficient of determination R²

# Correlations
lgp.PearsonCorrelation()     # Pearson correlation
lgp.SpearmanCorrelation()    # Spearman correlation

# Robust loss functions
lgp.HuberLoss(delta=1.0)     # Huber loss (robust to outliers)
lgp.PinballLoss(quantile=0.5) # Pinball loss (quantile regression)

# Penalized functions
lgp.LengthPenalizedMSE(alpha=0.01)  # MSE + penalty for program length
lgp.ClockPenalizedMSE(alpha=0.01)   # MSE + penalty for clock cycles
```

#### Classification
```python
# Basic metrics
lgp.Accuracy()               # Accuracy
lgp.BalancedAccuracy()       # Balanced accuracy for imbalanced classes

# F-score metrics
lgp.F1Score()                # F1-score (harmonic mean of precision/recall)
lgp.Precision()              # Precision
lgp.Recall()                 # Recall (sensitivity)

# Advanced metrics
lgp.MatthewsCorrelation()    # Matthews Correlation Coefficient
lgp.CohensKappa()            # Cohen's Kappa (inter-rater agreement)
lgp.LogLoss()                # Logarithmic loss
```

### Selection Methods (8+ methods)

#### Basic Selection
```python
# Tournament selection (recommended)
lgp.Tournament(tournament_size=3)

# Elitism (preserve the best)
lgp.Elitism(elite_size=10)
lgp.PercentualElitism(elite_percentage=0.1)

# Roulette wheel selection
lgp.Roulette(sampling_size=100)
```

#### Fitness Sharing (preserve diversity)
```python
# Tournament with fitness sharing
lgp.FitnessSharingTournament(
    tournament_size=3,
    alpha=1.0,    # Shape parameter
    beta=1.0,     # Scaling parameter  
    sigma=1.0     # Niche radius
)

# Elitism with fitness sharing
lgp.FitnessSharingElitism(
    elite_size=10,
    alpha=1.0, beta=1.0, sigma=1.0
)
```

## Complete Practical Examples

### Example 1: Symbolic Regression

```python
import lgp
import numpy as np

# 1. Dataset preparation
np.random.seed(42)
n_samples = 200
x1 = np.random.uniform(-2, 2, n_samples)
x2 = np.random.uniform(-2, 2, n_samples)
# Target function: y = x1² + 2*x2 + noise
y = x1**2 + 2*x2 + np.random.normal(0, 0.1, n_samples)

# 2. System initialization
lgp.random_init(seed=42, threadnum=1)

# 3. Optimized instruction set for regression
operations = [
    # Basic arithmetic
    lgp.Operation.ADD_F, lgp.Operation.SUB_F,
    lgp.Operation.MUL_F, lgp.Operation.DIV_F,
    # Memory access
    lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
    lgp.Operation.LOAD_ROM_F,  # To read input (x1, x2)
    lgp.Operation.MOV_F,        # Movement between registers
    # Advanced mathematical functions
    lgp.Operation.SQRT, lgp.Operation.POW,
    lgp.Operation.SIN, lgp.Operation.COS
]
instruction_set = lgp.InstructionSet(operations)

# 4. LGP input creation
X = np.column_stack([x1, x2])
lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=8)

# 5. Evolution component configuration
fitness = lgp.MSE()  # Minimize mean squared error
selection = lgp.Tournament(tournament_size=3)  # Tournament selection
initialization = lgp.UniquePopulation()  # Without duplicates

# 6. Evolution execution
result = lgp.evolve(
    lgp_input=lgp_input,
    fitness=fitness,
    selection=selection,
    initialization=initialization,
    init_params=(100, 5, 30),  # pop_size=100, min_len=5, max_len=30
    target=1e-6,              # Terminate if fitness < 1e-6
    mutation_prob=0.85,       # High mutation probability
    crossover_prob=0.95,      # High crossover probability
    max_clock=5000,           # Max VM cycles per program
    generations=200,          # Max generations
    verbose=1                 # Show progress
)

# 7. Results analysis
population, evaluations, generations, best_idx = result

print(f"Evolution completed in {generations} generations")
print(f"Total evaluations: {evaluations}")

# Analyze best individual
best_individual = population.get(best_idx)
print(f"Best fitness: {best_individual.fitness:.6f}")

# Print best program
print("\\nBest individual program:")
lgp.print_program(best_individual)
```

### Example 2: Binary Classification

```python
import lgp
import numpy as np
from sklearn.datasets import make_classification

# 1. Classification dataset
X, y = make_classification(
    n_samples=500, n_features=4, n_redundant=0, 
    n_informative=4, n_clusters_per_class=1, random_state=42
)

# 2. LGP setup for classification
lgp.random_init(123)

# Instruction set for classification
classification_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F,
    lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F, lgp.Operation.LOAD_ROM_F,
    lgp.Operation.MOV_F, lgp.Operation.CMP_F, lgp.Operation.TEST_F,
    # Control operations for logic
    lgp.Operation.JMP_L, lgp.Operation.JMP_G, lgp.Operation.JMP_Z
]
instruction_set = lgp.InstructionSet(classification_ops)

lgp_input = lgp.LGPInput.from_numpy(X, y, instruction_set, ram_size=6)

# Configuration for classification
result = lgp.evolve(
    lgp_input=lgp_input,
    fitness=lgp.Accuracy(),  # Maximize accuracy
    selection=lgp.FitnessSharingTournament(3, alpha=1.0, beta=1.0, sigma=0.1),
    initialization=lgp.UniquePopulation(),
    init_params=(80, 3, 25),
    target=0.98,  # Terminate if accuracy > 98%
    generations=150,
    verbose=1
)

population, evaluations, generations, best_idx = result
best = population.get(best_idx)
print(f"Best accuracy: {best.fitness:.3f}")
lgp.print_program(best)
```

### Example 3: Vector Distance Problem

```python
import lgp

# Setup for geometric problem
lgp.random_init(456)

# Instruction set optimized for distance calculations
distance_ops = [
    lgp.Operation.ADD_F, lgp.Operation.SUB_F, lgp.Operation.MUL_F,
    lgp.Operation.SQRT, lgp.Operation.POW,  # For distance calculations
    lgp.Operation.LOAD_RAM_F, lgp.Operation.STORE_RAM_F,
    lgp.Operation.LOAD_ROM_F, lgp.Operation.MOV_F
]
instruction_set = lgp.InstructionSet(distance_ops)

# Create 3D vector distance problem
vector_problem = lgp.VectorDistance(
    instruction_set=instruction_set,
    vector_len=3,    # 3D vectors
    instances=100    # 100 vector pairs
)

print(f"Problem created: {vector_problem.input_num} instances")
print(f"Input size: {vector_problem.rom_size} (2 vectors * 3 dimensions)")

# Evolution to find Euclidean distance formula
result = lgp.evolve(
    lgp_input=vector_problem,
    fitness=lgp.RMSE(),
    selection=lgp.Elitism(elite_size=15),
    initialization=lgp.UniquePopulation(),
    init_params=(60, 4, 20),
    target=1e-5,
    generations=100,
    verbose=1
)

population, evaluations, generations, best_idx = result
best = population.get(best_idx)
print(f"\\nBest RMSE: {best.fitness:.8f}")
print("Program that computes Euclidean distance:")
lgp.print_program(best)
```

## Technical Details

### Memory Management
The Python interface automatically manages C structure memory via ctypes:

- **C Structures**: Created and managed automatically
- **Dynamic Arrays**: Allocated via ctypes and deallocated automatically
- **Pointers**: Safely managed without memory leak risk
- **Garbage Collection**: Python structures deallocated when out of scope

### Performance
- **Zero-copy access**: Data passed by reference to C library
- **Minimal overhead**: Python interface adds minimal overhead (<1%)
- **Parallelization**: Multi-threading support via `threadnum` parameter
- **Optimizations**: C library compiled with optimization flags (-O3)

### Thread Safety
The interface is thread-safe for read-only operations. For concurrent operations:

```python
# Use different seeds for different threads
import threading

def worker(thread_id):
    lgp.random_init(seed=42 + thread_id, threadnum=1)
    # ... evolution ...

# Start multiple threads
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
```

## Troubleshooting

### Common Errors

#### `ImportError: cannot import liblgp.so`
**Cause**: C library not compiled or not found
**Solution**:
```bash
# Recompile library
make clean
make python

# Verify path
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
```

#### `AttributeError: 'Operation' has no attribute 'OP_*'`
**Cause**: Incorrect operation names
**Solution**: Use correct names (e.g., `ADD_F` instead of `OP_FADD`)

#### `TypeError: c_wrapper property not found`
**Cause**: Attempting to access c_wrapper on non-wrapper object
**Solution**: Verify object is a wrapper class

#### `ValueError: DataFrame cannot be empty`
**Cause**: Empty DataFrame passed to `from_df()`
**Solution**: Verify DataFrame contains data

#### `MemoryError: cannot allocate array`
**Cause**: Parameters too large (population size, max_clock, etc.)
**Solution**: Reduce parameters or increase available memory

### Debug and Diagnostics

#### Enable Verbose Mode
```python
# Show evolution progress
result = lgp.evolve(..., verbose=1)  # Basic info
result = lgp.evolve(..., verbose=2)  # Detailed info
```

#### Verify Configuration
```python
# Test basic configuration
import lgp

# Verify Operation enum
op = lgp.Operation.ADD_F
print(f"Operation {op.name()} - Code: {op.code()}")

# Verify instruction set
ops = [lgp.Operation.ADD_F, lgp.Operation.SUB_F]
iset = lgp.InstructionSet(ops)
print(f"InstructionSet size: {iset.size}")

# Verify fitness functions
fitness = lgp.MSE()
print(f"Fitness function: {type(fitness).__name__}")
```

## Known Limitations

1. **Instruction Set Size**: Maximum ~100 operations for performance
2. **Population Size**: Recommended max 1000 individuals 
3. **Program Length**: Max ~1000 instructions per program
4. **RAM/ROM Size**: Limited by available memory
5. **Threading**: C library not fully thread-safe for concurrent writes

## Future Extensions

- **Serialization**: Save/load populations and individuals
- **Visualization**: Automatic fitness and convergence plots
- **Multi-objective**: Support for multi-objective optimization
- **GPU Support**: GPU acceleration for fitness evaluation
- **Distributed**: Distributed evolution on clusters
