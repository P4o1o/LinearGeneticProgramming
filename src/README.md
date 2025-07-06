# Linear Genetic Programming (LGP) - C Core Implementation

This directory contains the complete C implementation of the Linear Genetic Programming framework. The system is designed for high performance, modularity, and extensibility while maintaining clean separation of concerns across different components.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Virtual Machine](#virtual-machine)
4. [Genetic Structures](#genetic-structures)
5. [Evolution Engine](#evolution-engine)
6. [Fitness Functions](#fitness-functions)
7. [Selection Methods](#selection-methods)
8. [API Reference](#api-reference)
9. [Build Configuration](#build-configuration)
10. [Performance Considerations](#performance-considerations)
11. [Extension Guidelines](#extension-guidelines)

## Architecture Overview

### Design Principles

- **Modular Architecture**: Each component (VM, genetics, fitness, selection) is self-contained
- **Performance-First**: Optimized data structures with memory alignment and minimal overhead
- **Type Safety**: Strong typing with compile-time checks where possible
- **OpenMP Parallelization**: Thread-safe parallel execution for population-based operations
- **Zero-Copy Operations**: Direct memory access patterns to minimize allocations

### Component Dependencies

```
main.c
├── evolution.h ── Core evolution loop and parameters
│   ├── genetics.h ── Genetic structures and population management
│   ├── selection.h ── Selection algorithms
│   ├── creation.h ── Population initialization
│   └── fitness.h ── Fitness evaluation functions
│       └── vm.h ── Virtual machine and instruction execution
│           ├── macros.h ── Common macros and compiler attributes
│           └── prob.h ── Probability distributions and random numbers
└── logger.h ── Logging and debugging utilities
```

## Core Components

### File Structure

| File | Purpose | Key Structures |
|------|---------|----------------|
| `vm.{c,h}` | Virtual Machine implementation | `VirtualMachine`, `Instruction`, `Operation` |
| `genetics.{c,h}` | Genetic data structures | `Individual`, `Population`, `LGPInput` |
| `evolution.{c,h}` | Main evolutionary algorithm | `LGPOptions`, `LGPResult` |
| `fitness.{c,h}` | Fitness function implementations | `FitnessAssessment`, `FitnessParams` |
| `selection.{c,h}` | Selection method implementations | `Selection`, `SelectionParams` |
| `creation.{c,h}` | Population initialization | `InitializationParams` |
| `main.c` | Command-line interface | - |

## Virtual Machine

### Overview

The Virtual Machine is the core execution engine that interprets linear genetic programs. It provides a complete instruction set with 87 operations covering arithmetic, logic, control flow, and mathematical functions.

### Architecture

```c
struct VirtualMachine {
    struct Core core;           // CPU state (registers, flags, PC)
    union Memblock *ram;        // Working memory (read-write)
    const union Memblock *rom;  // Problem data (read-only)
    const struct Instruction *program; // Program instructions
};

struct Core {
    uint64_t reg[4];           // Integer registers (R0-R3)
    double freg[4];            // Float registers (F0-F3)
    struct FlagReg flag;       // Status flags (zero, negative, etc.)
    uint64_t prcount;          // Program counter
};
```

### Instruction Set Architecture

#### Instruction Format
```c
struct Instruction {
    uint8_t op;        // Operation code (0-86)
    uint8_t reg[3];    // Register operands
    uint32_t addr;     // Memory address or immediate value
};
```

#### Operation Categories

**Memory Operations** (12 operations)
- `LOAD_RAM`, `STORE_RAM`: RAM memory access
- `LOAD_ROM`: Read-only memory access
- `MOV`, `MOV_F`: Register-to-register moves
- Conditional moves: `CMOV_Z`, `CMOV_NZ`, `CMOV_L`, `CMOV_G`, etc.

**Control Flow** (15 operations)
- Unconditional jump: `JMP`
- Conditional jumps: `JMP_Z`, `JMP_NZ`, `JMP_L`, `JMP_G`, etc.
- Flag operations: `CLC`, `CMP`, `TEST`

**Arithmetic Operations** (18 operations)
- Integer: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `INC`, `DEC`
- Float: `ADD_F`, `SUB_F`, `MUL_F`, `DIV_F`
- Advanced math: `SQRT`, `POW`, `EXP`, `LN`, `LOG`, `LOG10`

**Trigonometric Functions** (18 operations)
- Basic: `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`
- Hyperbolic: `SINH`, `COSH`, `TANH`, `ASINH`, `ACOSH`, `ATANH`

**Logical Operations** (8 operations)
- Bitwise: `AND`, `OR`, `XOR`, `NOT`
- Bit shifts: `SHL`, `SHR`
- Type conversion: `CAST`, `CAST_F`

**Utility Operations** (4 operations)
- `NOP`: No operation
- `EXIT`: Program termination
- `RAND`: Random number generation
- `ROUND`: Floating-point rounding

### VM Execution

```c
uint64_t run_vm(struct VirtualMachine *env, uint64_t clock_limit);
```

- **Clock-limited execution**: Prevents infinite loops
- **State preservation**: VM state maintained between calls
- **Return value**: Number of clock cycles consumed

## Genetic Structures

### Individual Representation

```c
struct Individual {
    struct Program prog;  // Linear sequence of instructions
    double fitness;       // Cached fitness value
};

struct Program {
    struct Instruction *content;  // Aligned instruction array
    uint64_t size;               // Number of instructions
};
```

### Population Management

```c
struct Population {
    struct Individual *individual;  // Array of individuals
    uint64_t size;                 // Population size
};
```

### Problem Definition

```c
struct LGPInput {
    const uint64_t input_num;    // Number of input samples
    const uint64_t rom_size;     // Input data size per sample
    const uint64_t res_size;     // Expected output size per sample
    const uint64_t ram_size;     // Working memory size
    const struct InstructionSet instr_set;  // Available operations
    union Memblock *memory;      // Interleaved input/output data
};
```

### Memory Layout

The `LGPInput.memory` array stores data in interleaved format:
```
[input1][output1][input2][output2]...[inputN][outputN]
```

This layout optimizes cache locality during fitness evaluation.

## Evolution Engine

### Main Evolution Function

```c
struct LGPResult evolve(const struct LGPInput *const in, 
                       const struct LGPOptions *const args);
```

### Evolution Parameters

```c
struct LGPOptions {
    const struct FitnessAssessment fitness;     // Fitness function
    const union FitnessParams fitness_param;   // Fitness parameters
    const struct Selection selection;           // Selection method
    const union SelectionParams select_param;  // Selection parameters
    const initialization_fn initialization_func; // Population init
    const struct InitializationParams init_params;
    const struct Population initial_pop;       // Pre-initialized population
    const double target;                       // Target fitness value
    const double mutation_prob;               // Mutation probability
    const double crossover_prob;              // Crossover probability
    const uint64_t max_clock;                 // VM execution limit
    const uint64_t max_individ_len;           // Maximum program length
    const uint64_t max_mutation_len;          // Maximum mutation size
    const uint64_t generations;               // Generation limit
    const unsigned verbose;                   // Verbosity level
};
```

### Evolution Result

```c
struct LGPResult {
    const struct Population pop;        // Final population
    const uint64_t evaluations;        // Total fitness evaluations
    const uint64_t generations;        // Generations completed
    const uint64_t best_individ;       // Index of best individual
};
```

## Fitness Functions

### Fitness Function Interface

```c
typedef double (*fitness_fn)(const struct LGPInput *const in,
                            const struct Program *const prog,
                            const uint64_t max_clock,
                            const union FitnessParams *const params);
```

### Available Fitness Functions

#### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R_SQUARED**: Coefficient of determination
- **PEARSON_CORRELATION**: Pearson correlation coefficient

#### Classification Metrics
- **ACCURACY**: Classification accuracy
- **BALANCED_ACCURACY**: Balanced accuracy for imbalanced datasets
- **F1_SCORE**: F1 score (harmonic mean of precision/recall)
- **MATTHEWS_CORRELATION**: Matthews correlation coefficient
- **G_MEAN**: Geometric mean of sensitivity and specificity

#### Robust Metrics
- **HUBER_LOSS**: Huber loss (robust to outliers)
- **PINBALL_LOSS**: Quantile regression loss
- **LOGCOSH**: Log-cosh loss

#### Penalized Metrics
- **LENGTH_PENALIZED_MSE**: MSE + α × program_length
- **CLOCK_PENALIZED_MSE**: MSE + α × execution_time

#### Advanced Metrics
- **BINARY_CROSS_ENTROPY**: Cross-entropy for binary classification
- **GAUSSIAN_LOG_LIKELIHOOD**: Log-likelihood assuming Gaussian errors
- **ADVERSARIAL_PERTURBATION_SENSITIVITY**: Robustness to input perturbations

### Fitness Assessment Structure

```c
struct FitnessAssessment {
    const fitness_fn fn;           // Function pointer
    const enum FitnessType type;   // MINIMIZE or MAXIMIZE
    const char *name;              // Human-readable name
};
```

## Selection Methods

### Selection Interface

```c
typedef void (*selection_fn)(struct Population* pop, 
                           const union SelectionParams *const params);
```

### Available Selection Methods

#### Tournament-Based Selection
- **tournament**: Basic tournament selection
- **fitness_sharing_tournament**: Tournament with fitness sharing for diversity

#### Elitism-Based Selection
- **elitism**: Fixed-size elite preservation
- **percentual_elitism**: Percentage-based elite preservation
- **fitness_sharing_elitism**: Elite selection with fitness sharing

#### Probabilistic Selection
- **roulette**: Fitness-proportional selection
- **fitness_sharing_roulette**: Roulette wheel with fitness sharing

### Selection Parameters

```c
union SelectionParams {
    const uint64_t size;                        // Tournament size, elite count
    const double val;                          // Elite percentage
    const struct FitnessSharingParams fs_params; // Fitness sharing parameters
};

struct FitnessSharingParams {
    const double alpha;     // Sharing function exponent
    const double beta;      // Distance scaling factor
    const double sigma;     // Sharing threshold
    union {
        const uint64_t size;  // Selection size
        const double val;     // Selection fraction
    } select_factor;
};
```

## API Reference

### Core Functions

#### VM Operations
```c
// Execute program on VM with clock limit
uint64_t run_vm(struct VirtualMachine *env, uint64_t clock_limit);
```

#### Genetic Operations
```c
// Generate random instruction
struct Instruction rand_instruction(const struct LGPInput *const in, 
                                  const uint64_t prog_size);
```

#### Evolution
```c
// Main evolution function
struct LGPResult evolve(const struct LGPInput *const in, 
                       const struct LGPOptions *const args);

// Print program in human-readable format
void print_program(const struct Program *const prog);
```

### Memory Management

All dynamic memory allocation is handled internally. Users should:

1. **LGPInput**: Allocate `memory` array with size `input_num * (rom_size + res_size)`
2. **Population**: Will be allocated/deallocated by evolution functions
3. **Programs**: Automatically managed during evolution

### Thread Safety

- **VM execution**: Thread-safe (each thread operates on separate VM instance)
- **Population operations**: Protected by OpenMP synchronization
- **Random number generation**: Thread-local RNG state
- **Fitness evaluation**: Fully parallelizable

## Build Configuration

### Compiler Requirements

- **GCC 7+** or **Clang 10+**
- **C11 standard** minimum (C2X features optional)
- **OpenMP 3.0+** for parallelization

### Compilation Flags

```makefile
# Standard build
CFLAGS = -std=c11 -O2 -fopenmp -Wall -Wextra

# Debug build  
CFLAGS = -std=c11 -O0 -g -fopenmp -Wall -Wextra -DDEBUG

# Performance build
CFLAGS = -std=c11 -O3 -march=native -fopenmp -DNDEBUG -flto
```

### Optional Features

```c
// Enable C2X enum typing (if supported)
#define C2X_SUPPORTED

// Enable vector alignment optimizations
#define VECT_ALIGNMENT 32

// Debug logging
#define DEBUG
```

## Performance Considerations

### Memory Optimization

1. **Instruction Alignment**: Instructions aligned to `VECT_ALIGNMENT` boundaries
2. **Cache-Friendly Layouts**: Interleaved data storage for optimal cache usage
3. **Memory Pool**: Pre-allocated instruction arrays to reduce allocation overhead
4. **Zero-Copy Operations**: Direct memory access without intermediate copies

### Computational Optimization

1. **Clock Limiting**: Prevents runaway program execution
2. **Efficient VM**: Optimized instruction dispatch with jump tables
3. **Parallel Evaluation**: OpenMP parallel fitness evaluation
4. **Early Termination**: Evolution stops when target fitness reached

### Scaling Characteristics

| Population Size | Memory Usage | Parallel Efficiency |
|----------------|--------------|-------------------|
| 100 | ~10 MB | 90%+ |
| 1,000 | ~100 MB | 85%+ |
| 10,000 | ~1 GB | 70%+ |

## Extension Guidelines

### Adding New Instructions

1. **Update `vm.h`**: Add entry to `INSTR_MACRO`
```c
INSTRUCTION(NEW_OP, 87, 2, 0, 0)
```

2. **Implement in `vm.c`**: Add case to instruction dispatch
```c
case I_NEW_OP:
    vm->core.freg[instr->reg[0]] = new_operation(
        vm->core.freg[instr->reg[1]]);
    break;
```

3. **Update `INSTR_NUM`**: Increment instruction count

### Adding New Fitness Functions

1. **Implement function** in `fitness.c`:
```c
double new_fitness(const struct LGPInput *const in,
                  const struct Program *const prog,
                  const uint64_t max_clock,
                  const union FitnessParams *const params) {
    // Implementation
}
```

2. **Declare in `fitness.h`**:
```c
extern const struct FitnessAssessment NEW_FITNESS;
```

3. **Define assessment structure**:
```c
const struct FitnessAssessment NEW_FITNESS = {
    .fn = new_fitness,
    .type = MINIMIZE,  // or MAXIMIZE
    .name = "New Fitness"
};
```

### Adding New Selection Methods

1. **Implement selection functions** for both minimize/maximize:
```c
void new_selection_MINIMIZE(struct Population* pop, 
                           const union SelectionParams *const params);
void new_selection_MAXIMIZE(struct Population* pop, 
                           const union SelectionParams *const params);
```

2. **Add to `SELECTION_MACRO`** in `selection.h`

3. **Export selection structure**:
```c
const struct Selection new_selection = {
    .type = {new_selection_MINIMIZE, new_selection_MAXIMIZE}
};
```

---

**Version**: 1.0.0  
**Last Updated**: 2025-07-06  
**Compatibility**: C11/C2X, GCC 7+, Clang 10+

For technical questions or contributions, see the main project README.md.
