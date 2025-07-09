# Linear Genetic Programming - C Core

This directory contains the C implementation of the Linear Genetic Programming (LGP) framework. The C core provides high-performance genetic programming capabilities with support for various fitness functions and automatic code generation.

## Architecture Overview

The C core is organized into several modules:

- **`main.c`** - Entry point and command-line interface
- **`vm.c/vm.h`** - Virtual machine for program execution
- **`genetics.c/genetics.h`** - Genetic operations (crossover, mutation)
- **`creation.c/creation.h`** - Individual creation and initialization
- **`evolution.c/evolution.h`** - Evolution loop and population management
- **`fitness.c/fitness.h`** - Fitness evaluation functions
- **`selection.c/selection.h`** - Selection algorithms
- **`prob.c/prob.h`** - Probability distributions and random number generation
- **`psb2.c/psb2.h`** - Problem-specific benchmark functions
- **`logger.c/logger.h`** - Logging and debugging utilities
- **`mt19937.c/mt19937.h`** - Mersenne Twister random number generator
- **`macros.h`** - Common macros and constants

## Build System

The C core supports two build systems that provide identical functionality:

### Quick Start
```bash
# Build executable (default)
make

# Build Python shared library
make python

# Show build configuration
make info

# Clean build artifacts
make clean
```

### Build Variables

Both Makefile and CMakeLists.txt support the following user-configurable variables:

#### `THREADS` (default: auto-detected)
Controls the number of threads used for parallel processing:
- **Auto**: Automatically detects available CPU cores
- **1**: Single-threaded execution
- **N**: Use N threads (where N > 1)

Examples:
```bash
make THREADS=4              # Use 4 threads
cmake -DTHREADS=1 ..        # Single-threaded build
```

#### `DEBUG` (default: 0)
Controls debug mode and optimizations:
- **0**: Release mode with full optimizations (`-O3`, `-DNDEBUG`)
- **1**: Debug mode with debug symbols (`-g`, `-O0`)

Examples:
```bash
make DEBUG=1                # Debug build
cmake -DDEBUG=1 ..          # Debug build with CMake
```

#### `C_STD` (default: auto-detected)
Override the C standard version:
- **Auto**: Automatically detects highest supported standard (C23 > C17 > C11 > C99 > C89)
- **c89**, **c99**, **c11**, **c17**, **c23**: Force specific standard

Examples:
```bash
make C_STD=c11              # Force C11 standard
cmake -DC_STD=c17 ..        # Force C17 standard
```

#### `CC` (default: auto-detected)
Override the compiler:
- **Auto**: Automatically selects best available compiler per platform
- **gcc**, **clang**, **cl**: Force specific compiler

Examples:
```bash
make CC=clang               # Use clang compiler
cmake -DCC=gcc ..           # Use gcc compiler
```

### Platform-Specific Behavior

The build system automatically optimizes for each platform:

#### **Linux/Ubuntu**
- **Default compiler**: `gcc` (preferred) > `clang`
- **Vector instructions**: AVX512 > AVX2 > AVX > SSE4.2 > SSE4.1 > SSE2
- **OpenMP**: Enabled if supported (`-fomp`)

#### **macOS**
- **Default compiler**: `clang` (preferred) > `gcc`
- **Vector instructions**: AVX512 > AVX2 > AVX > SSE4.2 (Intel), NEON (Apple Silicon)
- **OpenMP**: Enabled if supported (`-fomp`)

#### **Windows**
- **Default compiler**: `msvc` (preferred) > `clang` > `gcc`
- **Vector instructions**: AVX512 > AVX2 > AVX > SSE4.2 > SSE4.1 > SSE2
- **OpenMP**: Enabled if supported (`/openmp` for MSVC, `-fomp` for others)

#### **FreeBSD**
- **Default compiler**: `clang` (preferred) > `gcc`
- **Vector instructions**: AVX512 > AVX2 > AVX > SSE4.2 > SSE4.1 > SSE2
- **OpenMP**: Enabled if supported (`-fomp`)

## Fitness Functions

The fitness evaluation system supports multiple types of fitness functions with different output requirements. Understanding these types is crucial for proper usage.

### Fitness Function Types

#### 1. **Floating-Point Output Functions**
These functions expect programs to output floating-point values and compare them against target floating-point values.

**Supported Functions:**
- `koza1`, `koza2`, `koza3` - Koza's symbolic regression benchmarks
- `nguyen1` through `nguyen12` - Nguyen's symbolic regression suite
- `keijzer1` through `keijzer15` - Keijzer's symbolic regression benchmarks
- `vladislavleva1` through `vladislavleva8` - Vladislavleva's benchmarks
- `pagie1` - Pagie's symbolic regression function
- `korns1` through `korns15` - Korns' symbolic regression suite

**Expected Output:** Floating-point values (`double`)
**Fitness Calculation:** Mean Squared Error (MSE) between program output and target values

#### 2. **Integer Output Functions**
These functions expect programs to output integer values and compare them against target integer values.

**Supported Functions:**
- `median` - Find median of three integers
- `small_or_large` - Classify number as small or large
- `for_loop_index` - Generate sequence indices
- `compare_string_lengths` - Compare string lengths
- `double_letters` - Count double letters
- `collatz_numbers` - Collatz sequence operations
- `string_lengths_backwards` - String length operations
- `last_index_of_zero` - Find last zero index
- `vector_average` - Vector averaging operations
- `count_odds` - Count odd numbers
- `mirror_image` - Mirror image operations
- `super_anagrams` - Anagram operations
- `sum_of_squares` - Sum of squares calculation
- `vectors_summed` - Vector summation
- `x_word_lines` - Word line operations
- `pig_latin` - Pig Latin transformation
- `negative_to_zero` - Negative number handling
- `scrabble_score` - Scrabble scoring
- `word_stats` - Word statistics
- `checksum` - Checksum calculation
- `digits` - Digit operations
- `grade` - Grading operations
- `smallest` - Find smallest element
- `syllables` - Syllable counting

**Expected Output:** Integer values (`int`)
**Fitness Calculation:** Sum of absolute differences between program output and target values

#### 3. **Boolean Output Functions**
These functions expect programs to output boolean values (0 or 1) and compare them against target boolean values.

**Supported Functions:**
- `number_io` - Boolean number I/O operations
- `replace_space_with_newline` - Space replacement operations
- `string_differences` - String difference detection
- `even_squares` - Even square detection
- `wallis_pi` - Wallis pi approximation
- `string_lengths_backwards` - String length comparisons
- `last_index_of_zero` - Zero index detection
- `vector_average` - Vector averaging comparisons
- `count_odds` - Odd number detection
- `mirror_image` - Mirror image detection
- `super_anagrams` - Anagram detection
- `sum_of_squares` - Sum validation
- `vectors_summed` - Vector sum validation
- `x_word_lines` - Word line validation
- `pig_latin` - Pig Latin validation
- `negative_to_zero` - Negative number detection
- `scrabble_score` - Score validation
- `word_stats` - Word statistics validation
- `checksum` - Checksum validation
- `digits` - Digit validation
- `grade` - Grade validation
- `smallest` - Smallest element detection
- `syllables` - Syllable validation

**Expected Output:** Boolean values (0 or 1)
**Fitness Calculation:** Sum of mismatches between program output and target boolean values

### Fitness Function Implementation

#### Core Interface
```c
// Main fitness evaluation function
double evaluate_fitness(individual_t *individual, fitness_case_t *test_cases, 
                       int num_cases, char *function_name);

// Fitness case structure
typedef struct {
    double *inputs;      // Input values for the test case
    double target;       // Expected output value
    int input_size;      // Number of input values
} fitness_case_t;
```

#### Output Type Handling
The fitness evaluation system automatically handles different output types:

```c
// For floating-point functions
if (is_float_function(function_name)) {
    double program_output = execute_program(individual, inputs);
    double target_value = test_case->target;
    error += (program_output - target_value) * (program_output - target_value);
}

// For integer functions
else if (is_integer_function(function_name)) {
    int program_output = (int)execute_program(individual, inputs);
    int target_value = (int)test_case->target;
    error += abs(program_output - target_value);
}

// For boolean functions
else if (is_boolean_function(function_name)) {
    int program_output = (execute_program(individual, inputs) > 0.5) ? 1 : 0;
    int target_value = (test_case->target > 0.5) ? 1 : 0;
    error += (program_output != target_value) ? 1 : 0;
}
```

### Adding Custom Fitness Functions

To add a new fitness function:

1. **Add function declaration** in `fitness.h`:
```c
double my_custom_function(individual_t *individual, fitness_case_t *test_cases, int num_cases);
```

2. **Implement function** in `fitness.c`:
```c
double my_custom_function(individual_t *individual, fitness_case_t *test_cases, int num_cases) {
    double total_error = 0.0;
    
    for (int i = 0; i < num_cases; i++) {
        // Execute program with test case inputs
        double output = execute_program(individual, test_cases[i].inputs);
        
        // Calculate error based on output type
        double error = fabs(output - test_cases[i].target);  // For float
        // int error = abs((int)output - (int)test_cases[i].target);  // For int
        // int error = ((output > 0.5) != (test_cases[i].target > 0.5)) ? 1 : 0;  // For bool
        
        total_error += error;
    }
    
    return total_error;
}
```

3. **Register function** in the fitness function lookup table:
```c
static fitness_function_t fitness_functions[] = {
    {"my_custom_function", my_custom_function},
    // ... other functions
};
```

### Performance Considerations

#### Multi-threading
When `THREADS > 1`, fitness evaluation is automatically parallelized:
- Each thread evaluates a subset of the population
- Thread-safe random number generation is used
- Memory allocation is optimized for parallel access

#### Vector Instructions
The build system automatically enables vector instructions when supported:
- **AVX512**: 512-bit vector operations (Intel Skylake-X and newer)
- **AVX2**: 256-bit vector operations (Intel Haswell and newer)
- **AVX**: 256-bit vector operations (Intel Sandy Bridge and newer)
- **SSE4.2**: 128-bit vector operations (Intel Nehalem and newer)
- **NEON**: ARM vector instructions (ARM Cortex-A series)

#### Memory Management
- Stack-based allocation for temporary variables
- Pool-based allocation for individuals
- Garbage collection for unused genetic material

## Debug Mode

When `DEBUG=1`, the following debug features are enabled:

### Debug Output
- **Population statistics**: Size, diversity, convergence metrics
- **Fitness evaluation**: Detailed error breakdown per test case
- **Genetic operations**: Crossover and mutation success rates
- **Memory usage**: Allocation and deallocation tracking

### Debug Macros
```c
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
    #define DEBUG_ASSERT(condition) assert(condition)
    #define DEBUG_TRACE() printf("[TRACE] %s:%d\n", __FILE__, __LINE__)
#else
    #define DEBUG_PRINT(fmt, ...)
    #define DEBUG_ASSERT(condition)
    #define DEBUG_TRACE()
#endif
```

### Memory Debugging
- **Valgrind compatibility**: Clean memory access patterns
- **AddressSanitizer support**: Built-in memory error detection
- **Leak detection**: Automatic memory leak reporting

## Integration with Python

The C core can be compiled as a shared library for Python integration:

```bash
make python                 # Creates Python-compatible shared library
```

### Python Interface
The shared library exports the following functions:

```c
// Initialize LGP system
int lgp_init(int population_size, int max_generations, int num_threads);

// Run evolution
int lgp_evolve(char *fitness_function, double *fitness_cases, int num_cases);

// Get best individual
char* lgp_get_best_individual();

// Cleanup
void lgp_cleanup();
```

### C-Python Data Exchange
- **Fitness cases**: Passed as NumPy arrays
- **Individuals**: Returned as serialized strings
- **Statistics**: Returned as Python dictionaries

## Error Handling

The C core implements comprehensive error handling:

### Error Types
- **Memory errors**: Out-of-memory, invalid pointers
- **Fitness errors**: Invalid fitness function, malformed test cases
- **Evolution errors**: Population convergence, generation limits
- **I/O errors**: File access, parameter parsing

### Error Codes
```c
#define LGP_SUCCESS           0
#define LGP_ERROR_MEMORY     -1
#define LGP_ERROR_FITNESS    -2
#define LGP_ERROR_EVOLUTION  -3
#define LGP_ERROR_IO         -4
#define LGP_ERROR_INVALID    -5
```

### Error Handling Pattern
```c
int result = lgp_function();
if (result != LGP_SUCCESS) {
    fprintf(stderr, "Error: %s\n", lgp_error_string(result));
    return result;
}
```

## Testing

The C core includes comprehensive test coverage:

### Unit Tests
- **Fitness functions**: Verify correct output types and calculations
- **Genetic operations**: Test crossover and mutation correctness
- **Memory management**: Ensure no leaks or corruption
- **Random number generation**: Statistical validation

### Integration Tests
- **Evolution runs**: Complete evolution with known benchmarks
- **Python integration**: C-Python interface validation
- **Multi-threading**: Parallel execution correctness

### Performance Tests
- **Benchmark suite**: Standard GP benchmarks with timing
- **Memory profiling**: Memory usage under different configurations
- **Scalability tests**: Performance with varying population sizes

## Contributing

When contributing to the C core:

1. **Follow coding style**: Use consistent indentation and naming
2. **Add documentation**: Document all public functions
3. **Include tests**: Add tests for new functionality
4. **Check memory safety**: Use valgrind or AddressSanitizer
5. **Verify cross-platform**: Test on multiple operating systems

## License

This C core is part of the Linear Genetic Programming project and is subject to the same license terms as the overall project.
