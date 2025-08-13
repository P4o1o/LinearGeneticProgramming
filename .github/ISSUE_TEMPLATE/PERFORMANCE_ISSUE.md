---
name: Performance Issue
about: Report performance problems
title: '[PERFORMANCE] '
labels: ['performance']
assignees: ''
---

**Performance Problem**
What's slow? (evolution, fitness evaluation, compilation, etc.)

**Benchmark Code**
```python
# For Python interface
import lgp
import time
# Your benchmark here
```

OR

```c
// For C interface
#include "evolution.h"
// Your benchmark here
```

**Current Performance**
- Evaluations/sec: 
- Memory usage: 
- Time to complete: 

**Expected Performance**
What performance did you expect?

**System Info**
- OS:
- CPU:
- Compiler flags used:

**Interface Used**
- [ ] C interface (LGP executable)
- [ ] Python interface (import lgp)
- [ ] Build/compilation 
- Expected memory usage: 
- Expected completion time: 
- Baseline for comparison: 

**Performance Test Code**
Provide code to reproduce the performance issue:

```python
import lgp
import numpy as np
import time

# Your performance test code here
start_time = time.time()

# ... your code ...

duration = time.time() - start_time
print(f"Performance: {evaluations/duration:.0f} evals/sec")
```

**Profiling Information**
If you have profiling data, please include it:

```
# Example: Python cProfile output
# Example: perf report output
# Example: memory profiler output
```

**Environment Information**
- OS: [e.g. Ubuntu 22.04, macOS 13.0, Windows 11]
- CPU: [e.g. Intel i7-12700K, AMD Ryzen 7 5800X, Apple M2]
- RAM: [e.g. 32GB DDR4-3200]
- Python Version: [e.g. 3.11.2]
- NumPy Version: [e.g. 1.24.0]
- Compiler: [e.g. GCC 11.3.0 with -O3]
- Build flags: [e.g. RELEASE=1 THREADS=8]

**System Performance**
Run these commands and include output:

```bash
# CPU information
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|MHz)'

# Memory information
free -h

# Performance test
make test-performance
```

**Comparison Data**
If you have performance data from:
- Previous versions of LGP
- Other genetic programming libraries
- Different hardware/software configurations

Please include that information.

**Dataset Information**
- Dataset size: [e.g. 1000 samples, 10 features]
- Problem type: [e.g. symbolic regression, classification]
- Population size: [e.g. 100 individuals]
- Program size: [e.g. 5-20 instructions]
- Generations: [e.g. 100 generations]

**Potential Causes**
Do you suspect any specific causes?
- [ ] Algorithm inefficiency
- [ ] Memory allocation issues
- [ ] Compiler optimization problems
- [ ] Threading overhead
- [ ] Python/C interface overhead
- [ ] Other: ___________

**Regression Information**
If this is a performance regression:
- When did you first notice the issue?
- What was the last known good version?
- Any recent changes that might be related?

**Attempted Solutions**
What have you tried to improve performance?
- [ ] Different compiler flags
- [ ] Different thread counts
- [ ] Different population sizes
- [ ] Memory optimization
- [ ] Algorithm parameters tuning
- [ ] Other: ___________

**Additional Context**
Add any other context about the performance issue here.

**Checklist**
- [ ] I have searched for existing performance issues
- [ ] I have tried with the latest version
- [ ] I have provided reproducible test code
- [ ] I have included system specifications
- [ ] I have run basic performance tests
