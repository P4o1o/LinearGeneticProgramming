---
name: Bug Report
about: Report a bug in Linear Genetic Programming
title: '[BUG] '
labels: ['bug']
assignees: ''
---

**Bug Description**
What happened and what you expected.

**Minimal Code Example**
```python
# For Python interface
import lgp
import numpy as np

# Code that reproduces the bug
```

OR

```c
// For C interface  
#include "evolution.h"
// Code that reproduces the bug
```

**Error Message**
```
Paste error or stack trace here
```

**Environment**
- OS: [Ubuntu/macOS/Windows]
- Python: [version] (if using Python interface)
- Compiler: [GCC/Clang version]

**Interface Used**
- [ ] C interface (LGP executable)
- [ ] Python interface (import lgp)
- [ ] Build system
uname -a

# Python environment
python --version
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import lgp; print('LGP imported successfully')"

# Compiler information
gcc --version
make info  # If using our Makefile
```

**Build Information**
- Build flags used: [e.g. DEBUG=1, THREADS=8]
- Build method: [e.g. Makefile, CMake, Docker]
- Any custom modifications to build system

**Additional Context**
Add any other context about the problem here.

**Possible Solution**
If you have an idea of what might be causing the issue or how to fix it, please describe it here.

**Checklist**
- [ ] I have searched for existing issues
- [ ] I have read the documentation
- [ ] I have tried with the latest version
- [ ] I have provided a minimal reproducible example
