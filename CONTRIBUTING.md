# Contributing to Linear Genetic Programming (LGP)

Thank you for your interest in contributing to the Linear Genetic Programming (LGP) framework! This document outlines the guidelines and best practices for contributing to the project. Whether you're fixing bugs, adding features, or improving documentation, your contributions are greatly appreciated.

## Getting Started

1. **Understand the Project**: Familiarize yourself with the [README.md](README.md) and the project structure. The framework combines a high-performance C core with a Python interface.
2. **Set Up Your Environment**:
   - Use the provided `Dockerfile` or `docker-compose.yml` for a consistent development environment.
   - Alternatively, ensure you have the required dependencies installed (e.g., GCC/Clang, Python 3.8+, OpenMP, etc.).
3. **Clone the Repository**:
   ```bash
   git clone https://github.com/P4o1o/LinearGeneticProgramming.git
   cd LinearGeneticProgramming
   ```
4. **Build the Project**:
   - For the C core:
     ```bash
     make clean && make DEBUG=1 THREADS=4
     ```
   - For the Python interface:
     ```bash
     make python
     ```
5. **Run Tests**:
   - Execute the test suite to ensure everything is working:
     ```bash
     pytest tests/
     ```

## Contribution Guidelines

### Reporting Issues
- Use the [GitHub Issues](https://github.com/P4o1o/LinearGeneticProgramming/issues) page to report bugs or suggest features.
- Provide detailed information, including steps to reproduce the issue, expected behavior, and environment details.

### Submitting Changes
1. **Fork the Repository**: Create your own fork of the repository.
2. **Create a Branch**: Use a descriptive name for your branch (e.g., `fix-memory-leak`, `add-new-fitness-function`).
   ```bash
   git checkout -b your-branch-name
   ```
3. **Make Changes**: Follow the coding standards and ensure your changes are well-documented.
4. **Run Tests**: Verify that your changes do not break existing functionality.
5. **Submit a Pull Request (PR)**: Push your changes to your fork and open a PR against the `main` branch. Include a clear description of your changes and reference any related issues.

### Coding Standards
- **C Code**:
  - Follow the conventions in the existing codebase.
  - Use meaningful variable and function names.
  - Ensure thread safety and memory efficiency.
- **Python Code**:
  - Adhere to PEP 8 standards.
  - Use type hints and docstrings for all functions.
  - Write tests for new features or bug fixes.

### Testing
- Add unit tests for all new features and bug fixes.
- Ensure tests cover edge cases and are reproducible.
- Use `pytest` for Python tests and the `Makefile` targets for C tests.

## To-Do List

Here are some areas where contributions are needed:

- [ ] Complete support for multi-objective evolution.
- [ ] Test on various operating systems and hardware:
  - Linux (e.g., Ryzen 9700X with 64GB, i5-825U with 8GB).
  - Windows (e.g., i5-825U with 8GB).
- [ ] Expand and improve CI/CD tests.
- [ ] Add features to the Python interface.
- [ ] Enhance the virtual machine:
  - Implement new instructions (e.g., vectorized, GPU-based).
  - Expand instruction set.
- [ ] Add new selection and initialization methods.
- [ ] Develop additional fitness functions and evolutionary methods.
- [ ] Explore and implement other essential or innovative features.

## Community

- Join discussions on [GitHub Discussions](https://github.com/P4o1o/LinearGeneticProgramming/discussions).

