# Linear Genetic Programming - Development Environment
# Multi-stage Dockerfile for development and production environments

# Base image with Python and development tools
FROM ubuntu:22.04 AS base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    make \
    gcc \
    clang \
    # Development and debugging tools
    gdb \
    valgrind \
    perf-tools-unstable \
    # OpenMP for parallel processing
    libomp-dev \
    # Python and pip
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Utilities
    git \
    vim \
    htop \
    tree \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install scientific Python stack
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    jupyter>=1.0.0 \
    # Testing frameworks
    pytest>=6.0.0 \
    pytest-cov>=2.12.0 \
    pytest-xdist>=2.3.0 \
    # Code quality tools
    black>=21.0.0 \
    flake8>=3.9.0 \
    isort>=5.9.0 \
    mypy>=0.910 \
    # Profiling tools
    memory-profiler>=0.60.0 \
    line-profiler>=3.3.0

# Set up working directory
WORKDIR /workspace

# Copy build configuration files
COPY Makefile CMakeLists.txt ./

# Development stage
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Additional debugging tools
    strace \
    ltrace \
    # Documentation tools
    doxygen \
    graphviz \
    # Version control
    git-flow \
    # Additional profiling tools
    gperftools \
    google-perftools \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development packages
RUN pip install --no-cache-dir \
    # Jupyter extensions
    jupyterlab>=3.0.0 \
    ipywidgets>=7.6.0 \
    # Profiling and benchmarking
    snakeviz>=2.1.0 \
    py-spy>=0.3.0 \
    # Documentation generation
    sphinx>=4.0.0 \
    sphinx-rtd-theme>=0.5.0 \
    # Additional testing tools
    hypothesis>=6.0.0 \
    coverage>=5.5.0

# Copy source code
COPY src/ ./src/
COPY lgp/ ./lgp/
COPY tests/ ./tests/
COPY examples.py ./
COPY README.md ./

# Build the library in development mode
RUN make clean && make DEBUG=1 THREADS=4

# Build Python extension
RUN make python

# Verify installation
RUN python -c "import lgp; print('LGP development environment ready!')"

# Set up development environment
ENV LGP_DEV=1
ENV PYTHONPATH=/workspace

# Create a non-root user for development
RUN useradd -m -s /bin/bash lgpdev && \
    chown -R lgpdev:lgpdev /workspace

USER lgpdev

# Default command for development
CMD ["/bin/bash"]

# Production stage
FROM base AS production

# Copy only necessary files for production
COPY src/ ./src/
COPY lgp/ ./lgp/
COPY examples.py ./
COPY README.md ./

# Build optimized library
RUN make clean && make RELEASE=1 THREADS=8

# Build Python extension
RUN make python

# Verify installation
RUN python -c "import lgp; print('LGP production environment ready!')"

# Clean up build artifacts
RUN make clean && \
    rm -rf /workspace/bin/ && \
    rm -rf /workspace/build/

# Create a non-root user for production
RUN useradd -m -s /bin/bash lgpuser && \
    chown -R lgpuser:lgpuser /workspace

USER lgpuser

# Default command for production
CMD ["python", "examples.py"]

# Testing stage
FROM development AS testing

# Copy test files
COPY tests/ ./tests/
COPY test.sh ./

# Run comprehensive tests
RUN ./test.sh

# Benchmark stage  
FROM production AS benchmark

# Install benchmarking tools
USER root
RUN pip install --no-cache-dir \
    psutil>=5.8.0 \
    memory-profiler>=0.60.0

USER lgpuser

# Copy benchmark scripts
COPY psb2_benchmark.py ./

# Run benchmark
CMD ["python", "psb2_benchmark.py"]
