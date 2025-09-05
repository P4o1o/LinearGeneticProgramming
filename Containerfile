FROM ubuntu:22.04 AS base

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
    # Debugging tools
    gdb \
    valgrind \
    # Benchmark
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

RUN pip3 install --no-cache-dir --upgrade pip3 setuptools wheel

RUN pip3 install --no-cache-dir \
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

# Create a non-root user for development
RUN useradd -m -s /bin/bash lgpdev && \
    chown -R lgpdev:lgpdev /workspace

USER lgpdev

# Copy build configuration files
COPY . .

# Default command for development
CMD ["/bin/bash"]

