# Linear Genetic Programming - Simplified Cross-Platform Makefile
# Automatic SIMD detection and multi-platform support

# User configurable variables
DEBUG ?= 0
THREADS ?= 16
CC ?= 

# Detect OS
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE_EXT := .exe
    LIB_EXT := .dll
    LIB_PREFIX := 
    MKDIR := mkdir
    RM := del /Q /F
    RMDIR := rmdir /S /Q
    PATH_SEP := \\
    NULL_DEVICE := nul
    EXTRA_LIBS := -lwinmm
else
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    else ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    else ifeq ($(UNAME_S),FreeBSD)
        DETECTED_OS := FreeBSD
    else
        DETECTED_OS := Unix
    endif
    EXE_EXT := 
    LIB_EXT := .so
    LIB_PREFIX := lib
    MKDIR := mkdir -p
    RM := rm -f
    RMDIR := rm -rf
    PATH_SEP := /
    NULL_DEVICE := /dev/null
    EXTRA_LIBS := -lm
    ifeq ($(DETECTED_OS),macOS)
        LIB_EXT := .dylib
    endif
endif

# Auto-detect compiler if not specified
ifeq ($(CC),)
    ifeq ($(DETECTED_OS),Windows)
        CC := gcc
    else
        CC := $(shell command -v clang 2>$(NULL_DEVICE) && echo clang || echo gcc)
    endif
endif

# Detect CPU architecture
UNAME_M := $(shell uname -m 2>/dev/null || echo "x86_64")
ifeq ($(UNAME_M),x86_64)
    ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
    ARCH := x86_64
else ifeq ($(UNAME_M),arm64)
    ARCH := arm64
else ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
else ifneq ($(findstring arm,$(UNAME_M)),)
    ARCH := arm
else
    ARCH := x86_64
endif

# Function to test compiler flags
test_flag = $(shell $(CC) $(1) -x c -c $(NULL_DEVICE) -o $(NULL_DEVICE) 2>$(NULL_DEVICE) && echo "$(1)")

# Auto-detect C standard
C_STD := $(or $(call test_flag,-std=c2x),$(call test_flag,-std=c23),$(call test_flag,-std=c17),$(call test_flag,-std=c11),-std=c99)

# Detect OpenMP
OPENMP_FLAG := $(call test_flag,-fopenmp)

# Auto-detect SIMD flags based on architecture
SIMD_FLAGS := 
ifeq ($(ARCH),x86_64)
    # Test x86_64 SIMD instructions in order of preference
    SIMD_TESTS := -mavx512vpopcntdq -mavx512bitalg -mavx512vnni -mavx512vbmi2 -mavx512vbmi -mavx512ifma -mavx512cd -mavx512dq -mavx512bw -mavx512vl -mavx512f -mfma -mavx2 -mavx -msse4.2 -msse4.1 -mssse3 -msse3 -msse2
    $(foreach flag,$(SIMD_TESTS),$(eval SIMD_FLAGS += $(call test_flag,$(flag))))
else ifeq ($(ARCH),arm64)
    # ARM64 NEON
    SIMD_FLAGS += $(call test_flag,-march=armv8-a+simd)
    SIMD_FLAGS += $(call test_flag,-mfpu=neon)
else ifeq ($(ARCH),arm)
    # ARM32 NEON
    SIMD_FLAGS += $(call test_flag,-march=armv7-a)
    SIMD_FLAGS += $(call test_flag,-mfpu=neon)
endif

# Architecture optimization
ARCH_OPT := $(or $(call test_flag,-march=native),$(call test_flag,-mtune=native))

# Base compiler flags
BASE_FLAGS := -O3 -Wall -Wextra -DOMP_NUM_THREADS=$(THREADS) -DLGP_DEBUG=$(DEBUG) $(C_STD)
ifeq ($(DEBUG),1)
    BASE_FLAGS += -g -ggdb3 -fsanitize=undefined -fsanitize=signed-integer-overflow
endif

# Combine all flags
CFLAGS := $(BASE_FLAGS) $(SIMD_FLAGS) $(ARCH_OPT) $(OPENMP_FLAG)
LDFLAGS := $(EXTRA_LIBS) $(OPENMP_FLAG)

# Directories and files
SRCDIR := src
BINDIR := bin
SOURCES := $(wildcard $(SRCDIR)$(PATH_SEP)*.c)
OBJECTS := $(patsubst $(SRCDIR)$(PATH_SEP)%.c,$(BINDIR)$(PATH_SEP)%.o,$(SOURCES))

# Targets
.PHONY: all clean info python test

# Default target
all: LGP$(EXE_EXT)
	@echo ""
	@echo "=== Build Complete ==="
	@echo "OS: $(DETECTED_OS)"
	@echo "Architecture: $(ARCH)"
	@echo "Compiler: $(CC)"
	@echo "C Standard: $(C_STD)"
	@echo "OpenMP: $(if $(OPENMP_FLAG),ENABLED,DISABLED)"
	@echo "SIMD: $(if $(SIMD_FLAGS),$(SIMD_FLAGS),NONE)"
	@echo "Arch Opt: $(if $(ARCH_OPT),$(ARCH_OPT),NONE)"
	@echo "======================"

# Build info
info:
	@echo "=== Build Configuration ==="
	@echo "OS: $(DETECTED_OS)"
	@echo "Architecture: $(ARCH)"  
	@echo "Compiler: $(CC)"
	@echo "C Standard: $(C_STD)"
	@echo "OpenMP: $(if $(OPENMP_FLAG),ENABLED,DISABLED)"
	@echo "SIMD Flags: $(if $(SIMD_FLAGS),$(SIMD_FLAGS),NONE)"
	@echo "Arch Opt: $(if $(ARCH_OPT),$(ARCH_OPT),NONE)"
	@echo "Threads: $(THREADS)"
	@echo "Debug: $(DEBUG)"
	@echo "========================="

# Build executable
LGP$(EXE_EXT): $(OBJECTS) example.c
	@echo "Building executable..."
	$(CC) $(CFLAGS) -o $@ example.c $(OBJECTS) $(LDFLAGS)

# Build Python shared library  
python: $(LIB_PREFIX)lgp$(LIB_EXT)

$(LIB_PREFIX)lgp$(LIB_EXT): $(OBJECTS)
	@echo "Building Python library..."
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $(OBJECTS) $(LDFLAGS)

# Compile object files
$(BINDIR)$(PATH_SEP)%.o: $(SRCDIR)$(PATH_SEP)%.c | $(BINDIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Create bin directory
$(BINDIR):
	$(MKDIR) $(BINDIR)

# Run tests
test: all python
	@echo "Running tests..."
	@chmod +x test.sh 2>$(NULL_DEVICE) || true
	@./test.sh

# Clean
clean:
ifeq ($(DETECTED_OS),Windows)
	-$(RM) $(BINDIR)$(PATH_SEP)*.o LGP$(EXE_EXT) $(LIB_PREFIX)lgp$(LIB_EXT) 2>$(NULL_DEVICE)
	-$(RMDIR) $(BINDIR) 2>$(NULL_DEVICE)
else
	$(RM) $(BINDIR)$(PATH_SEP)*.o LGP$(EXE_EXT) $(LIB_PREFIX)lgp$(LIB_EXT)
	$(RMDIR) $(BINDIR)
endif
