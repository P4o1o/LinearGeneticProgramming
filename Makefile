# User configurable variables
DEBUG ?= 0
THREADS ?= 16
C_STD ?= auto

# Check if CC was set by user or is the default
ifeq ($(origin CC),default)
    CC_AUTO := true
else
    CC_AUTO := false
endif

# Detect OS and set platform-specific variables
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE_EXT := .exe
    LIB_EXT := .dll
    LIB_PREFIX := 
    MKDIR := mkdir
    RM := del /Q
    RMDIR := rmdir /S /Q
    PATH_SEP := \\
    NULL_DEVICE := nul
    LIBFLAGS_EXTRA := -lwinmm
else
    UNAME_S := $(shell uname -s)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
        LIB_EXT := .so
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
        LIB_EXT := .dylib
    endif
    ifeq ($(UNAME_S),FreeBSD)
        DETECTED_OS := FreeBSD
        LIB_EXT := .so
    endif
    EXE_EXT := 
    LIB_PREFIX := lib
    MKDIR := mkdir -p
    RM := rm -f
    RMDIR := rm -rf
    PATH_SEP := /
    NULL_DEVICE := /dev/null
    LIBFLAGS_EXTRA := -lm
endif

# Detect CPU architecture
ifeq ($(UNAME_M),x86_64)
    DETECTED_ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
    DETECTED_ARCH := x86_64
else ifeq ($(UNAME_M),i386)
    DETECTED_ARCH := x86
else ifeq ($(UNAME_M),i686)
    DETECTED_ARCH := x86
else ifeq ($(UNAME_M),arm64)
    DETECTED_ARCH := arm64
else ifeq ($(UNAME_M),aarch64)
    DETECTED_ARCH := arm64
else ifeq ($(findstring arm,$(UNAME_M)),arm)
    DETECTED_ARCH := arm
else
    DETECTED_ARCH := unknown
endif

# Function to check if a command exists
command_exists = $(shell command -v $(1) 2>$(NULL_DEVICE) && echo "yes" || echo "no")

# Auto-detect compiler based on OS preference
ifeq ($(CC_AUTO),true)
    ifeq ($(DETECTED_OS),Windows)
        # Windows: prefer MSVC > clang > gcc
        ifeq ($(call command_exists,cl),yes)
            CC := cl
        else ifeq ($(call command_exists,clang),yes)
            CC := clang
        else ifeq ($(call command_exists,gcc),yes)
            CC := gcc
        else
            CC := gcc
        endif
    else ifeq ($(DETECTED_OS),macOS)
        # macOS: prefer clang > gcc
        ifeq ($(call command_exists,clang),yes)
            CC := clang
        else ifeq ($(call command_exists,gcc),yes)
            CC := gcc
        else
            CC := clang
        endif
    else
        # Linux/FreeBSD: prefer gcc > clang
        ifeq ($(call command_exists,gcc),yes)
            CC := gcc
        else ifeq ($(call command_exists,clang),yes)
            CC := clang
        else
            CC := gcc
        endif
    endif
endif

# Detect compiler type
ifeq ($(CC),cl)
    COMPILER_TYPE := MSVC
else ifeq ($(CC),clang)
    COMPILER_TYPE := CLANG
else ifeq ($(CC),gcc)
    COMPILER_TYPE := GCC
else
    # For other compiler names, detect from actual output
    CC_VERSION := $(shell $(CC) --version 2>$(NULL_DEVICE) | head -1)
    ifneq ($(findstring clang,$(CC_VERSION)),)
        COMPILER_TYPE := CLANG
    else ifneq ($(findstring gcc,$(CC_VERSION)),)
        COMPILER_TYPE := GCC
    else
        COMPILER_TYPE := UNKNOWN
    endif
endif

# Function to test compiler support for flags
ifeq ($(COMPILER_TYPE),MSVC)
    test_flag = $(shell $(CC) $(1) -c $(NULL_DEVICE) 2>$(NULL_DEVICE) && echo "$(1)")
else
    test_flag = $(shell $(CC) $(1) -x c -c $(NULL_DEVICE) -o $(NULL_DEVICE) 2>$(NULL_DEVICE) && echo "$(1)")
endif

# Auto-detect best C standard supported
ifneq ($(C_STD),auto)
    # User specified a C standard
    ifeq ($(COMPILER_TYPE),MSVC)
        C_STD_FLAG := /std:$(C_STD)
    else
        C_STD_FLAG := -std=$(C_STD)
    endif
else
    # Auto-detect best C standard
    ifeq ($(COMPILER_TYPE),MSVC)
        C_STD_FLAG := $(call test_flag,/std:c17)
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,/std:c11)
        endif
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := /std:c90
        endif
    else
        C_STD_FLAG := $(call test_flag,-std=c2x)
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,-std=c23)
        endif
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,-std=c17)
        endif
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,-std=c11)
        endif
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,-std=c99)
        endif
        ifeq ($(C_STD_FLAG),)
            C_STD_FLAG := $(call test_flag,-std=c90)
        endif
        # If no standard is supported, fail the build
        ifeq ($(C_STD_FLAG),)
            $(error Compiler does not support C90 or higher. Compilation cannot proceed.)
        endif
    endif
endif

# Detect OpenMP support
ifeq ($(COMPILER_TYPE),MSVC)
    OPENMP_FLAG := $(call test_flag,/openmp)
else
    OPENMP_FLAG := $(call test_flag,-fopenmp)
endif

# Auto-detect best vector instructions based on architecture and compiler
VECTOR_FLAGS := 
ifeq ($(DETECTED_ARCH),x86_64)
    ifeq ($(COMPILER_TYPE),MSVC)
        # MSVC x86_64 vector instructions
        VECTOR_FLAGS += $(call test_flag,/arch:SSE2)
        VECTOR_FLAGS += $(call test_flag,/arch:AVX)
        VECTOR_FLAGS += $(call test_flag,/arch:AVX2)
        VECTOR_FLAGS += $(call test_flag,/arch:AVX512)
    else
        # GCC/Clang x86_64 vector instructions
        VECTOR_FLAGS += $(call test_flag,-msse2)
        VECTOR_FLAGS += $(call test_flag,-msse3)
        VECTOR_FLAGS += $(call test_flag,-mssse3)
        VECTOR_FLAGS += $(call test_flag,-msse4.1)
        VECTOR_FLAGS += $(call test_flag,-msse4.2)
        VECTOR_FLAGS += $(call test_flag,-mavx)
        VECTOR_FLAGS += $(call test_flag,-mavx2)
        VECTOR_FLAGS += $(call test_flag,-mfma)
        VECTOR_FLAGS += $(call test_flag,-mavx512f)
        VECTOR_FLAGS += $(call test_flag,-mavx512vl)
        VECTOR_FLAGS += $(call test_flag,-mavx512bw)
        VECTOR_FLAGS += $(call test_flag,-mavx512dq)
        VECTOR_FLAGS += $(call test_flag,-mavx512cd)
        VECTOR_FLAGS += $(call test_flag,-mavx512ifma)
        VECTOR_FLAGS += $(call test_flag,-mavx512vbmi)
        VECTOR_FLAGS += $(call test_flag,-mavx512vbmi2)
        VECTOR_FLAGS += $(call test_flag,-mavx512vnni)
        VECTOR_FLAGS += $(call test_flag,-mavx512bitalg)
        VECTOR_FLAGS += $(call test_flag,-mavx512vpopcntdq)
    endif
else ifeq ($(DETECTED_ARCH),arm64)
    # ARM64 NEON instructions
    ifneq ($(COMPILER_TYPE),MSVC)
        VECTOR_FLAGS += $(call test_flag,-mfpu=neon)
        VECTOR_FLAGS += $(call test_flag,-march=armv8-a+simd)
    endif
else ifeq ($(DETECTED_ARCH),arm)
    # ARM32 NEON instructions
    ifneq ($(COMPILER_TYPE),MSVC)
        VECTOR_FLAGS += $(call test_flag,-mfpu=neon)
        VECTOR_FLAGS += $(call test_flag,-march=armv7-a)
    endif
endif

# Auto-detect architecture-specific optimizations
ifeq ($(COMPILER_TYPE),MSVC)
    ARCH_FLAGS := $(call test_flag,/favor:INTEL64)
    ifeq ($(ARCH_FLAGS),)
        ARCH_FLAGS := $(call test_flag,/favor:AMD64)
    endif
else
    ARCH_FLAGS := $(call test_flag,-march=native)
    ifeq ($(ARCH_FLAGS),)
        ARCH_FLAGS := $(call test_flag,-mtune=native)
    endif
endif

# Set compiler-specific base flags
ifeq ($(COMPILER_TYPE),MSVC)
    BASE_CFLAGS = /O2 /W3 /DOMP_NUM_THREADS=$(THREADS) /DLGP_DEBUG=$(DEBUG) $(C_STD_FLAG)
    DFLAGS = /Zi /DEBUG
    PIC_FLAG = 
    SHARED_FLAG = /LD
else
    BASE_CFLAGS = -O3 -Wall -Wextra -pedantic -DOMP_NUM_THREADS=$(THREADS) -DLGP_DEBUG=$(DEBUG) $(C_STD_FLAG)
    DFLAGS = -ggdb3 -fsanitize=undefined -fsanitize=signed-integer-overflow -pg -g
    PIC_FLAG = -fPIC
    SHARED_FLAG = -shared
endif

# Combine all flags
CFLAGS = $(BASE_CFLAGS) $(VECTOR_FLAGS) $(ARCH_FLAGS) $(OPENMP_FLAG)
LIBFLAGS = $(LIBFLAGS_EXTRA) $(OPENMP_FLAG)

SRCDIR = src
BINDIR = bin
SRCFILES = $(wildcard $(SRCDIR)$(PATH_SEP)*.c)
OBJFILES = $(patsubst $(SRCDIR)$(PATH_SEP)%.c,$(BINDIR)$(PATH_SEP)%.o,$(SRCFILES))

.PHONY: all python clean info

# Default target - build executable and show info
all: LGP$(EXE_EXT)
	@echo ""
	@echo "=== Build Completed ==="
	@echo "OS: $(DETECTED_OS)"
	@echo "Architecture: $(DETECTED_ARCH)"
	@echo "Compiler: $(CC) ($(COMPILER_TYPE))"
	@echo "C Standard: $(C_STD_FLAG)"
	@echo "OpenMP: $(if $(OPENMP_FLAG),ENABLED,DISABLED)"
	@echo "Vector flags: $(VECTOR_FLAGS)"
	@echo "Architecture opts: $(ARCH_FLAGS)"
	@echo "Threads: $(THREADS)"
	@echo "Debug: $(DEBUG)"
	@echo "======================="

# Show build information
info:
	@echo "=== Build Configuration ==="
	@echo "OS: $(DETECTED_OS)"
	@echo "Architecture: $(DETECTED_ARCH)"
	@echo "Compiler: $(CC) ($(COMPILER_TYPE))"
	@echo "C Standard: $(C_STD_FLAG)"
	@echo "OpenMP: $(if $(OPENMP_FLAG),ENABLED,DISABLED)"
	@echo "Vector flags: $(VECTOR_FLAGS)"
	@echo "Architecture opts: $(ARCH_FLAGS)"
	@echo "Threads: $(THREADS)"
	@echo "Debug: $(DEBUG)"
	@echo "=========================="

# Build executable
LGP$(EXE_EXT): $(OBJFILES)
	@echo "=== Building executable ==="
	$(CC) $(CFLAGS) $(if $(DEBUG:0=),$(DFLAGS)) -o LGP$(EXE_EXT) $(OBJFILES) $(LIBFLAGS)

# Build Python shared library
python: $(LIB_PREFIX)lgp$(LIB_EXT)

$(LIB_PREFIX)lgp$(LIB_EXT): $(filter-out $(BINDIR)$(PATH_SEP)main.o,$(OBJFILES))
	@echo "=== Building Python library ==="
ifeq ($(COMPILER_TYPE),MSVC)
	$(CC) $(CFLAGS) $(SHARED_FLAG) -o $(LIB_PREFIX)lgp$(LIB_EXT) $(filter-out $(BINDIR)$(PATH_SEP)main.o,$(OBJFILES)) $(LIBFLAGS)
else
	$(CC) $(CFLAGS) $(SHARED_FLAG) $(PIC_FLAG) -o $(LIB_PREFIX)lgp$(LIB_EXT) $(filter-out $(BINDIR)$(PATH_SEP)main.o,$(OBJFILES)) $(LIBFLAGS)
endif

$(BINDIR)$(PATH_SEP)%.o: $(SRCDIR)$(PATH_SEP)%.c | $(BINDIR)
ifeq ($(COMPILER_TYPE),MSVC)
	$(CC) $(CFLAGS) $(if $(DEBUG:0=),$(DFLAGS)) -c $< -Fo$@
else
	$(CC) $(CFLAGS) $(if $(DEBUG:0=),$(DFLAGS)) $(PIC_FLAG) -c $< -o $@
endif

$(BINDIR):
	$(MKDIR) $(BINDIR)

clean:
ifeq ($(DETECTED_OS),Windows)
	-$(RM) $(BINDIR)$(PATH_SEP)*.o LGP$(EXE_EXT) $(LIB_PREFIX)lgp$(LIB_EXT) 2>$(NULL_DEVICE)
	-$(RMDIR) $(BINDIR) 2>$(NULL_DEVICE)
else
	$(RM) $(BINDIR)$(PATH_SEP)*.o LGP$(EXE_EXT) $(LIB_PREFIX)lgp$(LIB_EXT)
	$(RMDIR) $(BINDIR)
endif
