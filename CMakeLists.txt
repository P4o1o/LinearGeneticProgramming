# Linear Genetic Programming - Simplified CMake Configuration
cmake_minimum_required(VERSION 3.15)
project(LinearGeneticProgramming LANGUAGES C)

# Configuration options
option(DEBUG "Enable debug mode" OFF)
set(THREADS "16" CACHE STRING "Number of OpenMP threads")

# Set C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED OFF)  # Fallback to older standards

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
    set(ARCH "arm")
else()
    set(ARCH "x86_64")
endif()

# Include necessary modules
include(CheckCCompilerFlag)

# Function to test and add compiler flags
function(add_compiler_flag_if_supported flag)
    string(REGEX REPLACE "[-=+]" "_" flag_var "HAVE_FLAG_${flag}")
    check_c_compiler_flag(${flag} ${flag_var})
    if(${flag_var})
        add_compile_options(${flag})
    endif()
endfunction()

# Find OpenMP
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP: ENABLED")
else()
    message(STATUS "OpenMP: DISABLED")
endif()

# Base compiler flags
add_compile_options(-O3 -Wall -Wextra)
add_compile_definitions(OMP_NUM_THREADS=${THREADS})

# Debug flags  
if(DEBUG)
    add_compile_options(-g -ggdb3)
    add_compile_definitions(LGP_DEBUG=1)
else()
    add_compile_definitions(LGP_DEBUG=0)
endif()

# Auto-detect and add SIMD flags
if(ARCH STREQUAL "x86_64")
    # x86_64 SIMD instructions (in order of preference)
    set(X86_SIMD_FLAGS 
        -mavx512vpopcntdq -mavx512bitalg -mavx512vnni -mavx512vbmi2 -mavx512vbmi 
        -mavx512ifma -mavx512cd -mavx512dq -mavx512bw -mavx512vl -mavx512f 
        -mfma -mavx2 -mavx -msse4.2 -msse4.1 -mssse3 -msse3 -msse2)
    
    foreach(flag ${X86_SIMD_FLAGS})
        add_compiler_flag_if_supported(${flag})
    endforeach()
    
elseif(ARCH STREQUAL "arm64")
    # ARM64 NEON
    add_compiler_flag_if_supported(-march=armv8-a+simd)
    add_compiler_flag_if_supported(-mfpu=neon)
    
elseif(ARCH STREQUAL "arm")
    # ARM32 NEON  
    add_compiler_flag_if_supported(-march=armv7-a)
    add_compiler_flag_if_supported(-mfpu=neon)
endif()

# Architecture-specific optimizations
add_compiler_flag_if_supported(-march=native)
if(NOT HAVE_FLAG__march_native)
    add_compiler_flag_if_supported(-mtune=native)
endif()

# Source files
file(GLOB SOURCES "src/*.c" "src/fitness/*.c")

# Platform-specific libraries
set(PLATFORM_LIBS)
if(WIN32)
    list(APPEND PLATFORM_LIBS winmm)
else()
    list(APPEND PLATFORM_LIBS m)
endif()

# Executable target
add_executable(LGP example.c ${SOURCES})
target_link_libraries(LGP ${PLATFORM_LIBS})

# Add OpenMP if available
if(OpenMP_C_FOUND)
    target_link_libraries(LGP OpenMP::OpenMP_C)
endif()

# Python shared library target
add_library(lgp SHARED ${SOURCES})
set_target_properties(lgp PROPERTIES OUTPUT_NAME "lgp")

# Platform-specific library settings
if(WIN32)
    set_target_properties(lgp PROPERTIES PREFIX "" SUFFIX ".dll")
elseif(APPLE)
    set_target_properties(lgp PROPERTIES PREFIX "lib" SUFFIX ".dylib")
else()
    set_target_properties(lgp PROPERTIES PREFIX "lib" SUFFIX ".so")
endif()

target_link_libraries(lgp ${PLATFORM_LIBS})
if(OpenMP_C_FOUND)
    target_link_libraries(lgp OpenMP::OpenMP_C)
endif()

# Custom target for Python build
add_custom_target(python DEPENDS lgp)

# Display configuration info
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "OS: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "OpenMP: ${OpenMP_C_FOUND}")
message(STATUS "Threads: ${THREADS}")
message(STATUS "Debug: ${DEBUG}")
message(STATUS "===========================")
message(STATUS "")
