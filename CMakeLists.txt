cmake_minimum_required(VERSION 3.15)
project(LGP C)

# Default variables
set(THREADS 16 CACHE STRING "Number of threads")
set(DEBUG 0 CACHE BOOL "Enable debug mode")
set(C_STD "auto" CACHE STRING "C standard to use (auto, c90, c99, c11, c17, c23, c2x)")

# Detect CPU architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(DETECTED_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i386|i686|x86")
    set(DETECTED_ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DETECTED_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(DETECTED_ARCH "arm")
else()
    set(DETECTED_ARCH "unknown")
endif()

# Auto-detect compiler based on OS preference (if not explicitly set)
if(NOT CMAKE_C_COMPILER)
    if(WIN32)
        # Windows: prefer MSVC > clang > gcc
        find_program(MSVC_COMPILER cl)
        find_program(CLANG_COMPILER clang)
        find_program(GCC_COMPILER gcc)
        
        if(MSVC_COMPILER)
            set(CMAKE_C_COMPILER ${MSVC_COMPILER})
        elseif(CLANG_COMPILER)
            set(CMAKE_C_COMPILER ${CLANG_COMPILER})
        elseif(GCC_COMPILER)
            set(CMAKE_C_COMPILER ${GCC_COMPILER})
        endif()
    elseif(APPLE)
        # macOS: prefer clang > gcc
        find_program(CLANG_COMPILER clang)
        find_program(GCC_COMPILER gcc)
        
        if(CLANG_COMPILER)
            set(CMAKE_C_COMPILER ${CLANG_COMPILER})
        elseif(GCC_COMPILER)
            set(CMAKE_C_COMPILER ${GCC_COMPILER})
        endif()
    else()
        # Linux/FreeBSD: prefer gcc > clang
        find_program(GCC_COMPILER gcc)
        find_program(CLANG_COMPILER clang)
        
        if(GCC_COMPILER)
            set(CMAKE_C_COMPILER ${GCC_COMPILER})
        elseif(CLANG_COMPILER)
            set(CMAKE_C_COMPILER ${CLANG_COMPILER})
        endif()
    endif()
endif()

# Auto-detect best C standard supported
include(CheckCCompilerFlag)

function(test_c_standard std_flag result_var)
    if(MSVC)
        check_c_compiler_flag(${std_flag} ${result_var})
    else()
        check_c_compiler_flag(${std_flag} ${result_var})
    endif()
endfunction()

# Detect best C standard (try from newest to oldest)
set(C_STANDARD_DETECTED "")
set(C_STANDARD_FLAG "")

if(NOT C_STD STREQUAL "auto")
    # User specified a C standard
    if(MSVC)
        set(C_STANDARD_FLAG "/std:${C_STD}")
    else()
        set(C_STANDARD_FLAG "-std=${C_STD}")
    endif()
    set(C_STANDARD_DETECTED ${C_STD})
else()
    # Auto-detect best C standard
    if(MSVC)
        # MSVC standards
        test_c_standard("/std:c17" SUPPORTS_C17)
        if(SUPPORTS_C17)
            set(C_STANDARD_DETECTED "17")
            set(C_STANDARD_FLAG "/std:c17")
        else()
            test_c_standard("/std:c11" SUPPORTS_C11)
            if(SUPPORTS_C11)
                set(C_STANDARD_DETECTED "11")
                set(C_STANDARD_FLAG "/std:c11")
            else()
                set(C_STANDARD_DETECTED "90")
                set(C_STANDARD_FLAG "/std:c90")
            endif()
        endif()
    else()
        # GCC/Clang standards
        test_c_standard("-std=c2x" SUPPORTS_C2X)
        if(SUPPORTS_C2X)
            set(C_STANDARD_DETECTED "2x")
            set(C_STANDARD_FLAG "-std=c2x")
        else()
            test_c_standard("-std=c23" SUPPORTS_C23)
            if(SUPPORTS_C23)
                set(C_STANDARD_DETECTED "23")
                set(C_STANDARD_FLAG "-std=c23")
            else()
                test_c_standard("-std=c17" SUPPORTS_C17)
                if(SUPPORTS_C17)
                    set(C_STANDARD_DETECTED "17")
                    set(C_STANDARD_FLAG "-std=c17")
                else()
                    test_c_standard("-std=c11" SUPPORTS_C11)
                    if(SUPPORTS_C11)
                        set(C_STANDARD_DETECTED "11")
                        set(C_STANDARD_FLAG "-std=c11")
                    else()
                        test_c_standard("-std=c99" SUPPORTS_C99)
                        if(SUPPORTS_C99)
                            set(C_STANDARD_DETECTED "99")
                            set(C_STANDARD_FLAG "-std=c99")
                        else()
                            test_c_standard("-std=c90" SUPPORTS_C90)
                            if(SUPPORTS_C90)
                                set(C_STANDARD_DETECTED "90")
                                set(C_STANDARD_FLAG "-std=c90")
                            else()
                                message(FATAL_ERROR "Compiler does not support C90 or higher. Compilation cannot proceed.")
                            endif()
                        endif()
                    endif()
                endif()
            endif()
        endif()
    endif()
endif()

# Set the detected C standard
if(C_STANDARD_DETECTED STREQUAL "2x")
    set(CMAKE_C_STANDARD 23)  # Use C23 as fallback for C2x
elseif(C_STANDARD_DETECTED STREQUAL "23")
    set(CMAKE_C_STANDARD 23)
elseif(C_STANDARD_DETECTED STREQUAL "17")
    set(CMAKE_C_STANDARD 17)
elseif(C_STANDARD_DETECTED STREQUAL "11")
    set(CMAKE_C_STANDARD 11)
elseif(C_STANDARD_DETECTED STREQUAL "99")
    set(CMAKE_C_STANDARD 99)
else()
    set(CMAKE_C_STANDARD 90)
endif()

set(CMAKE_C_STANDARD_REQUIRED ON)

# Function to test compiler flag support
function(test_compiler_flag flag result)
    string(MAKE_C_IDENTIFIER "HAVE_FLAG_${flag}" FLAG_VAR)
    check_c_compiler_flag(${flag} ${FLAG_VAR})
    if(${FLAG_VAR})
        set(${result} ${flag} PARENT_SCOPE)
    else()
        set(${result} "" PARENT_SCOPE)
    endif()
endfunction()

# Detect OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    set(OPENMP_FLAG ${OpenMP_C_FLAGS})
    set(OPENMP_LIBS ${OpenMP_C_LIBRARIES})
    message(STATUS "OpenMP: ENABLED")
else()
    message(STATUS "OpenMP: DISABLED")
endif()

# Auto-detect best vector instructions based on architecture and compiler
set(VECTOR_FLAGS "")

if(DETECTED_ARCH STREQUAL "x86_64")
    if(MSVC)
        # MSVC x86_64 vector instructions
        set(VECTOR_TESTS "/arch:SSE2" "/arch:AVX" "/arch:AVX2" "/arch:AVX512")
        foreach(flag ${VECTOR_TESTS})
            test_compiler_flag(${flag} result)
            if(result)
                list(APPEND VECTOR_FLAGS ${result})
            endif()
        endforeach()
    else()
        # GCC/Clang x86_64 vector instructions
        set(VECTOR_TESTS 
            "-msse2" "-msse3" "-mssse3" "-msse4.1" "-msse4.2" 
            "-mavx" "-mavx2" "-mfma"
            "-mavx512f" "-mavx512vl" "-mavx512bw" "-mavx512dq" "-mavx512cd"
            "-mavx512ifma" "-mavx512vbmi" "-mavx512vbmi2" "-mavx512vnni"
            "-mavx512bitalg" "-mavx512vpopcntdq"
        )
        foreach(flag ${VECTOR_TESTS})
            test_compiler_flag(${flag} result)
            if(result)
                list(APPEND VECTOR_FLAGS ${result})
            endif()
        endforeach()
    endif()
elseif(DETECTED_ARCH STREQUAL "arm64")
    # ARM64 NEON instructions
    if(NOT MSVC)
        set(VECTOR_TESTS "-mfpu=neon" "-march=armv8-a+simd")
        foreach(flag ${VECTOR_TESTS})
            test_compiler_flag(${flag} result)
            if(result)
                list(APPEND VECTOR_FLAGS ${result})
            endif()
        endforeach()
    endif()
elseif(DETECTED_ARCH STREQUAL "arm")
    # ARM32 NEON instructions
    if(NOT MSVC)
        set(VECTOR_TESTS "-mfpu=neon" "-march=armv7-a")
        foreach(flag ${VECTOR_TESTS})
            test_compiler_flag(${flag} result)
            if(result)
                list(APPEND VECTOR_FLAGS ${result})
            endif()
        endforeach()
    endif()
endif()

# Auto-detect architecture-specific optimizations
set(ARCH_FLAGS "")
if(MSVC)
    test_compiler_flag("/favor:INTEL64" ARCH_FLAG)
    if(NOT ARCH_FLAG)
        test_compiler_flag("/favor:AMD64" ARCH_FLAG)
    endif()
else()
    test_compiler_flag("-march=native" ARCH_FLAG)
    if(NOT ARCH_FLAG)
        test_compiler_flag("-mtune=native" ARCH_FLAG)
    endif()
endif()

if(ARCH_FLAG)
    set(ARCH_FLAGS ${ARCH_FLAG})
endif()

# Set compiler-specific base flags
if(MSVC)
    set(BASE_CFLAGS "/O2" "/W3" "/DOMP_NUM_THREADS=${THREADS}" "/DLGP_DEBUG=${DEBUG}")
    if(NOT C_STANDARD_FLAG STREQUAL "")
        list(APPEND BASE_CFLAGS ${C_STANDARD_FLAG})
    endif()
    set(DEBUG_FLAGS "/Zi" "/DEBUG")
else()
    set(BASE_CFLAGS "-O3" "-Wall" "-Wextra" "-pedantic" "-DOMP_NUM_THREADS=${THREADS}" "-DLGP_DEBUG=${DEBUG}")
    if(NOT C_STANDARD_FLAG STREQUAL "")
        list(APPEND BASE_CFLAGS ${C_STANDARD_FLAG})
    endif()
    set(DEBUG_FLAGS "-ggdb3" "-fsanitize=undefined" "-fsanitize=signed-integer-overflow" "-pg" "-g")
endif()

# Combine all flags
set(ALL_CFLAGS ${BASE_CFLAGS} ${VECTOR_FLAGS} ${ARCH_FLAGS})
if(DEBUG)
    list(APPEND ALL_CFLAGS ${DEBUG_FLAGS})
endif()

# Add OpenMP flags to compilation if available
if(OpenMP_C_FOUND)
    list(APPEND ALL_CFLAGS ${OPENMP_FLAG})
endif()

# Apply flags to CMAKE_C_FLAGS
string(REPLACE ";" " " CMAKE_C_FLAGS "${ALL_CFLAGS}")

# Source files
file(GLOB SOURCES "src/*.c")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")

# Platform-specific libraries
set(PLATFORM_LIBS)
if(WIN32)
    list(APPEND PLATFORM_LIBS winmm)
else()
    list(APPEND PLATFORM_LIBS m)
endif()

# Executable target
add_executable(LGP src/main.c ${SOURCES})
target_link_libraries(LGP ${PLATFORM_LIBS})
if(OpenMP_C_FOUND)
    if(MSVC)
        target_link_libraries(LGP ${OpenMP_C_LIBRARIES})
    else()
        target_link_libraries(LGP OpenMP::OpenMP_C)
    endif()
endif()

# Python shared library target
add_library(lgp SHARED ${SOURCES})
set_target_properties(lgp PROPERTIES
    OUTPUT_NAME "lgp"
    POSITION_INDEPENDENT_CODE ON
)

# Platform-specific library settings
if(WIN32)
    set_target_properties(lgp PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
    )
elseif(APPLE)
    set_target_properties(lgp PROPERTIES
        PREFIX "lib"
        SUFFIX ".dylib"
    )
else()
    set_target_properties(lgp PROPERTIES
        PREFIX "lib"
        SUFFIX ".so"
    )
endif()

target_link_libraries(lgp ${PLATFORM_LIBS})
if(OpenMP_C_FOUND)
    if(MSVC)
        target_link_libraries(lgp ${OpenMP_C_LIBRARIES})
    else()
        target_link_libraries(lgp OpenMP::OpenMP_C)
    endif()
endif()

# Custom targets for convenience
add_custom_target(python DEPENDS lgp)

# Display configuration info
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "OS: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Architecture: ${DETECTED_ARCH}")
message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Standard: ${C_STANDARD_FLAG} (C${C_STANDARD_DETECTED})")
message(STATUS "OpenMP: ${OpenMP_C_FOUND}")
message(STATUS "Vector flags: ${VECTOR_FLAGS}")
message(STATUS "Architecture opts: ${ARCH_FLAGS}")
message(STATUS "Threads: ${THREADS}")
message(STATUS "Debug: ${DEBUG}")
message(STATUS "===========================")
message(STATUS "")