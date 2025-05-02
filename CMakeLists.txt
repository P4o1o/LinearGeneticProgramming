cmake_minimum_required(VERSION 3.20)
project(lgp C)

# 1) Directory sorgenti e output
set(SRCDIR src)
file(GLOB SRCFILES "${SRCDIR}/*.c")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(ASSEMBLY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/assembly")

# 2) Flag ottimizzazione e architettura
if (MSVC)
    add_compile_options(/W4 /O2 /std:c17 /openmp)
    add_compile_options(/arch:AVX512)
else()
    add_compile_options(-O3 -Wall -pedantic -std=c2x -msse2 -mavx2 -mavx512f -mavx512dq -mavx512vl -fopenmp)
endif()

# 3) Target eseguibile
add_executable(lgp ${SRCFILES})

# 4) Regola custom per generare .asm (solo MSVC)
if (MSVC)
    foreach(src ${SRCFILES})
        get_filename_component(fname ${src} NAME_WE)
        add_custom_command(
            OUTPUT "${ASSEMBLY_OUTPUT_DIR}/${fname}.asm"
            COMMAND ${CMAKE_C_COMPILER}
                    /FA /FAs
                    /Fo "${ASSEMBLY_OUTPUT_DIR}/${fname}.asm"
                    ${src}
            DEPENDS ${src}
        )
        list(APPEND ASMFILES "${ASSEMBLY_OUTPUT_DIR}/${fname}.asm")
    endforeach()
    add_custom_target(asm ALL DEPENDS ${ASMFILES})
endif()
