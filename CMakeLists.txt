cmake_minimum_required(VERSION 3.10)
project(lgp C)

# Configurazione standard C e output directories
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Trova il pacchetto OpenMP
find_package(OpenMP REQUIRED)

# Ricerca dei file sorgente
file(GLOB SRCFILES "src/*.c")

# Configurazione flags specifici per compilatore
if(MSVC)
    # Flags per MSVC
    add_compile_options(/O2 /W3 /std:c11 /arch:AVX /openmp)
else()
    # Flags per GCC/Clang
    add_compile_options(-O3 -Wall -pedantic -msse2 -mavx)
endif()

# Creazione eseguibile
add_executable(lgp ${SRCFILES})

# Link delle librerie
target_link_libraries(lgp PRIVATE OpenMP::OpenMP_C)

# Link della math library su sistemi UNIX
if(UNIX)
    target_link_libraries(lgp PRIVATE m)
endif()