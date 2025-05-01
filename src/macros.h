#ifndef MACROS_H_INCLUDED
#define MACROS_H_INCLUDED

#define MAX_OMP_THREAD 16

#if defined(__AVX512DQ__)
    #define INCLUDE_AVX512DQ
#endif
#if defined(__AVX512VL__)
    #define INCLUDE_AVX512VL
#endif
#if defined(__AVX512F__)
    #define INCLUDE_AVX512F
#endif
#if defined(__AVX2__)
    #define INCLUDE_AVX2
#endif
#if defined(__AVX__)
    #define INCLUDE_AVX
    #define INCLUDE_SSE2
    #define INCLUDE_SSE3
    #define INCLUDE_SSE4_1
#else
    #if defined(_M_X64) || defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
        #define INCLUDE_SSE2
    #elif defined(__SSE2__)
        #define INCLUDE_SSE2
    #endif
    #if defined(__SSE3__)
        #define INCLUDE_SSE3
    #endif
    #if defined(__SSE4_1__)
        #define INCLUDE_SSE4_1
    #endif
#endif

#ifdef INCLUDE_SSE2
    #include <xmmintrin.h> // SSE2
    #include <emmintrin.h>
    #include <mmintrin.h>
#endif
#ifdef INCLUDE_SSE3
    #include <pmmintrin.h> // SSE3
    #include <tmmintrin.h>
#endif
#ifdef INCLUDE_SSE4_1
    #include <smmintrin.h> // SSE4
#endif
#if defined(INCLUDE_AVX) || defined(INCLUDE_AVX2) || defined(INCLUDE_AVX512F)
    #include <immintrin.h> // AVX AVX2 AVX512
#endif

#endif