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

// C89, C90, C99
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
    #if defined(_MSC_VER)
        #define NORETURN_ATTRIBUTE __declspec(noreturn)
        #define alignas(n) __declspec(align(n))
        #define unreachable() (__assume(0))
    #elif defined(__GNUC__) || defined(__clang__)
        #define NORETURN_ATTRIBUTE __attribute__((noreturn))
        #define alignas(n) __attribute__((aligned(n)))
        #define unreachable() (__builtin_unreachable())
    #else
        #error "No alignment support, compile this program with GCC, clang, MSVC or a compiler with support >= C11"
    #endif
// C11, C17
#elif __STDC_VERSION__ <= 201710L
    #include <stdalign.h>
    #define NORETURN_ATTRIBUTE _Noreturn
    #if defined(_MSC_VER)
        #define unreachable() (__assume(0))
    #elif defined(__GNUC__) || defined(__clang__)
        #define unreachable() (__builtin_unreachable())
    #else
        [[noreturn]] inline void unreachable_impl() {}
        #define unreachable() (unreachable_impl())
    #endif
// C2X, C23
#else
    #define NORETURN_ATTRIBUTE [[noreturn]]
#endif

#define ASSERT(x) \
	do \
		if(!(x)) unreachable(); \
	while(0)


#endif
