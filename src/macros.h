#ifndef MACROS_H_INCLUDED
#define MACROS_H_INCLUDED

#ifdef _OPENMP
    #include <omp.h>
    #define INCLUDE_OMP
#else
    #define NUMBER_OF_OMP_THREADS 1
#endif

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

#if defined(INCLUDE_AVX512F)
    #define VECT_ALIGNMENT 64
#elif defined(INCLUDE_AVX2)
    #define VECT_ALIGNMENT 32
#elif defined(INCLUDE_SSE2)
    #define VECT_ALIGNMENT 16
#endif

// C89, C90, C99
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
    #if defined(_MSC_VER)
        #define NORETURN_ATTRIBUTE __declspec(noreturn)
        #define alignas(n) __declspec(align(n))
        #define unreachable() (__assume(0))
        #define UNUSED_ATTRIBUTE
        #include <malloc.h>
        #define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
        #define aligned_free(ptr) _aligned_free(ptr)
    #elif defined(__GNUC__) || defined(__clang__)
        #define NORETURN_ATTRIBUTE __attribute__((noreturn))
        #define alignas(n) __attribute__((aligned(n)))
        #define unreachable() (__builtin_unreachable())
        #define UNUSED_ATTRIBUTE __attribute__((unused)) 
        #include <stdlib.h>
        static inline void *aligned_alloc(size_t alignment, size_t size) {
            void *p;
            return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
        }
        #define aligned_free(ptr) free(ptr)
    #else
        #warning "No alignment support, compile this program with GCC, clang, MSVC or a compiler with support >= C11 for better performance."
        #define NORETURN_ATTRIBUTE 
        inline void unreachable_impl() {
            while(1){}
        }
        #define unreachable() (unreachable_impl())
        #define UNUSED_ATTRIBUTE 
    #endif
// C11, C17
#elif __STDC_VERSION__ <= 201710L
    #include <stdalign.h>
    #define NORETURN_ATTRIBUTE _Noreturn
    #if defined(_MSC_VER)
        #define unreachable() (__assume(0))
        #define UNUSED_ATTRIBUTE 
    #elif defined(__GNUC__) || defined(__clang__)
        #define unreachable() (__builtin_unreachable())
        #define UNUSED_ATTRIBUTE __attribute__((unused))
    #else
        NORETURN_ATTRIBUTE inline void unreachable_impl() {}
        #define unreachable() (unreachable_impl())
        #define UNUSED_ATTRIBUTE 
    #endif
    #include <stdlib.h>
    #define aligned_free(ptr) free(ptr)
// C2X, C23
#else
    #define C2X_SUPPORTED
    #define NORETURN_ATTRIBUTE [[noreturn]]
    #define UNUSED_ATTRIBUTE [[maybe_unused]]
    #include <stdlib.h>
    #define aligned_free(ptr) free(ptr)
#endif

#ifndef VECT_ALIGNMENT
    #define VECT_ALIGNMENT 0
    #define alignas(n) 
    #define aligned_alloc(alignment, size) malloc(size)
    #define aligned_free(ptr) free(ptr)
#endif

#define ASSERT(x) \
	do \
		if(!(x)) unreachable(); \
	while(0)

#endif
