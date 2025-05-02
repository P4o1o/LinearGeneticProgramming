#ifndef MACROS_H_INCLUDED
#define MACROS_H_INCLUDED

#define NUMBER_OF_OMP_THREADS 16

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

#define ASSERT(x) \
	do \
		if(!(x)) unreachable(); \
	while(0)


// C89, C90, C99
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
    #if defined(_MSC_VER)
        #define NORETURN_ATTRIBUTE __declspec(noreturn)
        #define alignas(n) __declspec(align(n))
        #define unreachable() (__assume(0))
        #include <malloc.h>
        #define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
        #define aligned_realloc(ptr, old_size, new_size, alignment) _aligned_realloc((ptr), (new_size), (alignment))
        #define aligned_free(ptr) _aligned_free(ptr)
    #elif defined(__GNUC__) || defined(__clang__)
        #define NORETURN_ATTRIBUTE __attribute__((noreturn))
        #define alignas(n) __attribute__((aligned(n)))
        #define unreachable() (__builtin_unreachable())
        #include <stdlib.h>
        #include <string.h>
        static inline void *aligned_alloc(size_t alignment, size_t size) {
            void *p;
            return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
        }
        static inline void *aligned_realloc(void *ptr, size_t old_size, size_t new_size, size_t alignment) {
            ASSERT(ptr != NULL);
            void *newp;
            if (posix_memalign(&newp, alignment, new_size) != 0)
                return NULL;
            size_t to_copy = old_size < new_size ? old_size : new_size;
            memcpy(newp, ptr, to_copy);
            free(ptr);
            return newp;
        }
        #define aligned_free(ptr) free(ptr)
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
        NORETURN_ATTRIBUTE inline void unreachable_impl() {}
        #define unreachable() (unreachable_impl())
    #endif
    #include <stdlib.h>
    #include <string.h>
    static inline void *aligned_realloc(void *ptr, size_t old_size, size_t new_size, size_t alignment) {
        ASSERT(ptr != NULL);
        void *__new = aligned_alloc((alignment),(new_size));
        if (__new) {
            size_t to_copy = old_size < new_size ? old_size : new_size;
            memcpy(__new, (ptr), (to_copy));
            free(ptr);
        }
        return __new;
    }
    #define aligned_free(ptr) free(ptr)
// C2X, C23
#else
    #define NORETURN_ATTRIBUTE [[noreturn]]
    #include <stdlib.h>
    #define aligned_realloc(ptr, old_size, new_size, alignment) reallocaligned((ptr),(size),(alignment))
    #define aligned_free(ptr) free(ptr)
#endif

#endif
