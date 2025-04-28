#ifndef MACROS_H_INCLUDED
#define MACROS_H_INCLUDED

#if defined(__AVX512F__) || defined(__AVX2__)
	#include <immintrin.h>
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    #define INCLUDE_SSE2
    #include <xmmintrin.h>
    #include <emmintrin.h>
#endif

#endif