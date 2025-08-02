#include "immintrin.h"

void saxpyAVX2(int N,
                float scale,
                float X[],
                float Y[],
                float result[])
{
    __m256 sc = _mm256_set1_ps(scale);
    
    for (int i=0; i<N; i+=8) {

        __m256 X_input = _mm256_loadu_ps(&X[i]);
        __m256 Y_input = _mm256_loadu_ps(&Y[i]);
        
        // __m256 output = _mm256_add_ps( _mm256_mul_ps(sc, X_input), Y_input);

         // use FMA if available
        __m256 output = _mm256_fmadd_ps(sc, X_input, Y_input);

        _mm256_storeu_ps(&result[i], output);
    }
}