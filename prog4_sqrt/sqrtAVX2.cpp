// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>

#include <immintrin.h>


void sqrtAVX2(int N,
                float initialGuess,
                float values[],
                float output[])
{
    __m256 const kThreshold = _mm256_set1_ps(0.00001f);
    __m256 const half = _mm256_set1_ps(0.5f);
    __m256 const one = _mm256_set1_ps(1.f);
    __m256 const three = _mm256_set1_ps(3.f);

    for (int i=0; i<N; i+=8) {

        __m256 x = _mm256_loadu_ps(&values[i]);
        __m256 guess = _mm256_set1_ps(initialGuess);

        __m256 error = _mm256_sub_ps(_mm256_mul_ps( guess,_mm256_mul_ps(guess, x) ), one);
        // TODO: Apply abs to error
        
        // TODO: Corret illegal comparison below
        while (error > kThreshold) {
            __m256 three_guess = _mm256_mul_ps(three , guess);
            __m256 g2 = _mm256_mul_ps(guess , guess);
            __m256 g3 = _mm256_mul_ps(g2 , guess);
            __m256 xg3 = _mm256_mul_ps(x , g3);

            guess = _mm256_mul_ps(_mm256_sub_ps( three_guess , xg3) , half );
            error = _mm256_abs_epi32( _mm256_sub_ps( _mm256_mul_ps( x, _mm256_mul_ps(guess , guess) ), one) );
            // TODO: Apply abs to error
        }
        
        __m256_storeu_ps(&output[i] , _mm256_mul_ps(x, guess));
    }
}