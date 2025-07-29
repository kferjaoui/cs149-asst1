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

    //  __m256 result = _mm256_loadu_ps(&output[i]);

    for (int i=0; i<N; i+=8) {

        __m256 x = _mm256_loadu_ps(&values[i]);
        __m256 guess = _mm256_set1_ps(initialGuess);

        __m256 error = _mm256_sub_ps(_mm256_mul_ps( guess,_mm256_mul_ps(guess, x) ), one);

        // Apply abs to error: 
        // Bitwise AND opetaion to convert the sign bit (bit 31) of each of the 8 32-bit floats into a 0 i.e. positive
        error = _mm256_and_ps(error, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        
        // mask is equal to zero only if condition error > kThreshols is false for all the gang lanes
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ));
        while (mask != 0) {
            __m256 three_guess = _mm256_mul_ps(three , guess);
            __m256 g2 = _mm256_mul_ps(guess , guess);
            __m256 g3 = _mm256_mul_ps(g2 , guess);
            __m256 xg3 = _mm256_mul_ps(x , g3);

            guess = _mm256_mul_ps(_mm256_sub_ps( three_guess , xg3) , half );
            error = _mm256_sub_ps( _mm256_mul_ps( x, _mm256_mul_ps(guess , guess) ), one) ;
            
            // Absolute value
            error = _mm256_and_ps(error, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            
            // Update mask 
            mask = _mm256_movemask_ps(_mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ));
        }
        
        _mm256_storeu_ps(&output[i] , _mm256_mul_ps(x, guess));
    }
}