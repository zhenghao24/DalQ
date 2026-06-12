#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef __AVX512F__
#include <immintrin.h>
#endif


class VectorUtils {
public:
    static float compute_l2_distance_squared(const float* a, const float* b, int dim);
    static float compute_l2_distance(const float* a, const float* b, int dim);
    static float compute_norm(const float* vector, int dim);
    static float compute_dot_product(const float* a, const float* b, int dim);
    
    static void compute_dataset_mean(const std::vector<std::vector<float>>& vectors,
                                     std::vector<float>& mean);
    
    static void center_vectors(const std::vector<std::vector<float>>& vectors,
                              const std::vector<float>& mean,
                              std::vector<std::vector<float>>& centered);
    
    static void compute_bounds(const float* vector, int dim,
                              float& min_val, float& max_val);
    
    static void scale_vector(const float* vector, float scale, float* result, int dim);

    static inline void subtract_vectors(const float* a, const float* b, float* result, int dim) {
    #ifdef __AVX512F__
        int i = 0;
        for (; i + 15 < dim; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vr = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(result + i, vr);
        }
        for (; i < dim; ++i) {
            result[i] = a[i] - b[i];
        }
    #else
        for (int i = 0; i < dim; ++i) {
            result[i] = a[i] - b[i];
        }
    #endif
    }
};


#endif
