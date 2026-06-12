#include "vector_utils.h"
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

float VectorUtils::compute_l2_distance_squared(const float* a, const float* b, int dim) {
#ifdef __AVX512F__
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    
    for (; i + 15 < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }
    
    float dist_sq = _mm512_reduce_add_ps(sum_vec);
    
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    
    return dist_sq;
#else
    float dist_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
#endif
}

float VectorUtils::compute_l2_distance(const float* a, const float* b, int dim) {
    return std::sqrt(compute_l2_distance_squared(a, b, dim));
}

float VectorUtils::compute_norm(const float* vector, int dim) {
#ifdef __AVX512F__
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    
    for (; i + 15 < dim; i += 16) {
        __m512 v = _mm512_loadu_ps(vector + i);
        sum_vec = _mm512_fmadd_ps(v, v, sum_vec);
    }
    
    float sum_sq = _mm512_reduce_add_ps(sum_vec);
    
    for (; i < dim; ++i) {
        sum_sq += vector[i] * vector[i];
    }
    
    return std::sqrt(sum_sq);
#else
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_sq += vector[i] * vector[i];
    }
    return std::sqrt(sum_sq);
#endif
}

float VectorUtils::compute_dot_product(const float* a, const float* b, int dim) {
#ifdef __AVX512F__
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    
    for (; i + 15 < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    
    float dot = _mm512_reduce_add_ps(sum_vec);
    
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    
    return dot;
#else

    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#endif
}

void VectorUtils::compute_dataset_mean(const std::vector<std::vector<float>>& vectors,
                                       std::vector<float>& mean) {
    if (vectors.empty()) return;
    
    int dim = vectors[0].size();
    mean.resize(dim, 0.0f);
    
    for (const auto& vec : vectors) {
        for (int i = 0; i < dim; ++i) {
            mean[i] += vec[i];
        }
    }
    
    for (int i = 0; i < dim; ++i) {
        mean[i] /= vectors.size();
    }
}

void VectorUtils::center_vectors(const std::vector<std::vector<float>>& vectors,
                                 const std::vector<float>& mean,
                                 std::vector<std::vector<float>>& centered) {
    centered.resize(vectors.size());
    int dim = mean.size();
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        centered[i].resize(dim);
        for (int j = 0; j < dim; ++j) {
            centered[i][j] = vectors[i][j] - mean[j];
        }
    }
}

void VectorUtils::compute_bounds(const float* vector, int dim,
                                 float& min_val, float& max_val) {
#ifdef __AVX512F__
    __m512 min_vec = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 max_vec = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 v = _mm512_loadu_ps(vector + i);
        min_vec = _mm512_min_ps(min_vec, v);
        max_vec = _mm512_max_ps(max_vec, v);
    }
    
    min_val = _mm512_reduce_min_ps(min_vec);
    max_val = _mm512_reduce_max_ps(max_vec);
    
    for (; i < dim; ++i) {
        min_val = std::min(min_val, vector[i]);
        max_val = std::max(max_val, vector[i]);
    }
#else
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < dim; ++i) {
        min_val = std::min(min_val, vector[i]);
        max_val = std::max(max_val, vector[i]);
    }
#endif
}


void VectorUtils::scale_vector(const float* vector, float scale, float* result, int dim) {
#ifdef __AVX512F__
    __m512 scale_vec = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 v = _mm512_loadu_ps(vector + i);
        __m512 vr = _mm512_mul_ps(v, scale_vec);
        _mm512_storeu_ps(result + i, vr);
    }
    for (; i < dim; ++i) {
        result[i] = vector[i] * scale;
    }
#else
    for (int i = 0; i < dim; ++i) {
        result[i] = vector[i] * scale;
    }
#endif
}
