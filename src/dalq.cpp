#include "dalq.h"
#include "vector_utils.h"
#include <cstring>
#include <iostream>
#include <limits>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

DALQ::DALQ(int dim, int bits, float clip_factor, bool enable_code_search, int search_rounds)
    : dim_(dim), bits_(bits), clip_factor_(clip_factor), 
      enable_code_search_(enable_code_search), search_rounds_(search_rounds) {
    num_levels_ = 1 << bits;
    inv_num_levels_ = 1.0f / (num_levels_ - 1);
}

void DALQ::decode_codes(const uint8_t* codes, float clipped_lower, 
                       float clipped_upper, float* output) const {
    float effective_range = clipped_upper - clipped_lower;
    
#ifdef __AVX512F__
    __m512 range_vec = _mm512_set1_ps(effective_range);
    __m512 lower_vec = _mm512_set1_ps(clipped_lower);
    __m512 inv_levels_vec = _mm512_set1_ps(inv_num_levels_);
    
    int i = 0;
    for (; i + 15 < dim_; i += 16) {
        __m128i codes_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_i32 = _mm512_cvtepu8_epi32(codes_u8);
        __m512 codes_f32 = _mm512_cvtepi32_ps(codes_i32);
        __m512 normalized = _mm512_mul_ps(codes_f32, inv_levels_vec);
        __m512 result = _mm512_fmadd_ps(normalized, range_vec, lower_vec);
        _mm512_storeu_ps(output + i, result);
    }
    for (; i < dim_; ++i) {
        float normalized = codes[i] * inv_num_levels_;
        output[i] = clipped_lower + normalized * effective_range;
    }
#else
    for (int i = 0; i < dim_; ++i) {
        float normalized = codes[i] * inv_num_levels_;
        output[i] = clipped_lower + normalized * effective_range;
    }
#endif
}

float DALQ::compute_cosine_similarity(const float* reconstructed, 
                                     const float* original, 
                                     int dim) const {
#ifdef __AVX512F__
    __m512 dot_vec = _mm512_setzero_ps();
    __m512 norm_x_vec = _mm512_setzero_ps();
    __m512 norm_o_vec = _mm512_setzero_ps();
    
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(reconstructed + i);
        __m512 vo = _mm512_loadu_ps(original + i);
        
        dot_vec = _mm512_fmadd_ps(vx, vo, dot_vec);
        norm_x_vec = _mm512_fmadd_ps(vx, vx, norm_x_vec);
        norm_o_vec = _mm512_fmadd_ps(vo, vo, norm_o_vec);
    }
    
    float dot_product = _mm512_reduce_add_ps(dot_vec);
    float norm_x = _mm512_reduce_add_ps(norm_x_vec);
    float norm_o = _mm512_reduce_add_ps(norm_o_vec);
    
    for (; i < dim; ++i) {
        dot_product += reconstructed[i] * original[i];
        norm_x += reconstructed[i] * reconstructed[i];
        norm_o += original[i] * original[i];
    }
    
    float denominator = std::sqrt(norm_x * norm_o);
    if (denominator < 1e-8f) {
        return 0.0f;
    }
    
    return dot_product / denominator;
#else
    float dot_product = 0.0f;
    float norm_x = 0.0f;
    float norm_o = 0.0f;
    
    for (int i = 0; i < dim; ++i) {
        dot_product += reconstructed[i] * original[i];
        norm_x += reconstructed[i] * reconstructed[i];
        norm_o += original[i] * original[i];
    }
    
    float denominator = std::sqrt(norm_x * norm_o);
    if (denominator < 1e-8f) {
        return 0.0f;
    }
    
    return dot_product / denominator;
#endif
}


void DALQ::code_search(const float* vector, uint8_t* codes,
                          float clipped_lower, float clipped_upper,
                          int rounds) const {
    std::vector<float> x(dim_);
    decode_codes(codes, clipped_lower, clipped_upper, x.data());
    
    float effective_range = clipped_upper - clipped_lower;
    if (effective_range < 1e-8f) {
        return;
    }
    
    float o_norm_sq = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        o_norm_sq += vector[i] * vector[i];
    }
    float o_norm = std::sqrt(o_norm_sq);
    
    if (o_norm < 1e-8f) {
        return;
    }
    
    float current_dot = 0.0f;
    float current_norm_sq = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        current_dot += x[i] * vector[i];
        current_norm_sq += x[i] * x[i];
    }
    float current_norm = std::sqrt(current_norm_sq);
    float current_similarity = (current_norm > 1e-8f) ? (current_dot / (current_norm * o_norm)) : 0.0f;
    
    for (int round = 0; round < rounds; ++round) {
        bool improved = false;
        
        for (int i = 0; i < dim_; ++i) {
            uint8_t original_code = codes[i];
            float original_value = x[i];
            
            int max_steps_neg = static_cast<int>(original_code);
            int max_steps_pos = num_levels_ - 1 - static_cast<int>(original_code);
            
            int best_delta = 0;
            float best_similarity = current_similarity;
            float best_new_value = original_value;
            float best_dot = current_dot;
            float best_norm_sq = current_norm_sq;
            
            for (int delta = -1; delta >= -max_steps_neg; --delta) {
                int new_code = static_cast<int>(original_code) + delta;
                if (new_code < 0) break;
                
                float normalized = new_code * inv_num_levels_;
                float new_value = clipped_lower + normalized * effective_range;
                float value_delta = new_value - original_value;
                
                float new_dot = current_dot + value_delta * vector[i];
                float new_norm_sq = current_norm_sq + value_delta * (2.0f * original_value + value_delta);
                
                if (new_norm_sq < 1e-16f) continue;
                
                float new_norm = std::sqrt(new_norm_sq);
                float new_similarity = new_dot / (new_norm * o_norm);
                
                if (new_similarity > best_similarity) {
                    best_delta = delta;
                    best_similarity = new_similarity;
                    best_new_value = new_value;
                    best_dot = new_dot;
                    best_norm_sq = new_norm_sq;
                } else {
                    break;
                }
            }
            
            for (int delta = 1; delta <= max_steps_pos; ++delta) {
                int new_code = static_cast<int>(original_code) + delta;
                if (new_code >= num_levels_) break;
                
                float normalized = new_code * inv_num_levels_;
                float new_value = clipped_lower + normalized * effective_range;
                float value_delta = new_value - original_value;
                
                float new_dot = current_dot + value_delta * vector[i];
                float new_norm_sq = current_norm_sq + value_delta * (2.0f * original_value + value_delta);
                
                if (new_norm_sq < 1e-16f) continue;
                
                float new_norm = std::sqrt(new_norm_sq);
                float new_similarity = new_dot / (new_norm * o_norm);
                
                if (new_similarity > best_similarity) {
                    best_delta = delta;
                    best_similarity = new_similarity;
                    best_new_value = new_value;
                    best_dot = new_dot;
                    best_norm_sq = new_norm_sq;
                } else {
                    break;
                }
            }
            
            if (best_delta != 0) {
                codes[i] = static_cast<uint8_t>(static_cast<int>(original_code) + best_delta);
                x[i] = best_new_value;
                current_dot = best_dot;
                current_norm_sq = best_norm_sq;
                current_similarity = best_similarity;
                improved = true;
            }
        }
        
        if (!improved) {
            break;
        }
    }
}




void DALQ::apply_clipping(float min_val, float max_val,
                         float& clipped_min, float& clipped_max) const {
    float range = max_val - min_val;
    float shrink = (1.0f - clip_factor_) / 2.0f;
    
    clipped_min = min_val + shrink * range;
    clipped_max = max_val - shrink * range;
}


void DALQ::quantize(const float* vector, uint8_t* codes, 
                    float& scale, float& bias) const {
    float original_min, original_max;
    VectorUtils::compute_bounds(vector, dim_, original_min, original_max);
    
    float clipped_lower, clipped_upper;
    apply_clipping(original_min, original_max, clipped_lower, clipped_upper);
    
    float effective_range = clipped_upper - clipped_lower;
    if (effective_range < 1e-8f) {
        effective_range = 1e-8f;
    }
    
    float quant_scale = (num_levels_ - 1) / effective_range;
    
#ifdef __AVX512F__
    __m512 lower_vec = _mm512_set1_ps(clipped_lower);
    __m512 upper_vec = _mm512_set1_ps(clipped_upper);
    __m512 scale_vec = _mm512_set1_ps(quant_scale);
    __m512i zero_vec = _mm512_setzero_si512();
    __m512i max_code_vec = _mm512_set1_epi32(num_levels_ - 1);
    
    int i = 0;
    for (; i + 15 < dim_; i += 16) {
        __m512 val = _mm512_loadu_ps(vector + i);
    
        val = _mm512_max_ps(lower_vec, _mm512_min_ps(val, upper_vec));
        __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(val, lower_vec), scale_vec);
        __m512i code_i32 = _mm512_cvtps_epi32(normalized);
        code_i32 = _mm512_max_epi32(zero_vec, _mm512_min_epi32(code_i32, max_code_vec));
        __m128i code_u8 = _mm512_cvtusepi32_epi8(code_i32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + i), code_u8);
    }
    
    for (; i < dim_; ++i) {
        float val = vector[i];
        val = std::max(clipped_lower, std::min(val, clipped_upper));
        
        float normalized = (val - clipped_lower) * quant_scale;
        int code = static_cast<int>(normalized + 0.5f);
        
        code = std::max(0, std::min(code, num_levels_ - 1));
        codes[i] = static_cast<uint8_t>(code);
    }
#else
    for (int i = 0; i < dim_; ++i) {
        float val = vector[i];
        val = std::max(clipped_lower, std::min(val, clipped_upper));
        
        float normalized = (val - clipped_lower) * quant_scale;
        int code = static_cast<int>(normalized + 0.5f);
        
        code = std::max(0, std::min(code, num_levels_ - 1));
        codes[i] = static_cast<uint8_t>(code);
    }
#endif

    if (enable_code_search_) {
        code_search(vector, codes, clipped_lower, clipped_upper, search_rounds_);
    }
    
    std::vector<float> x_hat(dim_);
    decode_codes(codes, clipped_lower, clipped_upper, x_hat.data());
    float x_hat_norm = VectorUtils::compute_norm(x_hat.data(), dim_);
    float projection_length;
    if (x_hat_norm < 1e-8f) {
        projection_length = 0.0f;
    } else {
        float dot_product = VectorUtils::compute_dot_product(vector, x_hat.data(), dim_);
        projection_length = dot_product / (x_hat_norm * x_hat_norm);
    }
    
    scale = (clipped_upper - clipped_lower) * inv_num_levels_ * projection_length;
    bias = clipped_lower * projection_length;
}

void DALQ::dequantize(const uint8_t* codes, float scale, float bias, float* output) const {

#ifdef __AVX512F__
    __m512 scale_vec = _mm512_set1_ps(scale);
    __m512 bias_vec = _mm512_set1_ps(bias);
    
    int i = 0;
    for (; i + 15 < dim_; i += 16) {
        __m128i codes_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_i32 = _mm512_cvtepu8_epi32(codes_u8);
        __m512 codes_f32 = _mm512_cvtepi32_ps(codes_i32);
        __m512 result = _mm512_fmadd_ps(codes_f32, scale_vec, bias_vec);
        _mm512_storeu_ps(output + i, result);
    }

    for (; i < dim_; ++i) {
        output[i] = bias + codes[i] * scale;
    }
#else
    for (int i = 0; i < dim_; ++i) {
        output[i] = bias + codes[i] * scale;
    }
#endif
}

float DALQ::compute_distance(const float* query, const uint8_t* codes,
                            float scale, float bias) const {
    std::vector<float> reconstructed(dim_);
    dequantize(codes, scale, bias, reconstructed.data());
    
    float dist_sq = VectorUtils::compute_l2_distance_squared(
        query, reconstructed.data(), dim_);
    
    return dist_sq;
}

