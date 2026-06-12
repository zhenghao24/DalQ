#ifndef DALQ_H
#define DALQ_H

#include <vector>
#include <cstdint>


class DALQ {
public:
    /**
     * @param dim: dimensionality of vectors
     * @param bits: number of bits per dimension
     * @param clip_factor: β ∈ (0, 1], controls range tightening
     * @param enable_code_search: whether to enable code search 
     * @param search_rounds: number of code search rounds
     */
    DALQ(int dim, int bits = 8, float clip_factor = 0.95f, 
         bool enable_code_search = false, int search_rounds = 2);
    

    void quantize(const float* vector, uint8_t* codes, 
                  float& scale, float& bias) const;
    
   
    void dequantize(const uint8_t* codes, float scale, float bias, float* output) const;
    float compute_distance(const float* query, const uint8_t* codes,
                          float scale, float bias) const;
    float compute_distance_fused(const float* query, const uint8_t* codes,
                                 float scale, float bias) const;
    
    int get_dim() const { return dim_; }
    int get_bits() const { return bits_; }
    float get_clip_factor() const { return clip_factor_; }
    bool is_code_search_enabled() const { return enable_code_search_; }
    int get_search_rounds() const { return search_rounds_; }
    
private:
    int dim_;
    int bits_;
    float clip_factor_;
    int num_levels_;
    float inv_num_levels_;
    bool enable_code_search_;
    int search_rounds_;
    
    void apply_clipping(float min_val, float max_val,
                       float& clipped_min, float& clipped_max) const;

    void code_search(const float* vector, uint8_t* codes,
                        float clipped_lower, float clipped_upper,
                        int rounds) const;
    float compute_cosine_similarity(const float* reconstructed, 
                                   const float* original, 
                                   int dim) const;
    void decode_codes(const uint8_t* codes, float clipped_lower, 
                     float clipped_upper, float* output) const;
};

#endif 
