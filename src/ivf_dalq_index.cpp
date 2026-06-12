#include "ivf_dalq_index.h"
#include "vector_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <xmmintrin.h>
#include <chrono>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <immintrin.h>

template <int START, int END>
inline float l2_sq_distance_unfold(const float* a, const float* b, float partial) {
    __m512 sum = _mm512_setzero_ps();
    
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps();
    __m512 s3 = _mm512_setzero_ps();

    for (int i = START; i < END; i += 64) {
        __m512 a0 = _mm512_loadu_ps(a + i);      __m512 b0 = _mm512_loadu_ps(b + i);
        __m512 a1 = _mm512_loadu_ps(a + i + 16); __m512 b1 = _mm512_loadu_ps(b + i + 16);
        __m512 a2 = _mm512_loadu_ps(a + i + 32); __m512 b2 = _mm512_loadu_ps(b + i + 32);
        __m512 a3 = _mm512_loadu_ps(a + i + 48); __m512 b3 = _mm512_loadu_ps(b + i + 48);

        __m512 d0 = _mm512_sub_ps(a0, b0);
        __m512 d1 = _mm512_sub_ps(a1, b1);
        __m512 d2 = _mm512_sub_ps(a2, b2);
        __m512 d3 = _mm512_sub_ps(a3, b3);

        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
        s2 = _mm512_fmadd_ps(d2, d2, s2);
        s3 = _mm512_fmadd_ps(d3, d3, s3);
    }
    
    sum = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(sum) + partial;
    
}

template <int START_DIM, int END_DIM>
__attribute__((always_inline)) inline float dalq_l2_sq_distance_unfold(
    const float* query_residual, 
    const uint8_t* codes, 
    float scale, 
    float bias,
    float partial_dist) { 
    
    __m512 scale_v = _mm512_set1_ps(scale);
    __m512 bias_v = _mm512_set1_ps(bias);
    
    
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps();
    __m512 s3 = _mm512_setzero_ps();

    for (int i = START_DIM; i < END_DIM; i += 64) {
    
        __m512 q0 = _mm512_loadu_ps(query_residual + i);
        __m128i c0_u8 = _mm_loadu_si128((const __m128i*)(codes + i));
        __m512 c0_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c0_u8));
        __m512 rec0 = _mm512_fmadd_ps(c0_f, scale_v, bias_v);
        __m512 diff0 = _mm512_sub_ps(q0, rec0);
        s0 = _mm512_fmadd_ps(diff0, diff0, s0);

        __m512 q1 = _mm512_loadu_ps(query_residual + i + 16);
        __m128i c1_u8 = _mm_loadu_si128((const __m128i*)(codes + i + 16));
        __m512 c1_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c1_u8));
        __m512 rec1 = _mm512_fmadd_ps(c1_f, scale_v, bias_v);
        __m512 diff1 = _mm512_sub_ps(q1, rec1);
        s1 = _mm512_fmadd_ps(diff1, diff1, s1);

        __m512 q2 = _mm512_loadu_ps(query_residual + i + 32);
        __m128i c2_u8 = _mm_loadu_si128((const __m128i*)(codes + i + 32));
        __m512 c2_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c2_u8));
        __m512 rec2 = _mm512_fmadd_ps(c2_f, scale_v, bias_v);
        __m512 diff2 = _mm512_sub_ps(q2, rec2);
        s2 = _mm512_fmadd_ps(diff2, diff2, s2);

        __m512 q3 = _mm512_loadu_ps(query_residual + i + 48);
        __m128i c3_u8 = _mm_loadu_si128((const __m128i*)(codes + i + 48));
        __m512 c3_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c3_u8));
        __m512 rec3 = _mm512_fmadd_ps(c3_f, scale_v, bias_v);
        __m512 diff3 = _mm512_sub_ps(q3, rec3);
        s3 = _mm512_fmadd_ps(diff3, diff3, s3);
    }
    
    __m512 sum = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return partial_dist + _mm512_reduce_add_ps(sum);
}


IVFDALQIndex::IVFDALQIndex() {}

bool IVFDALQIndex::load(const std::string& index_file) {
    std::ifstream in(index_file, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open index file: " << index_file << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(&header_), sizeof(IndexHeader));
    if (header_.magic != 0x44414C51) {
        std::cerr << "Invalid index file format" << std::endl;
        return false;
    }

    centroids_.resize(header_.num_clusters * header_.dimension);
    in.read(reinterpret_cast<char*>(centroids_.data()),
           header_.num_clusters * header_.dimension * sizeof(float));
    
    clusters_.resize(header_.num_clusters);
    vector_index_.resize(header_.num_base_vectors);
    
    size_t block_size = DALQBlockMap::block_bytes(header_.dimension);
    
    for (int cluster_id = 0; cluster_id < header_.num_clusters; ++cluster_id) {
        int32_t cluster_size;
        in.read(reinterpret_cast<char*>(&cluster_size), sizeof(int32_t));
        
        auto& cluster = clusters_[cluster_id];
        cluster.num_vectors = cluster_size;
        cluster.vector_ids.resize(cluster_size);
        
        int num_blocks = (cluster_size + kBlockSize - 1) / kBlockSize;
        cluster.block_data.resize(num_blocks * block_size);
        
        in.read(reinterpret_cast<char*>(cluster.vector_ids.data()),
               cluster_size * sizeof(int32_t));
        
        for (int block_id = 0; block_id < num_blocks; ++block_id) {
            int vectors_in_block = std::min(kBlockSize, cluster_size - block_id * kBlockSize);
            
            DALQBlockMap block_map(cluster.block_data.data() + block_id * block_size, 
                                  header_.dimension);
            
            in.read(reinterpret_cast<char*>(block_map.scales()),
                   vectors_in_block * sizeof(float));
            
            in.read(reinterpret_cast<char*>(block_map.biases()),
                   vectors_in_block * sizeof(float));
            
            for (int i = 0; i < vectors_in_block; ++i) {
                in.read(reinterpret_cast<char*>(block_map.vector_codes(i)),
                       header_.dimension * sizeof(uint8_t));
            }
            
            for (int i = 0; i < vectors_in_block; ++i) {
                int vec_idx = block_id * kBlockSize + i;
                int vector_id = cluster.vector_ids[vec_idx];
                
                if (vector_id >= 0 && vector_id < header_.num_base_vectors) {
                    vector_index_[vector_id] = {cluster_id, block_id, i};
                }
            }
        }
    }
    
    in.close();
    
    quantizer_ = std::make_unique<DALQ>(header_.dimension,
                                        header_.quantization_bits,
                                        header_.clip_factor);
    
    std::cout << "Index loaded successfully. Built global vector index for " 
              << vector_index_.size() << " vectors." << std::endl;
    
    return true;
}


template <int DIM>
void IVFDALQIndex::search_core(const float* query, 
                               int k, int nprobe,
                               std::vector<int>& result_ids) const {
    
    
    MaxHeap cluster_heap(nprobe);  
    float threshold;

    constexpr int PARTIAL_L1 = 256;                 
    float ratio_1 = DIM / PARTIAL_L1 * 0.9;

    const float* centroid_ptr = centroids_.data();
    const int num_clusters = header_.num_clusters;

    for (int i = 0; i < nprobe; ++i) {
        float dist = l2_sq_distance_unfold<0, DIM>(query, centroid_ptr, 0);
        cluster_heap.add(i, dist);           
        centroid_ptr += DIM;
    }

    threshold = cluster_heap.top_dist();
    const int remaining_count = num_clusters - nprobe;
    const float* remaining_ptr = centroid_ptr;
    for (int r = 0; r < remaining_count; ++r) {
        float partial = l2_sq_distance_unfold<0, PARTIAL_L1>(query, remaining_ptr, 0);

        if (partial * ratio_1>= threshold) {
            remaining_ptr += DIM;
            continue;
        }

        float full_dist = l2_sq_distance_unfold<PARTIAL_L1, DIM>(query, remaining_ptr, partial);

        if (full_dist < threshold) {
            int cluster_id = nprobe + r;
            cluster_heap.add(cluster_id, full_dist);
            threshold = cluster_heap.top_dist();
        }

        remaining_ptr += DIM;
    }

    std::vector<int> top_cluster_ids = cluster_heap.sorted_ids();

    MaxHeap topk_heap(k);
    alignas(64) float residual_buffer[DIM]; 
    threshold = std::numeric_limits<float>::max();

    constexpr int PARTIAL_L = 256;  
    constexpr int N_WARMUP = 3;     
    float ratio_initial = DIM / PARTIAL_L;
    float ratio = ratio_initial * 0.85;

    int warmup_end = std::min(N_WARMUP, nprobe);
    for (int i = 0; i < warmup_end; ++i) {
        int cluster_id = top_cluster_ids[i];
        const auto& cluster = clusters_[cluster_id];
        
        if (cluster.num_vectors == 0) continue;
        

        const float* c = centroids_.data() + cluster_id * DIM;
        VectorUtils::subtract_vectors(query, c, 
                                     residual_buffer, 
                                     DIM);
        
        int num_blocks = (cluster.num_vectors + kBlockSize - 1) / kBlockSize;
        size_t block_bytes = DALQBlockMap::block_bytes(DIM);
        const char* block_base_ptr = cluster.block_data.data();

        for (int b = 0; b < num_blocks; ++b) {
            const DALQBlockMap block_map(
                const_cast<char*>(block_base_ptr + b * block_bytes), 
                DIM);
            
            int vectors_in_block = std::min(kBlockSize, cluster.num_vectors - b * kBlockSize);
            
            for (int j = 0; j < vectors_in_block; ++j) {
                float scale = block_map.scales()[j];
                float bias = block_map.biases()[j];
                const uint8_t* codes = block_map.vector_codes(j);
            
                float dist = dalq_l2_sq_distance_unfold<0, DIM>(
                    residual_buffer, codes, scale, bias, 0);
                
                if (dist < threshold) {
                    int vector_id = cluster.vector_ids[b * kBlockSize + j];
                    topk_heap.add(vector_id, dist);
                    threshold = topk_heap.top_dist(); 
                }
            }
        }
    }

    for (int i = N_WARMUP; i < nprobe; ++i) {
        int cluster_id = top_cluster_ids[i];
        const auto& cluster = clusters_[cluster_id];
        
        const float* c = centroids_.data() + cluster_id * DIM;
        VectorUtils::subtract_vectors(query, c, 
                                     residual_buffer, 
                                     DIM);
        
        int num_blocks = (cluster.num_vectors + kBlockSize - 1) / kBlockSize;
        size_t block_bytes = DALQBlockMap::block_bytes(DIM);
        const char* block_base_ptr = cluster.block_data.data();

        for (int b = 0; b < num_blocks; ++b) {
            const DALQBlockMap block_map(
                const_cast<char*>(block_base_ptr + b * block_bytes), 
                DIM);
            
            int vectors_in_block = std::min(kBlockSize, cluster.num_vectors - b * kBlockSize);
            float partial_dists[vectors_in_block];  
            for (int j = 0; j < vectors_in_block; ++j) {
                float scale = block_map.scales()[j];
                float bias  = block_map.biases()[j];
                const uint8_t* codes = block_map.vector_codes(j);

                partial_dists[j] = dalq_l2_sq_distance_unfold<0, PARTIAL_L>(
                    residual_buffer, codes, scale, bias, 0);
            }

            for (int j = 0; j < vectors_in_block; ++j) {
                if ((partial_dists[j] * ratio) < threshold){
                    float scale = block_map.scales()[j];
                    float bias  = block_map.biases()[j];
                    const uint8_t* codes = block_map.vector_codes(j);

                    float full_dist = dalq_l2_sq_distance_unfold<PARTIAL_L, DIM>(
                        residual_buffer, codes, scale, bias, partial_dists[j]);

                    if (full_dist < threshold) {
                        int vector_id = cluster.vector_ids[b * kBlockSize + j];
                        topk_heap.add(vector_id, full_dist);
                        threshold = topk_heap.top_dist();
                    }
                }
            }
            
        }
    }
    
    result_ids = topk_heap.sorted_ids();
}


std::vector<int> IVFDALQIndex::search(const std::vector<float>& query, 
                                      int k, int nprobe) const {

    std::vector<int> result_ids;

    search_core<1536>(query.data(), k, nprobe, result_ids);
    return result_ids;
    
}



void IVFDALQIndex::get_reconstructed_vector(int vector_id, float* reconstructed){
    if (vector_id < 0 || vector_id >= (int)vector_index_.size()) {
        std::cerr << "Invalid vector_id: " << vector_id << std::endl;
        return;
    }
    
    const auto& loc = vector_index_[vector_id];
    const auto& cluster = clusters_[loc.cluster_id];
    
    const float* centroid = centroids_.data() + loc.cluster_id * header_.dimension;
    
    size_t block_size = DALQBlockMap::block_bytes(header_.dimension);
    const DALQBlockMap block_map(
        const_cast<char*>(cluster.block_data.data() + loc.block_id * block_size), 
        header_.dimension);
    
    float scale = block_map.scales()[loc.offset_in_block];
    float bias = block_map.biases()[loc.offset_in_block];
    const uint8_t* codes = block_map.vector_codes(loc.offset_in_block);
    
    std::vector<float> reconstructed_residual(header_.dimension);
    quantizer_->dequantize(codes, scale, bias, reconstructed_residual.data());
    
#ifdef __AVX512F__
    int i = 0;
    for (; i + 15 < header_.dimension; i += 16) {
        __m512 vc = _mm512_loadu_ps(centroid + i);
        __m512 vr = _mm512_loadu_ps(reconstructed_residual.data() + i);
        __m512 result = _mm512_add_ps(vc, vr);
        _mm512_storeu_ps(reconstructed + i, result);
    }
    
    for (; i < header_.dimension; ++i) {
        reconstructed[i] = centroid[i] + reconstructed_residual[i];
    }
#else
    for (int i = 0; i < header_.dimension; ++i) {
        reconstructed[i] = centroid[i] + reconstructed_residual[i];
    }
#endif
}

float IVFDALQIndex::compute_estimated_distance(const std::vector<float>& query, 
                                               int vector_id) const {
    if (vector_id < 0 || vector_id >= (int)vector_index_.size()) {
        std::cerr << "Invalid vector_id: " << vector_id << std::endl;
        return -1.0f;
    }
    
    const auto& loc = vector_index_[vector_id];
    const auto& cluster = clusters_[loc.cluster_id];
    
    const float* centroid = centroids_.data() + loc.cluster_id * header_.dimension;
    
    size_t block_size = DALQBlockMap::block_bytes(header_.dimension);
    const DALQBlockMap block_map(
        const_cast<char*>(cluster.block_data.data() + loc.block_id * block_size), 
        header_.dimension);
    
    float scale = block_map.scales()[loc.offset_in_block];
    float bias = block_map.biases()[loc.offset_in_block];
    const uint8_t* codes = block_map.vector_codes(loc.offset_in_block);

    std::vector<float> reconstructed_residual(header_.dimension);
    quantizer_->dequantize(codes, scale, bias, reconstructed_residual.data());
    
    std::vector<float> reconstructed_vector(header_.dimension);
    
#ifdef __AVX512F__
    int i = 0;
    for (; i + 15 < header_.dimension; i += 16) {
        __m512 vc = _mm512_loadu_ps(centroid + i);
        __m512 vr = _mm512_loadu_ps(reconstructed_residual.data() + i);
        __m512 result = _mm512_add_ps(vc, vr);
        _mm512_storeu_ps(reconstructed_vector.data() + i, result);
    }
    
    for (; i < header_.dimension; ++i) {
        reconstructed_vector[i] = centroid[i] + reconstructed_residual[i];
    }
#else
    for (int i = 0; i < header_.dimension; ++i) {
        reconstructed_vector[i] = centroid[i] + reconstructed_residual[i];
    }
#endif
    
    float dist_sq = VectorUtils::compute_l2_distance_squared(
        query.data(), 
        reconstructed_vector.data(),
        header_.dimension);
    
    return dist_sq;
}

float IVFDALQIndex::compute_real_distance(const std::vector<float>& query,
                                          const std::vector<float>& base_vector) const {
    return VectorUtils::compute_l2_distance_squared(
        query.data(), 
        base_vector.data(), 
        header_.dimension);
}
