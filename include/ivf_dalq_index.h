#ifndef IVF_DALQ_INDEX_H
#define IVF_DALQ_INDEX_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include "dalq.h"


struct IndexHeader {
    uint32_t magic = 0x44414C51;  
    int32_t dimension;
    int32_t num_base_vectors;
    int32_t num_clusters;
    int32_t quantization_bits;
    float clip_factor;
};

struct Cluster {
    int num_vectors;
    std::vector<int> vector_ids;
    std::vector<char> block_data;
};


constexpr int kBlockSize = 32;

struct DALQBlockMap {
   public:
    explicit DALQBlockMap(char* data, size_t dim)
        : scales_(reinterpret_cast<float*>(data))
        , biases_(scales_ + kBlockSize)
        , codes_(reinterpret_cast<uint8_t*>(biases_ + kBlockSize)) 
        , dim_(dim) {}

    [[nodiscard]] float* scales() { return scales_; }
    [[nodiscard]] float* biases() { return biases_; }
    
    [[nodiscard]] uint8_t* vector_codes(int i) { 
        return codes_ + i * dim_; 
    }

    [[nodiscard]] const float* scales() const { return scales_; }
    [[nodiscard]] const float* biases() const { return biases_; }
    
    [[nodiscard]] const uint8_t* vector_codes(int i) const { 
        return codes_ + i * dim_; 
    }

    static size_t block_bytes(size_t dim) {
        return (sizeof(float) * kBlockSize * 2) + 
               (sizeof(uint8_t) * kBlockSize * dim);
    }

   private:
    float* scales_;
    float* biases_;
    uint8_t* codes_;
    size_t dim_;
};


struct SearchResult {
    int vector_id;
    float distance;
};

class MaxHeap {
public:
    explicit MaxHeap(size_t k) : k_(k) {
        data_.reserve(k + 1);
    }

    inline void add(int id, float dist) {
        if (data_.size() < k_) {
            data_.push_back({id, dist});
            std::push_heap(data_.begin(), data_.end(), compare);
        } else {
            std::pop_heap(data_.begin(), data_.end(), compare);
            data_.back() = {id, dist};
            std::push_heap(data_.begin(), data_.end(), compare);
        }
    }

    inline float top_dist() const {
        return (data_.size() == k_) ? data_.front().distance : std::numeric_limits<float>::max();
    }
    inline int size() const { return data_.size(); }

    std::vector<int> sorted_ids() {
        std::sort_heap(data_.begin(), data_.end(), compare);
        std::vector<int> res;
        res.reserve(data_.size());
        for (const auto& item : data_) {
            res.push_back(item.vector_id);
        }
        return res;
    }

    static bool compare(const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance; 
    }

private:
    std::vector<SearchResult> data_;
    size_t k_;
};


class IVFDALQIndex {
public:
    IVFDALQIndex();
    ~IVFDALQIndex() = default;
    

    bool load(const std::string& index_file);
    std::vector<int> search(const std::vector<float>& query, int k, int nprobe) const;

    template <int DIM>
    void search_core(const float* query, 
                          int k, int nprobe,
                          std::vector<int>& result_ids) const;
    float compute_estimated_distance(const std::vector<float>& query, int vector_id) const;

    void get_reconstructed_vector(int vector_id, float* reconstructed);
    
    float compute_real_distance(const std::vector<float>& query,
                               const std::vector<float>& base_vector) const;
    
    int get_dimension() const { return header_.dimension; }
    int get_num_base_vectors() const { return header_.num_base_vectors; }
    int get_num_clusters() const { return header_.num_clusters; }
    
private:
    struct VectorLocation {
        int cluster_id;
        int block_id;
        int offset_in_block;
    };
    
    IndexHeader header_;
    std::vector<float> centroids_;
    std::vector<Cluster> clusters_;
    std::unique_ptr<DALQ> quantizer_;
    std::vector<VectorLocation> vector_index_;
};

#endif 
