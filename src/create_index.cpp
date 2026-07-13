#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <chrono>
#include <omp.h>

#include "ivf_dalq_index.h"
#include "dalq.h"
#include "fvecs_reader.h"
#include "vector_utils.h"


/**
 * IVF-DALQ Index
 * 
 * 
 * Index file format (.index):
 * - Header:
 *   - dimension (int32)
 *   - num_base_vectors (int32)
 *   - num_clusters (int32)
 *   - quantization_bits (int32)
 *   - clip_factor (float)
 * - Cluster centroids: num_clusters * dimension * float
 * - For each cluster:
 *   - cluster_size (int32)
 *   - vector_ids (cluster_size * int32)
 *   - For each block (32 vectors per block):
 *     - scales (min(32, remaining_vectors) * float)
 *     - biases (min(32, remaining_vectors) * float)
 *     - codes (min(32, remaining_vectors) * dimension * uint8)
 */

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] 
                  << " <base_fvecs> <centroids_fvecs> <cluster_id_ivecs> <output_index> <quantization_bits> [clip_factor]" 
                  << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " dataset1_base.fvecs dataset_centroids.fvecs dataset_cluster_id.ivecs dataset1.index 8 0.95" 
                  << std::endl;
        return 1;
    }
    
    std::string base_file = argv[1];
    std::string centroids_file = argv[2];
    std::string cluster_id_file = argv[3];
    std::string output_index = argv[4];
    int quantization_bits = std::atoi(argv[5]);
    float clip_factor = (argc > 6) ? std::atof(argv[6]) : 1.0f;
    
    std::cout << "========================================" << std::endl;
    std::cout << "IVF-DALQ Index Creation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Base vectors: " << base_file << std::endl;
    std::cout << "Centroids file: " << centroids_file << std::endl;
    std::cout << "Cluster ID file: " << cluster_id_file << std::endl;
    std::cout << "Output index: " << output_index << std::endl;
    std::cout << "Quantization bits: " << quantization_bits << std::endl;
    std::cout << "Block size: " << kBlockSize << std::endl;
    

    std::vector<std::vector<float>> base_vectors;
    int dim = FvecsReader::read_fvecs(base_file, base_vectors);
    if (dim <= 0) {
        std::cerr << "Failed to read base vectors" << std::endl;
        return 1;
    }
    
    int num_base = base_vectors.size();
    std::cout << "Loaded " << num_base << " vectors of dimension " << dim << std::endl;
    
    std::vector<std::vector<float>> centroids_vec;
    int centroid_dim = FvecsReader::read_fvecs(centroids_file, centroids_vec);
    if (centroid_dim != dim) {
        std::cerr << "Error: Centroid dimension (" << centroid_dim 
                  << ") does not match base vector dimension (" << dim << ")" << std::endl;
        return 1;
    }
    
    int num_clusters = centroids_vec.size();
    std::cout << "Loaded " << num_clusters << " centroids from file" << std::endl;
    

    std::vector<float> centroids(num_clusters * dim);
    for (int i = 0; i < num_clusters; ++i) {
        std::memcpy(&centroids[i * dim], centroids_vec[i].data(), dim * sizeof(float));
    }
    
    std::vector<std::vector<int>> cluster_ids_vec;
    int cluster_id_dim = FvecsReader::read_ivecs(cluster_id_file, cluster_ids_vec);
    if (cluster_id_dim != 1) {
        std::cerr << "Error: Expected cluster ID dimension to be 1, got " 
                  << cluster_id_dim << std::endl;
        return 1;
    }
    
    if ((int)cluster_ids_vec.size() != num_base) {
        std::cerr << "Error: Number of cluster IDs (" << cluster_ids_vec.size()
                  << ") does not match number of base vectors (" << num_base << ")" << std::endl;
        return 1;
    }
    
    std::vector<std::vector<int>> cluster_assignments(num_clusters);
    for (int i = 0; i < num_base; ++i) {
        int cluster_id = cluster_ids_vec[i][0];
        if (cluster_id < 0 || cluster_id >= num_clusters) {
            std::cerr << "Error: Invalid cluster ID " << cluster_id 
                      << " for vector " << i << std::endl;
            return 1;
        }
        cluster_assignments[cluster_id].push_back(i);
    }
    
    std::cout << "Cluster assignments loaded successfully." << std::endl; 
    auto start_time = std::chrono::high_resolution_clock::now();
    int rounds = 12;
    DALQ dalq_quantizer(dim, quantization_bits, clip_factor, true, rounds);
    

    std::vector<std::vector<uint8_t>> quantized_codes(num_base);
    std::vector<float> scales(num_base);
    std::vector<float> biases(num_base);
    

    #pragma omp parallel for 
    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
        const float* centroid = &centroids[cluster_id * dim];
        
        for (int vector_id : cluster_assignments[cluster_id]) {
            std::vector<float> residual(dim);
            const float* base_vec = base_vectors[vector_id].data();
            VectorUtils::subtract_vectors(base_vec, centroid, residual.data(), dim);

            quantized_codes[vector_id].resize(dim);
            dalq_quantizer.quantize(residual.data(), 
                                   quantized_codes[vector_id].data(),
                                   scales[vector_id], 
                                   biases[vector_id]); 
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;
    
    std::cout << "Quantization time: " << seconds << " seconds" << std::endl;
    std::cout << "Average time per vector: " << (seconds * 1000.0 / num_base) << " ms" << std::endl;
    
    std::ofstream out(output_index, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_index << std::endl;
        return 1;
    }
    
    IndexHeader header;
    header.dimension = dim;
    header.num_base_vectors = num_base;
    header.num_clusters = num_clusters;
    header.quantization_bits = quantization_bits;
    header.clip_factor = clip_factor;
    out.write(reinterpret_cast<const char*>(&header), sizeof(IndexHeader));
    
    out.write(reinterpret_cast<const char*>(centroids.data()), num_clusters * dim * sizeof(float));
    
    size_t block_size = DALQBlockMap::block_bytes(dim);
    
    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
        const auto& cluster_vecs = cluster_assignments[cluster_id];
        int32_t cluster_size = cluster_vecs.size();
        
        out.write(reinterpret_cast<const char*>(&cluster_size), sizeof(int32_t));
        
        out.write(reinterpret_cast<const char*>(cluster_vecs.data()), 
                 cluster_size * sizeof(int32_t));
        
        int num_blocks = (cluster_size + kBlockSize - 1) / kBlockSize;
        
        for (int block_id = 0; block_id < num_blocks; ++block_id) {
            int start_idx = block_id * kBlockSize;
            int vectors_in_block = std::min(kBlockSize, cluster_size - start_idx);
            
            std::vector<char> block_buffer(block_size, 0);
            DALQBlockMap block_map(block_buffer.data(), dim);
            
            for (int i = 0; i < vectors_in_block; ++i) {
                int vec_id = cluster_vecs[start_idx + i];
                
                block_map.scales()[i] = scales[vec_id];
                block_map.biases()[i] = biases[vec_id];
                std::memcpy(block_map.vector_codes(i), 
                           quantized_codes[vec_id].data(),
                           dim * sizeof(uint8_t));
            }
            
            out.write(reinterpret_cast<const char*>(block_map.scales()),
                     vectors_in_block * sizeof(float));
            
            out.write(reinterpret_cast<const char*>(block_map.biases()),
                     vectors_in_block * sizeof(float));
            
            for (int i = 0; i < vectors_in_block; ++i) {
                out.write(reinterpret_cast<const char*>(block_map.vector_codes(i)),
                         dim * sizeof(uint8_t));
            }
        }
    }
    
    out.close();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Index creation completed!" << std::endl;
    std::cout << "Index saved to: " << output_index << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nIndex Statistics:" << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Base vectors: " << num_base << std::endl;
    std::cout << "  Clusters: " << num_clusters << std::endl;
    std::cout << "  Quantization bits: " << quantization_bits << std::endl;
    std::cout << "  Clip factor: " << clip_factor << std::endl;
    std::cout << "  Block size: " << kBlockSize << " vectors" << std::endl;
    std::cout << "  Bytes per block: " << block_size << std::endl;
    
    std::cout << "\nCluster Distribution:" << std::endl;
    int min_cluster_size = num_base;
    int max_cluster_size = 0;
    double avg_cluster_size = 0.0;
    int empty_clusters = 0;
    
    for (int i = 0; i < num_clusters; ++i) {
        int size = cluster_assignments[i].size();
        if (size == 0) {
            empty_clusters++;
        } else {
            min_cluster_size = std::min(min_cluster_size, size);
            max_cluster_size = std::max(max_cluster_size, size);
            avg_cluster_size += size;
        }
    }
    
    if (num_clusters > empty_clusters) {
        avg_cluster_size /= (num_clusters - empty_clusters);
    }
    
    std::cout << "  Empty clusters: " << empty_clusters << std::endl;
    std::cout << "  Min cluster size: " << min_cluster_size << std::endl;
    std::cout << "  Max cluster size: " << max_cluster_size << std::endl;
    std::cout << "  Avg cluster size: " << avg_cluster_size << std::endl;
    
    size_t memory_bytes = 0;
    memory_bytes += sizeof(IndexHeader);
    memory_bytes += num_clusters * dim * sizeof(float);  // centroids
    memory_bytes += num_base * dim * sizeof(uint8_t);  // quantized codes
    memory_bytes += num_base * sizeof(float);  // scales
    memory_bytes += num_base * sizeof(float);  // biases
    memory_bytes += num_base * sizeof(int32_t);  // vector IDs
    memory_bytes += num_clusters * sizeof(int32_t);  // cluster sizes
    
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "  Total: " << (memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Per vector: " << (memory_bytes / (double)num_base) << " bytes" << std::endl;
    
    double compression_ratio = (num_base * dim * sizeof(float)) / (double)memory_bytes;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;
    
    return 0;
}
