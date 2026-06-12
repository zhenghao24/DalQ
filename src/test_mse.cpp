#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <fstream>

#include "ivf_dalq_index.h"
#include "fvecs_reader.h"
#include "vector_utils.h"


struct MSEStats {
    double mse;                    // Mean Square Error
    double rmse;                   // Root Mean Square Error
    double normalized_mse;         // Normalized MSE (MSE / average vector norm)
    double avg_vector_norm;        // Average norm of original vectors
    double max_error;              // Maximum reconstruction error
    double min_error;              // Minimum reconstruction error
    double median_error;           // Median reconstruction error
    double std_dev;                // Standard deviation of errors
    double percentile_90;          // 90th percentile error
    double percentile_95;          // 95th percentile error
    double percentile_99;          // 99th percentile error
    int num_samples;               // Number of samples
    

    std::vector<int> distribution_counts;
    std::vector<std::pair<double, double>> distribution_buckets;
};

double compute_reconstruction_error(IVFDALQIndex& index,
                                    int vector_id,
                                    const std::vector<float>& original_vector) {
    
    float error_sq = index.compute_estimated_distance(original_vector, vector_id) / index.get_dimension();
    
    if (error_sq < 0) {
        return -1.0;  
    }
    
    return error_sq;
}

MSEStats compute_mse(IVFDALQIndex& index,
                     const std::vector<std::vector<float>>& base_vectors,
                     int max_samples = -1) {
    MSEStats stats;
    std::vector<double> reconstruction_errors;
    std::vector<double> vector_norms;
    
    int num_vectors = (max_samples > 0) ? std::min(max_samples, (int)base_vectors.size())
                                        : base_vectors.size();
    
    std::cout << "\nComputing MSE (Mean Square Error) for " << num_vectors 
              << " vectors..." << std::endl;
    std::cout << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int dim = index.get_dimension();
    double sum_errors = 0.0;
    double sum_norms = 0.0;
    int valid_samples = 0;
    
    for (int i = 0; i < num_vectors; ++i) {
        const auto& original_vec = base_vectors[i];
        
        double error_sq = compute_reconstruction_error(index, i, original_vec);
        
        if (error_sq < 0) {
            std::cerr << "Warning: Failed to compute error for vector " << i << std::endl;
            continue;
        }
        
        float norm_sq = VectorUtils::compute_l2_distance_squared(
            original_vec.data(), 
            std::vector<float>(dim, 0.0f).data(), 
            dim);
        
        reconstruction_errors.push_back(error_sq);
        vector_norms.push_back(norm_sq);
        
        sum_errors += error_sq;
        sum_norms += norm_sq;
        valid_samples++;
        

    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    std::cout << "\nCompleted in " << total_time << " seconds." << std::endl;
    std::cout << std::endl;
    
    if (valid_samples == 0) {
        std::cerr << "Error: No valid samples collected!" << std::endl;
        stats.num_samples = 0;
        return stats;
    }
    
    stats.num_samples = valid_samples;
    
    stats.mse = sum_errors / valid_samples;
    stats.rmse = std::sqrt(stats.mse);
    
    stats.avg_vector_norm = sum_norms / valid_samples;
    
    stats.normalized_mse = (stats.avg_vector_norm > 0) ? 
                           (stats.mse / stats.avg_vector_norm) : 0.0;
    
    std::sort(reconstruction_errors.begin(), reconstruction_errors.end());
    
    stats.min_error = reconstruction_errors.front();
    stats.max_error = reconstruction_errors.back();
    stats.median_error = reconstruction_errors[valid_samples / 2];
    
    double sq_sum = 0.0;
    for (double error : reconstruction_errors) {
        double diff = error - stats.mse;
        sq_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(sq_sum / valid_samples);
    
    stats.percentile_90 = reconstruction_errors[std::min((int)(valid_samples * 0.90), 
                                                         valid_samples - 1)];
    stats.percentile_95 = reconstruction_errors[std::min((int)(valid_samples * 0.95), 
                                                         valid_samples - 1)];
    stats.percentile_99 = reconstruction_errors[std::min((int)(valid_samples * 0.99), 
                                                         valid_samples - 1)];
    

    stats.distribution_buckets = {
        {0.0, 0.01},
        {0.01, 0.05},
        {0.05, 0.1},
        {0.1, 0.5},
        {0.5, 1.0},
        {1.0, 5.0},
        {5.0, 10.0},
        {10.0, std::numeric_limits<double>::infinity()}
    };
    
    stats.distribution_counts.resize(stats.distribution_buckets.size(), 0);
    
    for (double error : reconstruction_errors) {
        for (size_t i = 0; i < stats.distribution_buckets.size(); ++i) {
            if (error >= stats.distribution_buckets[i].first && 
                error < stats.distribution_buckets[i].second) {
                stats.distribution_counts[i]++;
                break;
            }
        }
    }
    
    return stats;
}

void print_mse_statistics(const MSEStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "MSE (Mean Square Error) Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total samples:    " << stats.num_samples << std::endl;
    std::cout << std::endl;
    
    std::cout << "Primary Metrics:" << std::endl;
    std::cout << "  MSE:              " << std::scientific << std::setprecision(6)
              << stats.mse << std::endl;
    std::cout << "  RMSE:             " << std::scientific << std::setprecision(6)
              << stats.rmse << std::endl;
    std::cout << "  Normalized MSE:   " << std::scientific << std::setprecision(6)
              << stats.normalized_mse << std::endl;
    std::cout << "  Avg Vector Norm²: " << std::scientific << std::setprecision(6)
              << stats.avg_vector_norm << std::endl;
    std::cout << std::endl;
    
    std::cout << "Reconstruction Error Statistics:" << std::endl;
    std::cout << "  Minimum:          " << std::scientific << std::setprecision(6)
              << stats.min_error << std::endl;
    std::cout << "  Median:           " << std::scientific << std::setprecision(6)
              << stats.median_error << std::endl;
    std::cout << "  Maximum:          " << std::scientific << std::setprecision(6)
              << stats.max_error << std::endl;
    std::cout << "  Std Dev:          " << std::scientific << std::setprecision(6)
              << stats.std_dev << std::endl;
    std::cout << std::endl;
    
    std::cout << "Percentiles:" << std::endl;
    std::cout << "  90th percentile:  " << std::scientific << std::setprecision(6)
              << stats.percentile_90 << std::endl;
    std::cout << "  95th percentile:  " << std::scientific << std::setprecision(6)
              << stats.percentile_95 << std::endl;
    std::cout << "  99th percentile:  " << std::scientific << std::setprecision(6)
              << stats.percentile_99 << std::endl;
    std::cout << std::endl;
    

    std::cout << "Reconstruction Error Distribution:" << std::endl;
    for (size_t i = 0; i < stats.distribution_buckets.size(); ++i) {
        double percentage = 100.0 * stats.distribution_counts[i] / stats.num_samples;
        
        std::cout << "  [" << std::fixed << std::setprecision(2) 
                  << stats.distribution_buckets[i].first << ", ";
        if (std::isinf(stats.distribution_buckets[i].second)) {
            std::cout << "inf   ";
        } else {
            std::cout << std::setw(6) << stats.distribution_buckets[i].second;
        }
        std::cout << "): ";
        
        int bar_length = (int)(percentage / 2);  
        for (int j = 0; j < bar_length; ++j) {
            std::cout << "█";
        }
        if (bar_length == 0 && stats.distribution_counts[i] > 0) {
            std::cout << "▏";
        }
        
        std::cout << " " << std::setw(6) << std::setprecision(2) << percentage << "%"
                  << " (" << stats.distribution_counts[i] << " vectors)" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <index_file> <base_fvecs> [max_samples]" 
                  << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " dataset1.index dataset1_base.fvecs 10000" 
                  << std::endl;
        std::cerr << "\nDescription:" << std::endl;
        std::cerr << "  Computes Mean Square Error of DALQ quantization" << std::endl;
        std::cerr << "  by measuring reconstruction error" << std::endl;
        std::cerr << "\nOptional parameters:" << std::endl;
        std::cerr << "  max_samples: Maximum number of vectors to test" << std::endl;
        return 1;
    }
    
    std::string index_file = argv[1];
    std::string base_file = argv[2];
    int max_samples = (argc > 3) ? std::atoi(argv[3]) : -1;
    
    std::cout << "========================================" << std::endl;
    std::cout << "IVF-DALQ MSE Analysis" << std::endl;
    std::cout << "Mean Square Error" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Index file:   " << index_file << std::endl;
    std::cout << "Base vectors: " << base_file << std::endl;
    if (max_samples > 0) {
        std::cout << "Max samples:  " << max_samples << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Loading index..." << std::endl;
    IVFDALQIndex index;
    if (!index.load(index_file)) {
        return 1;
    }
    
    std::cout << "Index loaded successfully." << std::endl;
    std::cout << "  Dimension:    " << index.get_dimension() << std::endl;
    std::cout << "  Base vectors: " << index.get_num_base_vectors() << std::endl;
    std::cout << "  Clusters:     " << index.get_num_clusters() << std::endl;
    
    std::cout << "\nLoading base vectors..." << std::endl;
    std::vector<std::vector<float>> base_vectors;
    int base_dim = FvecsReader::read_fvecs(base_file, base_vectors);
    if (base_dim != index.get_dimension()) {
        std::cerr << "Dimension mismatch: base vectors have " << base_dim 
                  << " dimensions, but index has " << index.get_dimension() << std::endl;
        return 1;
    }
    std::cout << "Loaded " << base_vectors.size() << " base vectors." << std::endl;
    
    if ((int)base_vectors.size() != index.get_num_base_vectors()) {
        std::cerr << "Warning: Base file has " << base_vectors.size() 
                  << " vectors, but index was built with " 
                  << index.get_num_base_vectors() << " vectors." << std::endl;
        std::cerr << "Will process min(" << base_vectors.size() << ", " 
                  << index.get_num_base_vectors() << ") vectors." << std::endl;
    }
    
    MSEStats stats = compute_mse(index, base_vectors, max_samples);
    
    if (stats.num_samples == 0) {
        std::cerr << "Error: No valid samples!" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    print_mse_statistics(stats);
    
    return 0;
}
