#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include "ivf_dalq_index.h"
#include "fvecs_reader.h"

class CSVWriter {
private:
    std::ofstream file;
    
public:
    CSVWriter(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
        }
    }
    
    ~CSVWriter() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    void writeHeader() {
        file << "nprobe,recall,qps\n";
    }
    
    void writeRow(int nprobe, double recall, double qps) {
        file << nprobe << ","
             << std::fixed << std::setprecision(4) << recall << ","
             << std::fixed << std::setprecision(2) << qps << "\n";
        file.flush();  
    }
    
    bool isOpen() const {
        return file.is_open();
    }
};


double compute_recall(const std::vector<int>& results,
                     const std::vector<int>& groundtruth,
                     int k) {
    int hits = 0;
    int gt_size = std::min(k, (int)groundtruth.size());
    
    std::unordered_set<int> gt_set(groundtruth.begin(), 
                                    groundtruth.begin() + gt_size);
    
    int result_size = std::min(k, (int)results.size());
    for (int i = 0; i < result_size; ++i) {
        if (gt_set.count(results[i]) > 0) {
            hits++;
        }
    }
    
    return (double)hits / gt_size;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <index_file> <query_fvecs> <groundtruth_ivecs> [nprobe_min] [nprobe_max] [num_threads]" 
                  << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " dataset1.index dataset1_query.fvecs dataset1_groundtruth.ivecs 1 128 8" 
                  << std::endl;
        return 1;
    }
    
    std::string index_file = argv[1];
    std::string query_file = argv[2];
    std::string groundtruth_file = argv[3];

    std::string output_file = (argc > 7) ? argv[7] : "results.csv";
    CSVWriter csv_writer(output_file);
    if (!csv_writer.isOpen()) {
        std::cerr << "Failed to create output file!" << std::endl;
        return 1;
    }
    csv_writer.writeHeader();
    
    std::vector<int> nprobe_values;
    if (argc >= 6) {
        int nprobe_min = std::atoi(argv[4]);
        int nprobe_max = std::atoi(argv[5]);
        
        for (int n = nprobe_min; n <= nprobe_max; n *= 2) {
            nprobe_values.push_back(n);
        }
        if (nprobe_values.back() < nprobe_max) {
            nprobe_values.push_back(nprobe_max);
        }
    } else {
        nprobe_values = {1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600};
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "IVF-DALQ QPS Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Index file: " << index_file << std::endl;
    std::cout << "Query file: " << query_file << std::endl;
    std::cout << "Groundtruth file: " << groundtruth_file << std::endl;
    std::cout << "nprobe range: [" << nprobe_values.front() << ", " 
              << nprobe_values.back() << "]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "[1/3] Loading index..." << std::endl;
    IVFDALQIndex index;
    if (!index.load(index_file)) {
        return 1;
    }
    
    std::cout << "Index loaded successfully" << std::endl;
    std::cout << "  Dimension: " << index.get_dimension() << std::endl;
    std::cout << "  Base vectors: " << index.get_num_base_vectors() << std::endl;
    std::cout << "  Clusters: " << index.get_num_clusters() << std::endl;
    std::cout << std::endl;
    
    std::cout << "[2/3] Loading queries..." << std::endl;
    std::vector<std::vector<float>> queries;
    int query_dim = FvecsReader::read_fvecs(query_file, queries);
    if (query_dim != index.get_dimension()) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << queries.size() << " queries" << std::endl;
    std::cout << std::endl;
    
    std::cout << "[3/3] Loading groundtruth..." << std::endl;
    std::vector<std::vector<int>> groundtruth;
    FvecsReader::read_ivecs(groundtruth_file, groundtruth);
    std::cout << "Loaded " << groundtruth.size() << " groundtruth entries" << std::endl;
    std::cout << std::endl;
    
    const int k = 100;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Testing QPS and Recall@" << k << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "nprobe\t\tRecall@" << k << "\t\tQPS" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (int nprobe : nprobe_values) {
        for (int i = 0; i < std::min(10, (int)queries.size()); ++i) {
            index.search(queries[i], k, nprobe);
        }
        
        std::vector<std::vector<int>> all_results(queries.size());
        std::vector<double> query_times_us(queries.size());  

        auto query_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < queries.size(); ++i) {
            all_results[i] = index.search(queries[i], k, nprobe);
        }
        auto query_end = std::chrono::steady_clock::now();
            
        query_times_us[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            query_end - query_start).count() / 1000.0;  
        
        std::vector<double> recalls(queries.size());
        
        for (size_t i = 0; i < queries.size(); ++i) {
            recalls[i] = compute_recall(all_results[i], groundtruth[i], k);
        }
        
        double total_query_time_us = 0.0;
        for (double t : query_times_us) {
            total_query_time_us += t;
        }
        
        double total_recall = 0.0;
        for (double r : recalls) {
            total_recall += r;
        }
        double avg_recall = total_recall / queries.size();
        
        double qps = queries.size() * 1000000.0 / total_query_time_us;  
                
        std::cout << nprobe << "\t\t\t\t"
                  << std::fixed << std::setprecision(4) << avg_recall << "\t\t\t"
                  << std::fixed << std::setprecision(2) << qps << "\t\t\t"
                  << std::endl;
        csv_writer.writeRow(nprobe, avg_recall, qps);
    }
    return 0;
}
