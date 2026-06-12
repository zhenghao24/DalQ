#include "fvecs_reader.h"
#include <fstream>
#include <iostream>

int FvecsReader::read_fvecs(const std::string& filename,
                           std::vector<std::vector<float>>& vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    
    vectors.clear();
    int dim = -1;
    
    while (file.peek() != EOF) {
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (file.eof()) break;
        
        if (dim == -1) {
            dim = d;
        } else if (dim != d) {
            std::cerr << "Inconsistent dimensions in file" << std::endl;
            return -1;
        }
        std::vector<float> vec(d);
        file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        vectors.push_back(std::move(vec));
    }
    
    file.close();
    std::cout << "Read " << vectors.size() << " vectors of dimension " << dim 
              << " from " << filename << std::endl;
    return dim;
}

int FvecsReader::read_ivecs(const std::string& filename,
                           std::vector<std::vector<int>>& vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    
    vectors.clear();
    int dim = -1;
    
    while (file.peek() != EOF) {
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (file.eof()) break;
        
        if (dim == -1) {
            dim = d;
        } else if (dim != d) {
            std::cerr << "Inconsistent dimensions in file" << std::endl;
            return -1;
        }
        std::vector<int> vec(d);
        file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int32_t));
        vectors.push_back(std::move(vec));
    }
    
    file.close();
    std::cout << "Read " << vectors.size() << " vectors of dimension " << dim 
              << " from " << filename << std::endl;
    return dim;
}

int FvecsReader::read_bvecs(const std::string& filename,
                           std::vector<std::vector<uint8_t>>& vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    
    vectors.clear();
    int dim = -1;
    
    while (file.peek() != EOF) {
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (file.eof()) break;
        
        if (dim == -1) {
            dim = d;
        } else if (dim != d) {
            std::cerr << "Inconsistent dimensions in file" << std::endl;
            return -1;
        }
        
        std::vector<uint8_t> vec(d);
        file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(uint8_t));
        vectors.push_back(std::move(vec));
    }
    
    file.close();
    std::cout << "Read " << vectors.size() << " vectors of dimension " << dim 
              << " from " << filename << std::endl;
    return dim;
}

bool FvecsReader::write_fvecs(const std::string& filename,
                             const std::vector<std::vector<float>>& vectors) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    for (const auto& vec : vectors) {
        int32_t dim = vec.size();
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    
    file.close();
    return true;
}

bool FvecsReader::write_ivecs(const std::string& filename,
                             const std::vector<std::vector<int>>& vectors) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    for (const auto& vec : vectors) {
        int32_t dim = vec.size();
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(int32_t));
    }
    
    file.close();
    return true;
}
