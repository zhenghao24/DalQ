#ifndef FVECS_READER_H
#define FVECS_READER_H

#include <vector>
#include <string>
#include <cstdint>

class FvecsReader {
public:
    
    static int read_fvecs(const std::string& filename,
                         std::vector<std::vector<float>>& vectors);
    
    static int read_ivecs(const std::string& filename,
                         std::vector<std::vector<int>>& vectors);
    static int read_bvecs(const std::string& filename,
                         std::vector<std::vector<uint8_t>>& vectors);
    static bool write_fvecs(const std::string& filename,
                           const std::vector<std::vector<float>>& vectors);
    static bool write_ivecs(const std::string& filename,
                           const std::vector<std::vector<int>>& vectors);
};

#endif 
