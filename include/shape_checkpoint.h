#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

struct ShapeCheckpoint {
    uint32_t seed = 0;
    int resolution = 512;
    bool cascade = false;
    float gss = 7.5f;
    float gsh = 7.5f;
    int max_tokens = 49152;
    std::vector<std::array<int, 3>> coords;
    std::vector<float> features;
};

bool save_shape_checkpoint(const std::string& path, const ShapeCheckpoint& checkpoint, std::string& error);
bool load_shape_checkpoint(const std::string& path, ShapeCheckpoint& checkpoint, std::string& error);

}  // namespace trellis
