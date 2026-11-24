#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GltfScene
{
    std::vector<float> positions;
    std::vector<float> texcoords;
    std::vector<uint32_t> indices;
    std::vector<uint8_t> baseColorRgba{255, 255, 255, 255};
    uint32_t textureWidth = 1;
    uint32_t textureHeight = 1;
    bool hasBaseColorTexture = false;
};

GltfScene loadGltfScene(const std::string& path);
