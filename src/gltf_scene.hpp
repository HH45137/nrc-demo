#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GltfTexture
{
    uint32_t width = 1;
    uint32_t height = 1;
    std::vector<uint8_t> rgba{255, 255, 255, 255};
};

struct GltfScene
{
    std::vector<float> positions;
    std::vector<float> texcoords;
    std::vector<uint32_t> indices;

    // Slot 0 is always an opaque white fallback. Remaining entries are the
    // distinct base-color textures referenced by the scene.
    std::vector<GltfTexture> textures;
    // One base-color texture index per triangle, parallel to indices / 3.
    std::vector<uint32_t> materialIndices;
};

GltfScene loadGltfScene(const std::string& path);
