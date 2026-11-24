#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gltf_scene.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <filesystem>
#include <stdexcept>
#include <string_view>

GltfScene loadGltfScene(const std::string& path)
{
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success)
        throw std::runtime_error("Cannot parse glTF scene: " + path);

    if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success)
    {
        cgltf_free(data);
        throw std::runtime_error("Cannot load glTF buffers");
    }

    GltfScene scene;
    for (size_t meshIndex = 0; meshIndex < data->meshes_count && scene.positions.empty(); ++meshIndex)
    {
        const cgltf_mesh& mesh = data->meshes[meshIndex];
        for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives_count && scene.positions.empty();
             ++primitiveIndex)
        {
            const cgltf_primitive& primitive = mesh.primitives[primitiveIndex];
            if (primitive.type != cgltf_primitive_type_triangles)
                continue;

            const cgltf_accessor* positions = nullptr;
            const cgltf_accessor* texcoords = nullptr;
            for (size_t attributeIndex = 0; attributeIndex < primitive.attributes_count; ++attributeIndex)
            {
                const cgltf_attribute& attribute = primitive.attributes[attributeIndex];
                if (attribute.type == cgltf_attribute_type_position)
                    positions = attribute.data;
                else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0)
                    texcoords = attribute.data;
            }
            if (!positions)
                continue;

            scene.positions.resize(positions->count * 3);
            scene.texcoords.assign(positions->count * 2, 0.0f);
            for (size_t i = 0; i < positions->count; ++i)
            {
                if (!cgltf_accessor_read_float(positions, i, &scene.positions[i * 3], 3))
                    throw std::runtime_error("Cannot read POSITION accessor");
            }
            if (texcoords)
            {
                const size_t count = std::min<size_t>(texcoords->count, positions->count);
                for (size_t i = 0; i < count; ++i)
                {
                    if (!cgltf_accessor_read_float(texcoords, i, &scene.texcoords[i * 2], 2))
                        throw std::runtime_error("Cannot read TEXCOORD_0 accessor");
                }
            }

            if (primitive.indices)
            {
                scene.indices.resize(primitive.indices->count);
                for (size_t i = 0; i < primitive.indices->count; ++i)
                    scene.indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i));
            }
            else
            {
                scene.indices.resize(positions->count);
                for (uint32_t i = 0; i < scene.indices.size(); ++i)
                    scene.indices[i] = i;
            }

            if (primitive.material && primitive.material->has_pbr_metallic_roughness)
            {
                const cgltf_texture* texture = primitive.material->pbr_metallic_roughness.base_color_texture.texture;
                const cgltf_image* image = texture ? texture->image : nullptr;
                if (image)
                {
                    int width = 0, height = 0, channels = 0;
                    stbi_uc* pixels = nullptr;
                    if (image->buffer_view)
                    {
                        const uint8_t* bytes = cgltf_buffer_view_data(image->buffer_view);
                        if (bytes && image->buffer_view->size <= static_cast<size_t>(INT_MAX))
                        {
                            pixels = stbi_load_from_memory(bytes, static_cast<int>(image->buffer_view->size), &width,
                                                           &height, &channels, STBI_rgb_alpha);
                        }
                    }
                    else if (image->uri && std::string_view(image->uri).substr(0, 5) != "data:")
                    {
                        const std::filesystem::path texturePath =
                            std::filesystem::path(path).parent_path() / image->uri;
                        pixels = stbi_load(texturePath.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
                    }

                    if (pixels)
                    {
                        const size_t byteCount = static_cast<size_t>(width) * height * 4;
                        scene.baseColorRgba.assign(pixels, pixels + byteCount);
                        scene.textureWidth = static_cast<uint32_t>(width);
                        scene.textureHeight = static_cast<uint32_t>(height);
                        scene.hasBaseColorTexture = true;
                        stbi_image_free(pixels);
                    }
                }
            }
        }
    }
    cgltf_free(data);

    if (scene.positions.empty() || scene.indices.empty() || scene.indices.size() % 3 != 0)
        throw std::runtime_error("Scene contains no supported triangle primitive");

    std::array<float, 3> min = {scene.positions[0], scene.positions[1], scene.positions[2]};
    std::array<float, 3> max = min;
    for (size_t i = 0; i < scene.positions.size(); i += 3)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            min[axis] = std::min(min[axis], scene.positions[i + axis]);
            max[axis] = std::max(max[axis], scene.positions[i + axis]);
        }
    }
    const float extent = std::max({max[0] - min[0], max[1] - min[1], max[2] - min[2]});
    if (extent > 0.0f)
    {
        for (size_t i = 0; i < scene.positions.size(); i += 3)
        {
            for (size_t axis = 0; axis < 3; ++axis)
                scene.positions[i + axis] =
                    (scene.positions[i + axis] - (min[axis] + max[axis]) * 0.5f) * (1.8f / extent);
        }
    }
    return scene;
}
