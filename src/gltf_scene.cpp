#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gltf_scene.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace
{
    // Column-major 4x4 matrices, matching glTF's layout.
    using Mat4 = std::array<float, 16>;

    Mat4 identityMatrix()
    {
        Mat4 matrix{};
        matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
        return matrix;
    }

    Mat4 multiply(const Mat4& left, const Mat4& right)
    {
        Mat4 result{};
        for (size_t column = 0; column < 4; ++column)
        {
            for (size_t row = 0; row < 4; ++row)
            {
                result[column * 4 + row] = left[row] * right[column * 4] + left[4 + row] * right[column * 4 + 1] +
                    left[8 + row] * right[column * 4 + 2] + left[12 + row] * right[column * 4 + 3];
            }
        }
        return result;
    }

    Mat4 nodeLocalTransform(const cgltf_node& node)
    {
        if (node.has_matrix)
        {
            Mat4 matrix{};
            std::copy(std::begin(node.matrix), std::end(node.matrix), matrix.begin());
            return matrix;
        }

        const float tx = node.has_translation ? node.translation[0] : 0.0f;
        const float ty = node.has_translation ? node.translation[1] : 0.0f;
        const float tz = node.has_translation ? node.translation[2] : 0.0f;
        const float qx = node.has_rotation ? node.rotation[0] : 0.0f;
        const float qy = node.has_rotation ? node.rotation[1] : 0.0f;
        const float qz = node.has_rotation ? node.rotation[2] : 0.0f;
        const float qw = node.has_rotation ? node.rotation[3] : 1.0f;
        const float sx = node.has_scale ? node.scale[0] : 1.0f;
        const float sy = node.has_scale ? node.scale[1] : 1.0f;
        const float sz = node.has_scale ? node.scale[2] : 1.0f;

        // T * R * S, written directly in column-major order.
        Mat4 matrix{};
        matrix[0] = (1.0f - 2.0f * (qy * qy + qz * qz)) * sx;
        matrix[1] = 2.0f * (qx * qy + qz * qw) * sx;
        matrix[2] = 2.0f * (qx * qz - qy * qw) * sx;
        matrix[4] = 2.0f * (qx * qy - qz * qw) * sy;
        matrix[5] = (1.0f - 2.0f * (qx * qx + qz * qz)) * sy;
        matrix[6] = 2.0f * (qy * qz + qx * qw) * sy;
        matrix[8] = 2.0f * (qx * qz + qy * qw) * sz;
        matrix[9] = 2.0f * (qy * qz - qx * qw) * sz;
        matrix[10] = (1.0f - 2.0f * (qx * qx + qy * qy)) * sz;
        matrix[12] = tx;
        matrix[13] = ty;
        matrix[14] = tz;
        matrix[15] = 1.0f;
        return matrix;
    }

    void transformPosition(const Mat4& matrix, float* position)
    {
        const float x = position[0];
        const float y = position[1];
        const float z = position[2];
        position[0] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
        position[1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
        position[2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];

        const float w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
        if (std::abs(w) > 1e-8f)
        {
            position[0] /= w;
            position[1] /= w;
            position[2] /= w;
        }
    }

    bool loadTexture(const cgltf_image& image, const std::string& scenePath, GltfTexture& texture)
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        stbi_uc* pixels = nullptr;

        if (image.buffer_view)
        {
            const uint8_t* bytes = cgltf_buffer_view_data(image.buffer_view);
            if (bytes && image.buffer_view->size <= static_cast<size_t>(INT_MAX))
            {
                pixels = stbi_load_from_memory(bytes, static_cast<int>(image.buffer_view->size), &width, &height,
                                               &channels, STBI_rgb_alpha);
            }
        }
        else if (image.uri && std::string_view(image.uri).substr(0, 5) != "data:")
        {
            const std::filesystem::path texturePath = std::filesystem::path(scenePath).parent_path() / image.uri;
            pixels = stbi_load(texturePath.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
        }

        if (!pixels)
            return false;

        texture.width = static_cast<uint32_t>(width);
        texture.height = static_cast<uint32_t>(height);
        texture.rgba.assign(pixels, pixels + static_cast<size_t>(width) * height * 4);
        stbi_image_free(pixels);
        return true;
    }

    uint32_t resolveMaterialTexture(const cgltf_primitive& primitive, const std::string& scenePath, GltfScene& scene,
                                    std::unordered_map<const cgltf_image*, uint32_t>& textureIndices)
    {
        const cgltf_material* material = primitive.material;
        if (!material || !material->has_pbr_metallic_roughness)
            return 0;

        const cgltf_texture* texture = material->pbr_metallic_roughness.base_color_texture.texture;
        const cgltf_image* image = texture ? texture->image : nullptr;
        if (!image)
            return 0;

        const auto existing = textureIndices.find(image);
        if (existing != textureIndices.end())
            return existing->second;

        GltfTexture loaded;
        if (!loadTexture(*image, scenePath, loaded))
            return 0;

        const uint32_t index = static_cast<uint32_t>(scene.textures.size());
        scene.textures.push_back(std::move(loaded));
        textureIndices.emplace(image, index);
        return index;
    }

    void appendPrimitive(const cgltf_primitive& primitive, const Mat4& world, const std::string& scenePath,
                         GltfScene& scene, std::unordered_map<const cgltf_image*, uint32_t>& textureIndices)
    {
        if (primitive.type != cgltf_primitive_type_triangles)
            return;

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
        if (!positions || positions->count == 0)
            return;

        const size_t inputIndexCount = primitive.indices ? primitive.indices->count : positions->count;
        if (inputIndexCount % 3 != 0)
            return;

        const size_t vertexBase = scene.positions.size() / 3;
        const size_t indexBase = scene.indices.size();

        scene.positions.resize((vertexBase + positions->count) * 3);
        scene.texcoords.resize((vertexBase + positions->count) * 2, 0.0f);
        for (size_t i = 0; i < positions->count; ++i)
        {
            float* position = &scene.positions[(vertexBase + i) * 3];
            if (!cgltf_accessor_read_float(positions, i, position, 3))
                throw std::runtime_error("Cannot read POSITION accessor");
            transformPosition(world, position);
        }

        if (texcoords)
        {
            const size_t count = std::min<size_t>(texcoords->count, positions->count);
            for (size_t i = 0; i < count; ++i)
            {
                if (!cgltf_accessor_read_float(texcoords, i, &scene.texcoords[(vertexBase + i) * 2], 2))
                    throw std::runtime_error("Cannot read TEXCOORD_0 accessor");
            }
        }

        scene.indices.resize(indexBase + inputIndexCount);
        if (primitive.indices)
        {
            for (size_t i = 0; i < inputIndexCount; ++i)
            {
                scene.indices[indexBase + i] =
                    static_cast<uint32_t>(vertexBase + cgltf_accessor_read_index(primitive.indices, i));
            }
        }
        else
        {
            for (size_t i = 0; i < inputIndexCount; ++i)
                scene.indices[indexBase + i] = static_cast<uint32_t>(vertexBase + i);
        }

        const uint32_t texture = resolveMaterialTexture(primitive, scenePath, scene, textureIndices);
        scene.materialIndices.insert(scene.materialIndices.end(), inputIndexCount / 3, texture);
    }

    void appendNode(const cgltf_node& node, const std::string& scenePath, const Mat4& parent, GltfScene& scene,
                    std::unordered_map<const cgltf_image*, uint32_t>& textureIndices)
    {
        const Mat4 world = multiply(parent, nodeLocalTransform(node));
        if (node.mesh)
        {
            for (size_t primitiveIndex = 0; primitiveIndex < node.mesh->primitives_count; ++primitiveIndex)
                appendPrimitive(node.mesh->primitives[primitiveIndex], world, scenePath, scene, textureIndices);
        }
        for (size_t childIndex = 0; childIndex < node.children_count; ++childIndex)
            appendNode(*node.children[childIndex], scenePath, world, scene, textureIndices);
    }
} // namespace

GltfScene loadGltfScene(const std::string& path)
{
    cgltf_options options{};
    cgltf_data* rawData = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &rawData) != cgltf_result_success)
    {
        if (rawData)
            cgltf_free(rawData);
        throw std::runtime_error("Cannot parse glTF scene: " + path);
    }
    const std::unique_ptr<cgltf_data, decltype(&cgltf_free)> data(rawData, cgltf_free);

    if (cgltf_load_buffers(&options, data.get(), path.c_str()) != cgltf_result_success)
        throw std::runtime_error("Cannot load glTF buffers");

    GltfScene scene;
    // Slot 0 is the white fallback used by untextured materials.
    scene.textures.emplace_back();
    std::unordered_map<const cgltf_image*, uint32_t> textureIndices;

    if (data->scene)
    {
        for (size_t nodeIndex = 0; nodeIndex < data->scene->nodes_count; ++nodeIndex)
            appendNode(*data->scene->nodes[nodeIndex], path, identityMatrix(), scene, textureIndices);
    }
    else
    {
        for (size_t nodeIndex = 0; nodeIndex < data->nodes_count; ++nodeIndex)
        {
            if (!data->nodes[nodeIndex].parent)
                appendNode(data->nodes[nodeIndex], path, identityMatrix(), scene, textureIndices);
        }
    }

    // Some files contain meshes but no nodes/scene. Append them once in object space.
    if (scene.positions.empty())
    {
        for (size_t meshIndex = 0; meshIndex < data->meshes_count; ++meshIndex)
        {
            const cgltf_mesh& mesh = data->meshes[meshIndex];
            for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives_count; ++primitiveIndex)
            {
                appendPrimitive(mesh.primitives[primitiveIndex], identityMatrix(), path, scene, textureIndices);
            }
        }
    }

    if (scene.positions.empty() || scene.indices.empty() || scene.indices.size() % 3 != 0)
        throw std::runtime_error("Scene contains no supported triangle primitives");
    if (scene.materialIndices.size() * 3 != scene.indices.size())
        throw std::runtime_error("Scene material table does not match triangle count");

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
