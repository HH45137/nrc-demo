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
#include <stdexcept>
#include <string_view>
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

    void collectMeshTransforms(const cgltf_node& node, const cgltf_data& data, const Mat4& parent,
                               std::vector<Mat4>& transforms, std::vector<bool>& assigned)
    {
        const Mat4 world = multiply(parent, nodeLocalTransform(node));
        if (node.mesh)
        {
            const size_t meshIndex = static_cast<size_t>(node.mesh - data.meshes);
            if (meshIndex < transforms.size() && !assigned[meshIndex])
            {
                transforms[meshIndex] = world;
                assigned[meshIndex] = true;
            }
        }
        for (size_t i = 0; i < node.children_count; ++i)
            collectMeshTransforms(*node.children[i], data, world, transforms, assigned);
    }

    std::vector<Mat4> collectSceneMeshTransforms(const cgltf_data& data)
    {
        std::vector<Mat4> transforms(data.meshes_count, identityMatrix());
        std::vector<bool> assigned(data.meshes_count, false);

        if (data.scene)
        {
            for (size_t i = 0; i < data.scene->nodes_count; ++i)
                collectMeshTransforms(*data.scene->nodes[i], data, identityMatrix(), transforms, assigned);
        }
        else
        {
            // No explicit scene: treat nodes without a parent as roots.
            for (size_t i = 0; i < data.nodes_count; ++i)
            {
                if (!data.nodes[i].parent)
                    collectMeshTransforms(data.nodes[i], data, identityMatrix(), transforms, assigned);
            }
        }
        return transforms;
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
} // namespace

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
    // glTF stores mesh vertices in object space. Apply the world transform of the
    // first node that references each mesh so node rotations/scales are honored.
    const std::vector<Mat4> meshTransforms = collectSceneMeshTransforms(*data);
    for (size_t meshIndex = 0; meshIndex < data->meshes_count && scene.positions.empty(); ++meshIndex)
    {
        const cgltf_mesh& mesh = data->meshes[meshIndex];
        const Mat4& meshTransform = meshTransforms[meshIndex];
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
                transformPosition(meshTransform, &scene.positions[i * 3]);
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
