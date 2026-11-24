#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 radiance;
hitAttributeEXT vec2 barycentrics;
layout(set = 0, binding = 2) uniform sampler2D baseColorTexture;
layout(std430, set = 0, binding = 3) readonly buffer UvBuffer { vec2 uvs[]; };
layout(std430, set = 0, binding = 4) readonly buffer IndexBuffer { uint indices[]; };

void main() {
    const uint first = gl_PrimitiveID * 3u;
    const uvec3 tri = uvec3(indices[first], indices[first + 1u], indices[first + 2u]);
    const vec3 bary = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    const vec2 uv = uvs[tri.x] * bary.x + uvs[tri.y] * bary.y + uvs[tri.z] * bary.z;
    const vec3 albedo = texture(baseColorTexture, uv).rgb;
    const float facing = 0.35 + 0.65 * abs(dot(normalize(gl_WorldRayDirectionEXT), vec3(0.0, 0.0, 1.0)));
    radiance = pow(albedo, vec3(1.0 / 2.2)) * facing;
}
