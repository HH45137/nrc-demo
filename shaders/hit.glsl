#version 460
#extension GL_EXT_ray_tracing : require

// ---------------------------------------------------------------------------
// Path tracer: closest-hit shader.
//
// This shader only reports the surface hit to ray generation: base color,
// world-space position and geometric normal. The next path segment is traced
// by raygen.glsl, so the pipeline does not need recursive traceRayEXT calls.
//
// Keep HitPayload identical to the declaration in raygen.glsl and miss.glsl.
// ---------------------------------------------------------------------------

struct HitPayload
{
    vec3 albedo;
    vec3 worldPosition;
    vec3 worldNormal;
    bool miss;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 barycentrics;

layout(set = 0, binding = 2) uniform sampler2D baseColorTexture;
layout(std430, set = 0, binding = 3) readonly buffer UvBuffer { vec2 uvs[]; };
layout(std430, set = 0, binding = 4) readonly buffer IndexBuffer { uint indices[]; };
// Positions are tightly packed as float3, so expose them as a float array.
layout(std430, set = 0, binding = 5) readonly buffer PositionBuffer { float positions[]; };

vec3 objectPosition(uint vertexIndex)
{
    const uint base = vertexIndex * 3u;
    return vec3(positions[base], positions[base + 1u], positions[base + 2u]);
}

void main()
{
    // Interpolate UVs over the triangle that was hit.
    const uint first = gl_PrimitiveID * 3u;
    const uvec3 triangle = uvec3(indices[first], indices[first + 1u], indices[first + 2u]);
    const vec3 bary = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    const vec2 uv = uvs[triangle.x] * bary.x + uvs[triangle.y] * bary.y + uvs[triangle.z] * bary.z;

    // The texture view is sRGB, so sampling already returns linear values.
    payload.albedo = texture(baseColorTexture, uv).rgb;

    // Build the geometric normal from the object-space triangle. The normal is
    // transformed with the instance transform; non-uniform scaling would need
    // the inverse-transpose and is not handled by this sample.
    const vec3 p0 = objectPosition(triangle.x);
    const vec3 p1 = objectPosition(triangle.y);
    const vec3 p2 = objectPosition(triangle.z);
    vec3 normal = normalize(cross(p1 - p0, p2 - p0));
    normal = normalize(mat3(gl_ObjectToWorldEXT) * normal);

    // Flip the normal toward the incoming ray so that both triangle windings work.
    const vec3 rayDirection = normalize(gl_WorldRayDirectionEXT);
    if (dot(normal, rayDirection) > 0.0)
        normal = -normal;

    payload.worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.worldNormal = normal;
    payload.miss = false;
}
