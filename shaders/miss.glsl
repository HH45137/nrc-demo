#version 460
#extension GL_EXT_ray_tracing : require

// ---------------------------------------------------------------------------
// Path tracer: miss shader.
//
// The sky is evaluated in raygen.glsl; this shader only reports that the ray
// escaped the scene.
//
// Keep HitPayload identical to the declaration in raygen.glsl and hit.glsl.
// ---------------------------------------------------------------------------

struct HitPayload
{
    vec3 albedo;
    vec3 worldPosition;
    vec3 worldNormal;
    bool miss;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main()
{
    payload.albedo = vec3(0.0);
    payload.worldPosition = vec3(0.0);
    payload.worldNormal = vec3(0.0);
    payload.miss = true;
}
