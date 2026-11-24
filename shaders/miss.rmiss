#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 radiance;

void main() {
    const float t = 0.5 * (gl_WorldRayDirectionEXT.y + 1.0);
    radiance = mix(vec3(0.92, 0.94, 0.98), vec3(0.24, 0.48, 0.78), t);
}
