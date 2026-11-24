#version 460
#extension GL_EXT_ray_tracing : require

// ---------------------------------------------------------------------------
// Path tracer: ray generation.
//
// Each pixel averages several cosine-weighted random walks. The hit shader only
// reports surface data (albedo / position / normal); the next bounce is launched
// here, which keeps the RT pipeline at recursion depth 1.
//
// Keep HitPayload identical to the declaration in hit.glsl and miss.glsl.
// ---------------------------------------------------------------------------

struct HitPayload
{
    vec3 albedo;         // linear-space base color
    vec3 worldPosition;  // hit position, valid when miss == false
    vec3 worldNormal;    // geometric world-space normal, valid when miss == false
    bool miss;           // true when the ray escaped the scene
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;
layout(set = 0, binding = 6, rgba32f) uniform image2D samplesAccumulated;

layout(push_constant) uniform Camera
{
    vec4 origin;
    vec4 right;
    vec4 up;
    vec4 forward;
    uint sampleIndex;
    int _pad0;
    int _pad1;
    int _pad2;
} camera;

layout(location = 0) rayPayloadEXT HitPayload payload;

const int kMaxBounces = 4;      // maximum path length
const float kPi = 3.14159265359;
const float kRayTMin = 0.001;
const float kRayTMax = 10000.0;

// PCG-style integer hash: cheap, well-distributed random numbers on the GPU.
uint pcgHash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint seed)
{
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

// Compute a direction around `normal` with a cosine-weighted distribution.
// The cosine factor cancels against the Lambertian BRDF, so throughput is
// simply multiplied by the surface albedo at each bounce.
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed)
{
    const float phi = 2.0 * kPi * randomFloat(seed);
    const float r = sqrt(randomFloat(seed));
    const float z = sqrt(max(0.0, 1.0 - r * r));

    // Build an orthonormal basis around the normal.
    const vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3 tangent = normalize(cross(up, normal));
    const vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * (cos(phi) * r) + bitangent * (sin(phi) * r) + normal * z);
}

// Simple procedural sky used for rays that escape the scene.
vec3 skyRadiance(vec3 direction)
{
    const float horizon = 0.5 * (direction.y + 1.0);
    const vec3 sky = mix(vec3(0.55, 0.72, 0.95), vec3(0.10, 0.28, 0.65), horizon);

    // Small sun disk to provide a bright light source for the path tracer.
    const vec3 sunDirection = normalize(vec3(0.4, 0.8, 0.3));
    const float sun = pow(max(dot(direction, sunDirection), 0.0), 256.0);
    return sky + vec3(3.0, 2.7, 2.2) * sun;
}

// Trace one random walk and return its (untonemapped) radiance estimate.
vec3 tracePath(vec3 origin, vec3 direction, inout uint seed)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (int bounce = 0; bounce < kMaxBounces; ++bounce)
    {
        payload.albedo = vec3(0.0);
        payload.worldPosition = vec3(0.0);
        payload.worldNormal = vec3(0.0);
        payload.miss = false;

        traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin, kRayTMin, direction, kRayTMax, 0);

        if (payload.miss)
        {
            radiance += throughput * skyRadiance(direction);
            break;
        }

        // Diffuse bounce: Lambertian BRDF * cosine / pdf == albedo.
        throughput *= payload.albedo;

        // Russian roulette after a few bounces keeps the average path cheap.
        if (bounce >= 2)
        {
            const float survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (randomFloat(seed) > survival)
            break;
            throughput /= survival;
        }

        // Offset the new origin slightly along the normal to avoid self hits.
        origin = payload.worldPosition + payload.worldNormal * kRayTMin;
        direction = sampleCosineHemisphere(payload.worldNormal, seed);
    }

    return radiance;
}

void main()
{
    const vec2 resolution = vec2(gl_LaunchSizeEXT.xy);
    const uint pixelIndex = gl_LaunchIDEXT.y * uint(gl_LaunchSizeEXT.x) + gl_LaunchIDEXT.x;
    uint seed = pcgHash(pixelIndex ^ pcgHash(0x9e3779b9u));
    seed = pcgHash(seed ^ camera.sampleIndex);

    // One sample
    vec3 S = vec3(0.0);
    {
        // Jitter within the pixel for anti-aliasing.
        const vec2 pixel = (vec2(gl_LaunchIDEXT.xy) + vec2(randomFloat(seed), randomFloat(seed))) / resolution;
        const vec2 screen = vec2(pixel.x * 2.0 - 1.0, 1.0 - pixel.y * 2.0);
        const float aspect = resolution.x / resolution.y;
        const vec3 direction = normalize(camera.forward.xyz + screen.x * aspect * camera.right.xyz + screen.y * camera.up.xyz);

        S = tracePath(camera.origin.xyz, direction, seed);
    }

    // Skip the frame 1
    vec3 accumulated = imageLoad(samplesAccumulated, ivec2(gl_LaunchIDEXT.xy)).rgb;
    if (camera.sampleIndex == 1) {
        accumulated = vec3(0.0);
    }

    // Average samples
    const float n = float(camera.sampleIndex);
    vec3 color = mix(accumulated, S, 1.0 / n);
    imageStore(samplesAccumulated, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));

    // Apply a simple Reinhard tonemap and gamma encode.
    color = color / (vec3(1.0) + color);
    color = pow(color, vec3(1.0 / 2.2));
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
}
