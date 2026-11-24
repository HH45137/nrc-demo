# Vulkan glTF ray-tracing viewer

A GLFW `NO_API` window viewer using Vulkan 1.2, Vulkan-Hpp RAII, and Dear ImGui's Vulkan backend. Scene tracing, swapchain composition, and presentation are all Vulkan; the project has no OpenGL dependency or OpenGL context. The render viewport fills the whole window, with a small floating control panel. All triangle primitives from the glTF scene graph are merged into one scene (with node world transforms), BLAS/TLAS are built, and the path-traced output image is sampled directly by ImGui on the swapchain.

The renderer is a progressive Monte Carlo path tracer. Each frame adds one cosine-weighted sample per pixel (up to 4 diffuse bounces, Russian roulette, procedural sky and sun) into a floating-point accumulation image. Camera changes restart accumulation; leaving the camera idle converges the image over time. Bounces are launched iteratively from ray generation, so the ray-tracing pipeline stays at recursion depth 1.

> Vulkan ray tracing and DXR are different APIs. This project targets `VK_KHR_ray_tracing_pipeline`; it does not use DirectX or DXR.

## Requirements

- CMake 3.24+
- C++20 compiler
- Vulkan 1.2 SDK and a driver exposing `VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, `VK_KHR_deferred_host_operations`, and the Vulkan 1.2 descriptor-indexing features used for bindless textures
- `glslangValidator` (normally installed with the Vulkan SDK)
- Network access on first configure to fetch cgltf, GLFW, Dear ImGui, and stb_image

## Build and run

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\vulkan_gltf_rt.exe path\to\scene.glb
```

The window's **Load glTF / GLB** button opens a native file picker on Windows. Other platforms use the scene path field. Scene changes reuse the Vulkan window and renderer, rebuilding only scene resources when a new file is loaded.

Camera controls:

- **Orbit mode** (default): left-drag over the image to orbit, mouse wheel to zoom.
- **First-person mode**: press **F** or tick the checkbox. **WASD** moves, **Q/E** moves down/up, **Shift** boosts speed, left-drag looks around, and the mouse wheel changes movement speed. Press **F** again to return to orbit mode with the same view direction.

## Scope

- Loads `.gltf` with external buffers and `.glb` through cgltf.
- Renders all triangle primitives reachable from the glTF scene graph, applying each node's world transform. Instanced mesh nodes are duplicated into the merged geometry. Skinning, morph targets and animations are not handled, and material factors are ignored.
- Samples a per-triangle base-color texture through a bindless runtime descriptor array (`GL_EXT_nonuniform_qualifier`). The number of distinct textures is limited only by the device's sampler limits (capped at 65536). Embedded GLB bufferViews and external image URIs are supported; other PBR maps / texture transforms are not handled.
- All textures are decoded and uploaded at scene load and stay resident. Bindless removes the descriptor-count limit, not the VRAM/upload-time limit; there is no streaming or mipmap generation yet.
- Path-tracing shaders live in `shaders/raygen.glsl`, `shaders/miss.glsl`, and `shaders/hit.glsl`; CMake compiles them to SPIR-V and the RT pipeline uses recursion depth 1.
- Renders into a Vulkan storage/sampled image, then displays that image through the Dear ImGui Vulkan backend in the Vulkan swapchain.
- Uses Vulkan-Hpp RAII handles and generated dispatch; application code does not load Vulkan function pointers manually.
