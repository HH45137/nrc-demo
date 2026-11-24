# Vulkan glTF ray-tracing viewer

A GLFW `NO_API` window viewer using Vulkan 1.2, Vulkan-Hpp RAII, and Dear ImGui's Vulkan backend. Scene tracing, swapchain composition, and presentation are all Vulkan; the project has no OpenGL dependency or OpenGL context. The first triangle primitive from a glTF 2.0 / GLB scene is uploaded, BLAS/TLAS are built, and the ray-traced output image is sampled directly by ImGui on the swapchain.

> Vulkan ray tracing and DXR are different APIs. This project targets `VK_KHR_ray_tracing_pipeline`; it does not use DirectX or DXR.

## Requirements

- CMake 3.24+
- C++20 compiler
- Vulkan 1.2 SDK and a driver exposing `VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, and `VK_KHR_deferred_host_operations`
- `glslangValidator` (normally installed with the Vulkan SDK)
- Network access on first configure to fetch cgltf, GLFW, Dear ImGui, and stb_image

## Build and run

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\vulkan_gltf_rt.exe path\to\scene.glb
```

The window's **Load glTF / GLB** button opens a native file picker on Windows. Other platforms use the scene path field. Drag with the left mouse button over the image to orbit; use the mouse wheel over the image to zoom. Scene changes reuse the Vulkan window and renderer, rebuilding only scene resources when a new file is loaded.

## Scope

- Loads `.gltf` with external buffers and `.glb` through cgltf.
- Currently renders only the first triangle primitive; it does not apply node transforms or material factors.
- Loads `TEXCOORD_0` and samples the primitive material's base-color texture (embedded GLB bufferView or external image URI); other PBR maps and texture transforms are not handled.
- Renders into a Vulkan storage/sampled image, then displays that image through the Dear ImGui Vulkan backend in the Vulkan swapchain.
- Uses Vulkan-Hpp RAII handles and generated dispatch; application code does not load Vulkan function pointers manually.
