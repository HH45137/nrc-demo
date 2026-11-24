cmake_minimum_required(VERSION 3.24)
project(vulkan_gltf_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Vulkan REQUIRED)
include(FetchContent)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(cgltf
        GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
        GIT_TAG v1.15
        GIT_SHALLOW TRUE)
FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE)
FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.9b
        GIT_SHALLOW TRUE)
FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 6b5799bae8a1fe3792691ef4b72ae26434c3e799
        GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(cgltf glfw imgui stb)

add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

add_executable(vulkan_gltf_rt
        src/main.cpp
        src/gltf_scene.cpp
        src/vulkan_viewer.cpp)
target_link_libraries(vulkan_gltf_rt PRIVATE Vulkan::Vulkan glfw imgui)
target_include_directories(vulkan_gltf_rt PRIVATE ${cgltf_SOURCE_DIR} ${stb_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (WIN32)
    target_link_libraries(vulkan_gltf_rt PRIVATE comdlg32)
endif ()

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
# Stage is explicit because the .glsl suffix does not imply an RT shader stage.
set(RT_SHADERS raygen:rgen miss:rmiss hit:rchit)
set(RT_SPIRV_FILES)
foreach (PAIR IN LISTS RT_SHADERS)
    string(REPLACE ":" ";" PAIR_FIELDS "${PAIR}")
    list(GET PAIR_FIELDS 0 SHADER_NAME)
    list(GET PAIR_FIELDS 1 SHADER_STAGE)
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.glsl)
    set(SPIRV ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
    add_custom_command(OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
            COMMAND ${GLSLANG_VALIDATOR} --target-env vulkan1.2 -S ${SHADER_STAGE} -V ${SHADER_SOURCE} -o ${SPIRV}
            DEPENDS ${SHADER_SOURCE}
            VERBATIM)
    list(APPEND RT_SPIRV_FILES ${SPIRV})
endforeach ()
add_custom_target(rt_shaders ALL DEPENDS ${RT_SPIRV_FILES})
add_dependencies(vulkan_gltf_rt rt_shaders)
target_compile_definitions(vulkan_gltf_rt PRIVATE RT_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
