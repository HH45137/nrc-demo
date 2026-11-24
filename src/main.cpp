#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "gltf_scene.hpp"
#include "vulkan_viewer.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    bool chooseScene(std::string& path)
    {
#ifdef _WIN32
        char file[MAX_PATH * 4]{};
        const char filter[] = "glTF scenes (*.gltf;*.glb)\0*.gltf;*.glb\0All files (*.*)\0*.*\0";
        OPENFILENAMEA dialog{};
        dialog.lStructSize = sizeof(dialog);
        dialog.lpstrFilter = filter;
        dialog.lpstrFile = file;
        dialog.nMaxFile = static_cast<DWORD>(sizeof(file));
        dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
        if (!GetOpenFileNameA(&dialog))
            return false;
        path = file;
        return true;
#else
        (void)path;
        return false;
#endif
    }

    void loadScene(VulkanViewer& viewer, const std::string& path)
    {
        try
        {
            GltfScene data = loadGltfScene(path);
            viewer.loadScene(std::move(data));
        }
        catch (const std::exception& error)
        {
            std::cerr << "Failed to load scene: " << error.what() << '\n';
        }
    }
} // namespace

int main(int argc, char** argv)
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Vulkan glTF Ray Tracing", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window\n";
        return EXIT_FAILURE;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    int result = EXIT_SUCCESS;
    try
    {
        VulkanViewer viewer(window);
        if (argc > 1)
            loadScene(viewer, argv[1]);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowSize(ImVec2(1280, 720), ImGuiCond_FirstUseEver);
            ImGui::Begin("Vulkan glTF Ray Tracing", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
#ifdef _WIN32
            if (ImGui::Button("Load glTF / GLB"))
            {
                std::string path;
                if (chooseScene(path))
                    loadScene(viewer, path);
            }
#else
            static char pathBuffer[1024]{};
            ImGui::SetNextItemWidth(480);
            ImGui::InputText("Scene path", pathBuffer, sizeof(pathBuffer));
            ImGui::SameLine();
            if (ImGui::Button("Load glTF / GLB") && pathBuffer[0])
                loadScene(viewer, pathBuffer);
#endif
            ImGui::SameLine();
            ImGui::TextUnformatted(viewer.sceneStatus().c_str());

            if (viewer.hasScene())
            {
                const ImVec2 available = ImGui::GetContentRegionAvail();
                const float scale = std::min(available.x / 960.0f, available.y / 540.0f);
                if (scale > 0.0f)
                {
                    ImGui::Image(viewer.sceneTexture(), ImVec2(960.0f * scale, 540.0f * scale));
                    if (ImGui::IsItemHovered())
                    {
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0f))
                        {
                            const ImVec2 delta = ImGui::GetIO().MouseDelta;
                            viewer.orbit(delta.x, delta.y);
                        }
                        const float wheel = ImGui::GetIO().MouseWheel;
                        if (wheel != 0.0f)
                            viewer.zoom(wheel);
                    }
                }
            }
            ImGui::End();
            ImGui::Render();
            viewer.drawFrame(ImGui::GetDrawData());
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Fatal error: " << error.what() << '\n';
        result = EXIT_FAILURE;
    }

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
