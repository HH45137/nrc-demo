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
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Vulkan glTF Ray Tracing", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window\n";
        return EXIT_FAILURE;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().LogFilename = nullptr;
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

            ImGuiIO& io = ImGui::GetIO();
            if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F, false))
                viewer.setFirstPerson(!viewer.isFirstPerson());

            // Full-window scene viewport. The image is not nested in a decorated
            // ImGui window anymore: it fills the entire client area, with a small
            // floating control panel on top.
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            const ImGuiWindowFlags sceneFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
            ImGui::Begin("##scene", nullptr, sceneFlags);

            bool firstPerson = viewer.isFirstPerson();
            const ImVec2 sceneSize = ImGui::GetContentRegionAvail();
            if (viewer.hasScene() && sceneSize.x > 0.0f && sceneSize.y > 0.0f)
            {
                ImGui::Image(viewer.sceneTexture(), sceneSize);

                if (firstPerson)
                {
                    // Small aiming reticle for first-person roaming.
                    const ImVec2 imageMin = ImGui::GetItemRectMin();
                    const ImVec2 imageMax = ImGui::GetItemRectMax();
                    const ImVec2 center((imageMin.x + imageMax.x) * 0.5f, (imageMin.y + imageMax.y) * 0.5f);
                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    drawList->AddCircle(center, 7.0f, IM_COL32(0, 0, 0, 160), 12, 1.5f);
                    drawList->AddCircleFilled(center, 2.0f, IM_COL32(255, 255, 255, 220));
                }

                // Keep rotating while the button is held, even if the cursor
                // leaves the image during the drag.
                static bool rotating = false;
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    rotating = true;
                if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    rotating = false;
                if (rotating && (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f))
                    viewer.rotateView(io.MouseDelta.x, io.MouseDelta.y);

                if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f)
                    viewer.zoom(io.MouseWheel);

                if (firstPerson && !io.WantTextInput)
                {
                    const float forward =
                        (ImGui::IsKeyDown(ImGuiKey_W) ? 1.0f : 0.0f) - (ImGui::IsKeyDown(ImGuiKey_S) ? 1.0f : 0.0f);
                    const float right =
                        (ImGui::IsKeyDown(ImGuiKey_D) ? 1.0f : 0.0f) - (ImGui::IsKeyDown(ImGuiKey_A) ? 1.0f : 0.0f);
                    const float up =
                        (ImGui::IsKeyDown(ImGuiKey_E) ? 1.0f : 0.0f) - (ImGui::IsKeyDown(ImGuiKey_Q) ? 1.0f : 0.0f);
                    viewer.move(forward, right, up, io.DeltaTime, io.KeyShift);
                }
            }
            else
            {
                const char* message = "Load a glTF / GLB scene to begin";
                const ImVec2 textSize = ImGui::CalcTextSize(message);
                ImGui::SetCursorPos(ImVec2((sceneSize.x - textSize.x) * 0.5f, (sceneSize.y - textSize.y) * 0.5f));
                ImGui::TextUnformatted(message);
            }
            ImGui::End();
            ImGui::PopStyleVar(2);

            // Compact overlay for loading and camera mode.
            ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSizeConstraints(ImVec2(360.0f, 0.0f), ImVec2(720.0f, 1000.0f));
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("Controls", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove);
#ifdef _WIN32
            if (ImGui::Button("Load glTF / GLB"))
            {
                std::string path;
                if (chooseScene(path))
                    loadScene(viewer, path);
            }
#else
            static char pathBuffer[1024]{};
            ImGui::SetNextItemWidth(300.0f);
            ImGui::InputText("Scene path", pathBuffer, sizeof(pathBuffer));
            if (ImGui::Button("Load glTF / GLB") && pathBuffer[0])
                loadScene(viewer, pathBuffer);
#endif

            firstPerson = viewer.isFirstPerson();
            if (ImGui::Checkbox("First-person", &firstPerson))
                viewer.setFirstPerson(firstPerson);
            ImGui::SameLine();
            if (firstPerson)
                ImGui::TextDisabled("WASD | Q/E down/up | Shift boost | drag look | wheel speed %.1f | F toggle",
                                    viewer.moveSpeed());
            else
                ImGui::TextDisabled("drag orbit | wheel zoom | F toggle");

            ImGui::TextWrapped("%s", viewer.sceneStatus().c_str());
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
