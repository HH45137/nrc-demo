#pragma once

#include "gltf_scene.hpp"

#include <imgui.h>

#include <memory>
#include <string>

struct GLFWwindow;
struct ImDrawData;

class VulkanViewer
{
public:
    explicit VulkanViewer(GLFWwindow* window);
    ~VulkanViewer();

    VulkanViewer(const VulkanViewer&) = delete;
    VulkanViewer& operator=(const VulkanViewer&) = delete;

    void loadScene(GltfScene scene);
    void rotateView(float deltaX, float deltaY);
    void zoom(float wheelDelta);
    void move(float forwardAxis, float rightAxis, float upAxis, float deltaSeconds, bool boost);
    void setFirstPerson(bool enabled);
    void drawFrame(ImDrawData* drawData);

    [[nodiscard]] bool hasScene() const;
    [[nodiscard]] bool isFirstPerson() const;
    [[nodiscard]] float moveSpeed() const;
    [[nodiscard]] ImTextureID sceneTexture() const;
    [[nodiscard]] const std::string& sceneStatus() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
