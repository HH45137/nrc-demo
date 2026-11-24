#include "info.hpp"
#include "vulkan_viewer.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan.hpp>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    constexpr uint32_t kRenderWidth = RENDERER_RESOLUTION_WIDTH;
    constexpr uint32_t kRenderHeight = RENDERER_RESOLUTION_HEIGHT;
    constexpr uint32_t kFramesInFlight = 2;
    constexpr std::array<const char*, 4> kDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };

    struct CameraData
    {
        alignas(16) float origin[4];
        alignas(16) float right[4];
        alignas(16) float up[4];
        alignas(16) float forward[4];
        alignas(4) uint32_t sampleIndex;
        alignas(4) int _pad0;
        alignas(4) int _pad1;
        alignas(4) int _pad2;
    };

    // Shared camera state. Orbit mode derives the position from yaw/pitch/distance;
    // first-person mode keeps an explicit world position and uses yaw/pitch as the
    // look direction. Toggling between modes converts yaw/pitch so the view
    // direction is preserved.
    struct CameraState
    {
        float yaw{}, pitch{};
        float distance{3.0f};
        std::array<float, 3> position{0.0f, 0.0f, 3.0f};
        bool firstPerson{};
    };

    CameraData makeCamera(const CameraState& state, uint32_t sampleIndex)
    {
        const float cp = std::cos(state.pitch);
        const float sp = std::sin(state.pitch);
        const float sy = std::sin(state.yaw);
        const float cy = std::cos(state.yaw);

        float px = 0.0f, py = 0.0f, pz = 0.0f;
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        if (state.firstPerson)
        {
            px = state.position[0];
            py = state.position[1];
            pz = state.position[2];

            // yaw = 0 looks down -Z; positive yaw turns right, positive pitch looks up.
            fx = sy * cp;
            fy = sp;
            fz = -cy * cp;
        }
        else
        {
            px = state.distance * cp * sy;
            py = state.distance * sp;
            pz = state.distance * cp * cy;

            const float length = std::sqrt(px * px + py * py + pz * pz);
            fx = -px / length;
            fy = -py / length;
            fz = -pz / length;
        }

        // right = normalize(cross(forward, worldUp)); worldUp is +Y.
        float rx = -fz;
        float rz = fx;
        const float rightLength = std::sqrt(rx * rx + rz * rz);
        if (rightLength > 1e-6f)
        {
            rx /= rightLength;
            rz /= rightLength;
        }
        else
        {
            rx = 1.0f;
            rz = 0.0f;
        }

        CameraData camera{};
        camera.origin[0] = px;
        camera.origin[1] = py;
        camera.origin[2] = pz;
        camera.right[0] = rx;
        camera.right[2] = rz;
        camera.forward[0] = fx;
        camera.forward[1] = fy;
        camera.forward[2] = fz;
        // up = cross(right, forward)
        camera.up[0] = -rz * fy;
        camera.up[1] = rz * fx - rx * fz;
        camera.up[2] = rx * fy;
        camera.sampleIndex = sampleIndex;
        return camera;
    }

    std::vector<uint32_t> readSpirv(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open SPIR-V shader: " + path);
        const auto byteCount = file.tellg();
        if (byteCount <= 0 || byteCount % 4 != 0)
            throw std::runtime_error("Invalid SPIR-V shader: " + path);
        std::vector<uint32_t> words(static_cast<size_t>(byteCount) / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()), byteCount);
        return words;
    }

    vk::DeviceAddress alignDeviceAddress(vk::DeviceAddress value, uint32_t alignment)
    {
        return (value + alignment - 1) & ~vk::DeviceAddress(alignment - 1);
    }
} // namespace

struct VulkanViewer::Impl
{
    struct Buffer
    {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Buffer handle{nullptr};
        vk::DeviceAddress address{};
    };

    struct Image
    {
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Image handle{nullptr};
        vk::raii::ImageView view{nullptr};
    };

    struct Scene
    {
        Buffer vertices, texcoords, indices, materialIndices, instance, blasStorage, tlasStorage, sbt;
        vk::raii::AccelerationStructureKHR blas{nullptr}, tlas{nullptr};
        std::vector<Image> baseColorImages;
        vk::raii::Sampler sampler{nullptr};
        vk::raii::DescriptorPool descriptorPool{nullptr};
        vk::raii::DescriptorSet descriptorSet{nullptr};
        vk::StridedDeviceAddressRegionKHR raygenRegion{}, missRegion{}, hitRegion{}, callableRegion{};
        uint32_t triangleCount{};
    };

    struct FrameSync
    {
        vk::raii::Semaphore imageAvailable{nullptr};
        vk::raii::Fence fence{nullptr};
    };

    GLFWwindow* window{};
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::SurfaceKHR surface{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};
    uint32_t queueFamily{};
    uint32_t scratchAlignment{1};
    uint32_t maxBindlessTextures{1};

    vk::raii::CommandPool commandPool{nullptr};
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::array<FrameSync, kFramesInFlight> frames;
    std::vector<vk::raii::Semaphore> renderFinished;
    uint32_t frameIndex{};
    uint32_t sampleIndex{};

    vk::raii::SwapchainKHR swapchain{nullptr};
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::raii::ImageView> swapchainViews;
    vk::raii::RenderPass renderPass{nullptr};
    std::vector<vk::raii::Framebuffer> framebuffers;
    vk::Format swapchainFormat{vk::Format::eUndefined};
    vk::Extent2D swapchainExtent{};
    bool swapchainDirty{true};

    vk::raii::DescriptorSetLayout rtSetLayout{nullptr};
    vk::raii::PipelineLayout rtPipelineLayout{nullptr};
    vk::raii::Pipeline rtPipeline{nullptr};
    vk::raii::DescriptorPool rtDescriptorPool{nullptr};
    vk::raii::DescriptorPool imguiDescriptorPool{nullptr};
    Image output, samplesAccumulated;
    vk::raii::Sampler outputSampler{nullptr};
    VkDescriptorSet imguiOutputSet{};
    bool outputInitialized{};
    uint32_t minImageCount{2};

    std::unique_ptr<Scene> scene;
    std::string status{"No scene loaded"};
    CameraState camera;
    float moveSpeed{2.0f};

    explicit Impl(GLFWwindow* glfwWindow) : window(glfwWindow)
    {
        createInstanceAndDevice();
        createCommandPool();
        createAllImages();
        createRtPipeline();
        createSwapchain();
        createImGuiVulkanBackend();
        createFrameSync();
    }

    ~Impl()
    {
        if (static_cast<VkDevice>(*device) != VK_NULL_HANDLE)
            device.waitIdle();
        if (imguiOutputSet)
            ImGui_ImplVulkan_RemoveTexture(imguiOutputSet);
        ImGui_ImplVulkan_Shutdown();
        framebuffers.clear();
        swapchainViews.clear();
        scene.reset();
    }

    uint32_t findMemoryType(uint32_t bits, vk::MemoryPropertyFlags required) const
    {
        const vk::PhysicalDeviceMemoryProperties properties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        {
            if ((bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & required) == required)
                return i;
        }
        throw std::runtime_error("No Vulkan memory type satisfies requested properties");
    }

    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryFlags,
                        const void* initialData = nullptr)
    {
        Buffer result;
        result.handle = vk::raii::Buffer(device, vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive));
        const vk::MemoryRequirements requirements = result.handle.getMemoryRequirements();
        const bool needsAddress = static_cast<bool>(usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
        vk::MemoryAllocateFlagsInfo addressFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);
        vk::MemoryAllocateInfo allocation(requirements.size, findMemoryType(requirements.memoryTypeBits, memoryFlags));
        if (needsAddress)
            allocation.pNext = &addressFlags;
        result.memory = vk::raii::DeviceMemory(device, allocation);
        result.handle.bindMemory(*result.memory, 0);
        if (initialData)
        {
            void* mapped = result.memory.mapMemory(0, size);
            std::memcpy(mapped, initialData, static_cast<size_t>(size));
            result.memory.unmapMemory();
        }
        if (needsAddress)
            result.address = device.getBufferAddress(vk::BufferDeviceAddressInfo(*result.handle));
        return result;
    }

    Image createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage)
    {
        Image result;
        vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format, vk::Extent3D(width, height, 1), 1, 1,
                                      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage,
                                      vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);
        result.handle = vk::raii::Image(device, imageInfo);
        const vk::MemoryRequirements requirements = result.handle.getMemoryRequirements();
        result.memory =
            vk::raii::DeviceMemory(device,
                                   vk::MemoryAllocateInfo(requirements.size,
                                                          findMemoryType(requirements.memoryTypeBits,
                                                                         vk::MemoryPropertyFlagBits::eDeviceLocal)));
        result.handle.bindMemory(*result.memory, 0);
        result.view = vk::raii::ImageView(
            device,
            vk::ImageViewCreateInfo({}, *result.handle, vk::ImageViewType::e2D, format, {},
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
        return result;
    }

    void immediateSubmit(const std::function<void(vk::raii::CommandBuffer&)>& record)
    {
        auto buffers = device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1));
        auto& commandBuffer = buffers.front();
        commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        record(commandBuffer);
        commandBuffer.end();
        vk::CommandBuffer rawCommandBuffer = *commandBuffer;
        queue.submit(vk::SubmitInfo({}, {}, rawCommandBuffer), nullptr);
        queue.waitIdle();
    }

    std::vector<Image> uploadTextures(const std::vector<GltfTexture>& textures)
    {
        std::vector<Image> images;
        images.reserve(textures.size());

        size_t totalBytes = 0;
        for (const GltfTexture& texture : textures)
            totalBytes += texture.rgba.size();

        // Pack all pixels into one staging buffer and record every copy in a
        // single command buffer. This avoids per-texture allocations and queue
        // stalls when a scene has many textures.
        Buffer staging =
            createBuffer(totalBytes, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        {
            void* mapped = staging.memory.mapMemory(0, totalBytes);
            size_t offset = 0;
            for (const GltfTexture& texture : textures)
            {
                std::memcpy(static_cast<uint8_t*>(mapped) + offset, texture.rgba.data(), texture.rgba.size());
                offset += texture.rgba.size();
            }
            staging.memory.unmapMemory();
        }

        for (const GltfTexture& texture : textures)
        {
            images.push_back(createImage(texture.width, texture.height, vk::Format::eR8G8B8A8Srgb,
                                         vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
        }

        immediateSubmit(
            [&](vk::raii::CommandBuffer& commandBuffer)
            {
                size_t offset = 0;
                for (size_t i = 0; i < textures.size(); ++i)
                {
                    const GltfTexture& texture = textures[i];
                    Image& image = images[i];

                    vk::ImageMemoryBarrier toTransfer;
                    toTransfer.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
                    toTransfer.oldLayout = vk::ImageLayout::eUndefined;
                    toTransfer.newLayout = vk::ImageLayout::eTransferDstOptimal;
                    toTransfer.srcQueueFamilyIndex = toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toTransfer.image = *image.handle;
                    toTransfer.subresourceRange =
                        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                                  vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toTransfer);

                    vk::BufferImageCopy copy;
                    copy.bufferOffset = offset;
                    copy.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
                    copy.imageExtent = vk::Extent3D(texture.width, texture.height, 1);
                    commandBuffer.copyBufferToImage(*staging.handle, *image.handle,
                                                    vk::ImageLayout::eTransferDstOptimal, copy);

                    vk::ImageMemoryBarrier toShader;
                    toShader.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                    toShader.dstAccessMask = vk::AccessFlagBits::eShaderRead;
                    toShader.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                    toShader.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                    toShader.srcQueueFamilyIndex = toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toShader.image = *image.handle;
                    toShader.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                  vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {},
                                                  toShader);

                    offset += texture.rgba.size();
                }
            });
        return images;
    }

    void createInstanceAndDevice();
    void createCommandPool();
    void createAllImages();
    void createRtPipeline();
    void createSwapchain();
    void createImGuiVulkanBackend();
    void createFrameSync();
    void buildScene(GltfScene loaded);
    void recordTrace(vk::raii::CommandBuffer& commandBuffer);
    void recreateSwapchain();
    void drawFrame(ImDrawData* drawData);
};

void VulkanViewer::Impl::createInstanceAndDevice()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (!glfwExtensions || glfwExtensionCount == 0)
        throw std::runtime_error("GLFW did not report the Vulkan surface extensions");

    vk::ApplicationInfo application("Vulkan glTF Ray Tracing", 1, "Vulkan glTF RT", 1, VK_API_VERSION_1_2);
    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.pApplicationInfo = &application;
    instanceInfo.enabledExtensionCount = glfwExtensionCount;
    instanceInfo.ppEnabledExtensionNames = glfwExtensions;
    instance = vk::raii::Instance(context, instanceInfo);

    VkSurfaceKHR rawSurface{};
    const VkResult surfaceResult =
        glfwCreateWindowSurface(static_cast<VkInstance>(*instance), window, nullptr, &rawSurface);
    if (surfaceResult != VK_SUCCESS)
        throw std::runtime_error("GLFW failed to create a Vulkan surface");
    surface = vk::raii::SurfaceKHR(instance, rawSurface);

    auto physicalDevices = instance.enumeratePhysicalDevices();
    for (auto& candidate : physicalDevices)
    {
        const auto extensions = candidate.enumerateDeviceExtensionProperties();
        const bool hasAllExtensions = std::all_of(
            kDeviceExtensions.begin(), kDeviceExtensions.end(),
            [&](const char* name)
            {
                return std::any_of(extensions.begin(), extensions.end(), [&](const vk::ExtensionProperties& extension)
                                   { return std::strcmp(extension.extensionName, name) == 0; });
            });
        if (!hasAllExtensions)
            continue;

        const auto queueFamilies = candidate.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); ++i)
        {
            const bool graphics = static_cast<bool>(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics);
            if (graphics && candidate.getSurfaceSupportKHR(i, *surface))
            {
                physicalDevice = std::move(candidate);
                queueFamily = i;
                break;
            }
        }
        if (static_cast<VkPhysicalDevice>(*physicalDevice) != VK_NULL_HANDLE)
            break;
    }
    if (static_cast<VkPhysicalDevice>(*physicalDevice) == VK_NULL_HANDLE)
        throw std::runtime_error("No Vulkan ray-tracing device supports graphics and presentation");

    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &accelerationProperties;
    physicalDevice.getProperties2(&properties);
    scratchAlignment = std::max(1u, accelerationProperties.minAccelerationStructureScratchOffsetAlignment);

    const auto supportedFeatures =
        physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                                    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                                    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
    const vk::PhysicalDeviceVulkan12Features& supportedVulkan12 =
        supportedFeatures.get<vk::PhysicalDeviceVulkan12Features>();
    if (!supportedVulkan12.bufferDeviceAddress ||
        !supportedFeatures.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure ||
        !supportedFeatures.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline)
        throw std::runtime_error("The selected device does not support required Vulkan RT features");
    if (!supportedVulkan12.runtimeDescriptorArray || !supportedVulkan12.descriptorBindingVariableDescriptorCount ||
        !supportedVulkan12.shaderSampledImageArrayNonUniformIndexing)
        throw std::runtime_error("The selected device does not support bindless sampled-image arrays");

    const vk::PhysicalDeviceFeatures coreFeatures = physicalDevice.getFeatures();
    if (!coreFeatures.shaderSampledImageArrayDynamicIndexing)
        throw std::runtime_error("The selected device cannot dynamically index sampled-image arrays");

    // A runtime descriptor array still needs a non-zero layout bound for the RT
    // pipeline. Use the device limit as the layout maximum, but allocate only
    // the number of textures that the scene actually needs.
    const vk::PhysicalDeviceLimits limits = physicalDevice.getProperties().limits;
    maxBindlessTextures =
        std::max(1u, std::min({limits.maxPerStageDescriptorSamplers, limits.maxDescriptorSetSamplers, 65536u}));

    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vulkan12Features.bufferDeviceAddress = true;
    vulkan12Features.runtimeDescriptorArray = true;
    vulkan12Features.descriptorBindingVariableDescriptorCount = true;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = true;

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures;
    accelerationFeatures.pNext = &rayTracingFeatures;
    vulkan12Features.pNext = &accelerationFeatures;
    accelerationFeatures.accelerationStructure = true;
    rayTracingFeatures.rayTracingPipeline = true;

    // Materials are selected per hit through a runtime descriptor array.
    vk::PhysicalDeviceFeatures2 coreFeatureChain;
    coreFeatureChain.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    coreFeatureChain.pNext = &vulkan12Features;
    const float priority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, queueFamily, 1, &priority);
    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.pNext = &coreFeatureChain;
    deviceInfo.setQueueCreateInfos(queueInfo);
    deviceInfo.setPEnabledExtensionNames(kDeviceExtensions);
    device = physicalDevice.createDevice(deviceInfo);
    queue = device.getQueue(queueFamily, 0);
}

void VulkanViewer::Impl::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueFamily;
    commandPool = device.createCommandPool(poolInfo);
    commandBuffers = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, kFramesInFlight));
}

void VulkanViewer::Impl::createAllImages()
{
    // Output
    output = createImage(kRenderWidth, kRenderHeight, vk::Format::eR8G8B8A8Unorm,
                         vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW =
        vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.maxLod = 0.0f;
    outputSampler = device.createSampler(samplerInfo);

    // Samples accumulated: linear HDR mean of all samples for each pixel.
    samplesAccumulated =
        createImage(kRenderWidth, kRenderHeight, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage);
    immediateSubmit(
        [&](vk::raii::CommandBuffer& commandBuffer)
        {
            vk::ImageMemoryBarrier toGeneral;
            toGeneral.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
            toGeneral.oldLayout = vk::ImageLayout::eUndefined;
            toGeneral.newLayout = vk::ImageLayout::eGeneral;
            toGeneral.srcQueueFamilyIndex = toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = *samplesAccumulated.handle;
            toGeneral.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                          vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, toGeneral);
        });
}

void VulkanViewer::Impl::createRtPipeline()
{
    const std::array<vk::DescriptorSetLayoutBinding, 8> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eAccelerationStructureKHR, 1,
                                       vk::ShaderStageFlagBits::eRaygenKHR),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eClosestHitKHR),
        vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eClosestHitKHR),
        vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eClosestHitKHR),
        vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eClosestHitKHR),
        vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR),
        // Runtime array: the layout declares the device maximum, while each
        // descriptor set allocates only the number of textures it actually uses.
        vk::DescriptorSetLayoutBinding(7, vk::DescriptorType::eCombinedImageSampler, maxBindlessTextures,
                                       vk::ShaderStageFlagBits::eClosestHitKHR),
    };

    std::array<vk::DescriptorBindingFlags, 8> bindingFlags{};
    bindingFlags[7] = vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
    layoutInfo.pNext = &bindingFlagsInfo;
    rtSetLayout = device.createDescriptorSetLayout(layoutInfo);

    const vk::PushConstantRange cameraRange(vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(CameraData));
    rtPipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo({}, *rtSetLayout, cameraRange));

    const std::array<std::string, 3> shaderPaths = {
        CURRENT_RUN_PATH + "/" + SHADER_DIR + "/rt_raygen.spv",
        CURRENT_RUN_PATH + "/" + SHADER_DIR + "/rt_miss.spv",
        CURRENT_RUN_PATH + "/" + SHADER_DIR + "/rt_hit.spv",
    };
    std::array<vk::raii::ShaderModule, 3> shaderModules = {
        vk::raii::ShaderModule(nullptr), vk::raii::ShaderModule(nullptr), vk::raii::ShaderModule(nullptr)};
    std::array<vk::PipelineShaderStageCreateInfo, 3> stages{};
    const std::array<vk::ShaderStageFlagBits, 3> stageFlags = {vk::ShaderStageFlagBits::eRaygenKHR,
                                                               vk::ShaderStageFlagBits::eMissKHR,
                                                               vk::ShaderStageFlagBits::eClosestHitKHR};
    for (size_t i = 0; i < stages.size(); ++i)
    {
        const auto words = readSpirv(shaderPaths[i]);
        shaderModules[i] =
            device.createShaderModule(vk::ShaderModuleCreateInfo({}, words.size() * sizeof(uint32_t), words.data()));
        stages[i] = vk::PipelineShaderStageCreateInfo({}, stageFlags[i], *shaderModules[i], "main");
    }

    std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> groups{};
    groups[0].type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    groups[1] = groups[0];
    groups[1].generalShader = 1;
    groups[2].type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    vk::RayTracingPipelineCreateInfoKHR pipelineInfo;
    pipelineInfo.setStages(stages).setGroups(groups).setMaxPipelineRayRecursionDepth(1).setLayout(*rtPipelineLayout);
    auto pipelines = device.createRayTracingPipelinesKHR(nullptr, nullptr, pipelineInfo);
    rtPipeline = std::move(pipelines.front());
}

void VulkanViewer::Impl::createImGuiVulkanBackend()
{
    const std::array<vk::DescriptorPoolSize, 1> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 64)};
    vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 64, poolSizes);
    imguiDescriptorPool = device.createDescriptorPool(poolInfo);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_2;
    initInfo.Instance = static_cast<VkInstance>(*instance);
    initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(*physicalDevice);
    initInfo.Device = static_cast<VkDevice>(*device);
    initInfo.QueueFamily = queueFamily;
    initInfo.Queue = static_cast<VkQueue>(*queue);
    initInfo.DescriptorPool = static_cast<VkDescriptorPool>(*imguiDescriptorPool);
    initInfo.RenderPass = static_cast<VkRenderPass>(*renderPass);
    initInfo.MinImageCount = minImageCount;
    initInfo.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    if (!ImGui_ImplVulkan_Init(&initInfo))
        throw std::runtime_error("Dear ImGui Vulkan backend initialization failed");

    imguiOutputSet = ImGui_ImplVulkan_AddTexture(static_cast<VkSampler>(*outputSampler),
                                                 static_cast<VkImageView>(*output.view), VK_IMAGE_LAYOUT_GENERAL);
}

void VulkanViewer::Impl::createFrameSync()
{
    for (FrameSync& frame : frames)
    {
        frame.imageAvailable = device.createSemaphore(vk::SemaphoreCreateInfo{});
        frame.fence = device.createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
}

void VulkanViewer::Impl::createSwapchain()
{
    const vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    const auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    if (formats.empty())
        throw std::runtime_error("The Vulkan surface exposes no image formats");
    auto selectedFormat = formats.front();
    for (const auto& format : formats)
    {
        if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            selectedFormat = format;
            break;
        }
    }

    int framebufferWidth = 0, framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0)
        throw std::runtime_error("Cannot create swapchain for a zero-sized window");
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        swapchainExtent = capabilities.currentExtent;
    else
    {
        swapchainExtent.width = std::clamp(static_cast<uint32_t>(framebufferWidth), capabilities.minImageExtent.width,
                                           capabilities.maxImageExtent.width);
        swapchainExtent.height = std::clamp(static_cast<uint32_t>(framebufferHeight),
                                            capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    minImageCount = std::max(2u, capabilities.minImageCount);
    uint32_t imageCount = minImageCount + 1;
    if (capabilities.maxImageCount != 0)
        imageCount = std::min(imageCount, capabilities.maxImageCount);

    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.surface = *surface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = selectedFormat.format;
    swapchainInfo.imageColorSpace = selectedFormat.colorSpace;
    swapchainInfo.imageExtent = swapchainExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainInfo.preTransform = capabilities.currentTransform;
    swapchainInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchainInfo.presentMode = vk::PresentModeKHR::eFifo;
    swapchainInfo.clipped = true;
    if (static_cast<VkSwapchainKHR>(*swapchain) != VK_NULL_HANDLE)
        swapchainInfo.oldSwapchain = *swapchain;

    swapchain = device.createSwapchainKHR(swapchainInfo);
    swapchainImages = swapchain.getImages();
    // One present-semaphore per swapchain image avoids reusing a semaphore that
    // is still owned by an in-flight presentation.
    renderFinished.clear();
    for ([[maybe_unused]] vk::Image image : swapchainImages)
        renderFinished.push_back(device.createSemaphore(vk::SemaphoreCreateInfo{}));
    swapchainFormat = selectedFormat.format;
    swapchainViews.clear();
    swapchainViews.reserve(swapchainImages.size());
    for (vk::Image image : swapchainImages)
    {
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = swapchainFormat;
        viewInfo.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        swapchainViews.emplace_back(device, viewInfo);
    }

    if (static_cast<VkRenderPass>(*renderPass) == VK_NULL_HANDLE)
    {
        vk::AttachmentDescription colorAttachment;
        colorAttachment.format = swapchainFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
        const vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.setColorAttachments(colorReference);
        vk::SubpassDependency dependency;
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        renderPass = device.createRenderPass(vk::RenderPassCreateInfo({}, colorAttachment, subpass, dependency));
    }

    framebuffers.clear();
    for (const auto& view : swapchainViews)
    {
        const vk::ImageView attachment = *view;
        vk::FramebufferCreateInfo framebufferInfo;
        framebufferInfo.renderPass = *renderPass;
        framebufferInfo.setAttachments(attachment);
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;
        framebuffers.emplace_back(device, framebufferInfo);
    }
    swapchainDirty = false;
}

void VulkanViewer::Impl::recreateSwapchain()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }
    device.waitIdle();
    framebuffers.clear();
    swapchainViews.clear();
    createSwapchain();
    ImGui_ImplVulkan_SetMinImageCount(minImageCount);
}

void VulkanViewer::Impl::buildScene(GltfScene loaded)
{
    if (loaded.textures.empty())
        loaded.textures.emplace_back();
    if (loaded.materialIndices.size() * 3 != loaded.indices.size())
        throw std::runtime_error("Scene material table does not match triangle count");

    auto next = std::make_unique<Scene>();
    const auto hostVisible = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const auto buildInput = vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer;
    next->vertices =
        createBuffer(loaded.positions.size() * sizeof(float), buildInput, hostVisible, loaded.positions.data());
    next->texcoords = createBuffer(loaded.texcoords.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                                   hostVisible, loaded.texcoords.data());
    next->indices =
        createBuffer(loaded.indices.size() * sizeof(uint32_t), buildInput | vk::BufferUsageFlagBits::eIndexBuffer,
                     hostVisible, loaded.indices.data());
    next->materialIndices =
        createBuffer(loaded.materialIndices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                     hostVisible, loaded.materialIndices.data());

    vk::BufferUsageFlags scratchUsage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
    triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
    triangles.vertexData.deviceAddress = next->vertices.address;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = static_cast<uint32_t>(loaded.positions.size() / 3 - 1);
    triangles.indexType = vk::IndexType::eUint32;
    triangles.indexData.deviceAddress = next->indices.address;
    vk::AccelerationStructureGeometryKHR blasGeometry;
    blasGeometry.geometryType = vk::GeometryTypeKHR::eTriangles;
    blasGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque;
    blasGeometry.geometry.triangles = triangles;

    vk::AccelerationStructureBuildGeometryInfoKHR blasBuild;
    blasBuild.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    blasBuild.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    blasBuild.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    blasBuild.setGeometries(blasGeometry);
    const uint32_t triangleCount = static_cast<uint32_t>(loaded.indices.size() / 3);
    const vk::AccelerationStructureBuildSizesInfoKHR blasSizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, blasBuild, triangleCount);

    next->blasStorage = createBuffer(blasSizes.accelerationStructureSize,
                                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR blasCreate;
    blasCreate.buffer = *next->blasStorage.handle;
    blasCreate.size = blasSizes.accelerationStructureSize;
    blasCreate.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    next->blas = device.createAccelerationStructureKHR(blasCreate);

    Buffer blasScratch = createBuffer(blasSizes.buildScratchSize + scratchAlignment, scratchUsage,
                                      vk::MemoryPropertyFlagBits::eDeviceLocal);
    blasBuild.dstAccelerationStructure = *next->blas;
    blasBuild.scratchData.deviceAddress = alignDeviceAddress(blasScratch.address, scratchAlignment);
    vk::AccelerationStructureBuildRangeInfoKHR blasRange;
    blasRange.primitiveCount = triangleCount;
    const std::array<const vk::AccelerationStructureBuildRangeInfoKHR*, 1> blasRanges = {&blasRange};
    immediateSubmit(
        [&](vk::raii::CommandBuffer& commandBuffer)
        {
            commandBuffer.buildAccelerationStructuresKHR(blasBuild, blasRanges);
            const vk::MemoryBarrier barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                            vk::AccessFlagBits::eAccelerationStructureReadKHR);
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                          vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier, {},
                                          {});
        });

    const vk::DeviceAddress blasAddress =
        device.getAccelerationStructureAddressKHR(vk::AccelerationStructureDeviceAddressInfoKHR(*next->blas));
    vk::AccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xff;
    instance.flags =
        static_cast<VkGeometryInstanceFlagsKHR>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    instance.accelerationStructureReference = blasAddress;
    next->instance = createBuffer(sizeof(instance), buildInput, hostVisible, &instance);

    vk::AccelerationStructureGeometryInstancesDataKHR instanceData;
    instanceData.data.deviceAddress = next->instance.address;
    vk::AccelerationStructureGeometryKHR tlasGeometry;
    tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
    tlasGeometry.geometry.instances = instanceData;
    vk::AccelerationStructureBuildGeometryInfoKHR tlasBuild;
    tlasBuild.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    tlasBuild.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    tlasBuild.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    tlasBuild.setGeometries(tlasGeometry);
    const vk::AccelerationStructureBuildSizesInfoKHR tlasSizes =
        device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuild, 1u);
    next->tlasStorage = createBuffer(tlasSizes.accelerationStructureSize,
                                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::AccelerationStructureCreateInfoKHR tlasCreate;
    tlasCreate.buffer = *next->tlasStorage.handle;
    tlasCreate.size = tlasSizes.accelerationStructureSize;
    tlasCreate.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    next->tlas = device.createAccelerationStructureKHR(tlasCreate);
    Buffer tlasScratch = createBuffer(tlasSizes.buildScratchSize + scratchAlignment, scratchUsage,
                                      vk::MemoryPropertyFlagBits::eDeviceLocal);
    tlasBuild.dstAccelerationStructure = *next->tlas;
    tlasBuild.scratchData.deviceAddress = alignDeviceAddress(tlasScratch.address, scratchAlignment);
    vk::AccelerationStructureBuildRangeInfoKHR tlasRange;
    tlasRange.primitiveCount = 1;
    const std::array<const vk::AccelerationStructureBuildRangeInfoKHR*, 1> tlasRanges = {&tlasRange};
    immediateSubmit(
        [&](vk::raii::CommandBuffer& commandBuffer)
        {
            commandBuffer.buildAccelerationStructuresKHR(tlasBuild, tlasRanges);
            const vk::MemoryBarrier barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                            vk::AccessFlagBits::eAccelerationStructureReadKHR);
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                          vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, barrier, {}, {});
        });

    if (loaded.textures.size() > maxBindlessTextures)
        throw std::runtime_error("Scene requires " + std::to_string(loaded.textures.size()) +
                                 " textures, but this device supports at most " + std::to_string(maxBindlessTextures));

    next->baseColorImages = uploadTextures(loaded.textures);
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.maxLod = 0.0f;
    next->sampler = device.createSampler(samplerInfo);

    const uint32_t textureCount = static_cast<uint32_t>(loaded.textures.size());
    const std::array<vk::DescriptorPoolSize, 4> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eAccelerationStructureKHR, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 2),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, textureCount),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4),
    };
    next->descriptorPool = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes));

    // Bindless array: allocate exactly as many descriptors as the scene needs.
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo;
    variableCountInfo.setDescriptorSetCount(1);
    variableCountInfo.setPDescriptorCounts(&textureCount);
    vk::DescriptorSetAllocateInfo descriptorAllocateInfo(*next->descriptorPool, *rtSetLayout);
    descriptorAllocateInfo.pNext = &variableCountInfo;
    next->descriptorSet = std::move(device.allocateDescriptorSets(descriptorAllocateInfo).front());

    vk::WriteDescriptorSetAccelerationStructureKHR asDescriptor;
    asDescriptor.setAccelerationStructures(*next->tlas);
    const vk::DescriptorImageInfo outputInfo(*outputSampler, *output.view, vk::ImageLayout::eGeneral);
    const vk::DescriptorBufferInfo uvInfo(*next->texcoords.handle, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo indexInfo(*next->indices.handle, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo positionInfo(*next->vertices.handle, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo materialInfo(*next->materialIndices.handle, 0, VK_WHOLE_SIZE);
    vk::DescriptorImageInfo samplesAccumulatedInfo{};
    samplesAccumulatedInfo.imageView = *samplesAccumulated.view;
    samplesAccumulatedInfo.imageLayout = vk::ImageLayout::eGeneral;

    std::vector<vk::DescriptorImageInfo> textureInfos;
    textureInfos.reserve(textureCount);
    for (const Image& image : next->baseColorImages)
    {
        vk::DescriptorImageInfo info{};
        info.sampler = *next->sampler;
        info.imageView = *image.view;
        info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        textureInfos.push_back(info);
    }

    std::array<vk::WriteDescriptorSet, 8> writes{};
    writes[0]
        .setPNext(&asDescriptor)
        .setDstSet(*next->descriptorSet)
        .setDstBinding(0)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    writes[1]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(1)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(outputInfo);
    writes[2]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(2)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(materialInfo);
    writes[3]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(3)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(uvInfo);
    writes[4]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(4)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(indexInfo);
    writes[5]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(5)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(positionInfo);
    writes[6]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(6)
        .setDescriptorCount(1)
        .setDescriptorType(vk::DescriptorType::eStorageImage)
        .setImageInfo(samplesAccumulatedInfo);
    writes[7]
        .setDstSet(*next->descriptorSet)
        .setDstBinding(7)
        .setDescriptorCount(textureCount)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(textureInfos);
    device.updateDescriptorSets(writes, {});

    const auto properties =
        physicalDevice
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const auto& pipelineProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const uint32_t handleSize = pipelineProperties.shaderGroupHandleSize;
    const uint32_t handleAlignment = pipelineProperties.shaderGroupHandleAlignment;
    const uint32_t baseAlignment = pipelineProperties.shaderGroupBaseAlignment;
    const uint32_t stride = (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);
    const vk::DeviceSize sbtSize = vk::DeviceSize(baseAlignment) * 3;
    const auto handles = rtPipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, 3, size_t(handleSize) * 3);
    std::vector<uint8_t> sbtBytes(sbtSize);
    for (uint32_t i = 0; i < 3; ++i)
        std::memcpy(sbtBytes.data() + vk::DeviceSize(i) * baseAlignment, handles.data() + i * handleSize, handleSize);
    next->sbt = createBuffer(
        sbtSize + baseAlignment,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, hostVisible);
    void* mapped = next->sbt.memory.mapMemory(0, sbtSize + baseAlignment);
    const vk::DeviceAddress sbtBase = (next->sbt.address + baseAlignment - 1) & ~vk::DeviceAddress(baseAlignment - 1);
    std::memcpy(static_cast<uint8_t*>(mapped) + (sbtBase - next->sbt.address), sbtBytes.data(), sbtBytes.size());
    next->sbt.memory.unmapMemory();
    next->raygenRegion.deviceAddress = sbtBase;
    next->raygenRegion.stride = stride;
    next->raygenRegion.size = stride;
    next->missRegion.deviceAddress = sbtBase + baseAlignment;
    next->missRegion.stride = stride;
    next->missRegion.size = stride;
    next->hitRegion.deviceAddress = sbtBase + 2 * baseAlignment;
    next->hitRegion.stride = stride;
    next->hitRegion.size = stride;
    next->triangleCount = triangleCount;

    scene = std::move(next);
    const bool keepFirstPerson = camera.firstPerson;
    camera = CameraState{};
    camera.firstPerson = keepFirstPerson;
    // Look from -Z toward the origin: the DamagedHelmet node rotation leaves its
    // front facing that way. Keep the first-person spawn on the same side.
    camera.yaw = 3.14159265f;
    camera.position = {0.0f, 0.0f, -camera.distance};
    sampleIndex = 1;
    const size_t baseColorTextureCount = loaded.textures.size() - 1;
    status = "Loaded " + std::to_string(scene->triangleCount) + " triangles | " +
        std::to_string(baseColorTextureCount) +
        (baseColorTextureCount == 1 ? " base-color texture" : " base-color textures");
}

void VulkanViewer::Impl::recordTrace(vk::raii::CommandBuffer& commandBuffer)
{
    if (!scene)
        return;

    vk::ImageMemoryBarrier prepare;
    prepare.srcAccessMask = outputInitialized ? vk::AccessFlagBits::eShaderRead : vk::AccessFlags{};
    prepare.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
    prepare.oldLayout = outputInitialized ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined;
    prepare.newLayout = vk::ImageLayout::eGeneral;
    prepare.srcQueueFamilyIndex = prepare.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prepare.image = *output.handle;
    prepare.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    commandBuffer.pipelineBarrier(outputInitialized ? vk::PipelineStageFlagBits::eFragmentShader
                                                    : vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, prepare);

    // Make the previous frame's accumulation writes visible to this dispatch.
    const vk::ImageMemoryBarrier accumulate(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *samplesAccumulated.handle, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                                  vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, accumulate);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *rtPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *rtPipelineLayout, 0, *scene->descriptorSet,
                                     {});
    const std::array<CameraData, 1> cameraData = {makeCamera(camera, sampleIndex)};
    const vk::ArrayProxy<const CameraData> cameraProxy(1, cameraData.data());
    commandBuffer.pushConstants(*rtPipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, cameraProxy);
    commandBuffer.traceRaysKHR(scene->raygenRegion, scene->missRegion, scene->hitRegion, scene->callableRegion,
                               kRenderWidth, kRenderHeight, 1);
    const vk::ImageMemoryBarrier sampled({}, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eGeneral,
                                         vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                         *output.handle,
                                         vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                                  vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, sampled);
    outputInitialized = true;
}

void VulkanViewer::Impl::drawFrame(ImDrawData* drawData)
{
    int framebufferWidth = 0, framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth == 0 || framebufferHeight == 0)
        return;
    if (swapchainDirty || static_cast<uint32_t>(framebufferWidth) != swapchainExtent.width ||
        static_cast<uint32_t>(framebufferHeight) != swapchainExtent.height)
        recreateSwapchain();

    FrameSync& frame = frames[frameIndex];
    (void)device.waitForFences(*frame.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    uint32_t imageIndex = 0;
    bool suboptimal = false;
    try
    {
        const auto acquired = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable);
        imageIndex = acquired.value;
        suboptimal = acquired.result == vk::Result::eSuboptimalKHR;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        recreateSwapchain();
        return;
    }

    device.resetFences(*frame.fence);
    vk::raii::CommandBuffer& commandBuffer = commandBuffers[frameIndex];
    commandBuffer.reset();
    commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    recordTrace(commandBuffer);

    const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0.035f, 0.04f, 0.05f, 1.0f}));
    const vk::RenderPassBeginInfo renderBegin(*renderPass, *framebuffers[imageIndex],
                                              vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent), clear);
    commandBuffer.beginRenderPass(renderBegin, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(drawData, static_cast<VkCommandBuffer>(*commandBuffer));
    commandBuffer.endRenderPass();
    commandBuffer.end();

    const vk::Semaphore waitSemaphore = *frame.imageAvailable;
    const vk::Semaphore signalSemaphore = *renderFinished[imageIndex];
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    const vk::CommandBuffer rawCommandBuffer = *commandBuffer;
    vk::SubmitInfo submitInfo;
    submitInfo.setWaitSemaphores(waitSemaphore)
        .setWaitDstStageMask(waitStage)
        .setCommandBuffers(rawCommandBuffer)
        .setSignalSemaphores(signalSemaphore);
    queue.submit(submitInfo, *frame.fence);

    const vk::SwapchainKHR rawSwapchain = *swapchain;
    vk::PresentInfoKHR presentInfo;
    presentInfo.setWaitSemaphores(signalSemaphore).setSwapchains(rawSwapchain).setImageIndices(imageIndex);
    try
    {
        suboptimal = suboptimal || queue.presentKHR(presentInfo) == vk::Result::eSuboptimalKHR;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        swapchainDirty = true;
    }
    swapchainDirty = swapchainDirty || suboptimal;
    frameIndex = (frameIndex + 1) % kFramesInFlight;
    sampleIndex++;
}

VulkanViewer::VulkanViewer(GLFWwindow* window) : impl_(std::make_unique<Impl>(window)) {}
VulkanViewer::~VulkanViewer() = default;

void VulkanViewer::loadScene(GltfScene scene)
{
    impl_->buildScene(std::move(scene));
}
void VulkanViewer::rotateView(float deltaX, float deltaY)
{
    // Shared by both modes: drag right turns right, drag up looks up.
    // First-person uses a lower sensitivity because the camera rotates in place.
    CameraState& camera = impl_->camera;
    const float sensitivity = camera.firstPerson ? 0.0025f : 0.008f;
    camera.yaw += deltaX * sensitivity;
    camera.pitch = std::clamp(camera.pitch - deltaY * sensitivity, -1.5f, 1.5f);
    impl_->sampleIndex = 1;
}
void VulkanViewer::zoom(float wheelDelta)
{
    CameraState& camera = impl_->camera;
    if (camera.firstPerson)
        impl_->moveSpeed = std::clamp(impl_->moveSpeed * std::exp(wheelDelta * 0.15f), 0.2f, 20.0f);
    else
        camera.distance = std::clamp(camera.distance * std::exp(-wheelDelta * 0.12f), 1.4f, 12.0f);
    impl_->sampleIndex = 1;
}
void VulkanViewer::move(float forwardAxis, float rightAxis, float upAxis, float deltaSeconds, bool boost)
{
    CameraState& camera = impl_->camera;
    if (!camera.firstPerson || !impl_->scene)
        return;
    if (forwardAxis == 0.0f && rightAxis == 0.0f && upAxis == 0.0f)
        return;

    const float cp = std::cos(camera.pitch);
    const float sp = std::sin(camera.pitch);
    const float sy = std::sin(camera.yaw);
    const float cy = std::cos(camera.yaw);

    // Full look direction for forward/back, horizontal right for strafing,
    // and world +Y for Q/E so vertical movement stays intuitive.
    const float forwardX = sy * cp;
    const float forwardY = sp;
    const float forwardZ = -cy * cp;
    const float rightX = cy;
    const float rightZ = sy;

    // Clamp the combined input length so diagonal movement is not faster.
    float length = std::sqrt(forwardAxis * forwardAxis + rightAxis * rightAxis + upAxis * upAxis);
    if (length > 1.0f)
    {
        forwardAxis /= length;
        rightAxis /= length;
        upAxis /= length;
    }

    const float step = impl_->moveSpeed * (boost ? 4.0f : 1.0f) * deltaSeconds;
    camera.position[0] += (forwardX * forwardAxis + rightX * rightAxis) * step;
    camera.position[1] += (forwardY * forwardAxis + upAxis) * step;
    camera.position[2] += (forwardZ * forwardAxis + rightZ * rightAxis) * step;
    impl_->sampleIndex = 1;
}
void VulkanViewer::setFirstPerson(bool enabled)
{
    CameraState& camera = impl_->camera;
    if (camera.firstPerson == enabled)
        return;

    if (enabled)
    {
        // Start where the orbit camera is, looking in the same direction.
        const float cp = std::cos(camera.pitch);
        const float sp = std::sin(camera.pitch);
        camera.position = {camera.distance * cp * std::sin(camera.yaw), camera.distance * sp,
                           camera.distance * cp * std::cos(camera.yaw)};
    }
    else
    {
        const float length =
            std::sqrt(camera.position[0] * camera.position[0] + camera.position[1] * camera.position[1] +
                      camera.position[2] * camera.position[2]);
        camera.distance = std::clamp(length, 0.5f, 50.0f);
    }

    // The two modes use opposite yaw/pitch conventions; negating both keeps the
    // current view direction when switching.
    camera.yaw = -camera.yaw;
    camera.pitch = -camera.pitch;
    camera.firstPerson = enabled;
    impl_->sampleIndex = 1;
}
void VulkanViewer::drawFrame(ImDrawData* drawData)
{
    impl_->drawFrame(drawData);
}
bool VulkanViewer::hasScene() const
{
    return static_cast<bool>(impl_->scene);
}
bool VulkanViewer::isFirstPerson() const
{
    return impl_->camera.firstPerson;
}
float VulkanViewer::moveSpeed() const
{
    return impl_->moveSpeed;
}
ImTextureID VulkanViewer::sceneTexture() const
{
    return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(impl_->imguiOutputSet));
}
const std::string& VulkanViewer::sceneStatus() const
{
    return impl_->status;
}
