/*
 * Vulkan as each platform has it, for the display's device and the renderer's.
 *
 * Windows, Linux and Android have a Vulkan driver behind the system's loader,
 * found by volkInitialize. Apple's systems have Metal, and Vulkan on top of it
 * through MoltenVK, carried inside the app (iOS/iPadOS: Frameworks/
 * MoltenVK.framework) - a "portability" implementation, which an instance
 * lists only when asked to (VK_KHR_portability_enumeration) and whose devices
 * must be created naming VK_KHR_portability_subset.
 *
 * The renderer needs Vulkan 1.3's dynamic rendering. A Vulkan 1.2 device that
 * has it as VK_KHR_dynamic_rendering (MoltenVK) does too: its KHR entry points
 * stand in for the core ones.
 */

#include "gpu_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace ps2gpu {

namespace {

bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const VkExtensionProperties& e : list)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

void AddOnce(std::vector<const char*>* list, const char* name) {
    for (const char* n : *list)
        if (std::strcmp(n, name) == 0) return;
    list->push_back(name);
}

}  // namespace

const char* MoltenVkPath() {
#ifdef __APPLE__
    // Where an app carries it (embedded, Frameworks/), then a Mac's.
    static const char* const kPaths[] = {
        "@rpath/MoltenVK.framework/MoltenVK",
        "@executable_path/Frameworks/MoltenVK.framework/MoltenVK",
        "@executable_path/../Frameworks/MoltenVK.framework/MoltenVK",
        "@rpath/libMoltenVK.dylib",
        "@executable_path/Frameworks/libMoltenVK.dylib",
        "libMoltenVK.dylib",
        "libvulkan.1.dylib",
    };
    static const char* found = [] {
        for (const char* path : kPaths)
            if (void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
                if (dlsym(handle, "vkGetInstanceProcAddr")) return path;
                dlclose(handle);
            }
        return static_cast<const char*>(nullptr);
    }();
    return found;
#else
    return nullptr;
#endif
}

VkResult LoadVulkan() {
    static VkResult loaded = [] {
#ifdef __APPLE__
        // dlopen'ed by its own path: the app's bundle is not where volk looks,
        // and the executable's own "vkGetInstanceProcAddr" is volk's pointer.
        if (const char* path = MoltenVkPath()) {
            void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(handle, "vkGetInstanceProcAddr"));
            if (get) {
                volkInitializeCustom(get);
                return VK_SUCCESS;
            }
        }
#endif
        return volkInitialize();
    }();
    return loaded;
}

void AddInstancePortability(std::vector<const char*>* extensions, VkInstanceCreateFlags* flags) {
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());
    if (HasExtension(available, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        AddOnce(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        *flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    if (HasExtension(available, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        AddOnce(extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
}

bool DeviceRequirementsFor(VkPhysicalDevice gpu, bool swapchain, DeviceRequirements* out) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, available.data());
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu, &p);
    out->extensions.clear();
    out->chain = nullptr;
    if (swapchain) {
        if (!HasExtension(available, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) return false;
        out->extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    // A portability implementation's devices name its subset (the spec's rule).
    if (HasExtension(available, "VK_KHR_portability_subset")) out->extensions.push_back("VK_KHR_portability_subset");
    // PS2GPU_VULKAN12=1: as a Vulkan 1.2 device would be (the extension's path, tested anywhere).
    static const bool as_12 = std::getenv("PS2GPU_VULKAN12") != nullptr;
    if (p.apiVersion >= VK_API_VERSION_1_3 && !as_12) {
        out->v13 = VkPhysicalDeviceVulkan13Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        out->v13.dynamicRendering = VK_TRUE;
        out->v13.synchronization2 = VK_TRUE;
        out->chain = &out->v13;
        out->core13 = true;
        return true;
    }
    if (!HasExtension(available, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) return false;
    // VK_KHR_dynamic_rendering, before 1.2: its two dependencies by name too.
    if (p.apiVersion < VK_API_VERSION_1_2) {
        if (HasExtension(available, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME))
            out->extensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
        if (HasExtension(available, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME))
            out->extensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    }
    out->extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    out->dynamic = VkPhysicalDeviceDynamicRenderingFeaturesKHR{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    out->dynamic.dynamicRendering = VK_TRUE;
    out->chain = &out->dynamic;
    out->core13 = false;
    return true;
}

void UseDynamicRenderingExtension() {
    // The core names, where the device has only the extension's (volk loads
    // both through the instance; a 1.2 implementation leaves the core ones null).
    if (!vkCmdBeginRendering || std::getenv("PS2GPU_VULKAN12")) vkCmdBeginRendering = vkCmdBeginRenderingKHR;
    if (!vkCmdEndRendering || std::getenv("PS2GPU_VULKAN12")) vkCmdEndRendering = vkCmdEndRenderingKHR;
}

}  // namespace ps2gpu
