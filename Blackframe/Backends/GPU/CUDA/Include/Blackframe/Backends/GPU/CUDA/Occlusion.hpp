#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

struct CudaOcclusionQueryOptions final {
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

// Executes opaque any-hit queries entirely through the serialized CUDA scene
// and its matching device BVH. A true lane stops at its first eligible
// crossing and never reconstructs hit data. No CPU backend is selected.
[[nodiscard]] core::Result<std::vector<bool>>
trace_cuda_occlusions(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                      std::span<const renderer::Ray> rays, CudaOcclusionQueryOptions options = {});

[[nodiscard]] core::Result<bool> trace_cuda_occlusion(const CudaSceneSoA& scene,
                                                      const CudaSceneBvh& bvh,
                                                      const renderer::Ray& ray,
                                                      CudaOcclusionQueryOptions options = {});

static_assert(std::is_standard_layout_v<CudaOcclusionQueryOptions>);

} // namespace blackframe::engine
