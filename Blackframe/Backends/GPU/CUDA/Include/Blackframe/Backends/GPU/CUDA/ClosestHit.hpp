#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

struct CudaClosestHitQueryOptions final {
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

// Executes synchronous closest-hit queries entirely through the serialized
// CUDA scene and its matching device BVH. The Embree and analytic backends are
// independent validation oracles and are never selected by these functions.
[[nodiscard]] core::Result<std::vector<std::optional<AccelHit>>>
trace_cuda_closest_hits(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                        std::span<const renderer::Ray> rays,
                        CudaClosestHitQueryOptions options = {});

[[nodiscard]] core::Result<std::optional<AccelHit>>
trace_cuda_closest_hit(const CudaSceneSoA& scene, const CudaSceneBvh& bvh, const renderer::Ray& ray,
                       CudaClosestHitQueryOptions options = {});

static_assert(std::is_standard_layout_v<CudaClosestHitQueryOptions>);

} // namespace blackframe::engine
