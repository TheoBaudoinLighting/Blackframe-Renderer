#include "SceneQuery.hpp"

#include <Blackframe/Backends/GPU/CUDA/Occlusion.hpp>
#include <Blackframe/XPU/CUDA/SceneOcclusion.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::shared::SceneOcclusionResult;
using xpu::shared::SceneOcclusionStatus;
using xpu::shared::SceneTraversalStatus;

[[nodiscard]] core::Result<bool>
convert_result(const renderer::Ray&, const SceneOcclusionResult& result, const std::size_t index) {
    if (result.reserved != std::array<std::uint32_t, 3>{}) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA occlusion returned non-zero result padding in ray lane " + std::to_string(index) +
                "."));
    }

    const auto status = static_cast<SceneOcclusionStatus>(result.status);
    if (status == SceneOcclusionStatus::visible) {
        return false;
    }
    if (status == SceneOcclusionStatus::occluded) {
        return true;
    }

    auto error = cuda_scene_query_detail::lane_error(
        static_cast<SceneTraversalStatus>(result.status), index, "occlusion");
    return std::unexpected(std::move(error.error()));
}

} // namespace

core::Result<std::vector<bool>> trace_cuda_occlusions(const CudaSceneSoA& scene,
                                                      const CudaSceneBvh& bvh,
                                                      const std::span<const renderer::Ray> rays,
                                                      const CudaOcclusionQueryOptions options) {
    return cuda_scene_query_detail::execute<SceneOcclusionResult, bool>(
        scene, bvh, rays, options, "occlusion", blackframe_cuda_launch_scene_occlusion,
        convert_result);
}

core::Result<bool> trace_cuda_occlusion(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                                        const renderer::Ray& ray,
                                        const CudaOcclusionQueryOptions options) {
    auto results =
        trace_cuda_occlusions(scene, bvh, std::span<const renderer::Ray>{&ray, 1U}, options);
    if (!results) {
        return std::unexpected(std::move(results.error()));
    }
    if (results->size() != 1U) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA occlusion single-ray query returned an invalid result count."));
    }
    return (*results)[0];
}

} // namespace blackframe::engine
