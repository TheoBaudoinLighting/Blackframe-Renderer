#include "SceneQuery.hpp"

#include <Blackframe/Backends/GPU/CUDA/ClosestHit.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::shared::SceneClosestHitResult;
using xpu::shared::SceneClosestHitStatus;

[[nodiscard]] core::Result<std::optional<AccelHit>>
convert_result(const renderer::Ray& ray, const SceneClosestHitResult& result,
               const std::size_t index) {
    if (result.reserved != std::array<std::uint32_t, 3>{}) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit returned non-zero result padding in ray lane " +
                std::to_string(index) + "."));
    }
    const auto status = static_cast<SceneClosestHitStatus>(result.status);
    if (status == SceneClosestHitStatus::miss) {
        const auto empty = xpu::shared::ClosestHit{};
        if (std::memcmp(&result.hit, &empty, sizeof(empty)) != 0) {
            return std::unexpected(cuda_scene_query_detail::query_error(
                core::StatusCode::internal_error,
                "CUDA closest-hit returned a non-canonical miss in ray lane " +
                    std::to_string(index) + "."));
        }
        return std::optional<AccelHit>{};
    }
    if (status != SceneClosestHitStatus::hit) {
        auto error = cuda_scene_query_detail::lane_error(status, index, "closest-hit");
        return std::unexpected(std::move(error.error()));
    }

    const auto& hit = result.hit;
    if (hit.reserved != 0U || hit.identifiers.reserved[0] != 0U ||
        hit.identifiers.reserved[1] != 0U || hit.identifiers.reserved[2] != 0U ||
        !std::isfinite(hit.parameter) || !ray.contains_parameter(hit.parameter) ||
        !std::isfinite(hit.barycentric_vertex0) || !std::isfinite(hit.barycentric_vertex1) ||
        !std::isfinite(hit.barycentric_vertex2) || !std::isfinite(hit.geometric_normal_x) ||
        !std::isfinite(hit.geometric_normal_y) || !std::isfinite(hit.geometric_normal_z)) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit returned malformed hit data in ray lane " + std::to_string(index) +
                "."));
    }
    constexpr auto barycentric_tolerance = 8.0F * std::numeric_limits<float>::epsilon();
    constexpr auto normal_tolerance = 32.0F * std::numeric_limits<float>::epsilon();
    const auto barycentric_sum =
        hit.barycentric_vertex0 + hit.barycentric_vertex1 + hit.barycentric_vertex2;
    const auto normal_length_squared = hit.geometric_normal_x * hit.geometric_normal_x +
                                       hit.geometric_normal_y * hit.geometric_normal_y +
                                       hit.geometric_normal_z * hit.geometric_normal_z;
    if (hit.barycentric_vertex0 < -barycentric_tolerance ||
        hit.barycentric_vertex1 < -barycentric_tolerance ||
        hit.barycentric_vertex2 < -barycentric_tolerance ||
        hit.barycentric_vertex0 > 1.0F + barycentric_tolerance ||
        hit.barycentric_vertex1 > 1.0F + barycentric_tolerance ||
        hit.barycentric_vertex2 > 1.0F + barycentric_tolerance || !std::isfinite(barycentric_sum) ||
        std::abs(barycentric_sum - 1.0F) > barycentric_tolerance ||
        !std::isfinite(normal_length_squared) ||
        std::abs(normal_length_squared - 1.0F) > normal_tolerance) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit returned unnormalized surface data in ray lane " +
                std::to_string(index) + "."));
    }

    auto position = ray.at(hit.parameter);
    if (!position) {
        return std::unexpected(std::move(position.error()));
    }
    return std::optional<AccelHit>{AccelHit{
        .object = renderer::ObjectId{.value = hit.identifiers.object},
        .triangle =
            renderer::TriangleHit{
                .parameter = hit.parameter,
                .position = *position,
                .geometric_normal =
                    {
                        .x = hit.geometric_normal_x,
                        .y = hit.geometric_normal_y,
                        .z = hit.geometric_normal_z,
                    },
                .barycentrics =
                    {
                        .vertex0 = hit.barycentric_vertex0,
                        .vertex1 = hit.barycentric_vertex1,
                        .vertex2 = hit.barycentric_vertex2,
                    },
            },
        .identifiers =
            {
                .instance = renderer::InstanceId{.value = hit.identifiers.instance},
                .geometry = renderer::GeometryId{.value = hit.identifiers.geometry},
                .primitive = renderer::PrimitiveId{.value = hit.identifiers.primitive},
                .material = renderer::MaterialId{.value = hit.identifiers.material},
            },
    }};
}

} // namespace

core::Result<std::vector<std::optional<AccelHit>>>
trace_cuda_closest_hits(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                        const std::span<const renderer::Ray> rays,
                        const CudaClosestHitQueryOptions options) {
    return cuda_scene_query_detail::execute<SceneClosestHitResult, std::optional<AccelHit>>(
        scene, bvh, rays, options, "closest-hit", blackframe_cuda_launch_scene_closest_hit,
        convert_result);
}

core::Result<std::optional<AccelHit>>
trace_cuda_closest_hit(const CudaSceneSoA& scene, const CudaSceneBvh& bvh, const renderer::Ray& ray,
                       const CudaClosestHitQueryOptions options) {
    auto results =
        trace_cuda_closest_hits(scene, bvh, std::span<const renderer::Ray>{&ray, 1U}, options);
    if (!results) {
        return std::unexpected(std::move(results.error()));
    }
    if (results->size() != 1U) {
        return std::unexpected(cuda_scene_query_detail::query_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit single-ray query returned an invalid result count."));
    }
    return std::move(results->front());
}

} // namespace blackframe::engine
