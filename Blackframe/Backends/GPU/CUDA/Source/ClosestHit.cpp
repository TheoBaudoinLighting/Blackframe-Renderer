#include <Blackframe/Backends/GPU/CUDA/ClosestHit.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::shared::SceneClosestHitResult;
using xpu::shared::SceneClosestHitStatus;
using xpu::shared::TransportRay;

[[nodiscard]] core::Error closest_hit_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_query_error(const cudaError_t status, const char* const operation,
                                           const std::size_t byte_count) {
    return closest_hit_error(
        xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
        std::string{"CUDA closest-hit "} + operation + " failed for " + std::to_string(byte_count) +
            " bytes: " + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ").");
}

[[nodiscard]] bool checked_product(const std::size_t left, const std::size_t right,
                                   std::size_t& result) noexcept {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

[[nodiscard]] core::Status validate_binding(const CudaSceneSoA& scene, const CudaSceneBvh& bvh) {
    if (!scene || scene.device_data() == nullptr || scene.device_ordinal() < 0 ||
        scene.size_bytes() < sizeof(xpu::shared::SceneSoaHeader) ||
        xpu::shared::validate_scene_soa_header(scene.header()) !=
            xpu::shared::SceneSoaHeaderValidationStatus::valid ||
        scene.header().total_size_bytes != scene.size_bytes()) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::invalid_argument,
                              "CUDA closest-hit requires an open serialized CUDA scene."));
    }
    if (!bvh || bvh.device_data() == nullptr || bvh.device_ordinal() < 0 ||
        bvh.size_bytes() < sizeof(xpu::shared::SceneBvhHeader) ||
        xpu::shared::validate_scene_bvh_header(bvh.header()) !=
            xpu::shared::SceneBvhHeaderValidationStatus::valid ||
        bvh.header().total_size_bytes != bvh.size_bytes()) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::invalid_argument,
                              "CUDA closest-hit requires an open serialized CUDA BVH."));
    }
    if (bvh.header().source_scene_hash != scene.header().content_hash) {
        return std::unexpected(closest_hit_error(
            core::StatusCode::incompatible,
            "CUDA closest-hit requires a BVH built from the supplied serialized scene."));
    }
    if (scene.device_ordinal() != bvh.device_ordinal()) {
        return std::unexpected(closest_hit_error(
            core::StatusCode::incompatible,
            "CUDA closest-hit scene and BVH must reside on the same CUDA device."));
    }

    auto active_device = int{-1};
    const auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_query_error(status, "device query", 0U));
    }
    if (active_device != scene.device_ordinal()) {
        return std::unexpected(closest_hit_error(
            core::StatusCode::invalid_argument,
            "CUDA closest-hit requires the serialized scene's device to be active."));
    }
    return {};
}

[[nodiscard]] core::Result<TransportRay> transport_ray(const renderer::Ray& ray,
                                                       const std::size_t index) {
    const auto origin = ray.origin();
    const auto direction = ray.direction();
    if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
        !std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z) ||
        (direction.x == 0.0F && direction.y == 0.0F && direction.z == 0.0F) ||
        !std::isfinite(ray.t_min()) || ray.t_min() < 0.0F || std::isnan(ray.t_max()) ||
        ray.t_max() < ray.t_min() || !std::isfinite(ray.time()) || ray.time() < 0.0F ||
        ray.time() > 1.0F) {
        return std::unexpected(closest_hit_error(core::StatusCode::invalid_argument,
                                                 "CUDA closest-hit rejected invalid ray lane " +
                                                     std::to_string(index) + "."));
    }
    return TransportRay{
        .origin_x = origin.x,
        .origin_y = origin.y,
        .origin_z = origin.z,
        .t_min = ray.t_min(),
        .direction_x = direction.x,
        .direction_y = direction.y,
        .direction_z = direction.z,
        .t_max = ray.t_max(),
        .time = ray.time(),
        .visibility_mask = ray.mask(),
        .current_medium = ray.current_medium().value,
        .reserved = 0U,
    };
}

[[nodiscard]] core::Status lane_error(const SceneClosestHitStatus status, const std::size_t index) {
    const auto suffix = std::string{" in ray lane "} + std::to_string(index) + ".";
    switch (status) {
    case SceneClosestHitStatus::invalid_ray:
        return std::unexpected(
            closest_hit_error(core::StatusCode::invalid_argument,
                              "CUDA closest-hit device validation rejected an input ray" + suffix));
    case SceneClosestHitStatus::invalid_scene:
        return std::unexpected(closest_hit_error(
            core::StatusCode::incompatible,
            "CUDA closest-hit device validation rejected the serialized scene" + suffix));
    case SceneClosestHitStatus::invalid_bvh:
        return std::unexpected(closest_hit_error(
            core::StatusCode::incompatible,
            "CUDA closest-hit device validation rejected the serialized BVH" + suffix));
    case SceneClosestHitStatus::stack_overflow:
        return std::unexpected(closest_hit_error(
            core::StatusCode::resource_exhausted,
            "CUDA closest-hit traversal exhausted its bounded node stack" + suffix));
    case SceneClosestHitStatus::invalid_topology:
        return std::unexpected(closest_hit_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit traversal encountered invalid BVH topology" + suffix));
    case SceneClosestHitStatus::numerical_failure:
        return std::unexpected(closest_hit_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit traversal encountered an unrepresentable numerical result" + suffix));
    case SceneClosestHitStatus::miss:
    case SceneClosestHitStatus::hit:
        break;
    }
    return std::unexpected(closest_hit_error(core::StatusCode::internal_error,
                                             "CUDA closest-hit returned an unknown lane status."));
}

[[nodiscard]] core::Result<std::optional<AccelHit>>
convert_result(const renderer::Ray& ray, const SceneClosestHitResult& result,
               const std::size_t index) {
    if (result.reserved != std::array<std::uint32_t, 3>{}) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::internal_error,
                              "CUDA closest-hit returned non-zero result padding in ray lane " +
                                  std::to_string(index) + "."));
    }
    const auto status = static_cast<SceneClosestHitStatus>(result.status);
    if (status == SceneClosestHitStatus::miss) {
        const auto empty = xpu::shared::ClosestHit{};
        if (std::memcmp(&result.hit, &empty, sizeof(empty)) != 0) {
            return std::unexpected(
                closest_hit_error(core::StatusCode::internal_error,
                                  "CUDA closest-hit returned a non-canonical miss in ray lane " +
                                      std::to_string(index) + "."));
        }
        return std::optional<AccelHit>{};
    }
    if (status != SceneClosestHitStatus::hit) {
        auto error = lane_error(status, index);
        return std::unexpected(std::move(error.error()));
    }

    const auto& hit = result.hit;
    if (hit.reserved != 0U || hit.identifiers.reserved[0] != 0U ||
        hit.identifiers.reserved[1] != 0U || hit.identifiers.reserved[2] != 0U ||
        !std::isfinite(hit.parameter) || !ray.contains_parameter(hit.parameter) ||
        !std::isfinite(hit.barycentric_vertex0) || !std::isfinite(hit.barycentric_vertex1) ||
        !std::isfinite(hit.barycentric_vertex2) || !std::isfinite(hit.geometric_normal_x) ||
        !std::isfinite(hit.geometric_normal_y) || !std::isfinite(hit.geometric_normal_z)) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::internal_error,
                              "CUDA closest-hit returned malformed hit data in ray lane " +
                                  std::to_string(index) + "."));
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
        return std::unexpected(
            closest_hit_error(core::StatusCode::internal_error,
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
                        const CudaClosestHitQueryOptions options) try {
    if (auto status = validate_binding(scene, bvh); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (rays.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::resource_exhausted,
                              "CUDA closest-hit ray count exceeds its 32-bit launch domain."));
    }
    if (rays.empty()) {
        return std::vector<std::optional<AccelHit>>{};
    }

    auto ray_bytes = std::size_t{};
    auto result_bytes = std::size_t{};
    if (!checked_product(rays.size(), sizeof(TransportRay), ray_bytes) ||
        !checked_product(rays.size(), sizeof(SceneClosestHitResult), result_bytes) ||
        ray_bytes > std::numeric_limits<std::size_t>::max() - result_bytes) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::resource_exhausted,
                              "CUDA closest-hit query storage size overflowed."));
    }
    const auto total_bytes = ray_bytes + result_bytes;
    if (total_bytes > options.device_memory_budget.maximum_bytes ||
        total_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(
            closest_hit_error(core::StatusCode::resource_exhausted,
                              "CUDA closest-hit query exceeds its explicit device-memory budget."));
    }

    auto host_rays = std::vector<TransportRay>{};
    host_rays.reserve(rays.size());
    for (auto index = std::size_t{0}; index < rays.size(); ++index) {
        auto converted = transport_ray(rays[index], index);
        if (!converted) {
            return std::unexpected(std::move(converted.error()));
        }
        host_rays.push_back(*converted);
    }
    auto host_results = std::vector<SceneClosestHitResult>(rays.size());

    auto device_rays =
        xpu::cuda::DeviceBuffer<TransportRay>::allocate(rays.size(), options.device_memory_budget);
    if (!device_rays) {
        return std::unexpected(std::move(device_rays.error()));
    }
    auto device_results = xpu::cuda::DeviceBuffer<SceneClosestHitResult>::allocate(
        rays.size(), options.device_memory_budget);
    if (!device_results) {
        return std::unexpected(std::move(device_results.error()));
    }

    auto copy_status =
        cudaMemcpy(device_rays->data(), host_rays.data(), ray_bytes, cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_query_error(copy_status, "ray upload", ray_bytes));
    }
    const auto launch_status = static_cast<cudaError_t>(blackframe_cuda_launch_scene_closest_hit(
        scene.device_data(), scene.size_bytes(), bvh.device_data(), bvh.size_bytes(),
        device_rays->data(), static_cast<std::uint32_t>(rays.size()), device_results->data()));
    if (launch_status != cudaSuccess) {
        return std::unexpected(cuda_query_error(launch_status, "kernel execution", total_bytes));
    }
    copy_status = cudaMemcpy(host_results.data(), device_results->data(), result_bytes,
                             cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_query_error(copy_status, "result download", result_bytes));
    }

    auto result_close = device_results->close();
    auto ray_close = device_rays->close();
    if (!result_close) {
        return std::unexpected(std::move(result_close.error()));
    }
    if (!ray_close) {
        return std::unexpected(std::move(ray_close.error()));
    }

    auto results = std::vector<std::optional<AccelHit>>{};
    results.reserve(rays.size());
    for (auto index = std::size_t{0}; index < rays.size(); ++index) {
        auto converted = convert_result(rays[index], host_results[index], index);
        if (!converted) {
            return std::unexpected(std::move(converted.error()));
        }
        results.push_back(std::move(*converted));
    }
    return results;
} catch (const std::bad_alloc&) {
    return std::unexpected(closest_hit_error(core::StatusCode::resource_exhausted,
                                             "CUDA closest-hit exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(closest_hit_error(core::StatusCode::resource_exhausted,
                                             "CUDA closest-hit exceeded host container limits."));
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
        return std::unexpected(closest_hit_error(
            core::StatusCode::internal_error,
            "CUDA closest-hit single-ray query returned an invalid result count."));
    }
    return std::move(results->front());
}

} // namespace blackframe::engine
