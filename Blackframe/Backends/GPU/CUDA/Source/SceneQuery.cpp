#include "SceneQuery.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace blackframe::engine::cuda_scene_query_detail {

core::Error query_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

core::Error runtime_error(const cudaError_t status, const std::string_view query_name,
                          const char* const operation, const std::size_t byte_count) {
    return query_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                       "CUDA " + std::string{query_name} + " " + operation + " failed for " +
                           std::to_string(byte_count) + " bytes: " + cudaGetErrorName(status) +
                           " (" + cudaGetErrorString(status) + ").");
}

bool checked_product(const std::size_t left, const std::size_t right,
                     std::size_t& result) noexcept {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

core::Status validate_binding(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                              const std::string_view query_name) {
    const auto prefix = "CUDA " + std::string{query_name};
    if (!scene || scene.device_data() == nullptr || scene.device_ordinal() < 0 ||
        scene.size_bytes() < sizeof(xpu::shared::SceneSoaHeader) ||
        xpu::shared::validate_scene_soa_header(scene.header()) !=
            xpu::shared::SceneSoaHeaderValidationStatus::valid ||
        scene.header().total_size_bytes != scene.size_bytes()) {
        return std::unexpected(query_error(core::StatusCode::invalid_argument,
                                           prefix + " requires an open serialized CUDA scene."));
    }
    if (!bvh || bvh.device_data() == nullptr || bvh.device_ordinal() < 0 ||
        bvh.size_bytes() < sizeof(xpu::shared::SceneBvhHeader) ||
        xpu::shared::validate_scene_bvh_header(bvh.header()) !=
            xpu::shared::SceneBvhHeaderValidationStatus::valid ||
        bvh.header().total_size_bytes != bvh.size_bytes()) {
        return std::unexpected(query_error(core::StatusCode::invalid_argument,
                                           prefix + " requires an open serialized CUDA BVH."));
    }
    if (bvh.header().source_scene_hash != scene.header().content_hash) {
        return std::unexpected(
            query_error(core::StatusCode::incompatible,
                        prefix + " requires a BVH built from the supplied serialized scene."));
    }
    if (scene.device_ordinal() != bvh.device_ordinal()) {
        return std::unexpected(
            query_error(core::StatusCode::incompatible,
                        prefix + " scene and BVH must reside on the same CUDA device."));
    }

    auto active_device = int{-1};
    const auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(runtime_error(status, query_name, "device query", 0U));
    }
    if (active_device != scene.device_ordinal()) {
        return std::unexpected(
            query_error(core::StatusCode::invalid_argument,
                        prefix + " requires the serialized scene's device to be active."));
    }
    return {};
}

core::Result<xpu::shared::TransportRay> transport_ray(const renderer::Ray& ray,
                                                      const std::size_t index,
                                                      const std::string_view query_name) {
    const auto origin = ray.origin();
    const auto direction = ray.direction();
    if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
        !std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z) ||
        (direction.x == 0.0F && direction.y == 0.0F && direction.z == 0.0F) ||
        !std::isfinite(ray.t_min()) || ray.t_min() < 0.0F || std::isnan(ray.t_max()) ||
        ray.t_max() < ray.t_min() || !std::isfinite(ray.time()) || ray.time() < 0.0F ||
        ray.time() > 1.0F) {
        return std::unexpected(
            query_error(core::StatusCode::invalid_argument, "CUDA " + std::string{query_name} +
                                                                " rejected invalid ray lane " +
                                                                std::to_string(index) + "."));
    }
    return xpu::shared::TransportRay{
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

core::Status lane_error(const xpu::shared::SceneTraversalStatus status, const std::size_t index,
                        const std::string_view query_name) {
    const auto prefix = "CUDA " + std::string{query_name};
    const auto suffix = std::string{" in ray lane "} + std::to_string(index) + ".";
    switch (status) {
    case xpu::shared::SceneTraversalStatus::invalid_ray:
        return std::unexpected(
            query_error(core::StatusCode::invalid_argument,
                        prefix + " device validation rejected an input ray" + suffix));
    case xpu::shared::SceneTraversalStatus::invalid_scene:
        return std::unexpected(
            query_error(core::StatusCode::incompatible,
                        prefix + " device validation rejected the serialized scene" + suffix));
    case xpu::shared::SceneTraversalStatus::invalid_bvh:
        return std::unexpected(
            query_error(core::StatusCode::incompatible,
                        prefix + " device validation rejected the serialized BVH" + suffix));
    case xpu::shared::SceneTraversalStatus::stack_overflow:
        return std::unexpected(
            query_error(core::StatusCode::resource_exhausted,
                        prefix + " traversal exhausted its bounded node stack" + suffix));
    case xpu::shared::SceneTraversalStatus::invalid_topology:
        return std::unexpected(
            query_error(core::StatusCode::internal_error,
                        prefix + " traversal encountered invalid BVH topology" + suffix));
    case xpu::shared::SceneTraversalStatus::numerical_failure:
        return std::unexpected(query_error(
            core::StatusCode::internal_error,
            prefix + " traversal encountered an unrepresentable numerical result" + suffix));
    case xpu::shared::SceneTraversalStatus::miss:
    case xpu::shared::SceneTraversalStatus::hit:
        break;
    }
    return std::unexpected(query_error(core::StatusCode::internal_error,
                                       prefix + " returned an unknown lane status."));
}

} // namespace blackframe::engine::cuda_scene_query_detail
