#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine::cuda_scene_query_detail {

[[nodiscard]] core::Error query_error(core::StatusCode code, std::string message);

[[nodiscard]] core::Error runtime_error(cudaError_t status, std::string_view query_name,
                                        const char* operation, std::size_t byte_count);

[[nodiscard]] core::Status validate_binding(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                                            std::string_view query_name);

[[nodiscard]] core::Result<xpu::shared::TransportRay>
transport_ray(const renderer::Ray& ray, std::size_t index, std::string_view query_name);

[[nodiscard]] core::Status lane_error(xpu::shared::SceneTraversalStatus status, std::size_t index,
                                      std::string_view query_name);

[[nodiscard]] bool checked_product(std::size_t left, std::size_t right,
                                   std::size_t& result) noexcept;

template <typename DeviceResult, typename HostResult, typename Options, typename Launcher,
          typename Converter>
[[nodiscard]] core::Result<std::vector<HostResult>>
execute(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
        const std::span<const renderer::Ray> rays, const Options options,
        const std::string_view query_name, Launcher&& launcher, Converter&& converter) try {
    if (auto status = validate_binding(scene, bvh, query_name); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (rays.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(query_error(core::StatusCode::resource_exhausted,
                                           "CUDA " + std::string{query_name} +
                                               " ray count exceeds its 32-bit launch domain."));
    }
    if (rays.empty()) {
        return std::vector<HostResult>{};
    }

    auto ray_bytes = std::size_t{};
    auto result_bytes = std::size_t{};
    if (!checked_product(rays.size(), sizeof(xpu::shared::TransportRay), ray_bytes) ||
        !checked_product(rays.size(), sizeof(DeviceResult), result_bytes) ||
        ray_bytes > std::numeric_limits<std::size_t>::max() - result_bytes) {
        return std::unexpected(
            query_error(core::StatusCode::resource_exhausted,
                        "CUDA " + std::string{query_name} + " query storage size overflowed."));
    }
    const auto total_bytes = ray_bytes + result_bytes;
    if (total_bytes > options.device_memory_budget.maximum_bytes ||
        total_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(
            query_error(core::StatusCode::resource_exhausted,
                        "CUDA " + std::string{query_name} +
                            " query exceeds its explicit device-memory budget."));
    }

    auto host_rays = std::vector<xpu::shared::TransportRay>{};
    host_rays.reserve(rays.size());
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        auto converted = transport_ray(rays[index], index, query_name);
        if (!converted) {
            return std::unexpected(std::move(converted.error()));
        }
        host_rays.push_back(*converted);
    }
    auto host_results = std::vector<DeviceResult>(rays.size());

    auto device_rays = xpu::cuda::DeviceBuffer<xpu::shared::TransportRay>::allocate(
        rays.size(), options.device_memory_budget);
    if (!device_rays) {
        return std::unexpected(std::move(device_rays.error()));
    }
    auto device_results =
        xpu::cuda::DeviceBuffer<DeviceResult>::allocate(rays.size(), options.device_memory_budget);
    if (!device_results) {
        return std::unexpected(std::move(device_results.error()));
    }

    auto copy_status =
        cudaMemcpy(device_rays->data(), host_rays.data(), ray_bytes, cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(runtime_error(copy_status, query_name, "ray upload", ray_bytes));
    }
    const auto launch_status = static_cast<cudaError_t>(
        launcher(scene.device_data(), scene.size_bytes(), bvh.device_data(), bvh.size_bytes(),
                 device_rays->data(), static_cast<std::uint32_t>(rays.size()),
                 device_results->data(), nullptr));
    if (launch_status != cudaSuccess) {
        return std::unexpected(
            runtime_error(launch_status, query_name, "kernel execution", total_bytes));
    }
    copy_status = cudaMemcpy(host_results.data(), device_results->data(), result_bytes,
                             cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        return std::unexpected(
            runtime_error(copy_status, query_name, "result download", result_bytes));
    }

    auto result_close = device_results->close();
    auto ray_close = device_rays->close();
    if (!result_close) {
        return std::unexpected(std::move(result_close.error()));
    }
    if (!ray_close) {
        return std::unexpected(std::move(ray_close.error()));
    }

    auto results = std::vector<HostResult>{};
    results.reserve(rays.size());
    for (auto index = std::size_t{}; index < rays.size(); ++index) {
        auto converted = converter(rays[index], host_results[index], index);
        if (!converted) {
            return std::unexpected(std::move(converted.error()));
        }
        results.push_back(std::move(*converted));
    }
    return results;
} catch (const std::bad_alloc&) {
    return std::unexpected(
        query_error(core::StatusCode::resource_exhausted,
                    "CUDA " + std::string{query_name} + " exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        query_error(core::StatusCode::resource_exhausted,
                    "CUDA " + std::string{query_name} + " exceeded host container limits."));
}

} // namespace blackframe::engine::cuda_scene_query_detail
