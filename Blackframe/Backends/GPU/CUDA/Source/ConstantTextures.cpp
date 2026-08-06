#include <Blackframe/Backends/GPU/CUDA/ConstantTextures.hpp>
#include <Blackframe/XPU/CUDA/ConstantTextureKernel.hpp>
#include <Blackframe/XPU/Shared/ConstantTextureAbi.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace shared = xpu::shared;

[[nodiscard]] core::Error texture_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_error(const cudaError_t status, const char* const operation,
                                     const std::size_t byte_count) {
    return texture_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                         "CUDA constant texture " + std::string{operation} + " failed for " +
                             std::to_string(byte_count) + " bytes: " + cudaGetErrorName(status) +
                             " (" + cudaGetErrorString(status) + ").");
}

[[nodiscard]] bool checked_product(const std::size_t left, const std::size_t right,
                                   std::size_t& result) noexcept {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

[[nodiscard]] core::Status validate_scene(const CudaSceneSoA& scene) {
    if (!scene || scene.device_data() == nullptr || scene.device_ordinal() < 0 ||
        scene.size_bytes() < sizeof(shared::SceneSoaHeader) ||
        shared::validate_scene_soa_header(scene.header()) !=
            shared::SceneSoaHeaderValidationStatus::valid ||
        scene.header().total_size_bytes != scene.size_bytes()) {
        return std::unexpected(texture_error(
            core::StatusCode::invalid_argument,
            "CUDA constant texture evaluation requires an open serialized CUDA scene."));
    }
    auto active_device = int{-1};
    const auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "device query", 0U));
    }
    if (active_device != scene.device_ordinal()) {
        return std::unexpected(texture_error(
            core::StatusCode::invalid_argument,
            "CUDA constant texture evaluation requires the serialized scene's device to be "
            "active."));
    }
    return {};
}

[[nodiscard]] core::Status lane_status(const shared::ConstantTextureEvaluationResult& result,
                                       const renderer::TextureId texture_id,
                                       const shared::ConstantTextureKind expected_kind,
                                       const std::size_t lane) {
    const auto suffix = " in texture request lane " + std::to_string(lane) + " (id " +
                        std::to_string(texture_id.value) + ").";
    if (result.reserved[0U] != 0U || result.reserved[1U] != 0U) {
        return std::unexpected(texture_error(
            core::StatusCode::internal_error,
            "CUDA constant texture evaluation returned non-zero reserved storage" + suffix));
    }
    switch (result.status) {
    case shared::ConstantTextureEvaluationStatus::success:
        if (result.kind != expected_kind) {
            return std::unexpected(texture_error(
                core::StatusCode::internal_error,
                "CUDA constant texture evaluation returned a non-canonical success" + suffix));
        }
        return {};
    case shared::ConstantTextureEvaluationStatus::invalid_scene:
        return std::unexpected(texture_error(
            core::StatusCode::incompatible,
            "CUDA constant texture device validation rejected the serialized scene" + suffix));
    case shared::ConstantTextureEvaluationStatus::invalid_request:
        return std::unexpected(
            texture_error(core::StatusCode::invalid_argument,
                          "CUDA constant texture device validation rejected the request" + suffix));
    case shared::ConstantTextureEvaluationStatus::unknown_texture:
        return std::unexpected(
            texture_error(core::StatusCode::not_found,
                          "CUDA constant texture identifier was not found" + suffix));
    case shared::ConstantTextureEvaluationStatus::type_mismatch:
        return std::unexpected(texture_error(
            core::StatusCode::incompatible,
            "CUDA constant texture value kind does not match the typed request" + suffix));
    case shared::ConstantTextureEvaluationStatus::invalid_record:
        return std::unexpected(texture_error(
            core::StatusCode::incompatible,
            "CUDA constant texture device validation rejected a scene texture record" + suffix));
    }
    return std::unexpected(
        texture_error(core::StatusCode::internal_error,
                      "CUDA constant texture evaluation returned an unknown status" + suffix));
}

template <class ResultValue, class Convert>
[[nodiscard]] core::Result<std::vector<ResultValue>>
evaluate(const CudaSceneSoA& scene, const std::span<const renderer::TextureId> texture_ids,
         const shared::ConstantTextureKind kind, const CudaConstantTextureEvaluationOptions options,
         Convert&& convert) try {
    if (auto status = validate_scene(scene); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (texture_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(texture_error(
            core::StatusCode::resource_exhausted,
            "CUDA constant texture request count exceeds its fixed 32-bit launch domain."));
    }
    if (texture_ids.empty()) {
        return std::vector<ResultValue>{};
    }

    auto request_bytes = std::size_t{};
    auto result_bytes = std::size_t{};
    if (!checked_product(texture_ids.size(), sizeof(shared::ConstantTextureEvaluationRequest),
                         request_bytes) ||
        !checked_product(texture_ids.size(), sizeof(shared::ConstantTextureEvaluationResult),
                         result_bytes) ||
        request_bytes > std::numeric_limits<std::size_t>::max() - result_bytes) {
        return std::unexpected(
            texture_error(core::StatusCode::resource_exhausted,
                          "CUDA constant texture evaluation storage size overflowed."));
    }
    const auto total_bytes = request_bytes + result_bytes;
    if (total_bytes > options.device_memory_budget.maximum_bytes ||
        total_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(texture_error(
            core::StatusCode::resource_exhausted,
            "CUDA constant texture evaluation exceeds its explicit device-memory budget."));
    }

    auto requests = std::vector<shared::ConstantTextureEvaluationRequest>{};
    requests.reserve(texture_ids.size());
    for (const auto texture_id : texture_ids) {
        requests.push_back(shared::ConstantTextureEvaluationRequest{
            .texture_id = texture_id.value,
            .expected_kind = kind,
            .reserved = {},
        });
    }
    auto results = std::vector<shared::ConstantTextureEvaluationResult>(texture_ids.size());

    auto device_requests =
        xpu::cuda::DeviceBuffer<shared::ConstantTextureEvaluationRequest>::allocate(
            texture_ids.size(), options.device_memory_budget);
    if (!device_requests) {
        return std::unexpected(std::move(device_requests.error()));
    }
    auto device_results =
        xpu::cuda::DeviceBuffer<shared::ConstantTextureEvaluationResult>::allocate(
            texture_ids.size(), options.device_memory_budget);
    if (!device_results) {
        return std::unexpected(std::move(device_results.error()));
    }

    auto status =
        cudaMemcpy(device_requests->data(), requests.data(), request_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "request upload", request_bytes));
    }
    status = static_cast<cudaError_t>(blackframe_cuda_launch_constant_texture_evaluation(
        scene.device_data(), scene.size_bytes(), device_requests->data(),
        static_cast<std::uint32_t>(texture_ids.size()), device_results->data(), nullptr));
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "kernel launch", total_bytes));
    }
    status =
        cudaMemcpy(results.data(), device_results->data(), result_bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "result download", result_bytes));
    }

    auto result_close = device_results->close();
    auto request_close = device_requests->close();
    if (!result_close) {
        return std::unexpected(std::move(result_close.error()));
    }
    if (!request_close) {
        return std::unexpected(std::move(request_close.error()));
    }

    auto converted = std::vector<ResultValue>{};
    converted.reserve(texture_ids.size());
    for (auto lane = std::size_t{}; lane < results.size(); ++lane) {
        if (auto lane_result = lane_status(results[lane], texture_ids[lane], kind, lane);
            !lane_result) {
            return std::unexpected(std::move(lane_result.error()));
        }
        converted.push_back(convert(results[lane]));
    }
    return converted;
} catch (const std::bad_alloc&) {
    return std::unexpected(
        texture_error(core::StatusCode::resource_exhausted,
                      "CUDA constant texture evaluation exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        texture_error(core::StatusCode::resource_exhausted,
                      "CUDA constant texture evaluation exceeded a host container length limit."));
}

} // namespace

core::Result<std::vector<renderer::TransportScalar>>
evaluate_cuda_constant_float_textures(const CudaSceneSoA& scene,
                                      const std::span<const renderer::TextureId> texture_ids,
                                      const CudaConstantTextureEvaluationOptions options) {
    return evaluate<renderer::TransportScalar>(
        scene, texture_ids, shared::ConstantTextureKind::float_value, options,
        [](const shared::ConstantTextureEvaluationResult& result) { return result.values[0U]; });
}

core::Result<std::vector<renderer::LinearRGB>>
evaluate_cuda_constant_color_textures(const CudaSceneSoA& scene,
                                      const std::span<const renderer::TextureId> texture_ids,
                                      const CudaConstantTextureEvaluationOptions options) {
    return evaluate<renderer::LinearRGB>(scene, texture_ids,
                                         shared::ConstantTextureKind::linear_rgb, options,
                                         [](const shared::ConstantTextureEvaluationResult& result) {
                                             return renderer::LinearRGB{
                                                 .red = result.values[0U],
                                                 .green = result.values[1U],
                                                 .blue = result.values[2U],
                                             };
                                         });
}

core::Result<std::vector<renderer::TransportSpectrum>>
evaluate_cuda_constant_spectrum_textures(const CudaSceneSoA& scene,
                                         const std::span<const renderer::TextureId> texture_ids,
                                         const CudaConstantTextureEvaluationOptions options) {
    return evaluate<renderer::TransportSpectrum>(
        scene, texture_ids, shared::ConstantTextureKind::sampled_spectrum, options,
        [](const shared::ConstantTextureEvaluationResult& result) {
            return renderer::TransportSpectrum{
                .values = {result.values[0U], result.values[1U], result.values[2U],
                           result.values[3U]},
            };
        });
}

} // namespace blackframe::engine
