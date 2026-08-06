#include "SceneQuery.hpp"

#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <Blackframe/XPU/CUDA/AsyncRuntime.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/CUDA/SceneOcclusion.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueCompaction.hpp>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <new>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::cuda::WavefrontCameraInputDeviceSoa;
using xpu::cuda::WavefrontLaneControl;
using xpu::cuda::WavefrontLanePhase;
using xpu::cuda::WavefrontPendingShadow;
using xpu::cuda::WavefrontPreviousBsdfSample;
using xpu::cuda::WavefrontQueueCompactionResult;
using xpu::cuda::WavefrontQueueCompactionStatus;
using xpu::cuda::WavefrontStageAudit;
using xpu::cuda::WavefrontStageDeviceSoa;
using xpu::cuda::WavefrontStageKind;
using xpu::cuda::WavefrontStageOutcome;
using xpu::cuda::WavefrontStageRoute;
using xpu::cuda::WavefrontStageStatus;
using xpu::cuda::WavefrontTermination;
using xpu::cuda::WavefrontTransportConfig;
using xpu::shared::ClosestHit;
using xpu::shared::PathSlot;
using xpu::shared::QueueHeader;
using xpu::shared::SampleStreamIndex;
using xpu::shared::SceneClosestHitResult;
using xpu::shared::SceneOcclusionResult;
using xpu::shared::TransportPathStateLane;
using xpu::shared::TransportRay;

static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::camera) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::hit) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::miss) == 3U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade) == 4U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shadow) == 5U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::continuation) == 6U);
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::ray) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::hit) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::hit));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::miss) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::miss));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::shade) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::shadow) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shadow));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::continuation) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::continuation));
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::discrete) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::solid_angle) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::area) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::distance) == 3U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::volume) == 4U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::wavelength) == 5U);
static_assert(static_cast<std::uint32_t>(renderer::MisHeuristic::balance) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::MisHeuristic::power) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::uniform) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::power_weighted) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::spatial_tree) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::RussianRouletteMode::disabled) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::RussianRouletteMode::enabled) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::ScatteringLobe::diffuse) == 0x00000001U);
static_assert(static_cast<std::uint32_t>(renderer::PathDeltaFlags::previous_bounce_was_delta) ==
              1U);
static_assert(static_cast<std::uint32_t>(renderer::PathDeltaFlags::any_non_delta_bounces) == 2U);

[[nodiscard]] core::Error transport_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_transport_error(const cudaError_t status,
                                               const std::string_view operation,
                                               const std::size_t byte_count = 0U) {
    auto message = std::string{"CUDA wavefront transport "} + std::string{operation} + " failed";
    if (byte_count != 0U) {
        message += " for " + std::to_string(byte_count) + " bytes";
    }
    message += ": ";
    message += cudaGetErrorName(status);
    message += " (";
    message += cudaGetErrorString(status);
    message += ").";
    return transport_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                           std::move(message));
}

struct CudaWavefrontTransferCounters final {
    std::uint64_t upload_bytes{};
    std::uint64_t download_bytes{};
    std::uint64_t event_dependencies{};
};

struct CudaWavefrontProfilingResources;

struct CudaWavefrontExecutionContext final {
    CudaWavefrontTransferMode mode{CudaWavefrontTransferMode::synchronous};
    xpu::cuda::Stream* compute_stream{};
    xpu::cuda::Event* checkpoint{};
    CudaWavefrontTransferCounters* counters{};
    WavefrontStageAudit* host_stage_audit{};
    WavefrontQueueCompactionResult* host_compaction_result{};
    QueueHeader* host_queue_header{};
    std::uint32_t* host_clear_status{};
    CudaWavefrontProfilingResources* profiling{};
    bool* workspace_poisoned{};
};

[[nodiscard]] constexpr bool valid_transfer_mode(const CudaWavefrontTransferMode mode) noexcept {
    switch (mode) {
    case CudaWavefrontTransferMode::synchronous:
    case CudaWavefrontTransferMode::asynchronous:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool
valid_instrumentation_mode(const CudaWavefrontInstrumentationMode mode) noexcept {
    switch (mode) {
    case CudaWavefrontInstrumentationMode::disabled:
    case CudaWavefrontInstrumentationMode::nsight:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool
uses_asynchronous_transfers(const CudaWavefrontExecutionContext& context) noexcept {
    return context.mode == CudaWavefrontTransferMode::asynchronous;
}

[[nodiscard]] void* opaque_stream(const cudaStream_t stream) noexcept {
    return static_cast<void*>(stream);
}

[[nodiscard]] cudaStream_t native_stream(const xpu::cuda::Stream* const stream) noexcept {
    return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

[[nodiscard]] core::Status add_transfer_bytes(std::uint64_t& destination,
                                              const std::size_t byte_count,
                                              const std::string_view operation) {
    if (byte_count > std::numeric_limits<std::uint64_t>::max() - destination) {
        return std::unexpected(
            transport_error(core::StatusCode::resource_exhausted,
                            "CUDA wavefront transport " + std::string{operation} +
                                " byte counter overflowed its fixed 64-bit report domain."));
    }
    destination += static_cast<std::uint64_t>(byte_count);
    return {};
}

[[nodiscard]] core::Status contextual_runtime_status(core::Status status,
                                                     const std::string_view operation) {
    if (status) {
        return {};
    }
    auto error = std::move(status.error());
    return std::unexpected(transport_error(error.code, "CUDA wavefront transport " +
                                                           std::string{operation} +
                                                           " failed: " + std::move(error.message)));
}

enum class CudaWavefrontNvtxCategory : std::uint32_t {
    transport = 1U,
    transfer = 2U,
    camera = 10U,
    intersection = 11U,
    hit = 12U,
    miss = 13U,
    shade = 14U,
    shadow = 15U,
    continuation = 16U,
};

struct CudaWavefrontProfilingResources final {
    CudaWavefrontProfilingResources(xpu::cuda::TimingEventPair stage,
                                    xpu::cuda::TimingEventPair traversal,
                                    const nvtxDomainHandle_t domain_handle) noexcept
        : stage_timer(std::move(stage)), traversal_timer(std::move(traversal)),
          domain(domain_handle) {}

    CudaWavefrontProfilingResources(const CudaWavefrontProfilingResources&) = delete;
    CudaWavefrontProfilingResources& operator=(const CudaWavefrontProfilingResources&) = delete;
    CudaWavefrontProfilingResources(CudaWavefrontProfilingResources&&) = delete;
    CudaWavefrontProfilingResources& operator=(CudaWavefrontProfilingResources&&) = delete;

    ~CudaWavefrontProfilingResources() noexcept {
        static_cast<void>(close());
    }

    [[nodiscard]] bool valid(const std::int32_t device_ordinal) const noexcept {
        return static_cast<bool>(stage_timer) && static_cast<bool>(traversal_timer) &&
               stage_timer.device_ordinal() == device_ordinal &&
               traversal_timer.device_ordinal() == device_ordinal;
    }

    [[nodiscard]] core::Status close() {
        auto first_error = core::Status{};
        auto status = stage_timer.close();
        if (!status && first_error) {
            first_error = std::unexpected(std::move(status.error()));
        }
        status = traversal_timer.close();
        if (!status && first_error) {
            first_error = std::unexpected(std::move(status.error()));
        }
        if (domain != nullptr) {
            nvtxDomainDestroy(domain);
            domain = nullptr;
        }
        return first_error;
    }

    xpu::cuda::TimingEventPair stage_timer;
    xpu::cuda::TimingEventPair traversal_timer;
    nvtxDomainHandle_t domain{};
};

[[nodiscard]] core::Result<std::unique_ptr<CudaWavefrontProfilingResources>>
create_profiling_resources(const xpu::cuda::Stream* const compute_stream,
                           const xpu::cuda::Stream* const transfer_stream) {
    auto stage_timer = xpu::cuda::TimingEventPair::create();
    if (!stage_timer) {
        return std::unexpected(std::move(stage_timer.error()));
    }
    auto traversal_timer = xpu::cuda::TimingEventPair::create();
    if (!traversal_timer) {
        return std::unexpected(std::move(traversal_timer.error()));
    }

    nvtxInitialize(nullptr);
    const auto domain = nvtxDomainCreateA("blackframe.cuda.wavefront");
    nvtxDomainNameCategoryA(
        domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::transport), "transport");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::transfer),
                            "transfer");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::camera),
                            "camera");
    nvtxDomainNameCategoryA(domain,
                            static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::intersection),
                            "intersection");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::hit),
                            "hit");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::miss),
                            "miss");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::shade),
                            "shade");
    nvtxDomainNameCategoryA(domain, static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::shadow),
                            "shadow");
    nvtxDomainNameCategoryA(domain,
                            static_cast<std::uint32_t>(CudaWavefrontNvtxCategory::continuation),
                            "continuation");

    nvtxNameCudaStreamA(native_stream(compute_stream), "Blackframe CUDA compute");
    if (transfer_stream != nullptr) {
        nvtxNameCudaStreamA(native_stream(transfer_stream), "Blackframe CUDA transfer");
    }
    nvtxNameCudaEventA(static_cast<cudaEvent_t>(stage_timer->begin_native_handle()),
                       "Blackframe stage timer begin");
    nvtxNameCudaEventA(static_cast<cudaEvent_t>(stage_timer->end_native_handle()),
                       "Blackframe stage timer end");
    nvtxNameCudaEventA(static_cast<cudaEvent_t>(traversal_timer->begin_native_handle()),
                       "Blackframe traversal timer begin");
    nvtxNameCudaEventA(static_cast<cudaEvent_t>(traversal_timer->end_native_handle()),
                       "Blackframe traversal timer end");

    return std::make_unique<CudaWavefrontProfilingResources>(std::move(*stage_timer),
                                                             std::move(*traversal_timer), domain);
}

class CudaWavefrontNvtxRange final {
  public:
    CudaWavefrontNvtxRange(CudaWavefrontProfilingResources* const profiling, const char* const name,
                           const CudaWavefrontNvtxCategory category, const std::uint32_t color,
                           const std::uint64_t payload) noexcept
        : profiling_(profiling) {
        if (profiling_ == nullptr) {
            return;
        }
        auto attributes = nvtxEventAttributes_t{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.category = static_cast<std::uint32_t>(category);
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = color;
        attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        attributes.payload.ullValue = payload;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        static_cast<void>(nvtxDomainRangePushEx(profiling_->domain, &attributes));
    }

    CudaWavefrontNvtxRange(const CudaWavefrontNvtxRange&) = delete;
    CudaWavefrontNvtxRange& operator=(const CudaWavefrontNvtxRange&) = delete;

    ~CudaWavefrontNvtxRange() noexcept {
        if (profiling_ != nullptr) {
            static_cast<void>(nvtxDomainRangePop(profiling_->domain));
        }
    }

  private:
    CudaWavefrontProfilingResources* profiling_{};
};

[[nodiscard]] core::Status add_stage_metric(CudaWavefrontStageMetric& metric,
                                            const std::uint64_t kernel_lanes,
                                            const std::uint64_t elapsed_nanoseconds,
                                            const std::string_view stage_name) {
    if (metric.kernel_dispatches == std::numeric_limits<std::uint64_t>::max() ||
        kernel_lanes > std::numeric_limits<std::uint64_t>::max() - metric.kernel_lanes ||
        elapsed_nanoseconds >
            std::numeric_limits<std::uint64_t>::max() - metric.gpu_elapsed_nanoseconds) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront " + std::string{stage_name} +
                " profiling counters overflowed their fixed 64-bit report domain."));
    }
    ++metric.kernel_dispatches;
    metric.kernel_lanes += kernel_lanes;
    metric.gpu_elapsed_nanoseconds += elapsed_nanoseconds;
    return {};
}

void poison_profiling_workspace(const CudaWavefrontExecutionContext& context) noexcept {
    if (context.workspace_poisoned != nullptr) {
        *context.workspace_poisoned = true;
    }
}

[[nodiscard]] core::Status begin_timed_kernel(xpu::cuda::TimingEventPair& timer,
                                              const CudaWavefrontExecutionContext& context,
                                              const std::string_view stage_name) {
    auto status = contextual_runtime_status(timer.begin(context.compute_stream),
                                            std::string{stage_name} + " timing begin");
    if (!status) {
        poison_profiling_workspace(context);
    }
    return status;
}

[[nodiscard]] core::Status end_timed_kernel(xpu::cuda::TimingEventPair& timer,
                                            const CudaWavefrontExecutionContext& context,
                                            const std::string_view stage_name) {
    auto status = contextual_runtime_status(timer.end(context.compute_stream),
                                            std::string{stage_name} + " timing end");
    if (!status) {
        poison_profiling_workspace(context);
    }
    return status;
}

[[nodiscard]] core::Status collect_timed_kernel(xpu::cuda::TimingEventPair& timer,
                                                CudaWavefrontStageMetric& metric,
                                                const std::uint64_t kernel_lanes,
                                                const CudaWavefrontExecutionContext& context,
                                                const std::string_view stage_name) {
    auto elapsed = timer.elapsed_nanoseconds();
    if (!elapsed) {
        poison_profiling_workspace(context);
        auto error = std::move(elapsed.error());
        return std::unexpected(
            transport_error(error.code, "CUDA wavefront transport " + std::string{stage_name} +
                                            " timing query failed: " + std::move(error.message)));
    }
    auto status = add_stage_metric(metric, kernel_lanes, *elapsed, stage_name);
    if (!status) {
        poison_profiling_workspace(context);
    }
    return status;
}

[[nodiscard]] core::Status initialize_device_bytes(void* const destination, const int value,
                                                   const std::size_t byte_count,
                                                   const CudaWavefrontExecutionContext& context,
                                                   const std::string_view operation) {
    const auto status =
        uses_asynchronous_transfers(context)
            ? cudaMemsetAsync(destination, value, byte_count, native_stream(context.compute_stream))
            : cudaMemset(destination, value, byte_count);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(status, operation, byte_count));
    }
    return {};
}

[[nodiscard]] core::Status
copy_host_to_device(void* const destination, const void* const source, const std::size_t byte_count,
                    const cudaStream_t stream, const CudaWavefrontTransferMode mode,
                    CudaWavefrontTransferCounters& counters, const std::string_view operation) {
    if (mode == CudaWavefrontTransferMode::asynchronous) {
        if (auto status = add_transfer_bytes(counters.upload_bytes, byte_count, operation);
            !status) {
            return status;
        }
        const auto cuda_status =
            cudaMemcpyAsync(destination, source, byte_count, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
            return std::unexpected(cuda_transport_error(cuda_status, operation, byte_count));
        }
        return {};
    }
    const auto cuda_status = cudaMemcpy(destination, source, byte_count, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, operation, byte_count));
    }
    return {};
}

[[nodiscard]] core::Status copy_device_to_host_and_wait(
    void* const destination, const void* const source, const std::size_t byte_count,
    const CudaWavefrontExecutionContext& context, const std::string_view operation) {
    if (!uses_asynchronous_transfers(context)) {
        const auto cuda_status =
            cudaMemcpy(destination, source, byte_count, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            return std::unexpected(cuda_transport_error(cuda_status, operation, byte_count));
        }
        return {};
    }
    if (context.checkpoint == nullptr || context.counters == nullptr) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront asynchronous transfer context is incomplete."));
    }
    if (context.compute_stream == nullptr) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront asynchronous compute stream is unavailable."));
    }
    if (auto status = add_transfer_bytes(context.counters->download_bytes, byte_count, operation);
        !status) {
        return status;
    }
    auto cuda_status = cudaMemcpyAsync(destination, source, byte_count, cudaMemcpyDeviceToHost,
                                       native_stream(context.compute_stream));
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, operation, byte_count));
    }
    if (auto status = contextual_runtime_status(context.checkpoint->record(*context.compute_stream),
                                                operation);
        !status) {
        return status;
    }
    return contextual_runtime_status(context.checkpoint->synchronize(), operation);
}

[[nodiscard]] constexpr std::uint32_t
queue_index(const renderer::WavefrontQueueKind kind) noexcept {
    return static_cast<std::uint32_t>(kind);
}

[[nodiscard]] core::Status
validate_options_and_inputs(const std::span<const CudaWavefrontPathInput> inputs,
                            const CudaWavefrontTransportOptions options) {
    if (!valid_transfer_mode(options.transfer_mode)) {
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront transport requires an explicit synchronous or asynchronous transfer "
            "mode."));
    }
    if (!valid_instrumentation_mode(options.instrumentation_mode)) {
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront transport requires instrumentation to be explicitly disabled or set "
            "to Nsight."));
    }
    if (inputs.empty()) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport requires at least one explicit path input."));
    }
    if (inputs.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport path count exceeds its fixed 32-bit device domain."));
    }
    switch (options.heuristic) {
    case renderer::MisHeuristic::balance:
    case renderer::MisHeuristic::power:
        break;
    default:
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront transport requires the balance or power MIS heuristic."));
    }
    if (auto status = renderer::validate_russian_roulette_policy(options.roulette_policy);
        !status) {
        return std::unexpected(transport_error(
            status.error().code, "CUDA wavefront transport rejected its Russian roulette policy: " +
                                     status.error().message));
    }
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        if (input.primary_ray.current_medium() != input.initial_state.current_medium()) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport path and ray medium identities disagree at path " +
                    std::to_string(index) + "."));
        }
        if (input.initial_state.current_medium() != renderer::VacuumMedium) {
            return std::unexpected(transport_error(
                core::StatusCode::invalid_argument,
                "CUDA wavefront transport currently requires vacuum rays and path states; path " +
                    std::to_string(index) + " selected another medium."));
        }
        if (input.initial_state.depth() != 0U) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport must begin at primary path depth zero; path " +
                    std::to_string(index) + " supplied a resumed state."));
        }
        if (auto status = renderer::validate_path_depth_state(options.depth_limits,
                                                              input.initial_state.depth_counters(),
                                                              input.initial_state.depth());
            !status) {
            return std::unexpected(transport_error(
                status.error().code, "CUDA wavefront transport rejected depth state at path " +
                                         std::to_string(index) + ": " + status.error().message));
        }
        const auto finite_nonnegative = [](const renderer::TransportSpectrum& spectrum) {
            return std::ranges::all_of(spectrum.values, [](const auto value) {
                return std::isfinite(value) && value >= 0.0F;
            });
        };
        if (!finite_nonnegative(input.initial_state.beta()) ||
            !finite_nonnegative(input.initial_state.accumulated_radiance())) {
            return std::unexpected(transport_error(
                core::StatusCode::invalid_argument,
                "CUDA wavefront transport requires finite non-negative path spectra at path " +
                    std::to_string(index) + "."));
        }
    }
    return {};
}

struct ValidatedLightSampler final {
    std::uint32_t light_count{};
    bool present{};
    renderer::LightSamplingStrategy strategy{renderer::LightSamplingStrategy::uniform};
};

[[nodiscard]] core::Result<ValidatedLightSampler>
validate_light_sampler(const xpu::shared::SceneSoaHeader& scene,
                       const CudaWavefrontLightSampler light_sampler) {
    if (scene.punctual_light_count >
        std::numeric_limits<std::uint64_t>::max() - scene.mesh_area_light_count) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront light-registry size overflowed its serialized domain."));
    }
    const auto expected_count = scene.punctual_light_count + scene.mesh_area_light_count;
    if (expected_count == 0U) {
        if (light_sampler.has_value()) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport requires the light sampler to be explicitly absent "
                "when the serialized scene light registry is empty."));
        }
        return ValidatedLightSampler{};
    }
    if (expected_count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront light registry exceeds its fixed 32-bit device domain."));
    }
    const auto count = static_cast<std::uint32_t>(expected_count);
    if (!light_sampler.has_value()) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport requires an explicit light sampler for a non-empty "
            "serialized scene light registry."));
    }
    const auto& sampler = light_sampler->get();
    if (sampler.strategy() != renderer::LightSamplingStrategy::uniform) {
        return std::unexpected(transport_error(
            core::StatusCode::unavailable,
            "CUDA wavefront transport currently requires an explicit uniform light sampler; "
            "the requested strategy is not ported and will not be substituted."));
    }
    if (sampler.light_count() != count) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront light sampler count does not match the serialized punctual and "
            "mesh-area registry."));
    }
    return ValidatedLightSampler{
        .light_count = count,
        .present = true,
        .strategy = sampler.strategy(),
    };
}

[[nodiscard]] WavefrontTransportConfig
transport_config(const CudaWavefrontTransportOptions& options, const std::uint32_t light_count,
                 const renderer::LightSamplingStrategy strategy) noexcept {
    return WavefrontTransportConfig{
        .abi_major = xpu::cuda::WavefrontTransportConfigAbiMajor,
        .abi_minor = xpu::cuda::WavefrontTransportConfigAbiMinor,
        .struct_size = sizeof(WavefrontTransportConfig),
        .mis_heuristic = static_cast<std::uint32_t>(options.heuristic),
        .light_sampling_strategy = static_cast<std::uint32_t>(strategy),
        .light_count = light_count,
        .diffuse_depth_limit = options.depth_limits.diffuse,
        .glossy_depth_limit = options.depth_limits.glossy,
        .specular_depth_limit = options.depth_limits.specular,
        .transmission_depth_limit = options.depth_limits.transmission,
        .volume_depth_limit = options.depth_limits.volume,
        .russian_roulette_mode = static_cast<std::uint32_t>(options.roulette_policy.mode()),
        .russian_roulette_first_depth = options.roulette_policy.first_eligible_depth(),
        .russian_roulette_minimum_probability =
            options.roulette_policy.minimum_survival_probability(),
        .russian_roulette_maximum_probability =
            options.roulette_policy.maximum_survival_probability(),
        .reserved = {0U, 0U},
    };
}

[[nodiscard]] TransportPathStateLane transport_path_state(const renderer::PathState& state) {
    auto result = TransportPathStateLane{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        result.beta.values[lane] = state.beta()[lane];
        result.accumulated_radiance.values[lane] = state.accumulated_radiance()[lane];
        result.wavelength_nanometers.values[lane] = state.wavelengths()[lane].nanometers;
        result.wavelength_pdf_values.values[lane] = state.wavelengths()[lane].probability.value;
        result.wavelength_pdf_measures[lane] =
            static_cast<std::uint8_t>(state.wavelengths()[lane].probability.measure);
    }
    const auto& counters = state.depth_counters();
    result.diffuse_depth = counters.diffuse;
    result.glossy_depth = counters.glossy;
    result.specular_depth = counters.specular;
    result.transmission_depth = counters.transmission;
    result.volume_depth = counters.volume;
    result.depth = state.depth();
    result.eta_scale = state.eta_scale();
    result.current_medium = state.current_medium().value;
    result.delta_flags = static_cast<std::uint32_t>(state.delta_flags());
    return result;
}

[[nodiscard]] SampleStreamIndex
transport_sample_stream_index(const renderer::SampleStreamIndex index) noexcept {
    return SampleStreamIndex{
        .pixel_x = index.pixel_x,
        .pixel_y = index.pixel_y,
        .sample_index = index.sample_index,
        .seed = index.seed,
    };
}

[[nodiscard]] core::Result<renderer::PathState>
renderer_path_state(const TransportPathStateLane& state, const std::size_t path_index) {
    if (std::ranges::any_of(state.reserved, [](const auto value) { return value != 0U; })) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path state returned non-zero reserved data at path " +
                                std::to_string(path_index) + "."));
    }
    auto beta = renderer::TransportSpectrum{};
    auto radiance = renderer::TransportSpectrum{};
    auto wavelengths = renderer::SampledWavelengths{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        beta[lane] = state.beta.values[lane];
        radiance[lane] = state.accumulated_radiance.values[lane];
        const auto measure =
            static_cast<renderer::ProbabilityMeasure>(state.wavelength_pdf_measures[lane]);
        if (measure != renderer::ProbabilityMeasure::wavelength) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront path returned a non-wavelength spectral PDF measure."));
        }
        wavelengths[lane] = renderer::WavelengthSample{
            .nanometers = state.wavelength_nanometers.values[lane],
            .probability =
                renderer::ProbabilityDensity{
                    .value = state.wavelength_pdf_values.values[lane],
                    .measure = measure,
                },
        };
    }
    const auto delta_bits = state.delta_flags;
    if ((delta_bits & ~std::uint32_t{0x3U}) != 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path returned unsupported delta-history bits."));
    }
    auto converted = renderer::PathState::create(beta, radiance,
                                                 renderer::PathDepthCounters{
                                                     .diffuse = state.diffuse_depth,
                                                     .glossy = state.glossy_depth,
                                                     .specular = state.specular_depth,
                                                     .transmission = state.transmission_depth,
                                                     .volume = state.volume_depth,
                                                 },
                                                 state.eta_scale, wavelengths,
                                                 static_cast<renderer::PathDeltaFlags>(delta_bits),
                                                 renderer::MediumId{.value = state.current_medium});
    if (!converted) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path state " + std::to_string(path_index) +
                                " violated the renderer contract: " + converted.error().message));
    }
    if (converted->depth() != state.depth) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path returned an inconsistent total depth."));
    }
    return converted;
}

[[nodiscard]] core::Result<renderer::Ray> renderer_ray(const TransportRay& ray,
                                                       const std::size_t path_index) {
    if (ray.reserved != 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront terminal ray returned non-zero reserved data at path " +
                                std::to_string(path_index) + "."));
    }
    auto converted = renderer::Ray::create(
        renderer::Point3{.x = ray.origin_x, .y = ray.origin_y, .z = ray.origin_z},
        renderer::Vector3{
            .x = ray.direction_x,
            .y = ray.direction_y,
            .z = ray.direction_z,
        },
        ray.t_min, ray.t_max, ray.time, ray.visibility_mask,
        renderer::MediumId{.value = ray.current_medium});
    if (!converted) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront terminal ray " + std::to_string(path_index) +
                                " violated the renderer contract: " + converted.error().message));
    }
    return converted;
}

struct RendererTermination final {
    CudaWavefrontPathTermination reason{};
    renderer::ScatteringLobe blocked_depth_limits{renderer::ScatteringLobe::none};
};

[[nodiscard]] core::Result<RendererTermination>
renderer_termination(const WavefrontLaneControl& control, const std::size_t path_index) {
    if (control.phase != static_cast<std::uint32_t>(WavefrontLanePhase::terminated) ||
        control.flags != 0U) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront path " + std::to_string(path_index) +
                                                   " did not finish in canonical terminal state."));
    }
    constexpr auto depth_category_mask =
        renderer::ScatteringLobe::diffuse | renderer::ScatteringLobe::glossy |
        renderer::ScatteringLobe::specular | renderer::ScatteringLobe::transmission |
        renderer::ScatteringLobe::volume;
    const auto blocked = static_cast<renderer::ScatteringLobe>(control.blocked_depth_limits);
    if (!renderer::is_known_scattering_lobe_mask(blocked) ||
        (blocked & depth_category_mask) != blocked) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront path " + std::to_string(path_index) +
                                                   " returned invalid blocked depth-limit bits."));
    }
    const auto termination = static_cast<WavefrontTermination>(control.termination);
    if ((termination == WavefrontTermination::depth_limit &&
         blocked == renderer::ScatteringLobe::none) ||
        (termination != WavefrontTermination::depth_limit &&
         blocked != renderer::ScatteringLobe::none)) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path " + std::to_string(path_index) +
                                " returned blocked depth-limit bits incompatible with its "
                                "termination."));
    }
    switch (termination) {
    case WavefrontTermination::escaped_environment:
        return RendererTermination{.reason = CudaWavefrontPathTermination::escaped_environment};
    case WavefrontTermination::depth_limit:
        return RendererTermination{
            .reason = CudaWavefrontPathTermination::depth_limit,
            .blocked_depth_limits = blocked,
        };
    case WavefrontTermination::zero_throughput:
        return RendererTermination{.reason = CudaWavefrontPathTermination::zero_throughput};
    case WavefrontTermination::outside_bsdf_support:
        return RendererTermination{.reason = CudaWavefrontPathTermination::outside_bsdf_support};
    case WavefrontTermination::russian_roulette:
        return RendererTermination{.reason = CudaWavefrontPathTermination::russian_roulette};
    case WavefrontTermination::none:
        break;
    }
    return std::unexpected(
        transport_error(core::StatusCode::internal_error,
                        "CUDA wavefront path " + std::to_string(path_index) +
                            " returned an unknown or missing termination reason."));
}

[[nodiscard]] constexpr bool known_stage_route(const WavefrontStageRoute route) noexcept {
    switch (route) {
    case WavefrontStageRoute::none:
    case WavefrontStageRoute::ray:
    case WavefrontStageRoute::hit:
    case WavefrontStageRoute::miss:
    case WavefrontStageRoute::shade:
    case WavefrontStageRoute::shadow:
    case WavefrontStageRoute::continuation:
    case WavefrontStageRoute::terminated:
        return true;
    }
    return false;
}

using StageRouteMask = std::uint32_t;

[[nodiscard]] constexpr StageRouteMask route_mask(const WavefrontStageRoute route) noexcept {
    return StageRouteMask{1U} << static_cast<std::uint32_t>(route);
}

[[nodiscard]] core::Status stage_failure(const WavefrontStageOutcome& outcome,
                                         const std::string_view stage_name,
                                         const std::size_t work_index,
                                         const std::uint32_t path_capacity,
                                         const StageRouteMask allowed_routes) {
    const auto status = static_cast<WavefrontStageStatus>(outcome.status);
    if (status == WavefrontStageStatus::success) {
        const auto route = static_cast<WavefrontStageRoute>(outcome.route);
        if (!known_stage_route(route) || (allowed_routes & route_mask(route)) == 0U ||
            outcome.path_slot >= path_capacity) {
            return std::unexpected(
                transport_error(core::StatusCode::internal_error,
                                "CUDA wavefront " + std::string{stage_name} +
                                    " returned a non-canonical outcome at work item " +
                                    std::to_string(work_index) + "."));
        }
        return {};
    }

    auto code = core::StatusCode::internal_error;
    switch (status) {
    case WavefrontStageStatus::invalid_contract:
    case WavefrontStageStatus::invalid_path_slot:
    case WavefrontStageStatus::invalid_lane_state:
    case WavefrontStageStatus::invalid_ray:
        code = core::StatusCode::incompatible;
        break;
    case WavefrontStageStatus::invalid_scene:
    case WavefrontStageStatus::traversal_error:
        code = core::StatusCode::incompatible;
        break;
    case WavefrontStageStatus::unsupported_transport:
        code = core::StatusCode::unavailable;
        break;
    case WavefrontStageStatus::queue_overflow:
        code = core::StatusCode::resource_exhausted;
        break;
    case WavefrontStageStatus::numerical_failure:
        code = core::StatusCode::internal_error;
        break;
    case WavefrontStageStatus::success:
        break;
    }
    return std::unexpected(transport_error(
        code, "CUDA wavefront " + std::string{stage_name} + " failed at work item " +
                  std::to_string(work_index) + " (path slot " + std::to_string(outcome.path_slot) +
                  ", status " + std::to_string(outcome.status) + ", detail " +
                  std::to_string(outcome.detail) + ")."));
}

template <typename Launcher>
[[nodiscard]] core::Result<WavefrontStageAudit>
execute_stage(const WavefrontStageKind stage_kind, const std::string_view stage_name,
              const std::uint32_t work_count, const StageRouteMask allowed_routes,
              const std::uint32_t path_capacity, WavefrontStageOutcome* const device_outcomes,
              WavefrontStageAudit* const device_audit, const CudaWavefrontExecutionContext& context,
              CudaWavefrontStageMetric* const metric, Launcher&& launcher) {
    if (work_count == 0U) {
        return WavefrontStageAudit{
            .abi_major = xpu::cuda::WavefrontStageAuditAbiMajor,
            .abi_minor = xpu::cuda::WavefrontStageAuditAbiMinor,
            .struct_size = sizeof(WavefrontStageAudit),
            .stage_kind = static_cast<std::uint32_t>(stage_kind),
            .expected_work_count = 0U,
            .inspected_work_count = 0U,
            .first_failure_work_index = std::numeric_limits<std::uint32_t>::max(),
        };
    }
    const auto byte_count = static_cast<std::size_t>(work_count) * sizeof(WavefrontStageOutcome);
    if (auto status = initialize_device_bytes(device_outcomes, 0xFF, byte_count, context,
                                              "outcome initialization");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    auto* const stage_timer = context.profiling != nullptr && metric != nullptr
                                  ? &context.profiling->stage_timer
                                  : nullptr;
    if (stage_timer != nullptr) {
        if (auto status = begin_timed_kernel(*stage_timer, context, stage_name); !status) {
            return std::unexpected(std::move(status.error()));
        }
    }
    auto cuda_status = static_cast<cudaError_t>(launcher());
    if (cuda_status != cudaSuccess) {
        if (stage_timer != nullptr) {
            poison_profiling_workspace(context);
        }
        return std::unexpected(
            cuda_transport_error(cuda_status, std::string{stage_name} + " kernel launch"));
    }
    if (stage_timer != nullptr) {
        if (auto status = end_timed_kernel(*stage_timer, context, stage_name); !status) {
            return std::unexpected(std::move(status.error()));
        }
    }
    cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_wavefront_audit_stage(
        device_outcomes, work_count, allowed_routes, path_capacity,
        static_cast<std::uint32_t>(stage_kind), device_audit,
        opaque_stream(native_stream(context.compute_stream))));
    if (cuda_status != cudaSuccess) {
        if (stage_timer != nullptr) {
            poison_profiling_workspace(context);
        }
        return std::unexpected(cuda_transport_error(
            cuda_status, std::string{stage_name} + " outcome-audit kernel launch"));
    }
    if (context.host_stage_audit == nullptr) {
        if (stage_timer != nullptr) {
            poison_profiling_workspace(context);
        }
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront transport has no host staging slot for stage audits."));
    }
    *context.host_stage_audit = WavefrontStageAudit{};
    if (auto status = copy_device_to_host_and_wait(context.host_stage_audit, device_audit,
                                                   sizeof(WavefrontStageAudit), context,
                                                   "outcome-audit download");
        !status) {
        if (stage_timer != nullptr) {
            poison_profiling_workspace(context);
        }
        return std::unexpected(std::move(status.error()));
    }
    if (stage_timer != nullptr) {
        if (auto status =
                collect_timed_kernel(*stage_timer, *metric, work_count, context, stage_name);
            !status) {
            return std::unexpected(std::move(status.error()));
        }
    }
    const auto audit = *context.host_stage_audit;
    const auto expected_stage_kind = static_cast<std::uint32_t>(stage_kind);
    const auto no_failure = std::numeric_limits<std::uint32_t>::max();
    const auto canonical_empty_failure =
        audit.first_failure.status == 0U && audit.first_failure.route == 0U &&
        audit.first_failure.path_slot == 0U && audit.first_failure.detail == 0U;
    const auto counts_are_canonical = audit.closure_samples <= work_count &&
                                      audit.light_samples <= work_count &&
                                      (stage_kind == WavefrontStageKind::shade ||
                                       (audit.closure_samples == 0U && audit.light_samples == 0U));
    if (audit.abi_major != xpu::cuda::WavefrontStageAuditAbiMajor ||
        audit.abi_minor != xpu::cuda::WavefrontStageAuditAbiMinor ||
        audit.struct_size != sizeof(WavefrontStageAudit) ||
        audit.stage_kind != expected_stage_kind || audit.expected_work_count != work_count ||
        audit.inspected_work_count != work_count || !counts_are_canonical ||
        audit.reserved[0U] != 0U || audit.reserved[1U] != 0U || audit.reserved[2U] != 0U ||
        audit.reserved[3U] != 0U ||
        (audit.first_failure_work_index == no_failure && !canonical_empty_failure) ||
        (audit.first_failure_work_index != no_failure &&
         audit.first_failure_work_index >= work_count)) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront " + std::string{stage_name} +
                                                   " returned a non-canonical device audit."));
    }
    if (audit.first_failure_work_index != no_failure) {
        if (auto status =
                stage_failure(audit.first_failure, stage_name, audit.first_failure_work_index,
                              path_capacity, allowed_routes);
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront " + std::string{stage_name} +
                                                   " audit reported a non-failing outcome."));
    }
    return audit;
}

[[nodiscard]] core::Result<WavefrontQueueCompactionResult>
compact_route(const xpu::cuda::WavefrontQueueDeviceSoa queues,
              const WavefrontStageOutcome* const device_outcomes, const std::uint32_t work_count,
              const WavefrontStageRoute route, void* const device_scratch,
              const std::size_t scratch_bytes, WavefrontQueueCompactionResult* const device_result,
              const CudaWavefrontExecutionContext& context) {
    const auto route_value = static_cast<std::uint32_t>(route);
    auto cuda_status = static_cast<cudaError_t>(xpu::cuda::launch_wavefront_queue_compaction(
        queues, device_outcomes, work_count, route_value, device_scratch, scratch_bytes,
        device_result, opaque_stream(native_stream(context.compute_stream))));
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "queue-compaction kernel launch"));
    }

    if (context.host_compaction_result == nullptr) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront transport has no host staging slot for compaction results."));
    }
    *context.host_compaction_result = WavefrontQueueCompactionResult{};
    if (auto status = copy_device_to_host_and_wait(context.host_compaction_result, device_result,
                                                   sizeof(WavefrontQueueCompactionResult), context,
                                                   "queue-compaction result download");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto result = *context.host_compaction_result;

    const auto status = static_cast<WavefrontQueueCompactionStatus>(result.status);
    const auto reserved_is_zero = result.reserved[0U] == 0U && result.reserved[1U] == 0U &&
                                  result.reserved[2U] == 0U && result.reserved[3U] == 0U;
    const auto identity_is_canonical = result.queue_kind == route_value &&
                                       result.route == route_value &&
                                       result.input_count == work_count;
    const auto counts_are_bounded = result.initial_size <= queues.slot_stride &&
                                    result.selected_count <= work_count &&
                                    result.published_count <= result.selected_count &&
                                    result.rejected_count <= result.selected_count;
    if (!reserved_is_zero || !identity_is_canonical || !counts_are_bounded) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront queue compaction returned a non-canonical device result."));
    }

    switch (status) {
    case WavefrontQueueCompactionStatus::success:
        if (result.published_count != result.selected_count || result.rejected_count != 0U ||
            result.published_count > queues.slot_stride - result.initial_size) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue compaction returned inconsistent success counters."));
        }
        return result;
    case WavefrontQueueCompactionStatus::capacity_exhausted:
        if (result.selected_count == 0U || result.published_count != 0U ||
            result.rejected_count != result.selected_count ||
            result.selected_count <= queues.slot_stride - result.initial_size) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue compaction returned inconsistent overflow counters."));
        }
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront queue compaction rejected a complete append that exceeded its fixed "
            "capacity."));
    case WavefrontQueueCompactionStatus::invalid_contract:
        return std::unexpected(
            transport_error(core::StatusCode::incompatible,
                            "CUDA wavefront queue compaction rejected its device contract."));
    }
    return std::unexpected(
        transport_error(core::StatusCode::incompatible,
                        "CUDA wavefront queue compaction returned an unknown status."));
}

[[nodiscard]] core::Result<QueueHeader>
queue_header(const xpu::cuda::WavefrontQueueDeviceSoa queues,
             const renderer::WavefrontQueueKind kind, const std::uint32_t capacity,
             const CudaWavefrontExecutionContext& context) {
    if (context.host_queue_header == nullptr) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront transport has no host staging slot for queue headers."));
    }
    *context.host_queue_header = QueueHeader{};
    if (auto status = copy_device_to_host_and_wait(
            context.host_queue_header, queues.headers + queue_index(kind), sizeof(QueueHeader),
            context, "queue-header download");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto header = *context.host_queue_header;
    if (xpu::shared::validate_queue_header(header) !=
            xpu::shared::QueueHeaderValidationStatus::valid ||
        header.queue_kind != queue_index(kind) || header.capacity != capacity ||
        header.overflow_count != header.rejected_count) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront queue counters violated their fixed host/device contract."));
    }
    if (header.overflow_count != 0U) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront queue exhausted its fixed capacity without publishing a partial "
            "fallback result."));
    }
    return header;
}

[[nodiscard]] core::Status clear_queue(const xpu::cuda::WavefrontQueueDeviceSoa queues,
                                       const renderer::WavefrontQueueKind kind,
                                       std::uint32_t* const device_status,
                                       const CudaWavefrontExecutionContext& context,
                                       const std::uint32_t acknowledge_overflow = 0U) {
    constexpr auto sentinel = std::numeric_limits<std::uint32_t>::max();
    if (context.host_clear_status == nullptr || context.counters == nullptr) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront transport has no host staging slot for queue-clear status."));
    }
    *context.host_clear_status = sentinel;
    if (auto status =
            copy_host_to_device(device_status, context.host_clear_status, sizeof(sentinel),
                                native_stream(context.compute_stream), context.mode,
                                *context.counters, "queue-clear status initialization");
        !status) {
        return status;
    }
    auto status = static_cast<cudaError_t>(blackframe_cuda_launch_wavefront_clear_queue(
        queues, queue_index(kind), acknowledge_overflow, device_status,
        opaque_stream(native_stream(context.compute_stream))));
    if (status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(status, "queue-clear kernel launch"));
    }
    if (auto copy_status = copy_device_to_host_and_wait(context.host_clear_status, device_status,
                                                        sizeof(std::uint32_t), context,
                                                        "queue-clear status download");
        !copy_status) {
        return copy_status;
    }
    const auto host_status = *context.host_clear_status;
    if (host_status != static_cast<std::uint32_t>(WavefrontStageStatus::success)) {
        return std::unexpected(transport_error(
            host_status == static_cast<std::uint32_t>(WavefrontStageStatus::queue_overflow)
                ? core::StatusCode::resource_exhausted
                : core::StatusCode::incompatible,
            "CUDA wavefront queue clear refused a non-canonical consumed queue."));
    }
    return {};
}

struct CudaWavefrontWorkspaceColumns final {
    SampleStreamIndex* input_samples{};
    TransportRay* input_rays{};
    TransportPathStateLane* input_states{};
    SampleStreamIndex* stream_samples{};
    TransportRay* stream_rays{};
    TransportPathStateLane* stream_states{};
    ClosestHit* hits{};
    WavefrontPendingShadow* pending_shadows{};
    WavefrontPreviousBsdfSample* previous_bsdf_samples{};
    WavefrontLaneControl* controls{};
    PathSlot* compact_slots{};
    TransportRay* compact_rays{};
    SceneClosestHitResult* closest_results{};
    SceneOcclusionResult* occlusion_results{};
    WavefrontStageOutcome* outcomes{};
    WavefrontStageAudit* stage_audit{};
    WavefrontQueueCompactionResult* compaction_result{};
    void* compaction_scratch{};
    std::size_t compaction_scratch_bytes{};
    std::uint32_t* clear_status{};
};

[[nodiscard]] core::Result<CudaWavefrontWorkspaceColumns>
allocate_workspace_columns(xpu::cuda::DeviceScratchBuffer& scratch, const std::size_t capacity,
                           const std::size_t compaction_scratch_bytes) {
    auto allocate_column = [&]<typename Value>(const std::size_t count) -> core::Result<Value*> {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            return std::unexpected(
                transport_error(core::StatusCode::resource_exhausted,
                                "CUDA wavefront scratch column byte count overflowed."));
        }
        auto slice = scratch.allocate(count * sizeof(Value), alignof(Value));
        if (!slice) {
            return std::unexpected(std::move(slice.error()));
        }
        return static_cast<Value*>(slice->data);
    };

    auto input_samples = allocate_column.template operator()<SampleStreamIndex>(capacity);
    auto input_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto input_states = allocate_column.template operator()<TransportPathStateLane>(capacity);
    auto stream_samples = allocate_column.template operator()<SampleStreamIndex>(capacity);
    auto stream_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto stream_states = allocate_column.template operator()<TransportPathStateLane>(capacity);
    auto hits = allocate_column.template operator()<ClosestHit>(capacity);
    auto pending_shadows = allocate_column.template operator()<WavefrontPendingShadow>(capacity);
    auto previous_bsdf_samples =
        allocate_column.template operator()<WavefrontPreviousBsdfSample>(capacity);
    auto controls = allocate_column.template operator()<WavefrontLaneControl>(capacity);
    auto compact_slots = allocate_column.template operator()<PathSlot>(capacity);
    auto compact_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto closest_results = allocate_column.template operator()<SceneClosestHitResult>(capacity);
    auto occlusion_results = allocate_column.template operator()<SceneOcclusionResult>(capacity);
    auto outcomes = allocate_column.template operator()<WavefrontStageOutcome>(capacity);
    auto stage_audit = allocate_column.template operator()<WavefrontStageAudit>(1U);
    auto compaction_result =
        allocate_column.template operator()<WavefrontQueueCompactionResult>(1U);
    auto compaction_scratch = scratch.allocate(compaction_scratch_bytes,
                                               xpu::cuda::WavefrontQueueCompactionScratchAlignment);
    auto clear_status = allocate_column.template operator()<std::uint32_t>(1U);
    const auto allocation_error = [&]() -> const core::Error* {
        if (!input_samples)
            return &input_samples.error();
        if (!input_rays)
            return &input_rays.error();
        if (!input_states)
            return &input_states.error();
        if (!stream_samples)
            return &stream_samples.error();
        if (!stream_rays)
            return &stream_rays.error();
        if (!stream_states)
            return &stream_states.error();
        if (!hits)
            return &hits.error();
        if (!pending_shadows)
            return &pending_shadows.error();
        if (!previous_bsdf_samples)
            return &previous_bsdf_samples.error();
        if (!controls)
            return &controls.error();
        if (!compact_slots)
            return &compact_slots.error();
        if (!compact_rays)
            return &compact_rays.error();
        if (!closest_results)
            return &closest_results.error();
        if (!occlusion_results)
            return &occlusion_results.error();
        if (!outcomes)
            return &outcomes.error();
        if (!stage_audit)
            return &stage_audit.error();
        if (!compaction_result)
            return &compaction_result.error();
        if (!compaction_scratch)
            return &compaction_scratch.error();
        if (!clear_status)
            return &clear_status.error();
        return nullptr;
    }();
    if (allocation_error != nullptr) {
        return std::unexpected(*allocation_error);
    }

    return CudaWavefrontWorkspaceColumns{
        .input_samples = *input_samples,
        .input_rays = *input_rays,
        .input_states = *input_states,
        .stream_samples = *stream_samples,
        .stream_rays = *stream_rays,
        .stream_states = *stream_states,
        .hits = *hits,
        .pending_shadows = *pending_shadows,
        .previous_bsdf_samples = *previous_bsdf_samples,
        .controls = *controls,
        .compact_slots = *compact_slots,
        .compact_rays = *compact_rays,
        .closest_results = *closest_results,
        .occlusion_results = *occlusion_results,
        .outcomes = *outcomes,
        .stage_audit = *stage_audit,
        .compaction_result = *compaction_result,
        .compaction_scratch = compaction_scratch->data,
        .compaction_scratch_bytes = compaction_scratch->size_bytes,
        .clear_status = *clear_status,
    };
}

struct CudaWavefrontScratchRequirement final {
    std::size_t total_bytes{};
    std::size_t compaction_bytes{};
};

[[nodiscard]] core::Result<CudaWavefrontScratchRequirement>
scratch_capacity(const std::size_t path_count, const xpu::cuda::DeviceMemoryBudget memory_budget) {
    auto compaction_bytes = std::size_t{};
    const auto query_status =
        static_cast<cudaError_t>(xpu::cuda::query_wavefront_queue_compaction_scratch_bytes(
            static_cast<std::uint32_t>(path_count), &compaction_bytes));
    if (query_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(query_status, "queue-compaction scratch query"));
    }
    if (path_count != 0U && compaction_bytes == 0U) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront queue compaction reported zero scratch for non-zero work."));
    }

    constexpr auto column_element_bytes =
        sizeof(SampleStreamIndex) + sizeof(TransportRay) + sizeof(TransportPathStateLane) +
        sizeof(SampleStreamIndex) + sizeof(TransportRay) + sizeof(TransportPathStateLane) +
        sizeof(ClosestHit) + sizeof(WavefrontPendingShadow) + sizeof(WavefrontLaneControl) +
        sizeof(WavefrontPreviousBsdfSample) + sizeof(PathSlot) + sizeof(TransportRay) +
        sizeof(SceneClosestHitResult) + sizeof(SceneOcclusionResult) +
        sizeof(WavefrontStageOutcome);
    constexpr auto column_alignment_slack =
        (alignof(SampleStreamIndex) - 1U) + (alignof(TransportRay) - 1U) +
        (alignof(TransportPathStateLane) - 1U) + (alignof(SampleStreamIndex) - 1U) +
        (alignof(TransportRay) - 1U) + (alignof(TransportPathStateLane) - 1U) +
        (alignof(ClosestHit) - 1U) + (alignof(WavefrontPendingShadow) - 1U) +
        (alignof(WavefrontPreviousBsdfSample) - 1U) + (alignof(WavefrontLaneControl) - 1U) +
        (alignof(PathSlot) - 1U) + (alignof(TransportRay) - 1U) +
        (alignof(SceneClosestHitResult) - 1U) + (alignof(SceneOcclusionResult) - 1U) +
        (alignof(WavefrontStageOutcome) - 1U) + (alignof(WavefrontStageAudit) - 1U) +
        (alignof(WavefrontQueueCompactionResult) - 1U) +
        (xpu::cuda::WavefrontQueueCompactionScratchAlignment - 1U) + (alignof(std::uint32_t) - 1U);
    constexpr auto fixed_bytes = sizeof(WavefrontStageAudit) +
                                 sizeof(WavefrontQueueCompactionResult) + sizeof(std::uint32_t);
    if (path_count >
        (std::numeric_limits<std::size_t>::max() - column_alignment_slack - fixed_bytes) /
            column_element_bytes) {
        return std::unexpected(
            transport_error(core::StatusCode::resource_exhausted,
                            "CUDA wavefront scratch byte count overflowed its host size domain."));
    }
    const auto column_bytes =
        path_count * column_element_bytes + column_alignment_slack + fixed_bytes;
    if (compaction_bytes > std::numeric_limits<std::size_t>::max() - column_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront compaction scratch byte count overflowed its host size domain."));
    }
    const auto scratch_bytes = column_bytes + compaction_bytes;
    constexpr auto queue_header_bytes =
        static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(QueueHeader);
    constexpr auto queue_slot_bytes_per_path =
        static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(PathSlot);
    if (path_count > (std::numeric_limits<std::size_t>::max() - queue_header_bytes) /
                         queue_slot_bytes_per_path) {
        return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                               "CUDA wavefront queue byte count overflowed."));
    }
    const auto queue_bytes = queue_header_bytes + path_count * queue_slot_bytes_per_path;
    if (scratch_bytes > std::numeric_limits<std::size_t>::max() - queue_bytes ||
        scratch_bytes + queue_bytes > memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport exceeds its explicit aggregate device-memory budget."));
    }
    return CudaWavefrontScratchRequirement{
        .total_bytes = scratch_bytes,
        .compaction_bytes = compaction_bytes,
    };
}

[[nodiscard]] constexpr std::size_t queue_capacity_bytes(const std::size_t capacity) noexcept {
    return static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(QueueHeader) +
           capacity * static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) *
               sizeof(PathSlot);
}

} // namespace

namespace {

template <typename Resource>
void retain_first_close_error(Resource& resource, core::Status& first_error) {
    auto status = resource.close();
    if (!status && first_error) {
        first_error = std::unexpected(std::move(status.error()));
    }
}

template <xpu::cuda::PinnedHostBufferElement Element>
[[nodiscard]] core::Result<std::unique_ptr<xpu::cuda::PinnedHostBuffer<Element>>>
allocate_pinned_buffer(const std::size_t count) {
    auto buffer = xpu::cuda::PinnedHostBuffer<Element>::allocate(count);
    if (!buffer) {
        return std::unexpected(std::move(buffer.error()));
    }
    return std::make_unique<xpu::cuda::PinnedHostBuffer<Element>>(std::move(*buffer));
}

struct CudaWavefrontAsyncStaging final {
    ~CudaWavefrontAsyncStaging() noexcept {
        static_cast<void>(close());
    }

    [[nodiscard]] bool valid() const noexcept {
        return compute_stream != nullptr && static_cast<bool>(*compute_stream) &&
               transfer_stream != nullptr && static_cast<bool>(*transfer_stream) &&
               uploads_ready != nullptr && static_cast<bool>(*uploads_ready) &&
               checkpoint != nullptr && static_cast<bool>(*checkpoint) && compute_done != nullptr &&
               static_cast<bool>(*compute_done) && downloads_ready != nullptr &&
               static_cast<bool>(*downloads_ready) && host_samples != nullptr &&
               static_cast<bool>(*host_samples) && host_rays != nullptr &&
               static_cast<bool>(*host_rays) && host_states != nullptr &&
               static_cast<bool>(*host_states) && final_states != nullptr &&
               static_cast<bool>(*final_states) && final_rays != nullptr &&
               static_cast<bool>(*final_rays) && final_controls != nullptr &&
               static_cast<bool>(*final_controls) && stage_audit != nullptr &&
               static_cast<bool>(*stage_audit) && compaction_result != nullptr &&
               static_cast<bool>(*compaction_result) && queue_header != nullptr &&
               static_cast<bool>(*queue_header) && clear_status != nullptr &&
               static_cast<bool>(*clear_status);
    }

    [[nodiscard]] core::Status synchronize() {
        auto first_error = core::Status{};
        if (compute_stream != nullptr && static_cast<bool>(*compute_stream)) {
            auto status = compute_stream->synchronize();
            if (!status && first_error) {
                first_error = std::unexpected(std::move(status.error()));
            }
        }
        if (transfer_stream != nullptr && static_cast<bool>(*transfer_stream)) {
            auto status = transfer_stream->synchronize();
            if (!status && first_error) {
                first_error = std::unexpected(std::move(status.error()));
            }
        }
        return first_error;
    }

    [[nodiscard]] core::Status close() {
        auto first_error = core::Status{};
        if (compute_stream != nullptr)
            retain_first_close_error(*compute_stream, first_error);
        if (transfer_stream != nullptr)
            retain_first_close_error(*transfer_stream, first_error);
        if ((compute_stream != nullptr && static_cast<bool>(*compute_stream)) ||
            (transfer_stream != nullptr && static_cast<bool>(*transfer_stream))) {
            return first_error;
        }
        if (uploads_ready != nullptr)
            retain_first_close_error(*uploads_ready, first_error);
        if (checkpoint != nullptr)
            retain_first_close_error(*checkpoint, first_error);
        if (compute_done != nullptr)
            retain_first_close_error(*compute_done, first_error);
        if (downloads_ready != nullptr)
            retain_first_close_error(*downloads_ready, first_error);
        if (host_samples != nullptr)
            retain_first_close_error(*host_samples, first_error);
        if (host_rays != nullptr)
            retain_first_close_error(*host_rays, first_error);
        if (host_states != nullptr)
            retain_first_close_error(*host_states, first_error);
        if (final_states != nullptr)
            retain_first_close_error(*final_states, first_error);
        if (final_rays != nullptr)
            retain_first_close_error(*final_rays, first_error);
        if (final_controls != nullptr)
            retain_first_close_error(*final_controls, first_error);
        if (stage_audit != nullptr)
            retain_first_close_error(*stage_audit, first_error);
        if (compaction_result != nullptr)
            retain_first_close_error(*compaction_result, first_error);
        if (queue_header != nullptr)
            retain_first_close_error(*queue_header, first_error);
        if (clear_status != nullptr)
            retain_first_close_error(*clear_status, first_error);
        return first_error;
    }

    std::unique_ptr<xpu::cuda::PinnedHostBuffer<SampleStreamIndex>> host_samples;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<TransportRay>> host_rays;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<TransportPathStateLane>> host_states;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<TransportPathStateLane>> final_states;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<TransportRay>> final_rays;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<WavefrontLaneControl>> final_controls;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<WavefrontStageAudit>> stage_audit;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<WavefrontQueueCompactionResult>> compaction_result;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<QueueHeader>> queue_header;
    std::unique_ptr<xpu::cuda::PinnedHostBuffer<std::uint32_t>> clear_status;
    std::unique_ptr<xpu::cuda::Event> uploads_ready;
    std::unique_ptr<xpu::cuda::Event> checkpoint;
    std::unique_ptr<xpu::cuda::Event> compute_done;
    std::unique_ptr<xpu::cuda::Event> downloads_ready;
    // Streams are declared last so their destructor backstops run before event and pinned-storage
    // destruction if an explicit close cannot establish quiescence.
    std::unique_ptr<xpu::cuda::Stream> compute_stream;
    std::unique_ptr<xpu::cuda::Stream> transfer_stream;
};

class CudaWavefrontAsyncDrain final {
  public:
    explicit CudaWavefrontAsyncDrain(CudaWavefrontAsyncStaging* staging,
                                     bool* const workspace_poisoned) noexcept
        : staging_(staging), workspace_poisoned_(workspace_poisoned) {}
    ~CudaWavefrontAsyncDrain() noexcept {
        try {
            static_cast<void>(drain());
        } catch (...) {
            poison_workspace();
        }
    }

    CudaWavefrontAsyncDrain(const CudaWavefrontAsyncDrain&) = delete;
    CudaWavefrontAsyncDrain& operator=(const CudaWavefrontAsyncDrain&) = delete;

    void arm() noexcept {
        armed_ = staging_ != nullptr;
    }
    void disarm() noexcept {
        armed_ = false;
    }
    [[nodiscard]] bool armed() const noexcept {
        return armed_;
    }
    [[nodiscard]] core::Status drain() {
        if (!armed_ || staging_ == nullptr) {
            return {};
        }
        auto status = staging_->synchronize();
        if (status) {
            armed_ = false;
        } else {
            poison_workspace();
        }
        return status;
    }

  private:
    void poison_workspace() noexcept {
        if (workspace_poisoned_ != nullptr) {
            *workspace_poisoned_ = true;
        }
    }

    CudaWavefrontAsyncStaging* staging_{};
    bool* workspace_poisoned_{};
    bool armed_{};
};

[[nodiscard]] core::Result<std::unique_ptr<CudaWavefrontAsyncStaging>>
create_async_staging(const std::size_t capacity) {
    auto staging = std::make_unique<CudaWavefrontAsyncStaging>();
    auto compute_stream = xpu::cuda::Stream::create();
    if (!compute_stream)
        return std::unexpected(std::move(compute_stream.error()));
    staging->compute_stream = std::make_unique<xpu::cuda::Stream>(std::move(*compute_stream));
    auto transfer_stream = xpu::cuda::Stream::create();
    if (!transfer_stream)
        return std::unexpected(std::move(transfer_stream.error()));
    staging->transfer_stream = std::make_unique<xpu::cuda::Stream>(std::move(*transfer_stream));

    const auto create_event = []() -> core::Result<std::unique_ptr<xpu::cuda::Event>> {
        auto event = xpu::cuda::Event::create();
        if (!event)
            return std::unexpected(std::move(event.error()));
        return std::make_unique<xpu::cuda::Event>(std::move(*event));
    };
    auto uploads_ready = create_event();
    if (!uploads_ready)
        return std::unexpected(std::move(uploads_ready.error()));
    staging->uploads_ready = std::move(*uploads_ready);
    auto checkpoint = create_event();
    if (!checkpoint)
        return std::unexpected(std::move(checkpoint.error()));
    staging->checkpoint = std::move(*checkpoint);
    auto compute_done = create_event();
    if (!compute_done)
        return std::unexpected(std::move(compute_done.error()));
    staging->compute_done = std::move(*compute_done);
    auto downloads_ready = create_event();
    if (!downloads_ready)
        return std::unexpected(std::move(downloads_ready.error()));
    staging->downloads_ready = std::move(*downloads_ready);

    auto host_samples = allocate_pinned_buffer<SampleStreamIndex>(capacity);
    if (!host_samples)
        return std::unexpected(std::move(host_samples.error()));
    staging->host_samples = std::move(*host_samples);
    auto host_rays = allocate_pinned_buffer<TransportRay>(capacity);
    if (!host_rays)
        return std::unexpected(std::move(host_rays.error()));
    staging->host_rays = std::move(*host_rays);
    auto host_states = allocate_pinned_buffer<TransportPathStateLane>(capacity);
    if (!host_states)
        return std::unexpected(std::move(host_states.error()));
    staging->host_states = std::move(*host_states);
    auto final_states = allocate_pinned_buffer<TransportPathStateLane>(capacity);
    if (!final_states)
        return std::unexpected(std::move(final_states.error()));
    staging->final_states = std::move(*final_states);
    auto final_rays = allocate_pinned_buffer<TransportRay>(capacity);
    if (!final_rays)
        return std::unexpected(std::move(final_rays.error()));
    staging->final_rays = std::move(*final_rays);
    auto final_controls = allocate_pinned_buffer<WavefrontLaneControl>(capacity);
    if (!final_controls)
        return std::unexpected(std::move(final_controls.error()));
    staging->final_controls = std::move(*final_controls);
    auto stage_audit = allocate_pinned_buffer<WavefrontStageAudit>(1U);
    if (!stage_audit)
        return std::unexpected(std::move(stage_audit.error()));
    staging->stage_audit = std::move(*stage_audit);
    auto compaction_result = allocate_pinned_buffer<WavefrontQueueCompactionResult>(1U);
    if (!compaction_result)
        return std::unexpected(std::move(compaction_result.error()));
    staging->compaction_result = std::move(*compaction_result);
    auto queue_header = allocate_pinned_buffer<QueueHeader>(1U);
    if (!queue_header)
        return std::unexpected(std::move(queue_header.error()));
    staging->queue_header = std::move(*queue_header);
    auto clear_status = allocate_pinned_buffer<std::uint32_t>(1U);
    if (!clear_status)
        return std::unexpected(std::move(clear_status.error()));
    staging->clear_status = std::move(*clear_status);
    return staging;
}

} // namespace

struct CudaWavefrontTransportWorkspace::Storage final {
    Storage(xpu::cuda::DeviceScratchBuffer scratch_storage, CudaWavefrontQueues queue_storage,
            CudaWavefrontWorkspaceColumns column_views, const std::size_t path_capacity,
            const std::size_t allocated_device_bytes,
            const CudaWavefrontTransferMode selected_transfer_mode,
            const CudaWavefrontInstrumentationMode selected_instrumentation_mode,
            std::unique_ptr<CudaWavefrontAsyncStaging> asynchronous_staging,
            std::unique_ptr<CudaWavefrontProfilingResources> profiling_resources)
        : scratch(std::move(scratch_storage)), queues(std::move(queue_storage)),
          columns(column_views), capacity(path_capacity), device_size_bytes(allocated_device_bytes),
          transfer_mode(selected_transfer_mode),
          instrumentation_mode(selected_instrumentation_mode),
          async_staging(std::move(asynchronous_staging)), profiling(std::move(profiling_resources)),
          host_samples(selected_transfer_mode == CudaWavefrontTransferMode::synchronous
                           ? path_capacity
                           : 0U),
          host_rays(selected_transfer_mode == CudaWavefrontTransferMode::synchronous ? path_capacity
                                                                                     : 0U),
          host_states(selected_transfer_mode == CudaWavefrontTransferMode::synchronous
                          ? path_capacity
                          : 0U),
          final_states(selected_transfer_mode == CudaWavefrontTransferMode::synchronous
                           ? path_capacity
                           : 0U),
          final_rays(selected_transfer_mode == CudaWavefrontTransferMode::synchronous
                         ? path_capacity
                         : 0U),
          final_controls(selected_transfer_mode == CudaWavefrontTransferMode::synchronous
                             ? path_capacity
                             : 0U) {}

    [[nodiscard]] SampleStreamIndex* host_samples_data() noexcept {
        return async_staging != nullptr ? async_staging->host_samples->data() : host_samples.data();
    }
    [[nodiscard]] TransportRay* host_rays_data() noexcept {
        return async_staging != nullptr ? async_staging->host_rays->data() : host_rays.data();
    }
    [[nodiscard]] TransportPathStateLane* host_states_data() noexcept {
        return async_staging != nullptr ? async_staging->host_states->data() : host_states.data();
    }
    [[nodiscard]] TransportPathStateLane* final_states_data() noexcept {
        return async_staging != nullptr ? async_staging->final_states->data() : final_states.data();
    }
    [[nodiscard]] TransportRay* final_rays_data() noexcept {
        return async_staging != nullptr ? async_staging->final_rays->data() : final_rays.data();
    }
    [[nodiscard]] WavefrontLaneControl* final_controls_data() noexcept {
        return async_staging != nullptr ? async_staging->final_controls->data()
                                        : final_controls.data();
    }
    [[nodiscard]] WavefrontStageAudit* host_stage_audit_data() noexcept {
        return async_staging != nullptr ? async_staging->stage_audit->data() : &host_stage_audit;
    }
    [[nodiscard]] WavefrontQueueCompactionResult* host_compaction_result_data() noexcept {
        return async_staging != nullptr ? async_staging->compaction_result->data()
                                        : &host_compaction_result;
    }
    [[nodiscard]] QueueHeader* host_queue_header_data() noexcept {
        return async_staging != nullptr ? async_staging->queue_header->data() : &host_queue_header;
    }
    [[nodiscard]] std::uint32_t* host_clear_status_data() noexcept {
        return async_staging != nullptr ? async_staging->clear_status->data() : &host_clear_status;
    }

    xpu::cuda::DeviceScratchBuffer scratch;
    CudaWavefrontQueues queues;
    CudaWavefrontWorkspaceColumns columns{};
    std::size_t capacity{};
    std::size_t device_size_bytes{};
    CudaWavefrontTransferMode transfer_mode{CudaWavefrontTransferMode::synchronous};
    CudaWavefrontInstrumentationMode instrumentation_mode{
        CudaWavefrontInstrumentationMode::disabled};
    std::unique_ptr<CudaWavefrontAsyncStaging> async_staging;
    std::unique_ptr<CudaWavefrontProfilingResources> profiling;
    std::vector<SampleStreamIndex> host_samples;
    std::vector<TransportRay> host_rays;
    std::vector<TransportPathStateLane> host_states;
    std::vector<TransportPathStateLane> final_states;
    std::vector<TransportRay> final_rays;
    std::vector<WavefrontLaneControl> final_controls;
    WavefrontStageAudit host_stage_audit{};
    WavefrontQueueCompactionResult host_compaction_result{};
    QueueHeader host_queue_header{};
    std::uint32_t host_clear_status{};
    bool queues_dirty{};
    bool poisoned{};
};

CudaWavefrontTransportWorkspace::CudaWavefrontTransportWorkspace(
    std::unique_ptr<Storage> storage) noexcept
    : storage_(std::move(storage)) {}

CudaWavefrontTransportWorkspace::CudaWavefrontTransportWorkspace(
    CudaWavefrontTransportWorkspace&& other) noexcept = default;

CudaWavefrontTransportWorkspace::~CudaWavefrontTransportWorkspace() noexcept = default;

core::Result<CudaWavefrontTransportWorkspace> CudaWavefrontTransportWorkspace::create(
    const std::size_t capacity, const xpu::cuda::DeviceMemoryBudget device_memory_budget,
    const CudaWavefrontTransferMode transfer_mode,
    const CudaWavefrontInstrumentationMode instrumentation_mode) try {
    if (!valid_transfer_mode(transfer_mode)) {
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront workspace requires an explicit synchronous or asynchronous transfer "
            "mode."));
    }
    if (!valid_instrumentation_mode(instrumentation_mode)) {
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront workspace requires instrumentation to be explicitly disabled or set "
            "to Nsight."));
    }
    if (capacity == 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport workspace capacity must be non-zero."));
    }
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport workspace capacity exceeds its 32-bit device domain."));
    }
    const auto required_scratch = scratch_capacity(capacity, device_memory_budget);
    if (!required_scratch) {
        return std::unexpected(required_scratch.error());
    }
    const auto required_queues = queue_capacity_bytes(capacity);
    if (required_scratch->total_bytes > std::numeric_limits<std::size_t>::max() - required_queues ||
        required_scratch->total_bytes + required_queues > device_memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront workspace exceeds its aggregate device-memory budget."));
    }
    auto scratch =
        xpu::cuda::DeviceScratchBuffer::create(required_scratch->total_bytes, device_memory_budget);
    if (!scratch) {
        return std::unexpected(std::move(scratch.error()));
    }
    auto columns =
        allocate_workspace_columns(*scratch, capacity, required_scratch->compaction_bytes);
    if (!columns) {
        return std::unexpected(std::move(columns.error()));
    }
    auto queues = CudaWavefrontQueues::create(
        capacity, CudaWavefrontQueueCreateOptions{.device_memory_budget = device_memory_budget});
    if (!queues) {
        return std::unexpected(std::move(queues.error()));
    }
    if (scratch->device_ordinal() != queues->device_ordinal()) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront workspace allocations unexpectedly landed on different devices."));
    }
    auto async_staging = std::unique_ptr<CudaWavefrontAsyncStaging>{};
    if (transfer_mode == CudaWavefrontTransferMode::asynchronous) {
        auto created_staging = create_async_staging(capacity);
        if (!created_staging) {
            return std::unexpected(std::move(created_staging.error()));
        }
        async_staging = std::move(*created_staging);
        if (!async_staging->valid() ||
            async_staging->compute_stream->device_ordinal() != scratch->device_ordinal() ||
            async_staging->transfer_stream->device_ordinal() != scratch->device_ordinal()) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront asynchronous resources do not belong to the workspace device."));
        }
    }
    auto profiling = std::unique_ptr<CudaWavefrontProfilingResources>{};
    if (instrumentation_mode == CudaWavefrontInstrumentationMode::nsight) {
        auto created_profiling = create_profiling_resources(
            async_staging != nullptr ? async_staging->compute_stream.get() : nullptr,
            async_staging != nullptr ? async_staging->transfer_stream.get() : nullptr);
        if (!created_profiling) {
            return std::unexpected(std::move(created_profiling.error()));
        }
        profiling = std::move(*created_profiling);
        if (!profiling->valid(scratch->device_ordinal())) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront profiling resources do not belong to the workspace device."));
        }
        if (async_staging != nullptr) {
            nvtxNameCudaEventA(
                static_cast<cudaEvent_t>(async_staging->uploads_ready->native_handle()),
                "Blackframe uploads ready");
            nvtxNameCudaEventA(static_cast<cudaEvent_t>(async_staging->checkpoint->native_handle()),
                               "Blackframe compute checkpoint");
            nvtxNameCudaEventA(
                static_cast<cudaEvent_t>(async_staging->compute_done->native_handle()),
                "Blackframe compute done");
            nvtxNameCudaEventA(
                static_cast<cudaEvent_t>(async_staging->downloads_ready->native_handle()),
                "Blackframe downloads ready");
        }
    }
    const auto allocated_device_bytes = required_scratch->total_bytes + required_queues;
    auto storage = std::make_unique<Storage>(
        std::move(*scratch), std::move(*queues), *columns, capacity, allocated_device_bytes,
        transfer_mode, instrumentation_mode, std::move(async_staging), std::move(profiling));
    return CudaWavefrontTransportWorkspace{std::move(storage)};
} catch (const std::bad_alloc&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport workspace exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(transport_error(
        core::StatusCode::resource_exhausted,
        "CUDA wavefront transport workspace exceeded a host container length limit."));
}

std::size_t CudaWavefrontTransportWorkspace::capacity() const noexcept {
    return storage_ != nullptr ? storage_->capacity : 0U;
}

std::size_t CudaWavefrontTransportWorkspace::device_size_bytes() const noexcept {
    return storage_ != nullptr ? storage_->device_size_bytes : 0U;
}

std::int32_t CudaWavefrontTransportWorkspace::device_ordinal() const noexcept {
    return storage_ != nullptr ? storage_->scratch.device_ordinal() : -1;
}

CudaWavefrontTransferMode CudaWavefrontTransportWorkspace::transfer_mode() const noexcept {
    return storage_ != nullptr ? storage_->transfer_mode : CudaWavefrontTransferMode::synchronous;
}

CudaWavefrontInstrumentationMode
CudaWavefrontTransportWorkspace::instrumentation_mode() const noexcept {
    return storage_ != nullptr ? storage_->instrumentation_mode
                               : CudaWavefrontInstrumentationMode::disabled;
}

CudaWavefrontTransportWorkspace::operator bool() const noexcept {
    return storage_ != nullptr && storage_->capacity != 0U && !storage_->scratch.empty() &&
           static_cast<bool>(storage_->queues) &&
           storage_->scratch.device_ordinal() == storage_->queues.device_ordinal() &&
           (storage_->transfer_mode == CudaWavefrontTransferMode::synchronous ||
            (storage_->async_staging != nullptr && storage_->async_staging->valid())) &&
           (storage_->instrumentation_mode == CudaWavefrontInstrumentationMode::disabled ||
            (storage_->profiling != nullptr &&
             storage_->profiling->valid(storage_->scratch.device_ordinal())));
}

core::Status CudaWavefrontTransportWorkspace::close() {
    if (storage_ == nullptr) {
        return {};
    }
    auto first_error = core::Status{};
    if (storage_->async_staging != nullptr) {
        auto async_status = storage_->async_staging->close();
        if (!async_status) {
            return std::unexpected(std::move(async_status.error()));
        }
    }
    if (storage_->profiling != nullptr) {
        auto profiling_status = storage_->profiling->close();
        if (!profiling_status && first_error) {
            first_error = std::unexpected(std::move(profiling_status.error()));
        }
    }
    auto queue_status = storage_->queues.close();
    if (!queue_status && first_error) {
        first_error = std::unexpected(std::move(queue_status.error()));
    }
    auto scratch_status = storage_->scratch.close();
    if (!scratch_status && first_error) {
        first_error = std::unexpected(std::move(scratch_status.error()));
    }
    if (first_error) {
        storage_.reset();
    }
    return first_error;
}

core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(CudaWavefrontTransportWorkspace& workspace,
                               const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                               const std::span<const CudaWavefrontPathInput> inputs,
                               const CudaWavefrontLightSampler light_sampler,
                               const CudaWavefrontTransportOptions options) try {
    if (auto status = cuda_scene_query_detail::validate_binding(scene, bvh, "wavefront transport");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = validate_options_and_inputs(inputs, options); !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto validated_light_sampler = validate_light_sampler(scene.header(), light_sampler);
    if (!validated_light_sampler) {
        return std::unexpected(validated_light_sampler.error());
    }
    const auto config = transport_config(options, validated_light_sampler->light_count,
                                         validated_light_sampler->strategy);

    const auto path_count = inputs.size();
    const auto device_path_count = static_cast<std::uint32_t>(path_count);
    if (!workspace) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport requires an open reusable workspace."));
    }
    auto& storage = *workspace.storage_;
    if (storage.poisoned) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport workspace is poisoned because a prior asynchronous drain or "
            "profiling interval could not be completed; it must be closed and recreated."));
    }
    if (storage.transfer_mode != options.transfer_mode) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport transfer mode does not match its reusable workspace; the "
            "requested mode was not substituted."));
    }
    if (storage.instrumentation_mode != options.instrumentation_mode) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport instrumentation mode does not match its reusable workspace; "
            "the requested mode was not substituted."));
    }
    if (path_count > storage.capacity) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport path count exceeds the reusable workspace capacity."));
    }
    if (storage.device_size_bytes > options.device_memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport workspace exceeds the requested device-memory budget."));
    }
    if (storage.scratch.device_ordinal() != scene.device_ordinal()) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport workspace and scene must reside on the same device."));
    }

    auto transfer_counters = CudaWavefrontTransferCounters{};
    auto* const asynchronous = storage.async_staging.get();
    auto execution_context = CudaWavefrontExecutionContext{
        .mode = storage.transfer_mode,
        .compute_stream = asynchronous != nullptr ? asynchronous->compute_stream.get() : nullptr,
        .checkpoint = asynchronous != nullptr ? asynchronous->checkpoint.get() : nullptr,
        .counters = &transfer_counters,
        .host_stage_audit = storage.host_stage_audit_data(),
        .host_compaction_result = storage.host_compaction_result_data(),
        .host_queue_header = storage.host_queue_header_data(),
        .host_clear_status = storage.host_clear_status_data(),
        .profiling = storage.profiling.get(),
        .workspace_poisoned = &storage.poisoned,
    };
    auto async_drain = CudaWavefrontAsyncDrain{asynchronous, &storage.poisoned};
    auto traced = [&]() -> core::Result<CudaWavefrontTransportBatch> {
        try {
            const auto transport_range = CudaWavefrontNvtxRange{
                execution_context.profiling, "transport", CudaWavefrontNvtxCategory::transport,
                0xFF607D8BU, path_count};
            const auto device_capacity = static_cast<std::uint32_t>(storage.capacity);
            const auto& columns = storage.columns;
            const auto queue_view = storage.queues.device_view();
            auto* const stream_rays = columns.stream_rays;
            auto* const stream_states = columns.stream_states;
            auto* const controls = columns.controls;
            auto* const compact_slots = columns.compact_slots;
            auto* const compact_rays = columns.compact_rays;
            auto* const closest_results = columns.closest_results;
            auto* const occlusion_results = columns.occlusion_results;
            auto* const outcomes = columns.outcomes;
            auto* const stage_audit = columns.stage_audit;
            auto* const compaction_result = columns.compaction_result;
            auto* const compaction_scratch = columns.compaction_scratch;
            const auto compaction_scratch_bytes = columns.compaction_scratch_bytes;
            auto* const clear_status = columns.clear_status;
            auto* const host_samples = storage.host_samples_data();
            auto* const host_rays = storage.host_rays_data();
            auto* const host_states = storage.host_states_data();
            const auto transfer_stream = asynchronous != nullptr
                                             ? native_stream(asynchronous->transfer_stream.get())
                                             : cudaStream_t{};
            if (asynchronous != nullptr) {
                async_drain.arm();
            }
            for (auto index = std::size_t{}; index < path_count; ++index) {
                host_samples[index] = transport_sample_stream_index(inputs[index].sample);
                auto converted_ray = cuda_scene_query_detail::transport_ray(
                    inputs[index].primary_ray, index, "wavefront transport");
                if (!converted_ray) {
                    return std::unexpected(std::move(converted_ray.error()));
                }
                host_rays[index] = *converted_ray;
                host_states[index] = transport_path_state(inputs[index].initial_state);
            }
            if (storage.queues_dirty) {
                if (asynchronous != nullptr) {
                    for (auto queue = std::uint32_t{}; queue < xpu::cuda::CudaWavefrontQueueCount;
                         ++queue) {
                        if (auto status = clear_queue(
                                queue_view, static_cast<renderer::WavefrontQueueKind>(queue),
                                clear_status, execution_context, 1U);
                            !status) {
                            return std::unexpected(std::move(status.error()));
                        }
                    }
                } else {
                    // The preceding trace already returned the device-stage failure to the caller.
                    // Synchronous reset may acknowledge its diagnostics while still validating
                    // every counter before the workspace is reused.
                    auto reset_status =
                        storage.queues.reset(CudaWavefrontQueueResetPolicy::acknowledge_overflow);
                    if (!reset_status) {
                        return std::unexpected(std::move(reset_status.error()));
                    }
                }
                storage.queues_dirty = false;
            }

            const auto sample_bytes = path_count * sizeof(SampleStreamIndex);
            const auto ray_bytes = path_count * sizeof(TransportRay);
            const auto state_bytes = path_count * sizeof(TransportPathStateLane);
            const auto control_bytes = path_count * sizeof(WavefrontLaneControl);
            {
                const auto upload_range = CudaWavefrontNvtxRange{
                    execution_context.profiling, "upload", CudaWavefrontNvtxCategory::transfer,
                    0xFF78909CU, sample_bytes + ray_bytes + state_bytes};
                if (auto status = copy_host_to_device(
                        columns.input_samples, host_samples, sample_bytes, transfer_stream,
                        storage.transfer_mode, transfer_counters, "sample upload");
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                if (auto status = copy_host_to_device(columns.input_rays, host_rays, ray_bytes,
                                                      transfer_stream, storage.transfer_mode,
                                                      transfer_counters, "ray upload");
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                if (auto status = copy_host_to_device(
                        columns.input_states, host_states, state_bytes, transfer_stream,
                        storage.transfer_mode, transfer_counters, "path-state upload");
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                if (auto status =
                        initialize_device_bytes(controls, 0, control_bytes, execution_context,
                                                "lane-control initialization");
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                if (asynchronous != nullptr) {
                    if (auto status = contextual_runtime_status(
                            asynchronous->uploads_ready->record(*asynchronous->transfer_stream),
                            "upload-ready event record");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = contextual_runtime_status(
                            asynchronous->compute_stream->wait(*asynchronous->uploads_ready),
                            "upload-ready event wait");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    ++transfer_counters.event_dependencies;
                }
            }
            auto cuda_status = cudaSuccess;

            const auto stream_view = WavefrontStageDeviceSoa{
                .sample_streams = columns.stream_samples,
                .rays = columns.stream_rays,
                .path_states = columns.stream_states,
                .hits = columns.hits,
                .pending_shadows = columns.pending_shadows,
                .previous_bsdf_samples = columns.previous_bsdf_samples,
                .controls = columns.controls,
                .capacity = device_capacity,
                .reserved = 0U,
            };
            const auto camera_inputs = WavefrontCameraInputDeviceSoa{
                .sample_streams = columns.input_samples,
                .rays = columns.input_rays,
                .path_states = columns.input_states,
                .count = device_path_count,
                .reserved = 0U,
            };
            void* const kernel_stream =
                opaque_stream(native_stream(execution_context.compute_stream));
            auto report = CudaWavefrontTransportReport{
                .schema_version = CurrentCudaWavefrontTransportReportSchemaVersion,
                .has_light_sampler = validated_light_sampler->present,
                .registered_light_count = validated_light_sampler->light_count,
                .heuristic = options.heuristic,
                .light_sampling_strategy = validated_light_sampler->strategy,
                .depth_limits = options.depth_limits,
                .roulette_policy = options.roulette_policy,
                .path_count = path_count,
                .transfer_mode = options.transfer_mode,
                .instrumentation_mode = options.instrumentation_mode,
            };

            storage.queues_dirty = true;
            {
                const auto camera_range = CudaWavefrontNvtxRange{
                    execution_context.profiling, "camera", CudaWavefrontNvtxCategory::camera,
                    0xFF42A5F5U, device_path_count};
                if (auto status = execute_stage(
                        WavefrontStageKind::camera_seed, "camera seed", device_path_count,
                        route_mask(WavefrontStageRoute::none), device_path_count, outcomes,
                        stage_audit, execution_context, nullptr,
                        [&] {
                            return blackframe_cuda_launch_wavefront_seed_camera(
                                queue_view, stream_view, 0U, device_path_count, outcomes,
                                kernel_stream);
                        });
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                const auto camera_header =
                    queue_header(queue_view, renderer::WavefrontQueueKind::camera, device_capacity,
                                 execution_context);
                if (!camera_header || camera_header->size != device_path_count) {
                    return std::unexpected(
                        camera_header
                            ? transport_error(core::StatusCode::internal_error,
                                              "CUDA wavefront camera seed lost path lanes.")
                            : camera_header.error());
                }
                if (auto status = execute_stage(
                        WavefrontStageKind::camera, "camera", device_path_count,
                        route_mask(WavefrontStageRoute::ray), device_path_count, outcomes,
                        stage_audit, execution_context, &report.stage_metrics.camera,
                        [&] {
                            return blackframe_cuda_launch_wavefront_camera_stage(
                                queue_view, camera_inputs, stream_view, device_path_count, outcomes,
                                kernel_stream);
                        });
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
                if (auto compacted = compact_route(queue_view, outcomes, device_path_count,
                                                   WavefrontStageRoute::ray, compaction_scratch,
                                                   compaction_scratch_bytes, compaction_result,
                                                   execution_context);
                    !compacted) {
                    return std::unexpected(std::move(compacted.error()));
                }
                report.stage_lanes.camera = device_path_count;
                if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::camera,
                                              clear_status, execution_context);
                    !status) {
                    return std::unexpected(std::move(status.error()));
                }
            }

            auto ray_header = queue_header(queue_view, renderer::WavefrontQueueKind::ray,
                                           device_capacity, execution_context);
            if (!ray_header) {
                return std::unexpected(ray_header.error());
            }
            auto iteration = std::uint64_t{};
            // Transmission is an orthogonal counter attached to a surface-family event, so it does
            // not add another bounce to the dispatch bound. The four primary families do.
            const auto maximum_iterations =
                static_cast<std::uint64_t>(options.depth_limits.diffuse) +
                static_cast<std::uint64_t>(options.depth_limits.glossy) +
                static_cast<std::uint64_t>(options.depth_limits.specular) +
                static_cast<std::uint64_t>(options.depth_limits.volume) + 2U;
            while (ray_header->size != 0U) {
                if (iteration++ >= maximum_iterations) {
                    return std::unexpected(transport_error(
                        core::StatusCode::internal_error,
                        "CUDA wavefront transport exceeded its depth-derived dispatch bound."));
                }
                const auto ray_count = ray_header->size;
                const auto bounce_range = CudaWavefrontNvtxRange{
                    execution_context.profiling, "bounce", CudaWavefrontNvtxCategory::transport,
                    0xFF90A4AEU, iteration};
                {
                    const auto intersection_range = CudaWavefrontNvtxRange{
                        execution_context.profiling, "intersection",
                        CudaWavefrontNvtxCategory::intersection, 0xFF5C6BC0U, ray_count};
                    report.stage_lanes.intersection += ray_count;
                    if (auto status = execute_stage(
                            WavefrontStageKind::intersection_gather, "intersection gather",
                            ray_count, route_mask(WavefrontStageRoute::ray), device_path_count,
                            outcomes, stage_audit, execution_context,
                            &report.stage_metrics.intersection,
                            [&] {
                                return blackframe_cuda_launch_wavefront_gather_rays(
                                    queue_view, stream_view, ray_count, compact_slots, compact_rays,
                                    outcomes, kernel_stream);
                            });
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (execution_context.profiling != nullptr) {
                        if (auto status =
                                begin_timed_kernel(execution_context.profiling->traversal_timer,
                                                   execution_context, "intersection traversal");
                            !status) {
                            return std::unexpected(std::move(status.error()));
                        }
                    }
                    cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_scene_closest_hit(
                        scene.device_data(), scene.size_bytes(), bvh.device_data(),
                        bvh.size_bytes(), compact_rays, ray_count, closest_results, kernel_stream));
                    if (cuda_status != cudaSuccess) {
                        if (execution_context.profiling != nullptr) {
                            poison_profiling_workspace(execution_context);
                        }
                        return std::unexpected(
                            cuda_transport_error(cuda_status, "closest-hit kernel launch"));
                    }
                    if (execution_context.profiling != nullptr) {
                        if (auto status =
                                end_timed_kernel(execution_context.profiling->traversal_timer,
                                                 execution_context, "intersection traversal");
                            !status) {
                            return std::unexpected(std::move(status.error()));
                        }
                    }
                    // Classification uses the same stream. Its compact stage audit is the
                    // synchronization boundary for both kernels, so an intermediate device-wide
                    // barrier is redundant.
                    auto classification = execute_stage(
                        WavefrontStageKind::intersection_classify, "intersection classify",
                        ray_count,
                        route_mask(WavefrontStageRoute::hit) |
                            route_mask(WavefrontStageRoute::miss),
                        device_path_count, outcomes, stage_audit, execution_context,
                        &report.stage_metrics.intersection, [&] {
                            return blackframe_cuda_launch_wavefront_classify_closest_hit(
                                queue_view, stream_view, compact_slots, closest_results, ray_count,
                                outcomes, kernel_stream);
                        });
                    if (execution_context.profiling != nullptr) {
                        auto traversal_status =
                            collect_timed_kernel(execution_context.profiling->traversal_timer,
                                                 report.stage_metrics.intersection, ray_count,
                                                 execution_context, "intersection traversal");
                        if (!traversal_status) {
                            if (!classification) {
                                auto error = std::move(classification.error());
                                error.message += " Profiling cleanup also failed: " +
                                                 std::move(traversal_status.error().message);
                                return std::unexpected(std::move(error));
                            }
                            return std::unexpected(std::move(traversal_status.error()));
                        }
                    }
                    if (!classification) {
                        return std::unexpected(std::move(classification.error()));
                    }
                    if (auto compacted =
                            compact_route(queue_view, outcomes, ray_count, WavefrontStageRoute::hit,
                                          compaction_scratch, compaction_scratch_bytes,
                                          compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto compacted = compact_route(queue_view, outcomes, ray_count,
                                                       WavefrontStageRoute::miss,
                                                       compaction_scratch, compaction_scratch_bytes,
                                                       compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::ray,
                                                  clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }

                auto miss_header = queue_header(queue_view, renderer::WavefrontQueueKind::miss,
                                                device_capacity, execution_context);
                auto hit_header = queue_header(queue_view, renderer::WavefrontQueueKind::hit,
                                               device_capacity, execution_context);
                if (!miss_header)
                    return std::unexpected(miss_header.error());
                if (!hit_header)
                    return std::unexpected(hit_header.error());
                if (static_cast<std::uint64_t>(miss_header->size) + hit_header->size != ray_count) {
                    return std::unexpected(transport_error(
                        core::StatusCode::internal_error, "CUDA closest-hit classification did not "
                                                          "partition every ray exactly once."));
                }
                report.stage_lanes.miss += miss_header->size;
                report.stage_lanes.hit += hit_header->size;

                if (miss_header->size != 0U) {
                    const auto miss_range = CudaWavefrontNvtxRange{
                        execution_context.profiling, "miss", CudaWavefrontNvtxCategory::miss,
                        0xFFB0BEC5U, miss_header->size};
                    if (auto status = execute_stage(
                            WavefrontStageKind::miss, "miss", miss_header->size,
                            route_mask(WavefrontStageRoute::terminated), device_path_count,
                            outcomes, stage_audit, execution_context, &report.stage_metrics.miss,
                            [&] {
                                return blackframe_cuda_launch_wavefront_miss_stage(
                                    scene.device_data(), scene.size_bytes(), queue_view,
                                    stream_view, miss_header->size, outcomes, kernel_stream);
                            });
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::miss,
                                                  clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }

                if (hit_header->size != 0U) {
                    const auto hit_range = CudaWavefrontNvtxRange{
                        execution_context.profiling, "hit", CudaWavefrontNvtxCategory::hit,
                        0xFF66BB6AU, hit_header->size};
                    if (auto status = execute_stage(
                            WavefrontStageKind::hit, "hit", hit_header->size,
                            route_mask(WavefrontStageRoute::shade), device_path_count, outcomes,
                            stage_audit, execution_context, &report.stage_metrics.hit,
                            [&] {
                                return blackframe_cuda_launch_wavefront_hit_stage(
                                    queue_view, stream_view, hit_header->size, outcomes,
                                    kernel_stream);
                            });
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto compacted = compact_route(queue_view, outcomes, hit_header->size,
                                                       WavefrontStageRoute::shade,
                                                       compaction_scratch, compaction_scratch_bytes,
                                                       compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::hit,
                                                  clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }

                auto shade_header = queue_header(queue_view, renderer::WavefrontQueueKind::shade,
                                                 device_capacity, execution_context);
                if (!shade_header)
                    return std::unexpected(shade_header.error());
                if (shade_header->size != hit_header->size) {
                    return std::unexpected(transport_error(
                        core::StatusCode::internal_error,
                        "CUDA hit stage did not route every surface lane to shading."));
                }
                report.stage_lanes.shade += shade_header->size;
                if (shade_header->size != 0U) {
                    const auto shade_range = CudaWavefrontNvtxRange{
                        execution_context.profiling, "shade", CudaWavefrontNvtxCategory::shade,
                        0xFFFFA726U, shade_header->size};
                    const auto audit = execute_stage(
                        WavefrontStageKind::shade, "shade", shade_header->size,
                        route_mask(WavefrontStageRoute::shadow) |
                            route_mask(WavefrontStageRoute::continuation) |
                            route_mask(WavefrontStageRoute::terminated),
                        device_path_count, outcomes, stage_audit, execution_context,
                        &report.stage_metrics.shade, [&] {
                            return blackframe_cuda_launch_wavefront_shade_stage(
                                scene.device_data(), scene.size_bytes(), queue_view, stream_view,
                                config, shade_header->size, outcomes, kernel_stream);
                        });
                    if (!audit) {
                        return std::unexpected(std::move(audit.error()));
                    }
                    report.closure_samples += audit->closure_samples;
                    report.light_samples += audit->light_samples;
                    if (auto compacted = compact_route(queue_view, outcomes, shade_header->size,
                                                       WavefrontStageRoute::shadow,
                                                       compaction_scratch, compaction_scratch_bytes,
                                                       compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto compacted = compact_route(queue_view, outcomes, shade_header->size,
                                                       WavefrontStageRoute::continuation,
                                                       compaction_scratch, compaction_scratch_bytes,
                                                       compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::shade,
                                                  clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }

                auto shadow_header = queue_header(queue_view, renderer::WavefrontQueueKind::shadow,
                                                  device_capacity, execution_context);
                if (!shadow_header)
                    return std::unexpected(shadow_header.error());
                report.stage_lanes.shadow += shadow_header->size;
                report.shadow_queries += shadow_header->size;
                if (shadow_header->size != 0U) {
                    const auto shadow_range = CudaWavefrontNvtxRange{
                        execution_context.profiling, "shadow", CudaWavefrontNvtxCategory::shadow,
                        0xFF7E57C2U, shadow_header->size};
                    if (auto status = execute_stage(
                            WavefrontStageKind::shadow_gather, "shadow gather", shadow_header->size,
                            route_mask(WavefrontStageRoute::shadow), device_path_count, outcomes,
                            stage_audit, execution_context, &report.stage_metrics.shadow,
                            [&] {
                                return blackframe_cuda_launch_wavefront_gather_shadow_rays(
                                    queue_view, stream_view, shadow_header->size, compact_slots,
                                    compact_rays, outcomes, kernel_stream);
                            });
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (execution_context.profiling != nullptr) {
                        if (auto status =
                                begin_timed_kernel(execution_context.profiling->traversal_timer,
                                                   execution_context, "shadow traversal");
                            !status) {
                            return std::unexpected(std::move(status.error()));
                        }
                    }
                    cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_scene_occlusion(
                        scene.device_data(), scene.size_bytes(), bvh.device_data(),
                        bvh.size_bytes(), compact_rays, shadow_header->size, occlusion_results,
                        kernel_stream));
                    if (cuda_status != cudaSuccess) {
                        if (execution_context.profiling != nullptr) {
                            poison_profiling_workspace(execution_context);
                        }
                        return std::unexpected(
                            cuda_transport_error(cuda_status, "shadow any-hit kernel launch"));
                    }
                    if (execution_context.profiling != nullptr) {
                        if (auto status =
                                end_timed_kernel(execution_context.profiling->traversal_timer,
                                                 execution_context, "shadow traversal");
                            !status) {
                            return std::unexpected(std::move(status.error()));
                        }
                    }
                    // Shadow processing is ordered on the same stream and the following stage audit
                    // provides the required completion/error boundary.
                    auto shadow_process = execute_stage(
                        WavefrontStageKind::shadow_process, "shadow", shadow_header->size,
                        route_mask(WavefrontStageRoute::continuation) |
                            route_mask(WavefrontStageRoute::terminated),
                        device_path_count, outcomes, stage_audit, execution_context,
                        &report.stage_metrics.shadow, [&] {
                            return blackframe_cuda_launch_wavefront_process_shadow(
                                queue_view, stream_view, compact_slots, occlusion_results,
                                shadow_header->size, outcomes, kernel_stream);
                        });
                    if (execution_context.profiling != nullptr) {
                        auto traversal_status =
                            collect_timed_kernel(execution_context.profiling->traversal_timer,
                                                 report.stage_metrics.shadow, shadow_header->size,
                                                 execution_context, "shadow traversal");
                        if (!traversal_status) {
                            if (!shadow_process) {
                                auto error = std::move(shadow_process.error());
                                error.message += " Profiling cleanup also failed: " +
                                                 std::move(traversal_status.error().message);
                                return std::unexpected(std::move(error));
                            }
                            return std::unexpected(std::move(traversal_status.error()));
                        }
                    }
                    if (!shadow_process) {
                        return std::unexpected(std::move(shadow_process.error()));
                    }
                    if (auto compacted = compact_route(queue_view, outcomes, shadow_header->size,
                                                       WavefrontStageRoute::continuation,
                                                       compaction_scratch, compaction_scratch_bytes,
                                                       compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::shadow,
                                                  clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }

                auto continuation_header =
                    queue_header(queue_view, renderer::WavefrontQueueKind::continuation,
                                 device_capacity, execution_context);
                if (!continuation_header)
                    return std::unexpected(continuation_header.error());
                report.stage_lanes.continuation += continuation_header->size;
                if (continuation_header->size != 0U) {
                    const auto continuation_range =
                        CudaWavefrontNvtxRange{execution_context.profiling, "continuation",
                                               CudaWavefrontNvtxCategory::continuation, 0xFF26A69AU,
                                               continuation_header->size};
                    if (auto status = execute_stage(
                            WavefrontStageKind::continuation, "continuation",
                            continuation_header->size, route_mask(WavefrontStageRoute::ray),
                            device_path_count, outcomes, stage_audit, execution_context,
                            &report.stage_metrics.continuation,
                            [&] {
                                return blackframe_cuda_launch_wavefront_continuation_stage(
                                    queue_view, stream_view, continuation_header->size, outcomes,
                                    kernel_stream);
                            });
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto compacted = compact_route(
                            queue_view, outcomes, continuation_header->size,
                            WavefrontStageRoute::ray, compaction_scratch, compaction_scratch_bytes,
                            compaction_result, execution_context);
                        !compacted) {
                        return std::unexpected(std::move(compacted.error()));
                    }
                    if (auto status =
                            clear_queue(queue_view, renderer::WavefrontQueueKind::continuation,
                                        clear_status, execution_context);
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }
                ray_header = queue_header(queue_view, renderer::WavefrontQueueKind::ray,
                                          device_capacity, execution_context);
                if (!ray_header)
                    return std::unexpected(ray_header.error());
            }

            auto* const final_states = storage.final_states_data();
            auto* const final_rays = storage.final_rays_data();
            auto* const final_controls = storage.final_controls_data();
            {
                const auto download_range = CudaWavefrontNvtxRange{
                    execution_context.profiling, "download", CudaWavefrontNvtxCategory::transfer,
                    0xFF78909CU, state_bytes + ray_bytes + control_bytes};
                if (asynchronous != nullptr) {
                    if (auto status = contextual_runtime_status(
                            asynchronous->compute_done->record(*asynchronous->compute_stream),
                            "compute-done event record");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = contextual_runtime_status(
                            asynchronous->transfer_stream->wait(*asynchronous->compute_done),
                            "compute-done event wait");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    ++transfer_counters.event_dependencies;
                    const auto enqueue_download =
                        [&](void* const destination, const void* const source,
                            const std::size_t byte_count,
                            const std::string_view operation) -> core::Status {
                        if (auto status = add_transfer_bytes(transfer_counters.download_bytes,
                                                             byte_count, operation);
                            !status) {
                            return status;
                        }
                        const auto status =
                            cudaMemcpyAsync(destination, source, byte_count, cudaMemcpyDeviceToHost,
                                            transfer_stream);
                        if (status != cudaSuccess) {
                            return std::unexpected(
                                cuda_transport_error(status, operation, byte_count));
                        }
                        return {};
                    };
                    if (auto status = enqueue_download(final_states, stream_states, state_bytes,
                                                       "final state download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = enqueue_download(final_rays, stream_rays, ray_bytes,
                                                       "terminal ray download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = enqueue_download(final_controls, controls, control_bytes,
                                                       "lane-control download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = contextual_runtime_status(
                            asynchronous->downloads_ready->record(*asynchronous->transfer_stream),
                            "download-ready event record");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status =
                            contextual_runtime_status(asynchronous->downloads_ready->synchronize(),
                                                      "download-ready event synchronization");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    async_drain.disarm();
                } else {
                    if (auto status =
                            copy_device_to_host_and_wait(final_states, stream_states, state_bytes,
                                                         execution_context, "final state download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = copy_device_to_host_and_wait(final_rays, stream_rays,
                                                                   ray_bytes, execution_context,
                                                                   "terminal ray download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                    if (auto status = copy_device_to_host_and_wait(final_controls, controls,
                                                                   control_bytes, execution_context,
                                                                   "lane-control download");
                        !status) {
                        return std::unexpected(std::move(status.error()));
                    }
                }
            }

            auto paths = std::vector<CudaWavefrontPathResult>{};
            paths.reserve(path_count);
            for (auto index = std::size_t{}; index < path_count; ++index) {
                auto state = renderer_path_state(final_states[index], index);
                auto ray = renderer_ray(final_rays[index], index);
                auto termination = renderer_termination(final_controls[index], index);
                if (!state)
                    return std::unexpected(std::move(state.error()));
                if (!ray)
                    return std::unexpected(std::move(ray.error()));
                if (!termination)
                    return std::unexpected(std::move(termination.error()));
                if (state->current_medium() != renderer::VacuumMedium ||
                    ray->current_medium() != state->current_medium() ||
                    state->wavelengths() != inputs[index].initial_state.wavelengths()) {
                    return std::unexpected(
                        transport_error(core::StatusCode::internal_error,
                                        "CUDA wavefront path " + std::to_string(index) +
                                            " returned terminal transport identities inconsistent "
                                            "with its input."));
                }
                paths.push_back(CudaWavefrontPathResult{
                    .state = *state,
                    .terminal_ray = *ray,
                    .termination = termination->reason,
                    .blocked_depth_limits = termination->blocked_depth_limits,
                });
            }
            report.terminated_paths = path_count;
            report.asynchronous_upload_bytes = transfer_counters.upload_bytes;
            report.asynchronous_download_bytes = transfer_counters.download_bytes;
            report.cross_stream_event_dependencies = transfer_counters.event_dependencies;
            storage.queues_dirty = false;
            return CudaWavefrontTransportBatch{.paths = std::move(paths), .report = report};
        } catch (const std::bad_alloc&) {
            return std::unexpected(
                transport_error(core::StatusCode::resource_exhausted,
                                "CUDA wavefront transport exhausted host memory during dispatch."));
        } catch (const std::length_error&) {
            return std::unexpected(transport_error(
                core::StatusCode::resource_exhausted,
                "CUDA wavefront transport exceeded a host container length limit during "
                "dispatch."));
        }
    }();
    if (async_drain.armed()) {
        auto drain_status = async_drain.drain();
        if (!drain_status) {
            storage.poisoned = true;
            auto drain_error = std::move(drain_status.error());
            if (!traced) {
                auto trace_error = std::move(traced.error());
                trace_error.message +=
                    " CUDA asynchronous cleanup also failed: " + std::move(drain_error.message);
                return std::unexpected(std::move(trace_error));
            }
            return std::unexpected(
                transport_error(drain_error.code, "CUDA wavefront asynchronous cleanup failed: " +
                                                      std::move(drain_error.message)));
        }
    }
    return traced;
} catch (const std::bad_alloc&) {
    return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront transport exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport exceeded a host container length limit."));
}

core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                               const std::span<const CudaWavefrontPathInput> inputs,
                               const CudaWavefrontLightSampler light_sampler,
                               const CudaWavefrontTransportOptions options) try {
    // Preserve the original contract that rejects the scene and inputs before allocating any
    // per-call transport storage. The workspace overload repeats this inexpensive preflight so it
    // remains independently safe for direct callers.
    if (auto status = cuda_scene_query_detail::validate_binding(scene, bvh, "wavefront transport");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = validate_options_and_inputs(inputs, options); !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto validated_light_sampler = validate_light_sampler(scene.header(), light_sampler);
    if (!validated_light_sampler) {
        return std::unexpected(validated_light_sampler.error());
    }

    auto workspace = CudaWavefrontTransportWorkspace::create(
        inputs.size(), options.device_memory_budget, options.transfer_mode,
        options.instrumentation_mode);
    if (!workspace) {
        return std::unexpected(std::move(workspace.error()));
    }
    auto traced =
        trace_cuda_wavefront_transport(*workspace, scene, bvh, inputs, light_sampler, options);
    auto close_status = workspace->close();
    if (!traced) {
        auto trace_error = std::move(traced.error());
        if (!close_status) {
            trace_error.message += " CUDA wavefront workspace cleanup also failed: " +
                                   std::move(close_status.error().message);
        }
        return std::unexpected(std::move(trace_error));
    }
    if (!close_status) {
        return std::unexpected(std::move(close_status.error()));
    }
    return traced;
} catch (const std::bad_alloc&) {
    return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront transport exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport exceeded a host container length limit."));
}

} // namespace blackframe::engine
