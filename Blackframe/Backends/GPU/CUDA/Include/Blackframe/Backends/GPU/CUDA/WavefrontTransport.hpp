#pragma once

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/RayCone.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

inline constexpr std::uint32_t CurrentCudaWavefrontTransportReportSchemaVersion = 4U;

struct CudaWavefrontPathInput final {
    renderer::Ray primary_ray;
    renderer::RayCone primary_cone;
    renderer::PathState initial_state;
    renderer::SampleStreamIndex sample;
};

enum class CudaWavefrontPathTermination : std::uint8_t {
    escaped_environment = 0U,
    depth_limit = 1U,
    zero_throughput = 2U,
    outside_bsdf_support = 3U,
    russian_roulette = 4U,
};

enum class CudaWavefrontTransferMode : std::uint8_t {
    synchronous = 0U,
    asynchronous = 1U,
};

enum class CudaWavefrontInstrumentationMode : std::uint8_t {
    disabled = 0U,
    nsight = 1U,
};

struct CudaWavefrontPathResult final {
    renderer::PathState state;
    renderer::Ray terminal_ray;
    CudaWavefrontPathTermination termination{};
    renderer::ScatteringLobe blocked_depth_limits{renderer::ScatteringLobe::none};
};

struct CudaWavefrontStageLaneCounts final {
    std::uint64_t camera{};
    std::uint64_t intersection{};
    std::uint64_t hit{};
    std::uint64_t miss{};
    std::uint64_t shade{};
    std::uint64_t shadow{};
    std::uint64_t continuation{};

    [[nodiscard]] constexpr bool
    operator==(const CudaWavefrontStageLaneCounts&) const noexcept = default;
};

struct CudaWavefrontStageMetric final {
    std::uint64_t kernel_dispatches{};
    std::uint64_t kernel_lanes{};
    std::uint64_t gpu_elapsed_nanoseconds{};

    // GPU duration is observational and naturally varies between identical replays. Dispatch and
    // lane counts remain part of deterministic report equality.
    [[nodiscard]] constexpr bool operator==(const CudaWavefrontStageMetric& other) const noexcept {
        return kernel_dispatches == other.kernel_dispatches && kernel_lanes == other.kernel_lanes;
    }
};

struct CudaWavefrontStageMetrics final {
    CudaWavefrontStageMetric camera{};
    CudaWavefrontStageMetric intersection{};
    CudaWavefrontStageMetric hit{};
    CudaWavefrontStageMetric miss{};
    CudaWavefrontStageMetric shade{};
    CudaWavefrontStageMetric shadow{};
    CudaWavefrontStageMetric continuation{};

    [[nodiscard]] constexpr bool
    operator==(const CudaWavefrontStageMetrics&) const noexcept = default;
};

struct CudaWavefrontTransportReport final {
    std::uint32_t schema_version{};
    bool has_light_sampler{};
    std::uint32_t registered_light_count{};
    renderer::MisHeuristic heuristic{renderer::MisHeuristic::power};
    renderer::LightSamplingStrategy light_sampling_strategy{
        renderer::LightSamplingStrategy::uniform};
    renderer::PathDepthLimits depth_limits{};
    renderer::RussianRoulettePolicy roulette_policy{renderer::RussianRoulettePolicy::disabled()};
    std::size_t path_count{};
    CudaWavefrontTransferMode transfer_mode{CudaWavefrontTransferMode::synchronous};
    CudaWavefrontInstrumentationMode instrumentation_mode{
        CudaWavefrontInstrumentationMode::disabled};
    std::uint64_t asynchronous_upload_bytes{};
    std::uint64_t asynchronous_download_bytes{};
    // Counts explicit event waits that establish dependencies between distinct CUDA streams.
    std::uint64_t cross_stream_event_dependencies{};
    CudaWavefrontStageLaneCounts stage_lanes{};
    CudaWavefrontStageMetrics stage_metrics{};
    std::uint64_t closure_samples{};
    std::uint64_t light_samples{};
    std::uint64_t shadow_queries{};
    std::uint64_t terminated_paths{};
    std::uint64_t queue_overflow_attempts{};
    std::uint64_t queue_rejected_lanes{};

    [[nodiscard]] constexpr bool
    operator==(const CudaWavefrontTransportReport&) const noexcept = default;
};

struct CudaWavefrontTransportBatch final {
    std::vector<CudaWavefrontPathResult> paths;
    // Parallel to paths. Each cone is defined at the origin of its path's terminal_ray.
    std::vector<renderer::RayCone> terminal_cones;
    CudaWavefrontTransportReport report;
};

struct CudaWavefrontTransportOptions final {
    renderer::MisHeuristic heuristic{renderer::MisHeuristic::power};
    renderer::PathDepthLimits depth_limits{.diffuse = 5U};
    renderer::RussianRoulettePolicy roulette_policy{renderer::RussianRoulettePolicy::disabled()};
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
    CudaWavefrontTransferMode transfer_mode{CudaWavefrontTransferMode::synchronous};
    CudaWavefrontInstrumentationMode instrumentation_mode{
        CudaWavefrontInstrumentationMode::disabled};
};

using CudaWavefrontLightSampler =
    std::optional<std::reference_wrapper<const renderer::LightSampler>>;

// Owns fixed-capacity CUDA transport queues, scratch columns, host staging buffers, and any stream
// resources required by its fixed transfer mode. A workspace is not thread-safe: callers must not
// use the same instance concurrently. The active path count may vary up to capacity without
// reallocating. Explicit close reports teardown errors; the destructor remains the no-throw
// ownership backstop.
class CudaWavefrontTransportWorkspace final {
  public:
    CudaWavefrontTransportWorkspace(const CudaWavefrontTransportWorkspace&) = delete;
    CudaWavefrontTransportWorkspace& operator=(const CudaWavefrontTransportWorkspace&) = delete;
    CudaWavefrontTransportWorkspace(CudaWavefrontTransportWorkspace&& other) noexcept;
    CudaWavefrontTransportWorkspace& operator=(CudaWavefrontTransportWorkspace&& other) = delete;
    ~CudaWavefrontTransportWorkspace() noexcept;

    [[nodiscard]] static core::Result<CudaWavefrontTransportWorkspace>
    create(std::size_t capacity, xpu::cuda::DeviceMemoryBudget device_memory_budget = {},
           CudaWavefrontTransferMode transfer_mode = CudaWavefrontTransferMode::synchronous,
           CudaWavefrontInstrumentationMode instrumentation_mode =
               CudaWavefrontInstrumentationMode::disabled);

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t device_size_bytes() const noexcept;
    [[nodiscard]] std::int32_t device_ordinal() const noexcept;
    [[nodiscard]] CudaWavefrontTransferMode transfer_mode() const noexcept;
    [[nodiscard]] CudaWavefrontInstrumentationMode instrumentation_mode() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] core::Status close();

  private:
    struct Storage;

    explicit CudaWavefrontTransportWorkspace(std::unique_ptr<Storage> storage) noexcept;

    std::unique_ptr<Storage> storage_{};

    friend core::Result<CudaWavefrontTransportBatch> trace_cuda_wavefront_transport(
        CudaWavefrontTransportWorkspace& workspace, const CudaSceneSoA& scene,
        const CudaSceneBvh& bvh, std::span<const CudaWavefrontPathInput> inputs,
        CudaWavefrontLightSampler light_sampler, CudaWavefrontTransportOptions options);
};

// Executes the currently representable FrameScene transport with explicit CUDA kernels and the
// serialized CUDA BLAS/TLAS. Inputs are primary rays, their explicit ray cones, and indexed sample
// addresses generated by the renderer contracts. The camera stage initializes device path streams
// from those immutable inputs; every later physical stage remains on the selected CUDA device. The
// initial CUDA port is
// deliberately limited to vacuum, bounded spectral closure mixtures, one-sided emission, the
// current punctual and mesh-area light registry, an explicitly uniform light sampler, balance or
// power MIS, and a constant environment. The sampler must be absent for an empty registry and
// present for a non-empty registry. Unsupported scene or path capabilities are rejected explicitly
// and are never substituted by a CPU backend.
[[nodiscard]] core::Result<CudaWavefrontTransportBatch> trace_cuda_wavefront_transport(
    CudaWavefrontTransportWorkspace& workspace, const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
    std::span<const CudaWavefrontPathInput> inputs, CudaWavefrontLightSampler light_sampler,
    CudaWavefrontTransportOptions options = {});

// Convenience overload retaining the original allocate-per-call behavior. Repeated batches should
// provide an explicit workspace to keep device and staging allocations resident.
[[nodiscard]] core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                               std::span<const CudaWavefrontPathInput> inputs,
                               CudaWavefrontLightSampler light_sampler,
                               CudaWavefrontTransportOptions options = {});

static_assert(std::is_standard_layout_v<CudaWavefrontPathInput>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontPathInput>);
static_assert(sizeof(CudaWavefrontPathTermination) == sizeof(std::uint8_t));
static_assert(sizeof(CudaWavefrontTransferMode) == sizeof(std::uint8_t));
static_assert(sizeof(CudaWavefrontInstrumentationMode) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<CudaWavefrontStageLaneCounts>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontStageLaneCounts>);
static_assert(std::is_standard_layout_v<CudaWavefrontStageMetric>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontStageMetric>);
static_assert(std::is_standard_layout_v<CudaWavefrontStageMetrics>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontStageMetrics>);
static_assert(std::is_standard_layout_v<CudaWavefrontTransportReport>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontTransportReport>);
static_assert(std::is_standard_layout_v<CudaWavefrontTransportOptions>);
static_assert(std::is_trivially_copyable_v<CudaWavefrontTransportOptions>);
static_assert(!std::is_copy_constructible_v<CudaWavefrontTransportWorkspace>);
static_assert(!std::is_copy_assignable_v<CudaWavefrontTransportWorkspace>);
static_assert(std::is_nothrow_move_constructible_v<CudaWavefrontTransportWorkspace>);
static_assert(!std::is_move_assignable_v<CudaWavefrontTransportWorkspace>);
static_assert(std::is_nothrow_destructible_v<CudaWavefrontTransportWorkspace>);

} // namespace blackframe::engine
