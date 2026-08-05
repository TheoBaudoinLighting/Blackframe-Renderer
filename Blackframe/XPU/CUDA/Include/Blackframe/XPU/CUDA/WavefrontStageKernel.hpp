#pragma once

#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

enum class WavefrontStageStatus : std::uint32_t {
    success = 0U,
    invalid_contract = 1U,
    invalid_path_slot = 2U,
    invalid_lane_state = 3U,
    invalid_ray = 4U,
    invalid_scene = 5U,
    unsupported_transport = 6U,
    numerical_failure = 7U,
    queue_overflow = 8U,
    traversal_error = 9U,
};

enum class WavefrontStageRoute : std::uint32_t {
    none = 0U,
    ray = 1U,
    hit = 2U,
    miss = 3U,
    shade = 4U,
    shadow = 5U,
    continuation = 6U,
    terminated = 7U,
};

enum class WavefrontLanePhase : std::uint32_t {
    empty = 0U,
    camera = 1U,
    ray = 2U,
    hit = 3U,
    miss = 4U,
    shade = 5U,
    shadow = 6U,
    continuation = 7U,
    terminated = 8U,
    processing = 9U,
};

enum class WavefrontTermination : std::uint32_t {
    none = 0U,
    escaped_environment = 1U,
    diffuse_depth_limit = 2U,
    zero_throughput = 3U,
    outside_bsdf_support = 4U,
    russian_roulette = 5U,
};

enum class WavefrontStageKind : std::uint32_t {
    camera_seed = 0U,
    camera = 1U,
    intersection_gather = 2U,
    intersection_classify = 3U,
    hit = 4U,
    miss = 5U,
    shade = 6U,
    shadow_gather = 7U,
    shadow_process = 8U,
    continuation = 9U,
};

inline constexpr std::uint16_t WavefrontTransportConfigAbiMajor = 1U;
inline constexpr std::uint16_t WavefrontTransportConfigAbiMinor = 0U;
inline constexpr std::uint16_t WavefrontStageAuditAbiMajor = 1U;
inline constexpr std::uint16_t WavefrontStageAuditAbiMinor = 0U;
inline constexpr std::uint32_t WavefrontLaneContinuationPending = 1U << 0U;
inline constexpr std::uint32_t WavefrontLaneShadowPending = 1U << 1U;
inline constexpr std::uint32_t WavefrontShadeDetailClosureSampled = 1U << 30U;
inline constexpr std::uint32_t WavefrontShadeDetailLightSampled = 1U << 31U;

// Backend-local transport configuration copied by value into the shading launch. Numeric enum
// values are validated and translated by the host before launch; device code never depends on
// host-only renderer layouts.
struct alignas(16) WavefrontTransportConfig final {
    std::uint16_t abi_major{};
    std::uint16_t abi_minor{};
    std::uint32_t struct_size{};
    std::uint32_t mis_heuristic{};
    std::uint32_t light_sampling_strategy{};
    std::uint32_t light_count{};
    std::uint32_t diffuse_depth_limit{};
    std::uint32_t glossy_depth_limit{};
    std::uint32_t specular_depth_limit{};
    std::uint32_t transmission_depth_limit{};
    std::uint32_t volume_depth_limit{};
    std::uint32_t russian_roulette_mode{};
    std::uint32_t russian_roulette_first_depth{};
    float russian_roulette_minimum_probability{};
    float russian_roulette_maximum_probability{};
    std::uint32_t reserved[2U]{};
};

struct alignas(16) WavefrontStageOutcome final {
    std::uint32_t status{};
    std::uint32_t route{};
    std::uint32_t path_slot{};
    std::uint32_t detail{};
};

// Fixed-size summary produced on the device after a stage. The first failure is the outcome with
// the smallest failing work-item index, matching the deterministic order of the former host scan.
// UINT32_MAX denotes a successful stage. Per-stage sample counts remain 32-bit because work_count
// is itself bounded to the same domain; the host report accumulates them in 64-bit counters.
struct alignas(16) WavefrontStageAudit final {
    std::uint16_t abi_major{};
    std::uint16_t abi_minor{};
    std::uint32_t struct_size{};
    std::uint32_t stage_kind{};
    std::uint32_t expected_work_count{};
    std::uint32_t inspected_work_count{};
    std::uint32_t first_failure_work_index{};
    std::uint32_t closure_samples{};
    std::uint32_t light_samples{};
    WavefrontStageOutcome first_failure{};
    std::uint32_t reserved[4U]{};
};

struct alignas(16) WavefrontLaneControl final {
    std::uint32_t phase{};
    std::uint32_t termination{};
    std::uint32_t flags{};
    std::uint32_t blocked_depth_limits{};
};

struct alignas(16) WavefrontPendingShadow final {
    shared::TransportRay ray{};
    shared::TransportSpectrum beta{};
    shared::TransportSpectrum reflectance{};
    shared::TransportSpectrum incident_radiance{};
    float receiver_cosine{};
    float estimator_weight{};
    float selection_probability{};
    float conditional_probability{};
    std::uint32_t continuation_pending{};
    std::uint32_t termination{};
    std::uint32_t reserved[2U]{};
};

// The previous continuous BSDF sample is backend-local MIS state. It deliberately stays outside
// the shared PathState ABI because public paths cannot be resumed without this vertex context.
struct alignas(16) WavefrontPreviousBsdfSample final {
    float context_x{};
    float context_y{};
    float context_z{};
    float context_time{};
    float incoming_x{};
    float incoming_y{};
    float incoming_z{};
    float probability_value{};
    std::uint32_t probability_measure{};
    std::uint32_t valid{};
    std::uint32_t reserved[2U]{};
};

// Backend-local device views. Every pointer names an independent array on the active CUDA device.
// Persistent stream columns are indexed by PathSlot, while compact traversal arrays are indexed by
// work item. No pointer may alias an outcome or compact traversal range used by the same launch.
struct alignas(16) WavefrontStageDeviceSoa final {
    shared::SampleStreamIndex* sample_streams{};
    shared::TransportRay* rays{};
    shared::TransportPathStateLane* path_states{};
    shared::ClosestHit* hits{};
    WavefrontPendingShadow* pending_shadows{};
    WavefrontPreviousBsdfSample* previous_bsdf_samples{};
    WavefrontLaneControl* controls{};
    std::uint32_t capacity{};
    std::uint32_t reserved{};
};

struct alignas(16) WavefrontCameraInputDeviceSoa final {
    const shared::SampleStreamIndex* sample_streams{};
    const shared::TransportRay* rays{};
    const shared::TransportPathStateLane* path_states{};
    std::uint32_t count{};
    std::uint32_t reserved{};
};

#define BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(record)                                                \
    static_assert(std::is_standard_layout_v<record>);                                              \
    static_assert(std::is_trivially_copyable_v<record>);                                           \
    static_assert(std::is_trivially_destructible_v<record>)

BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontStageOutcome);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontStageAudit);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontTransportConfig);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontLaneControl);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontPendingShadow);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontPreviousBsdfSample);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontStageDeviceSoa);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontCameraInputDeviceSoa);

#undef BLACKFRAME_ASSERT_CUDA_STAGE_RECORD

static_assert(sizeof(WavefrontStageStatus) == 4U);
static_assert(sizeof(WavefrontStageRoute) == 4U);
static_assert(sizeof(WavefrontLanePhase) == 4U);
static_assert(sizeof(WavefrontTermination) == 4U);
static_assert(sizeof(WavefrontStageKind) == 4U);
static_assert(sizeof(WavefrontTransportConfig) == 64U);
static_assert(alignof(WavefrontTransportConfig) == 16U);
static_assert(offsetof(WavefrontTransportConfig, abi_major) == 0U);
static_assert(offsetof(WavefrontTransportConfig, abi_minor) == 2U);
static_assert(offsetof(WavefrontTransportConfig, struct_size) == 4U);
static_assert(offsetof(WavefrontTransportConfig, mis_heuristic) == 8U);
static_assert(offsetof(WavefrontTransportConfig, light_sampling_strategy) == 12U);
static_assert(offsetof(WavefrontTransportConfig, light_count) == 16U);
static_assert(offsetof(WavefrontTransportConfig, diffuse_depth_limit) == 20U);
static_assert(offsetof(WavefrontTransportConfig, glossy_depth_limit) == 24U);
static_assert(offsetof(WavefrontTransportConfig, specular_depth_limit) == 28U);
static_assert(offsetof(WavefrontTransportConfig, transmission_depth_limit) == 32U);
static_assert(offsetof(WavefrontTransportConfig, volume_depth_limit) == 36U);
static_assert(offsetof(WavefrontTransportConfig, russian_roulette_mode) == 40U);
static_assert(offsetof(WavefrontTransportConfig, russian_roulette_first_depth) == 44U);
static_assert(offsetof(WavefrontTransportConfig, russian_roulette_minimum_probability) == 48U);
static_assert(offsetof(WavefrontTransportConfig, russian_roulette_maximum_probability) == 52U);
static_assert(offsetof(WavefrontTransportConfig, reserved) == 56U);
static_assert(sizeof(WavefrontStageOutcome) == 16U);
static_assert(alignof(WavefrontStageOutcome) == 16U);
static_assert(offsetof(WavefrontStageOutcome, status) == 0U);
static_assert(offsetof(WavefrontStageOutcome, route) == 4U);
static_assert(offsetof(WavefrontStageOutcome, path_slot) == 8U);
static_assert(offsetof(WavefrontStageOutcome, detail) == 12U);
static_assert(sizeof(WavefrontStageAudit) == 64U);
static_assert(alignof(WavefrontStageAudit) == 16U);
static_assert(offsetof(WavefrontStageAudit, abi_major) == 0U);
static_assert(offsetof(WavefrontStageAudit, abi_minor) == 2U);
static_assert(offsetof(WavefrontStageAudit, struct_size) == 4U);
static_assert(offsetof(WavefrontStageAudit, stage_kind) == 8U);
static_assert(offsetof(WavefrontStageAudit, expected_work_count) == 12U);
static_assert(offsetof(WavefrontStageAudit, inspected_work_count) == 16U);
static_assert(offsetof(WavefrontStageAudit, first_failure_work_index) == 20U);
static_assert(offsetof(WavefrontStageAudit, closure_samples) == 24U);
static_assert(offsetof(WavefrontStageAudit, light_samples) == 28U);
static_assert(offsetof(WavefrontStageAudit, first_failure) == 32U);
static_assert(offsetof(WavefrontStageAudit, reserved) == 48U);
static_assert(sizeof(WavefrontLaneControl) == 16U);
static_assert(alignof(WavefrontLaneControl) == 16U);
static_assert(offsetof(WavefrontLaneControl, phase) == 0U);
static_assert(offsetof(WavefrontLaneControl, termination) == 4U);
static_assert(offsetof(WavefrontLaneControl, flags) == 8U);
static_assert(offsetof(WavefrontLaneControl, blocked_depth_limits) == 12U);
static_assert(sizeof(WavefrontPendingShadow) == 128U);
static_assert(alignof(WavefrontPendingShadow) == 16U);
static_assert(offsetof(WavefrontPendingShadow, ray) == 0U);
static_assert(offsetof(WavefrontPendingShadow, beta) == 48U);
static_assert(offsetof(WavefrontPendingShadow, reflectance) == 64U);
static_assert(offsetof(WavefrontPendingShadow, incident_radiance) == 80U);
static_assert(offsetof(WavefrontPendingShadow, receiver_cosine) == 96U);
static_assert(offsetof(WavefrontPendingShadow, estimator_weight) == 100U);
static_assert(offsetof(WavefrontPendingShadow, selection_probability) == 104U);
static_assert(offsetof(WavefrontPendingShadow, conditional_probability) == 108U);
static_assert(offsetof(WavefrontPendingShadow, continuation_pending) == 112U);
static_assert(offsetof(WavefrontPendingShadow, termination) == 116U);
static_assert(offsetof(WavefrontPendingShadow, reserved) == 120U);
static_assert(sizeof(WavefrontPreviousBsdfSample) == 48U);
static_assert(alignof(WavefrontPreviousBsdfSample) == 16U);
static_assert(offsetof(WavefrontPreviousBsdfSample, context_x) == 0U);
static_assert(offsetof(WavefrontPreviousBsdfSample, context_y) == 4U);
static_assert(offsetof(WavefrontPreviousBsdfSample, context_z) == 8U);
static_assert(offsetof(WavefrontPreviousBsdfSample, context_time) == 12U);
static_assert(offsetof(WavefrontPreviousBsdfSample, incoming_x) == 16U);
static_assert(offsetof(WavefrontPreviousBsdfSample, incoming_y) == 20U);
static_assert(offsetof(WavefrontPreviousBsdfSample, incoming_z) == 24U);
static_assert(offsetof(WavefrontPreviousBsdfSample, probability_value) == 28U);
static_assert(offsetof(WavefrontPreviousBsdfSample, probability_measure) == 32U);
static_assert(offsetof(WavefrontPreviousBsdfSample, valid) == 36U);
static_assert(offsetof(WavefrontPreviousBsdfSample, reserved) == 40U);
static_assert(sizeof(void*) == 8U);
static_assert(sizeof(WavefrontStageDeviceSoa) == 64U);
static_assert(alignof(WavefrontStageDeviceSoa) == 16U);
static_assert(offsetof(WavefrontStageDeviceSoa, sample_streams) == 0U);
static_assert(offsetof(WavefrontStageDeviceSoa, rays) == 8U);
static_assert(offsetof(WavefrontStageDeviceSoa, path_states) == 16U);
static_assert(offsetof(WavefrontStageDeviceSoa, hits) == 24U);
static_assert(offsetof(WavefrontStageDeviceSoa, pending_shadows) == 32U);
static_assert(offsetof(WavefrontStageDeviceSoa, previous_bsdf_samples) == 40U);
static_assert(offsetof(WavefrontStageDeviceSoa, controls) == 48U);
static_assert(offsetof(WavefrontStageDeviceSoa, capacity) == 56U);
static_assert(offsetof(WavefrontStageDeviceSoa, reserved) == 60U);
static_assert(sizeof(WavefrontCameraInputDeviceSoa) == 32U);
static_assert(alignof(WavefrontCameraInputDeviceSoa) == 16U);
static_assert(offsetof(WavefrontCameraInputDeviceSoa, sample_streams) == 0U);
static_assert(offsetof(WavefrontCameraInputDeviceSoa, rays) == 8U);
static_assert(offsetof(WavefrontCameraInputDeviceSoa, path_states) == 16U);
static_assert(offsetof(WavefrontCameraInputDeviceSoa, count) == 24U);
static_assert(offsetof(WavefrontCameraInputDeviceSoa, reserved) == 28U);

} // namespace blackframe::xpu::cuda

extern "C" int blackframe_cuda_launch_wavefront_seed_camera(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t first_path_slot,
    std::uint32_t path_count, blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_clear_queue(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues, std::uint32_t queue_kind,
    std::uint32_t acknowledge_overflow, std::uint32_t* device_status) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_camera_stage(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontCameraInputDeviceSoa inputs,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_gather_rays(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::shared::PathSlot* compact_path_slots,
    blackframe::xpu::shared::TransportRay* compact_rays,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_classify_closest_hit(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams,
    const blackframe::xpu::shared::PathSlot* compact_path_slots,
    const blackframe::xpu::shared::SceneClosestHitResult* compact_results, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_hit_stage(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_miss_stage(
    const std::uint8_t* scene_bytes, std::size_t scene_size,
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_shade_stage(
    const std::uint8_t* scene_bytes, std::size_t scene_size,
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams,
    blackframe::xpu::cuda::WavefrontTransportConfig config, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_gather_shadow_rays(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::shared::PathSlot* compact_path_slots,
    blackframe::xpu::shared::TransportRay* compact_rays,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_process_shadow(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams,
    const blackframe::xpu::shared::PathSlot* compact_path_slots,
    const blackframe::xpu::shared::SceneOcclusionResult* compact_results, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_continuation_stage(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t work_count,
    blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

extern "C" int blackframe_cuda_launch_wavefront_audit_stage(
    const blackframe::xpu::cuda::WavefrontStageOutcome* outcomes, std::uint32_t work_count,
    std::uint32_t allowed_route_mask, std::uint32_t path_capacity, std::uint32_t stage_kind,
    blackframe::xpu::cuda::WavefrontStageAudit* audit) noexcept;
