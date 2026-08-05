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
};

inline constexpr std::uint32_t WavefrontLaneContinuationPending = 1U << 0U;
inline constexpr std::uint32_t WavefrontLaneShadowPending = 1U << 1U;
inline constexpr std::uint32_t WavefrontShadeDetailLightSampled = 1U << 31U;

struct alignas(16) WavefrontStageOutcome final {
    std::uint32_t status{};
    std::uint32_t route{};
    std::uint32_t path_slot{};
    std::uint32_t detail{};
};

struct alignas(16) WavefrontLaneControl final {
    std::uint32_t phase{};
    std::uint32_t termination{};
    std::uint32_t flags{};
    std::uint32_t reserved{};
};

struct alignas(16) WavefrontPendingShadow final {
    shared::TransportRay ray{};
    shared::TransportSpectrum visible_contribution{};
    std::uint32_t continuation_pending{};
    std::uint32_t termination{};
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
    WavefrontLaneControl* controls{};
    std::uint32_t capacity{};
    std::uint32_t reserved[3U]{};
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
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontLaneControl);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontPendingShadow);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontStageDeviceSoa);
BLACKFRAME_ASSERT_CUDA_STAGE_RECORD(WavefrontCameraInputDeviceSoa);

#undef BLACKFRAME_ASSERT_CUDA_STAGE_RECORD

static_assert(sizeof(WavefrontStageStatus) == 4U);
static_assert(sizeof(WavefrontStageRoute) == 4U);
static_assert(sizeof(WavefrontLanePhase) == 4U);
static_assert(sizeof(WavefrontTermination) == 4U);
static_assert(sizeof(WavefrontStageOutcome) == 16U);
static_assert(alignof(WavefrontStageOutcome) == 16U);
static_assert(offsetof(WavefrontStageOutcome, status) == 0U);
static_assert(offsetof(WavefrontStageOutcome, route) == 4U);
static_assert(offsetof(WavefrontStageOutcome, path_slot) == 8U);
static_assert(offsetof(WavefrontStageOutcome, detail) == 12U);
static_assert(sizeof(WavefrontLaneControl) == 16U);
static_assert(alignof(WavefrontLaneControl) == 16U);
static_assert(offsetof(WavefrontLaneControl, phase) == 0U);
static_assert(offsetof(WavefrontLaneControl, termination) == 4U);
static_assert(offsetof(WavefrontLaneControl, flags) == 8U);
static_assert(offsetof(WavefrontLaneControl, reserved) == 12U);
static_assert(sizeof(WavefrontPendingShadow) == 80U);
static_assert(alignof(WavefrontPendingShadow) == 16U);
static_assert(offsetof(WavefrontPendingShadow, ray) == 0U);
static_assert(offsetof(WavefrontPendingShadow, visible_contribution) == 48U);
static_assert(offsetof(WavefrontPendingShadow, continuation_pending) == 64U);
static_assert(offsetof(WavefrontPendingShadow, termination) == 68U);
static_assert(offsetof(WavefrontPendingShadow, reserved) == 72U);
static_assert(sizeof(void*) == 8U);
static_assert(sizeof(WavefrontStageDeviceSoa) == 64U);
static_assert(alignof(WavefrontStageDeviceSoa) == 16U);
static_assert(offsetof(WavefrontStageDeviceSoa, sample_streams) == 0U);
static_assert(offsetof(WavefrontStageDeviceSoa, rays) == 8U);
static_assert(offsetof(WavefrontStageDeviceSoa, path_states) == 16U);
static_assert(offsetof(WavefrontStageDeviceSoa, hits) == 24U);
static_assert(offsetof(WavefrontStageDeviceSoa, pending_shadows) == 32U);
static_assert(offsetof(WavefrontStageDeviceSoa, controls) == 40U);
static_assert(offsetof(WavefrontStageDeviceSoa, capacity) == 48U);
static_assert(offsetof(WavefrontStageDeviceSoa, reserved) == 52U);
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
    blackframe::xpu::cuda::WavefrontStageDeviceSoa streams, std::uint32_t max_diffuse_depth,
    std::uint32_t work_count, blackframe::xpu::cuda::WavefrontStageOutcome* outcomes) noexcept;

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
