#include <Blackframe/Core/Status.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::xpu::cuda {
namespace {

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

[[nodiscard]] core::Error audit_error(const cudaError_t status, const char* const operation) {
    return core::Error{
        .code = cuda_memory_status_code(static_cast<std::int32_t>(status)),
        .message = std::string{"CUDA wavefront stage audit "} + operation + " failed: " +
                   cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ").",
    };
}

[[nodiscard]] constexpr std::uint32_t route_mask(const WavefrontStageRoute route) noexcept {
    return std::uint32_t{1U} << static_cast<std::uint32_t>(route);
}

[[nodiscard]] constexpr WavefrontStageOutcome
successful_outcome(const WavefrontStageRoute route, const std::uint32_t path_slot,
                   const std::uint32_t detail = 0U) noexcept {
    return WavefrontStageOutcome{
        .status = static_cast<std::uint32_t>(WavefrontStageStatus::success),
        .route = static_cast<std::uint32_t>(route),
        .path_slot = path_slot,
        .detail = detail,
    };
}

[[nodiscard]] core::Result<WavefrontStageAudit>
run_audit(const std::span<const WavefrontStageOutcome> outcomes,
          const std::uint32_t allowed_route_mask, const std::uint32_t path_capacity,
          const WavefrontStageKind stage_kind) {
    if (outcomes.empty() || outcomes.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The CUDA wavefront stage audit test requires a non-empty 32-bit batch.",
        });
    }

    auto device_outcomes_result = DeviceBuffer<WavefrontStageOutcome>::allocate(outcomes.size());
    if (!device_outcomes_result) {
        return std::unexpected(std::move(device_outcomes_result.error()));
    }
    auto device_outcomes = std::move(*device_outcomes_result);
    auto device_audit_result = DeviceBuffer<WavefrontStageAudit>::allocate(1U);
    if (!device_audit_result) {
        return std::unexpected(std::move(device_audit_result.error()));
    }
    auto device_audit = std::move(*device_audit_result);

    auto status = cudaMemcpy(device_outcomes.data(), outcomes.data(), outcomes.size_bytes(),
                             cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return std::unexpected(audit_error(status, "outcome upload"));
    }
    status = static_cast<cudaError_t>(blackframe_cuda_launch_wavefront_audit_stage(
        device_outcomes.data(), static_cast<std::uint32_t>(outcomes.size()), allowed_route_mask,
        path_capacity, static_cast<std::uint32_t>(stage_kind), device_audit.data()));
    if (status != cudaSuccess) {
        return std::unexpected(audit_error(status, "launch"));
    }

    auto audit = WavefrontStageAudit{};
    status = cudaMemcpy(&audit, device_audit.data(), sizeof(audit), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return std::unexpected(audit_error(status, "download"));
    }

    auto audit_close = device_audit.close();
    auto outcomes_close = device_outcomes.close();
    if (!audit_close) {
        return std::unexpected(std::move(audit_close.error()));
    }
    if (!outcomes_close) {
        return std::unexpected(std::move(outcomes_close.error()));
    }
    return audit;
}

void expect_outcome(const WavefrontStageOutcome& actual, const WavefrontStageOutcome& expected) {
    EXPECT_EQ(actual.status, expected.status);
    EXPECT_EQ(actual.route, expected.route);
    EXPECT_EQ(actual.path_slot, expected.path_slot);
    EXPECT_EQ(actual.detail, expected.detail);
}

void expect_canonical_header(const WavefrontStageAudit& audit, const WavefrontStageKind stage_kind,
                             const std::uint32_t work_count) {
    EXPECT_EQ(audit.abi_major, WavefrontStageAuditAbiMajor);
    EXPECT_EQ(audit.abi_minor, WavefrontStageAuditAbiMinor);
    EXPECT_EQ(audit.struct_size, sizeof(WavefrontStageAudit));
    EXPECT_EQ(audit.stage_kind, static_cast<std::uint32_t>(stage_kind));
    EXPECT_EQ(audit.expected_work_count, work_count);
    EXPECT_EQ(audit.inspected_work_count, work_count);
    EXPECT_EQ(audit.reserved[0U], 0U);
    EXPECT_EQ(audit.reserved[1U], 0U);
    EXPECT_EQ(audit.reserved[2U], 0U);
    EXPECT_EQ(audit.reserved[3U], 0U);
}

class CudaWavefrontStageAuditTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_TRUE(select_test_device());
    }
};

template <typename Value> [[nodiscard]] Value* non_null_device_address() noexcept {
    return reinterpret_cast<Value*>(static_cast<std::uintptr_t>(alignof(Value)));
}

[[nodiscard]] WavefrontQueueDeviceSoa camera_validation_queues() noexcept {
    return WavefrontQueueDeviceSoa{
        .headers = non_null_device_address<shared::QueueHeader>(),
        .path_slots = non_null_device_address<shared::PathSlot>(),
        .queue_count = CudaWavefrontQueueCount,
        .slot_stride = 1U,
    };
}

[[nodiscard]] WavefrontStageDeviceSoa camera_validation_streams() noexcept {
    return WavefrontStageDeviceSoa{
        .sample_streams = non_null_device_address<shared::SampleStreamIndex>(),
        .rays = non_null_device_address<shared::TransportRay>(),
        .ray_cones = non_null_device_address<WavefrontRayCone>(),
        .path_states = non_null_device_address<shared::TransportPathStateLane>(),
        .hits = non_null_device_address<shared::ClosestHit>(),
        .pending_shadows = non_null_device_address<WavefrontPendingShadow>(),
        .previous_bsdf_samples = non_null_device_address<WavefrontPreviousBsdfSample>(),
        .controls = non_null_device_address<WavefrontLaneControl>(),
        .capacity = 1U,
        .reserved = 0U,
        .reserved_tail = {0U, 0U},
    };
}

[[nodiscard]] WavefrontCameraInputDeviceSoa camera_validation_inputs() noexcept {
    return WavefrontCameraInputDeviceSoa{
        .sample_streams = non_null_device_address<shared::SampleStreamIndex>(),
        .rays = non_null_device_address<shared::TransportRay>(),
        .ray_cones = non_null_device_address<WavefrontRayCone>(),
        .path_states = non_null_device_address<shared::TransportPathStateLane>(),
        .count = 1U,
        .reserved = 0U,
        .reserved_tail = {0U, 0U},
    };
}

TEST_F(CudaWavefrontStageAuditTest, CameraLauncherRejectsMissingRayConeColumns) {
    const auto queues = camera_validation_queues();
    const auto valid_streams = camera_validation_streams();
    const auto valid_inputs = camera_validation_inputs();
    EXPECT_EQ(blackframe_cuda_launch_wavefront_camera_stage(queues, valid_inputs, valid_streams, 0U,
                                                            nullptr, nullptr),
              static_cast<int>(cudaSuccess));

    auto missing_stream_cones = valid_streams;
    missing_stream_cones.ray_cones = nullptr;
    EXPECT_EQ(blackframe_cuda_launch_wavefront_camera_stage(
                  queues, valid_inputs, missing_stream_cones, 0U, nullptr, nullptr),
              static_cast<int>(cudaErrorInvalidValue));

    auto missing_input_cones = valid_inputs;
    missing_input_cones.ray_cones = nullptr;
    EXPECT_EQ(blackframe_cuda_launch_wavefront_camera_stage(queues, missing_input_cones,
                                                            valid_streams, 0U, nullptr, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

TEST_F(CudaWavefrontStageAuditTest, CameraLauncherRejectsNonCanonicalReservedTails) {
    const auto queues = camera_validation_queues();
    const auto valid_streams = camera_validation_streams();
    const auto valid_inputs = camera_validation_inputs();

    auto noncanonical_streams = valid_streams;
    noncanonical_streams.reserved_tail[1U] = 1U;
    EXPECT_EQ(blackframe_cuda_launch_wavefront_camera_stage(
                  queues, valid_inputs, noncanonical_streams, 0U, nullptr, nullptr),
              static_cast<int>(cudaErrorInvalidValue));

    auto noncanonical_inputs = valid_inputs;
    noncanonical_inputs.reserved_tail[0U] = 1U;
    EXPECT_EQ(blackframe_cuda_launch_wavefront_camera_stage(queues, noncanonical_inputs,
                                                            valid_streams, 0U, nullptr, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

TEST_F(CudaWavefrontStageAuditTest, SummarizesSuccessfulOutcomes) {
    const auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::ray, 0U),
        successful_outcome(WavefrontStageRoute::ray, 2U),
        successful_outcome(WavefrontStageRoute::ray, 1U),
        successful_outcome(WavefrontStageRoute::ray, 3U),
    };
    const auto audit =
        run_audit(outcomes, route_mask(WavefrontStageRoute::ray), 4U, WavefrontStageKind::camera);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    expect_canonical_header(*audit, WavefrontStageKind::camera,
                            static_cast<std::uint32_t>(outcomes.size()));
    EXPECT_EQ(audit->first_failure_work_index, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(audit->closure_samples, 0U);
    EXPECT_EQ(audit->light_samples, 0U);
    expect_outcome(audit->first_failure, WavefrontStageOutcome{});
}

TEST_F(CudaWavefrontStageAuditTest, CountsShadeClosureAndLightSamples) {
    const auto closure = WavefrontShadeDetailClosureSampled;
    const auto light = WavefrontShadeDetailLightSampled;
    const auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::shadow, 0U, closure),
        successful_outcome(WavefrontStageRoute::shadow, 1U, light),
        successful_outcome(WavefrontStageRoute::continuation, 2U, closure | light),
        successful_outcome(WavefrontStageRoute::terminated, 3U),
    };
    const auto allowed = route_mask(WavefrontStageRoute::shadow) |
                         route_mask(WavefrontStageRoute::continuation) |
                         route_mask(WavefrontStageRoute::terminated);
    const auto audit = run_audit(outcomes, allowed, 4U, WavefrontStageKind::shade);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    expect_canonical_header(*audit, WavefrontStageKind::shade,
                            static_cast<std::uint32_t>(outcomes.size()));
    EXPECT_EQ(audit->first_failure_work_index, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(audit->closure_samples, 2U);
    EXPECT_EQ(audit->light_samples, 2U);
    expect_outcome(audit->first_failure, WavefrontStageOutcome{});
}

TEST_F(CudaWavefrontStageAuditTest, ReportsForbiddenRoute) {
    auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::ray, 0U),
        successful_outcome(WavefrontStageRoute::shade, 1U),
        successful_outcome(WavefrontStageRoute::ray, 2U),
    };
    const auto audit = run_audit(outcomes, route_mask(WavefrontStageRoute::ray), 3U,
                                 WavefrontStageKind::intersection_gather);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    EXPECT_EQ(audit->first_failure_work_index, 1U);
    expect_outcome(audit->first_failure, outcomes[1U]);
}

TEST_F(CudaWavefrontStageAuditTest, ReportsOutOfRangePathSlot) {
    auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::ray, 0U),
        successful_outcome(WavefrontStageRoute::ray, 3U),
        successful_outcome(WavefrontStageRoute::ray, 1U),
    };
    const auto audit = run_audit(outcomes, route_mask(WavefrontStageRoute::ray), 3U,
                                 WavefrontStageKind::continuation);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    EXPECT_EQ(audit->first_failure_work_index, 1U);
    expect_outcome(audit->first_failure, outcomes[1U]);
}

TEST_F(CudaWavefrontStageAuditTest, ReportsFailedStatus) {
    auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::ray, 0U),
        successful_outcome(WavefrontStageRoute::ray, 1U),
        successful_outcome(WavefrontStageRoute::ray, 2U),
    };
    outcomes[2U] = WavefrontStageOutcome{
        .status = static_cast<std::uint32_t>(WavefrontStageStatus::numerical_failure),
        .route = static_cast<std::uint32_t>(WavefrontStageRoute::none),
        .path_slot = 2U,
        .detail = 0xA11D17U,
    };
    const auto audit = run_audit(outcomes, route_mask(WavefrontStageRoute::ray), 3U,
                                 WavefrontStageKind::continuation);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    EXPECT_EQ(audit->first_failure_work_index, 2U);
    expect_outcome(audit->first_failure, outcomes[2U]);
}

TEST_F(CudaWavefrontStageAuditTest, DetectsOutcomeLeftAtUnwrittenSentinel) {
    auto outcomes = std::array{
        successful_outcome(WavefrontStageRoute::ray, 0U),
        WavefrontStageOutcome{
            .status = std::numeric_limits<std::uint32_t>::max(),
            .route = std::numeric_limits<std::uint32_t>::max(),
            .path_slot = std::numeric_limits<std::uint32_t>::max(),
            .detail = std::numeric_limits<std::uint32_t>::max(),
        },
        successful_outcome(WavefrontStageRoute::ray, 2U),
    };
    const auto audit = run_audit(outcomes, route_mask(WavefrontStageRoute::ray), 3U,
                                 WavefrontStageKind::intersection_gather);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    EXPECT_EQ(audit->first_failure_work_index, 1U);
    expect_outcome(audit->first_failure, outcomes[1U]);
}

TEST_F(CudaWavefrontStageAuditTest, SelectsSmallestFailureIndexAcrossBlocks) {
    constexpr auto OutcomeCount = std::size_t{2048U};
    auto outcomes = std::vector<WavefrontStageOutcome>{};
    outcomes.reserve(OutcomeCount);
    for (auto index = std::uint32_t{}; index < OutcomeCount; ++index) {
        outcomes.push_back(successful_outcome(WavefrontStageRoute::ray, index));
    }
    outcomes[1537U].status = static_cast<std::uint32_t>(WavefrontStageStatus::invalid_ray);
    outcomes[777U].path_slot = static_cast<std::uint32_t>(OutcomeCount);
    outcomes[17U].route = static_cast<std::uint32_t>(WavefrontStageRoute::shade);

    const auto audit = run_audit(outcomes, route_mask(WavefrontStageRoute::ray),
                                 static_cast<std::uint32_t>(OutcomeCount),
                                 WavefrontStageKind::intersection_gather);
    ASSERT_TRUE(audit.has_value()) << audit.error().message;
    EXPECT_EQ(audit->first_failure_work_index, 17U);
    expect_outcome(audit->first_failure, outcomes[17U]);
}

TEST_F(CudaWavefrontStageAuditTest, RejectsInvalidLauncherParameters) {
    auto device_outcomes_result = DeviceBuffer<WavefrontStageOutcome>::allocate(1U);
    ASSERT_TRUE(device_outcomes_result.has_value()) << device_outcomes_result.error().message;
    auto device_outcomes = std::move(*device_outcomes_result);
    auto device_audit_result = DeviceBuffer<WavefrontStageAudit>::allocate(1U);
    ASSERT_TRUE(device_audit_result.has_value()) << device_audit_result.error().message;
    auto device_audit = std::move(*device_audit_result);

    const auto allowed = route_mask(WavefrontStageRoute::ray);
    EXPECT_EQ(blackframe_cuda_launch_wavefront_audit_stage(
                  device_outcomes.data(), 1U, allowed, 1U,
                  static_cast<std::uint32_t>(WavefrontStageKind::camera), nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_wavefront_audit_stage(
                  nullptr, 1U, allowed, 1U, static_cast<std::uint32_t>(WavefrontStageKind::camera),
                  device_audit.data()),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_launch_wavefront_audit_stage(
                  device_outcomes.data(), 1U, allowed, 1U,
                  static_cast<std::uint32_t>(WavefrontStageKind::continuation) + 1U,
                  device_audit.data()),
              static_cast<int>(cudaErrorInvalidValue));

    const auto audit_close = device_audit.close();
    EXPECT_TRUE(audit_close.has_value()) << audit_close.error().message;
    const auto outcomes_close = device_outcomes.close();
    EXPECT_TRUE(outcomes_close.has_value()) << outcomes_close.error().message;
}

} // namespace
} // namespace blackframe::xpu::cuda
