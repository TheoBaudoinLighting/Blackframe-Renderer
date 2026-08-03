#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

namespace shared = blackframe::xpu::shared;
namespace renderer = blackframe::renderer;

template <typename Record> void expect_host_device_record() {
    EXPECT_TRUE(std::is_standard_layout_v<Record>);
    EXPECT_TRUE(std::is_trivially_copyable_v<Record>);
    EXPECT_TRUE(std::is_trivially_destructible_v<Record>);
}

TEST(HostDeviceTransportAbi, RecordsAreTriviallyCopyable) {
    expect_host_device_record<shared::TransportSpectrum>();
    expect_host_device_record<shared::TransportRay>();
    expect_host_device_record<shared::SampleStreamIndex>();
    expect_host_device_record<shared::PathSlot>();
    expect_host_device_record<shared::QueueHeader>();
    expect_host_device_record<shared::SurfaceIdentifiers>();
    expect_host_device_record<shared::ClosestHit>();
    expect_host_device_record<shared::TransportPathStateLane>();
    expect_host_device_record<shared::LayoutManifest>();
}

TEST(HostDeviceTransportAbi, HostLayoutMatchesFrozenManifest) {
    constexpr auto expected = std::array<std::uint32_t, shared::HostDeviceLayoutValueCount>{
        16U, 16U, 0U, // TransportSpectrum
        48U, 16U, 0U,  4U,  8U,  12U, 16U, 20U, 24U, 28U, 32U, 36U,  40U,  44U,
        24U, 8U,  0U,  4U,  8U,  16U, // SampleStreamIndex
        4U,  4U,  0U,                 // PathSlot
        32U, 16U, 0U,  2U,  4U,  8U,  12U, 16U, 20U, 24U, 28U, 32U,  16U,  0U,
        4U,  8U,  12U, 16U, 20U, // SurfaceIdentifiers
        64U, 16U, 0U,  4U,  8U,  12U, 16U, 20U, 24U, 28U, 32U, 128U, 16U,  0U,
        16U, 32U, 48U, 64U, 68U, 72U, 76U, 80U, 84U, 88U, 92U, 96U,  100U, 104U,
    };
    constexpr auto manifest = shared::host_layout_manifest(__cplusplus);

    EXPECT_EQ(manifest.abi_major, shared::HostDeviceTransportAbiMajor);
    EXPECT_EQ(manifest.abi_minor, shared::HostDeviceTransportAbiMinor);
    EXPECT_EQ(manifest.device_cxx_standard, __cplusplus);
    EXPECT_EQ(manifest.value_count, expected.size());
    EXPECT_EQ(manifest.reserved_header, 0U);
    EXPECT_EQ(manifest.reserved_tail[0], 0U);
    EXPECT_EQ(manifest.reserved_tail[1], 0U);
    EXPECT_EQ(manifest.reserved_tail[2], 0U);

    for (auto index = std::size_t{0}; index < expected.size(); ++index) {
        EXPECT_EQ(manifest.values[index], expected[index]) << "layout value " << index;
    }
}

TEST(HostDeviceTransportAbi, RawFieldsCarryCanonicalHostTransportCodes) {
    constexpr auto queue_kinds = std::array{
        renderer::WavefrontQueueKind::camera,       renderer::WavefrontQueueKind::ray,
        renderer::WavefrontQueueKind::hit,          renderer::WavefrontQueueKind::miss,
        renderer::WavefrontQueueKind::shade,        renderer::WavefrontQueueKind::shadow,
        renderer::WavefrontQueueKind::continuation,
    };
    for (auto index = std::size_t{0}; index < queue_kinds.size(); ++index) {
        EXPECT_EQ(static_cast<std::uint32_t>(queue_kinds[index]), index);
    }

    constexpr auto path = [] {
        auto result = shared::TransportPathStateLane{};
        result.delta_flags =
            static_cast<std::uint32_t>(renderer::PathDeltaFlags::previous_bounce_was_delta |
                                       renderer::PathDeltaFlags::any_non_delta_bounces);
        for (auto& measure : result.wavelength_pdf_measures) {
            measure = static_cast<std::uint8_t>(renderer::ProbabilityMeasure::wavelength);
        }
        return result;
    }();
    EXPECT_EQ(path.delta_flags, 3U);
    for (const auto measure : path.wavelength_pdf_measures) {
        EXPECT_EQ(measure, 5U);
    }
}

TEST(HostDeviceTransportAbi, QueueHeaderRejectsEveryIncompatibleContract) {
    constexpr auto valid = [] {
        auto result = shared::QueueHeader{};
        result.abi_major = shared::HostDeviceTransportAbiMajor;
        result.abi_minor = shared::HostDeviceTransportAbiMinor;
        result.struct_size = static_cast<std::uint32_t>(sizeof(shared::QueueHeader));
        result.queue_kind = static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade);
        result.capacity = 64U;
        result.size = 16U;
        return result;
    }();
    EXPECT_EQ(shared::validate_queue_header(valid), shared::QueueHeaderValidationStatus::valid);

    auto incompatible = valid;
    ++incompatible.abi_major;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::unsupported_abi_version);
    incompatible = valid;
    ++incompatible.abi_minor;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::unsupported_abi_version);
    incompatible = valid;
    --incompatible.struct_size;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::unexpected_struct_size);
    incompatible = valid;
    incompatible.queue_kind = 7U;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::unknown_queue_kind);
    incompatible = valid;
    incompatible.size = incompatible.capacity + 1U;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::size_exceeds_capacity);
    incompatible = valid;
    incompatible.reserved = 1U;
    EXPECT_EQ(shared::validate_queue_header(incompatible),
              shared::QueueHeaderValidationStatus::nonzero_reserved);
}

TEST(HostDeviceTransportAbi, ReservedStorageIsExplicitlyZeroInitialized) {
    constexpr auto ray = shared::TransportRay{};
    constexpr auto queue = shared::QueueHeader{};
    constexpr auto identifiers = shared::SurfaceIdentifiers{};
    constexpr auto hit = shared::ClosestHit{};
    constexpr auto path = shared::TransportPathStateLane{};

    EXPECT_EQ(ray.reserved, 0U);
    EXPECT_EQ(queue.reserved, 0U);
    EXPECT_EQ(hit.reserved, 0U);
    for (const auto value : identifiers.reserved) {
        EXPECT_EQ(value, 0U);
    }
    for (const auto value : path.reserved) {
        EXPECT_EQ(value, 0U);
    }
}

} // namespace
