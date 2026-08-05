#pragma once

#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

enum class SampleStreamDumpStatus : std::uint32_t {
    success = 0U,
    bounce_out_of_range = 1U,
};

struct alignas(8) SampleStreamPrimaryDimensions final {
    std::uint64_t camera_raster_x;
    std::uint64_t camera_raster_y;
    std::uint64_t lens_u;
    std::uint64_t lens_v;
    std::uint64_t time;
    std::uint64_t wavelength;
};

struct alignas(8) SampleStreamBounceDimensions final {
    std::uint64_t light_selection;
    std::uint64_t light_u;
    std::uint64_t light_v;
    std::uint64_t bsdf_component;
    std::uint64_t bsdf_u;
    std::uint64_t bsdf_v;
    std::uint64_t medium_distance;
    std::uint64_t medium_phase_u;
    std::uint64_t medium_phase_v;
    std::uint64_t russian_roulette;
};

// This manifest is a value contract, not a serialization of the renderer's host-only map.
// Unsupported requested schemas are rejected by the launcher instead of returning this layout.
struct alignas(8) SampleStreamDimensionManifest final {
    std::uint32_t schema_version;
    std::uint32_t reserved;
    SampleStreamPrimaryDimensions primary_dimensions;
    std::uint64_t first_bounce_dimension;
    std::uint64_t dimensions_per_bounce;
    std::uint64_t maximum_bounce_index;
    SampleStreamBounceDimensions bounce_offsets;
};

struct alignas(8) SampleStreamDumpRequest final {
    shared::SampleStreamIndex sample_stream;
    std::uint64_t dimension;
    std::uint64_t bounce_index;
};

struct alignas(8) SampleStreamDumpResult final {
    SampleStreamDumpStatus status;
    std::uint32_t reserved;
    std::uint64_t indexed_bits;
    float sample;
    std::uint32_t reserved_tail;
    SampleStreamBounceDimensions bounce_dimensions;
};

namespace sample_stream {

inline constexpr std::uint32_t DimensionMapSchemaVersion = 1U;
inline constexpr std::uint64_t MaximumBounceIndex = 0xFFFFFFFFULL;
inline constexpr std::uint64_t FirstBounceDimension = 4U;
inline constexpr std::uint64_t DimensionsPerBounce = 10U;

inline constexpr auto PrimaryDimensions = SampleStreamPrimaryDimensions{
    .camera_raster_x = 0xA24BAED4963EE407ULL,
    .camera_raster_y = 0x9FB21C651E98DF25ULL,
    .lens_u = 0U,
    .lens_v = 1U,
    .time = 2U,
    .wavelength = 3U,
};

inline constexpr auto BounceOffsets = SampleStreamBounceDimensions{
    .light_selection = 0U,
    .light_u = 1U,
    .light_v = 2U,
    .bsdf_component = 3U,
    .bsdf_u = 4U,
    .bsdf_v = 5U,
    .medium_distance = 6U,
    .medium_phase_u = 7U,
    .medium_phase_v = 8U,
    .russian_roulette = 9U,
};

} // namespace sample_stream

#define BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(record)                                        \
    static_assert(std::is_standard_layout_v<record>);                                              \
    static_assert(std::is_trivially_copyable_v<record>);                                           \
    static_assert(std::is_trivially_destructible_v<record>)

BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(SampleStreamPrimaryDimensions);
BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(SampleStreamBounceDimensions);
BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(SampleStreamDimensionManifest);
BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(SampleStreamDumpRequest);
BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD(SampleStreamDumpResult);

#undef BLACKFRAME_ASSERT_CUDA_SAMPLE_STREAM_RECORD

static_assert(sizeof(SampleStreamDumpStatus) == 4U);

static_assert(sizeof(SampleStreamPrimaryDimensions) == 48U);
static_assert(alignof(SampleStreamPrimaryDimensions) == 8U);
static_assert(offsetof(SampleStreamPrimaryDimensions, camera_raster_x) == 0U);
static_assert(offsetof(SampleStreamPrimaryDimensions, camera_raster_y) == 8U);
static_assert(offsetof(SampleStreamPrimaryDimensions, lens_u) == 16U);
static_assert(offsetof(SampleStreamPrimaryDimensions, lens_v) == 24U);
static_assert(offsetof(SampleStreamPrimaryDimensions, time) == 32U);
static_assert(offsetof(SampleStreamPrimaryDimensions, wavelength) == 40U);

static_assert(sizeof(SampleStreamBounceDimensions) == 80U);
static_assert(alignof(SampleStreamBounceDimensions) == 8U);
static_assert(offsetof(SampleStreamBounceDimensions, light_selection) == 0U);
static_assert(offsetof(SampleStreamBounceDimensions, light_u) == 8U);
static_assert(offsetof(SampleStreamBounceDimensions, light_v) == 16U);
static_assert(offsetof(SampleStreamBounceDimensions, bsdf_component) == 24U);
static_assert(offsetof(SampleStreamBounceDimensions, bsdf_u) == 32U);
static_assert(offsetof(SampleStreamBounceDimensions, bsdf_v) == 40U);
static_assert(offsetof(SampleStreamBounceDimensions, medium_distance) == 48U);
static_assert(offsetof(SampleStreamBounceDimensions, medium_phase_u) == 56U);
static_assert(offsetof(SampleStreamBounceDimensions, medium_phase_v) == 64U);
static_assert(offsetof(SampleStreamBounceDimensions, russian_roulette) == 72U);

static_assert(sizeof(SampleStreamDimensionManifest) == 160U);
static_assert(alignof(SampleStreamDimensionManifest) == 8U);
static_assert(offsetof(SampleStreamDimensionManifest, schema_version) == 0U);
static_assert(offsetof(SampleStreamDimensionManifest, reserved) == 4U);
static_assert(offsetof(SampleStreamDimensionManifest, primary_dimensions) == 8U);
static_assert(offsetof(SampleStreamDimensionManifest, first_bounce_dimension) == 56U);
static_assert(offsetof(SampleStreamDimensionManifest, dimensions_per_bounce) == 64U);
static_assert(offsetof(SampleStreamDimensionManifest, maximum_bounce_index) == 72U);
static_assert(offsetof(SampleStreamDimensionManifest, bounce_offsets) == 80U);

static_assert(sizeof(SampleStreamDumpRequest) == 40U);
static_assert(alignof(SampleStreamDumpRequest) == 8U);
static_assert(offsetof(SampleStreamDumpRequest, sample_stream) == 0U);
static_assert(offsetof(SampleStreamDumpRequest, dimension) == 24U);
static_assert(offsetof(SampleStreamDumpRequest, bounce_index) == 32U);

static_assert(sizeof(SampleStreamDumpResult) == 104U);
static_assert(alignof(SampleStreamDumpResult) == 8U);
static_assert(offsetof(SampleStreamDumpResult, status) == 0U);
static_assert(offsetof(SampleStreamDumpResult, reserved) == 4U);
static_assert(offsetof(SampleStreamDumpResult, indexed_bits) == 8U);
static_assert(offsetof(SampleStreamDumpResult, sample) == 16U);
static_assert(offsetof(SampleStreamDumpResult, reserved_tail) == 20U);
static_assert(offsetof(SampleStreamDumpResult, bounce_dimensions) == 24U);

static_assert(sample_stream::BounceOffsets.russian_roulette + 1U ==
              sample_stream::DimensionsPerBounce);
static_assert(sample_stream::FirstBounceDimension +
                  sample_stream::MaximumBounceIndex * sample_stream::DimensionsPerBounce +
                  sample_stream::BounceOffsets.russian_roulette <
              sample_stream::PrimaryDimensions.camera_raster_x);
static_assert(sample_stream::FirstBounceDimension +
                  sample_stream::MaximumBounceIndex * sample_stream::DimensionsPerBounce +
                  sample_stream::BounceOffsets.russian_roulette <
              sample_stream::PrimaryDimensions.camera_raster_y);

} // namespace blackframe::xpu::cuda

// All pointers name non-overlapping storage on the active CUDA device. A zero request count,
// unsupported schema, null pointer, or launch failure is returned explicitly as a CUDA status.
extern "C" int blackframe_cuda_launch_sample_stream_dump(
    std::uint32_t requested_schema_version,
    const blackframe::xpu::cuda::SampleStreamDumpRequest* requests, std::uint32_t request_count,
    blackframe::xpu::cuda::SampleStreamDimensionManifest* manifest,
    blackframe::xpu::cuda::SampleStreamDumpResult* results) noexcept;
