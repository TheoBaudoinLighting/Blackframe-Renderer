#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace blackframe::renderer {

using SampleDimension = std::uint64_t;

inline constexpr std::uint32_t CurrentSampleDimensionMapSchemaVersion = 1;
inline constexpr std::uint64_t MaximumMappedBounceIndex = std::numeric_limits<std::uint32_t>::max();
inline constexpr SampleDimension FirstBounceSampleDimension = 4;
inline constexpr SampleDimension SampleDimensionsPerBounce = 10;

struct PrimarySampleDimensions final {
    SampleDimension camera_raster_x;
    SampleDimension camera_raster_y;
    SampleDimension lens_u;
    SampleDimension lens_v;
    SampleDimension time;
    SampleDimension wavelength;

    [[nodiscard]] constexpr bool
    operator==(const PrimarySampleDimensions&) const noexcept = default;
};

inline constexpr auto PrimarySampleDimensionMap = PrimarySampleDimensions{
    // These two established tags preserve the public pixel-jitter sequence.
    .camera_raster_x = 0xA24BAED4963EE407ULL,
    .camera_raster_y = 0x9FB21C651E98DF25ULL,
    .lens_u = 0,
    .lens_v = 1,
    .time = 2,
    .wavelength = 3,
};

struct BounceSampleDimensions final {
    SampleDimension light_selection;
    SampleDimension light_u;
    SampleDimension light_v;
    SampleDimension bsdf_component;
    SampleDimension bsdf_u;
    SampleDimension bsdf_v;
    SampleDimension medium_distance;
    SampleDimension medium_phase_u;
    SampleDimension medium_phase_v;
    SampleDimension russian_roulette;

    [[nodiscard]] constexpr bool operator==(const BounceSampleDimensions&) const noexcept = default;
};

// The uint64 input makes an out-of-contract bounce observable instead of narrowing it implicitly.
// Every accepted bounce maps to ten contiguous dimensions without wrapping or saturation.
[[nodiscard]] core::Result<BounceSampleDimensions>
sample_dimensions_for_bounce(std::uint64_t bounce_index);

// The requested schema version is mandatory: unsupported versions fail instead of returning the
// current map under another version number.
[[nodiscard]] core::Result<std::string> dump_sample_dimension_map(std::uint32_t schema_version);

static_assert(std::is_standard_layout_v<PrimarySampleDimensions>);
static_assert(std::is_trivially_copyable_v<PrimarySampleDimensions>);
static_assert(std::is_standard_layout_v<BounceSampleDimensions>);
static_assert(std::is_trivially_copyable_v<BounceSampleDimensions>);

} // namespace blackframe::renderer
