#include <Blackframe/Renderer/PixelJitter.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <string>
#include <type_traits>

namespace blackframe::renderer {
namespace {

[[nodiscard]] constexpr auto dimensions_as_array(const BounceSampleDimensions dimensions) {
    return std::array{
        dimensions.light_selection,  dimensions.light_u,        dimensions.light_v,
        dimensions.bsdf_component,   dimensions.bsdf_u,         dimensions.bsdf_v,
        dimensions.medium_distance,  dimensions.medium_phase_u, dimensions.medium_phase_v,
        dimensions.russian_roulette,
    };
}

TEST(SampleDimensionMapTest, FixesTheClosedVersionOneLayout) {
    static_assert(CurrentSampleDimensionMapSchemaVersion == 1);
    static_assert(std::is_same_v<SampleDimension, std::uint64_t>);
    static_assert(std::is_standard_layout_v<PrimarySampleDimensions>);
    static_assert(std::is_trivially_copyable_v<PrimarySampleDimensions>);
    static_assert(std::is_standard_layout_v<BounceSampleDimensions>);
    static_assert(std::is_trivially_copyable_v<BounceSampleDimensions>);

    EXPECT_EQ(PrimarySampleDimensionMap.camera_raster_x, 0xA24BAED4963EE407ULL);
    EXPECT_EQ(PrimarySampleDimensionMap.camera_raster_y, 0x9FB21C651E98DF25ULL);
    EXPECT_EQ(PrimarySampleDimensionMap.lens_u, 0U);
    EXPECT_EQ(PrimarySampleDimensionMap.lens_v, 1U);
    EXPECT_EQ(PrimarySampleDimensionMap.time, 2U);
    EXPECT_EQ(PrimarySampleDimensionMap.wavelength, 3U);
    EXPECT_EQ(FirstBounceSampleDimension, 4U);
    EXPECT_EQ(SampleDimensionsPerBounce, 10U);
    EXPECT_EQ(MaximumMappedBounceIndex, std::numeric_limits<std::uint32_t>::max());

    const auto first = sample_dimensions_for_bounce(0);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(dimensions_as_array(*first),
              (std::array<SampleDimension, 10>{4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));

    const auto second = sample_dimensions_for_bounce(1);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(dimensions_as_array(*second),
              (std::array<SampleDimension, 10>{14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));
}

TEST(SampleDimensionMapTest, KeepsEveryReservationDisjointThroughTheMaximumBounce) {
    const auto first = sample_dimensions_for_bounce(0);
    const auto second = sample_dimensions_for_bounce(1);
    const auto last = sample_dimensions_for_bounce(MaximumMappedBounceIndex);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(last.has_value());

    auto selected_dimensions = std::set<SampleDimension>{
        PrimarySampleDimensionMap.camera_raster_x,
        PrimarySampleDimensionMap.camera_raster_y,
        PrimarySampleDimensionMap.lens_u,
        PrimarySampleDimensionMap.lens_v,
        PrimarySampleDimensionMap.time,
        PrimarySampleDimensionMap.wavelength,
    };
    for (const auto dimension : dimensions_as_array(*first)) {
        EXPECT_TRUE(selected_dimensions.insert(dimension).second);
    }
    for (const auto dimension : dimensions_as_array(*second)) {
        EXPECT_TRUE(selected_dimensions.insert(dimension).second);
    }
    for (const auto dimension : dimensions_as_array(*last)) {
        EXPECT_TRUE(selected_dimensions.insert(dimension).second);
    }

    EXPECT_EQ(first->russian_roulette + 1, second->light_selection);
    EXPECT_EQ(last->light_selection, 0x00000009FFFFFFFAULL);
    EXPECT_EQ(last->russian_roulette, 0x0000000A00000003ULL);
    EXPECT_LT(last->russian_roulette, PrimarySampleDimensionMap.camera_raster_y);
    EXPECT_LT(last->russian_roulette, PrimarySampleDimensionMap.camera_raster_x);
}

TEST(SampleDimensionMapTest, RejectsAnUnrepresentableBounceWithoutWrapping) {
    const auto one_past = sample_dimensions_for_bounce(MaximumMappedBounceIndex + 1);
    ASSERT_FALSE(one_past.has_value());
    EXPECT_EQ(one_past.error().code, core::StatusCode::resource_exhausted);
    EXPECT_EQ(one_past.error().message,
              "Sample dimension map supports bounce indices up to 4294967295.");

    const auto maximum = sample_dimensions_for_bounce(std::numeric_limits<std::uint64_t>::max());
    ASSERT_FALSE(maximum.has_value());
    EXPECT_EQ(maximum.error().code, core::StatusCode::resource_exhausted);
    EXPECT_EQ(maximum.error().message, one_past.error().message);
}

TEST(SampleDimensionMapTest, PixelJitterUsesTheReservedCameraDimensionsBitForBit) {
    constexpr auto index =
        PixelSampleIndex{.pixel_x = 17, .pixel_y = 29, .sample_index = 27, .seed = 42};
    const auto stream = SampleStream{index};
    const auto sample = generate_pixel_sample<TransportScalar>(index, PixelJitterMode::uniform);
    ASSERT_TRUE(sample.has_value());

    EXPECT_EQ(sample->offset_x, stream.sample_1d(PrimarySampleDimensionMap.camera_raster_x));
    EXPECT_EQ(sample->offset_y, stream.sample_1d(PrimarySampleDimensionMap.camera_raster_y));
    EXPECT_EQ(sample->offset_x, static_cast<TransportScalar>(0x4D67A7U) * 0x1p-24F);
    EXPECT_EQ(sample->offset_y, static_cast<TransportScalar>(0x6B1427U) * 0x1p-24F);
}

TEST(SampleDimensionMapTest, DumpsTheExactVersionedLayoutDeterministically) {
    constexpr auto expected =
        R"({"schema_version":1,"primary":{"camera":{"raster_x":"0xa24baed4963ee407","raster_y":"0x9fb21c651e98df25"},"lens":{"u":"0x0000000000000000","v":"0x0000000000000001"},"time":"0x0000000000000002","wavelength":"0x0000000000000003"},"bounce":{"first":"0x0000000000000004","stride":10,"maximum_index":4294967295,"light":{"selection":0,"u":1,"v":2},"bsdf":{"component":3,"u":4,"v":5},"medium":{"distance":6,"phase_u":7,"phase_v":8},"russian_roulette":9}})";

    const auto first = dump_sample_dimension_map(CurrentSampleDimensionMapSchemaVersion);
    const auto replay = dump_sample_dimension_map(CurrentSampleDimensionMapSchemaVersion);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(replay.has_value());
    EXPECT_EQ(*first, expected);
    EXPECT_EQ(*replay, expected);
}

TEST(SampleDimensionMapTest, RejectsUnsupportedDumpVersionsWithoutFallback) {
    for (const auto version : std::array{std::uint32_t{0}, std::uint32_t{2}}) {
        const auto dump = dump_sample_dimension_map(version);
        ASSERT_FALSE(dump.has_value());
        EXPECT_EQ(dump.error().code, core::StatusCode::incompatible);
        EXPECT_EQ(dump.error().message, "Unsupported sample dimension map schema version " +
                                            std::to_string(version) + "; expected 1.");
    }
}

} // namespace
} // namespace blackframe::renderer
