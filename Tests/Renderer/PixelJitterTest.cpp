#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Result<OrthonormalFrame> identity_camera_frame() {
    return OrthonormalFrame::from_normal_and_tangent(Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
                                                     Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
}

template <GeometryScalar Scalar> void expect_uniform_two_dimensional_distribution() {
    constexpr auto bin_count = std::size_t{16};
    constexpr auto sample_count = std::size_t{65'536};
    constexpr auto expected_per_bin =
        static_cast<double>(sample_count) / static_cast<double>(bin_count * bin_count);
    std::array<std::uint32_t, bin_count * bin_count> histogram{};
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x_squared = 0.0;
    double sum_y_squared = 0.0;
    double sum_xy = 0.0;

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const auto sample = generate_pixel_sample<Scalar>(
            PixelSampleIndex{.pixel_x = 113,
                             .pixel_y = 47,
                             .sample_index = static_cast<std::uint64_t>(sample_index),
                             .seed = 0xA5A5F00D12345678ULL},
            PixelJitterMode::uniform);
        ASSERT_TRUE(sample.has_value());
        ASSERT_GE(sample->offset_x, Scalar{0});
        ASSERT_LT(sample->offset_x, Scalar{1});
        ASSERT_GE(sample->offset_y, Scalar{0});
        ASSERT_LT(sample->offset_y, Scalar{1});

        const auto bin_x =
            static_cast<std::size_t>(sample->offset_x * static_cast<Scalar>(bin_count));
        const auto bin_y =
            static_cast<std::size_t>(sample->offset_y * static_cast<Scalar>(bin_count));
        ASSERT_LT(bin_x, bin_count);
        ASSERT_LT(bin_y, bin_count);
        ++histogram[bin_y * bin_count + bin_x];

        sum_x += sample->offset_x;
        sum_y += sample->offset_y;
        sum_x_squared += sample->offset_x * sample->offset_x;
        sum_y_squared += sample->offset_y * sample->offset_y;
        sum_xy += sample->offset_x * sample->offset_y;
    }

    const auto inverse_count = 1.0 / static_cast<double>(sample_count);
    const auto mean_x = sum_x * inverse_count;
    const auto mean_y = sum_y * inverse_count;
    const auto covariance = sum_xy * inverse_count - mean_x * mean_y;
    double chi_squared = 0.0;
    for (const auto observed : histogram) {
        const auto difference = static_cast<double>(observed) - expected_per_bin;
        chi_squared += difference * difference / expected_per_bin;
    }

    EXPECT_NEAR(mean_x, 0.5, 0.005);
    EXPECT_NEAR(mean_y, 0.5, 0.005);
    EXPECT_NEAR(sum_x_squared * inverse_count, 1.0 / 3.0, 0.006);
    EXPECT_NEAR(sum_y_squared * inverse_count, 1.0 / 3.0, 0.006);
    EXPECT_NEAR(covariance, 0.0, 0.003);
    EXPECT_LT(chi_squared, 400.0);
}

TEST(PixelJitterTest, CenterModeReturnsTheExactDeterministicPixelCenter) {
    static_assert(!std::is_same_v<PixelSample, ReferencePixelSample>);

    constexpr auto expected = PixelSample{
        .pixel_x = 1,
        .pixel_y = 1,
        .offset_x = 0.5F,
        .offset_y = 0.5F,
    };
    for (const auto index : std::array{
             PixelSampleIndex{.pixel_x = 1, .pixel_y = 1, .sample_index = 0, .seed = 0},
             PixelSampleIndex{.pixel_x = 1,
                              .pixel_y = 1,
                              .sample_index = std::numeric_limits<std::uint64_t>::max(),
                              .seed = std::numeric_limits<std::uint64_t>::max()},
         }) {
        const auto sample = generate_pixel_sample<TransportScalar>(index, PixelJitterMode::center);
        ASSERT_TRUE(sample.has_value());
        EXPECT_EQ(*sample, expected);
    }

    const auto frame = identity_camera_frame();
    ASSERT_TRUE(frame.has_value());
    const auto camera =
        PinholeCamera::create(Point3{}, *frame, RenderExtent{.width = 3, .height = 3}, 1.0F, 0.0F,
                              1.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());
    const auto ray = camera->generate_primary_ray(
        PixelSampleIndex{.pixel_x = 1, .pixel_y = 1, .sample_index = 27, .seed = 42},
        PixelJitterMode::center, 0.0F);
    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(ray->direction(), (Vector3{.x = 0.0F, .y = 0.0F, .z = -1.0F}));
}

TEST(PixelJitterTest, UniformModeIsIndexedAndOrderIndependent) {
    constexpr auto sample_count = std::size_t{4096};
    std::vector<PixelSample> forward;
    forward.reserve(sample_count);

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const auto sample = generate_pixel_sample<TransportScalar>(
            PixelSampleIndex{.pixel_x = 17,
                             .pixel_y = 29,
                             .sample_index = static_cast<std::uint64_t>(sample_index),
                             .seed = 0x0123456789ABCDEFULL},
            PixelJitterMode::uniform);
        ASSERT_TRUE(sample.has_value());
        ASSERT_GE(sample->offset_x, 0.0F);
        ASSERT_LT(sample->offset_x, 1.0F);
        ASSERT_GE(sample->offset_y, 0.0F);
        ASSERT_LT(sample->offset_y, 1.0F);
        forward.push_back(*sample);
    }

    for (std::size_t reverse_index = sample_count; reverse_index > 0; --reverse_index) {
        const auto sample_index = reverse_index - 1;
        const auto replay = generate_pixel_sample<TransportScalar>(
            PixelSampleIndex{.pixel_x = 17,
                             .pixel_y = 29,
                             .sample_index = static_cast<std::uint64_t>(sample_index),
                             .seed = 0x0123456789ABCDEFULL},
            PixelJitterMode::uniform);
        ASSERT_TRUE(replay.has_value());
        EXPECT_EQ(*replay, forward[sample_index]);
    }
}

TEST(PixelJitterTest, UniformTransportModeHasAUniformTwoDimensionalDistribution) {
    expect_uniform_two_dimensional_distribution<TransportScalar>();
}

TEST(PixelJitterTest, UniformReferenceModeHasAUniformTwoDimensionalDistribution) {
    expect_uniform_two_dimensional_distribution<ReferenceScalar>();
}

TEST(PixelJitterTest, RejectsInvalidInputsWithoutFallback) {
    const auto invalid_mode = generate_pixel_sample<TransportScalar>(
        PixelSampleIndex{}, static_cast<PixelJitterMode>(255));
    ASSERT_FALSE(invalid_mode.has_value());
    EXPECT_EQ(invalid_mode.error().code, core::StatusCode::invalid_argument);

    const auto frame = identity_camera_frame();
    ASSERT_TRUE(frame.has_value());
    constexpr auto maximum_test_extent = RenderExtent{.width = 1U << 20U, .height = 1};
    const auto camera = PinholeCamera::create(Point3{}, *frame, maximum_test_extent, 1.0F, 0.0F,
                                              1.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    const auto outside_pixel = camera->generate_primary_ray(
        PixelSample{
            .pixel_x = maximum_test_extent.width, .pixel_y = 0, .offset_x = 0.5F, .offset_y = 0.5F},
        0.0F);
    ASSERT_FALSE(outside_pixel.has_value());
    EXPECT_EQ(outside_pixel.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(
        camera
            ->generate_primary_ray(
                PixelSample{.pixel_x = 0, .pixel_y = 0, .offset_x = -0.25F, .offset_y = 0.5F}, 0.0F)
            .has_value());
    EXPECT_FALSE(
        camera
            ->generate_primary_ray(
                PixelSample{.pixel_x = 0, .pixel_y = 0, .offset_x = 1.0F, .offset_y = 0.5F}, 0.0F)
            .has_value());
    EXPECT_FALSE(camera
                     ->generate_primary_ray(
                         PixelSample{.pixel_x = 0,
                                     .pixel_y = 0,
                                     .offset_x = std::numeric_limits<TransportScalar>::quiet_NaN(),
                                     .offset_y = 0.5F},
                         0.0F)
                     .has_value());

    const auto last_pixel_sample = generate_pixel_sample<TransportScalar>(
        PixelSampleIndex{.pixel_x = maximum_test_extent.width - 1,
                         .pixel_y = 0,
                         .sample_index = 99,
                         .seed = 1234},
        PixelJitterMode::uniform);
    ASSERT_TRUE(last_pixel_sample.has_value());
    EXPECT_TRUE(camera->generate_primary_ray(*last_pixel_sample, 0.0F).has_value());
    EXPECT_TRUE(camera
                    ->generate_primary_ray(PixelSample{.pixel_x = maximum_test_extent.width - 1,
                                                       .pixel_y = 0,
                                                       .offset_x = std::nextafter(1.0F, 0.0F),
                                                       .offset_y = 0.5F},
                                           0.0F)
                    .has_value());
}

} // namespace
} // namespace blackframe::renderer
