#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

TEST(RenderConfigurationTest, AcceptsTheDefaultFoundationConfiguration) {
    EXPECT_TRUE(validate_render_configuration(RenderConfiguration{}).has_value());
}

TEST(RenderConfigurationTest, RejectsZeroWork) {
    auto expect_invalid_argument = [](const RenderConfiguration& configuration) {
        const auto validation = validate_render_configuration(configuration);
        ASSERT_FALSE(validation.has_value());
        EXPECT_EQ(validation.error().code, core::StatusCode::invalid_argument);
    };

    auto configuration = RenderConfiguration{};
    configuration.samples_per_pixel = 0;
    expect_invalid_argument(configuration);

    configuration = RenderConfiguration{};
    configuration.maximum_path_depth = 0;
    expect_invalid_argument(configuration);

    configuration = RenderConfiguration{};
    configuration.tile_edge_length = 0;
    expect_invalid_argument(configuration);
}

TEST(RenderConfigurationTest, RejectsUnsafeImageDimensions) {
    auto configuration = RenderConfiguration{};
    configuration.extent = {
        .width = 1U << 20U,
        .height = 1U << 20U,
    };

    const auto validation = validate_render_configuration(configuration);

    ASSERT_FALSE(validation.has_value());
    EXPECT_EQ(validation.error().code, core::StatusCode::resource_exhausted);
}

TEST(RenderConfigurationTest, RejectsUnboundedScalarFields) {
    auto expect_resource_exhausted = [](const RenderConfiguration& configuration) {
        const auto validation = validate_render_configuration(configuration);
        ASSERT_FALSE(validation.has_value());
        EXPECT_EQ(validation.error().code, core::StatusCode::resource_exhausted);
    };

    auto configuration = RenderConfiguration{};
    configuration.samples_per_pixel = std::numeric_limits<std::uint32_t>::max();
    expect_resource_exhausted(configuration);

    configuration = RenderConfiguration{};
    configuration.maximum_path_depth = std::numeric_limits<std::uint32_t>::max();
    expect_resource_exhausted(configuration);

    configuration = RenderConfiguration{};
    configuration.tile_edge_length = std::numeric_limits<std::uint32_t>::max();
    expect_resource_exhausted(configuration);
}

TEST(RenderConfigurationTest, RejectsUnsafeWorkProducts) {
    auto configuration = RenderConfiguration{};
    configuration.extent = {
        .width = 65'535,
        .height = 65'535,
    };
    configuration.samples_per_pixel = 1U << 20U;

    const auto sample_validation = validate_render_configuration(configuration);

    ASSERT_FALSE(sample_validation.has_value());
    EXPECT_EQ(sample_validation.error().code, core::StatusCode::resource_exhausted);

    configuration = RenderConfiguration{};
    configuration.extent = {
        .width = 8192,
        .height = 4096,
    };
    configuration.samples_per_pixel = 1U << 20U;
    configuration.maximum_path_depth = 4096;

    const auto transport_validation = validate_render_configuration(configuration);

    ASSERT_FALSE(transport_validation.has_value());
    EXPECT_EQ(transport_validation.error().code, core::StatusCode::resource_exhausted);
}

} // namespace
} // namespace blackframe::renderer
