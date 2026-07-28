#include <Blackframe/Renderer/RenderConfiguration.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

TEST(RenderConfigurationTest, AcceptsTheDefaultFoundationConfiguration) {
    EXPECT_TRUE(validate_render_configuration(RenderConfiguration{}).has_value());
}

TEST(RenderConfigurationSchemaTest, ExposesAClosedVersionOneSchema) {
    const auto schema = render_configuration_schema();

    EXPECT_EQ(schema.version, 1U);
    EXPECT_EQ(schema.version, CurrentRenderConfigurationSchemaVersion);
    EXPECT_FALSE(schema.allows_unknown_keys);
    ASSERT_FALSE(schema.fields.empty());
    EXPECT_EQ(schema.fields.front().key, "schema_version");
    EXPECT_TRUE(schema.fields.front().required);
}

TEST(RenderConfigurationSchemaTest, ParsesAndValidatesVersionOne) {
    constexpr auto encoded_configuration = R"({
        "schema_version": 1,
        "width": 64,
        "height": 32,
        "samples_per_pixel": 8,
        "maximum_path_depth": 6,
        "tile_edge_length": 8,
        "seed": 42,
        "xpu_device_id": "gpu\u002d0"
    })";

    const auto configuration = parse_and_validate_render_configuration(encoded_configuration);

    ASSERT_TRUE(configuration.has_value());
    EXPECT_EQ(configuration->schema_version, 1U);
    EXPECT_EQ(configuration->extent.width, 64U);
    EXPECT_EQ(configuration->extent.height, 32U);
    EXPECT_EQ(configuration->samples_per_pixel, 8U);
    EXPECT_EQ(configuration->maximum_path_depth, 6U);
    EXPECT_EQ(configuration->tile_edge_length, 8U);
    EXPECT_EQ(configuration->seed, 42U);
    EXPECT_EQ(configuration->xpu_device_id, "gpu-0");
}

TEST(RenderConfigurationSchemaTest, RejectsAnUnknownKeyWithItsName) {
    constexpr auto encoded_configuration = R"({
        "schema_version": 1,
        "sample_count": 8
    })";

    const auto configuration = parse_and_validate_render_configuration(encoded_configuration);

    ASSERT_FALSE(configuration.has_value());
    EXPECT_EQ(configuration.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(configuration.error().message, "Unknown render configuration key 'sample_count'.");
}

TEST(RenderConfigurationSchemaTest, RejectsAnInvalidValueWithItsKey) {
    constexpr auto encoded_configuration = R"({
        "schema_version": 1,
        "samples_per_pixel": 0
    })";

    const auto configuration = parse_and_validate_render_configuration(encoded_configuration);

    ASSERT_FALSE(configuration.has_value());
    EXPECT_EQ(configuration.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(configuration.error().message,
              "Render configuration key 'samples_per_pixel' must be between 1 and 16777216.");
}

TEST(RenderConfigurationSchemaTest, RejectsAnUnsupportedVersionBeforeUse) {
    constexpr auto encoded_configuration = R"({"schema_version": 2})";

    const auto configuration = parse_and_validate_render_configuration(encoded_configuration);

    ASSERT_FALSE(configuration.has_value());
    EXPECT_EQ(configuration.error().code, core::StatusCode::incompatible);
    EXPECT_EQ(configuration.error().message,
              "Unsupported render configuration schema version 2; expected 1.");
}

TEST(RenderConfigurationTest, RejectsAnUnsupportedTypedSchemaVersion) {
    auto configuration = RenderConfiguration{};
    configuration.schema_version = 2;

    const auto validation = validate_render_configuration(configuration);

    ASSERT_FALSE(validation.has_value());
    EXPECT_EQ(validation.error().code, core::StatusCode::incompatible);
    EXPECT_EQ(validation.error().message,
              "Unsupported render configuration schema version 2; expected 1.");
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
