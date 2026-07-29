#include <Blackframe/Renderer/DebugIntegrators.hpp>
#include <Blackframe/Renderer/Sphere.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Result<SurfaceInteraction> make_debug_interaction() {
    return SurfaceInteraction::create(
        Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, Normal3{.z = 1.0F}, Normal3{.y = 0.6F, .z = 0.8F},
        Point2{.x = -3.0F, .y = 8.0F}, Vector3{.x = 1.0F}, Vector3{.y = 1.0F},
        SurfaceIdentifiers{
            .instance = {.value = 0},
            .geometry = {.value = std::numeric_limits<std::uint32_t>::max()},
            .primitive = {.value = 2048},
            .material = {.value = std::uint32_t{1} << 22U},
        },
        0.25F);
}

TEST(DebugIntegratorsTest, MapsNormalDepthUvAndBarycentricsWithoutSubstitution) {
    const auto interaction = make_debug_interaction();
    ASSERT_TRUE(interaction.has_value());

    EXPECT_EQ(debug_normal_color(*interaction),
              (LinearRGB{.red = 0.5F, .green = 0.5F, .blue = 1.0F}));
    const auto depth = debug_depth_color(0.5F);
    ASSERT_TRUE(depth.has_value());
    EXPECT_EQ(*depth, (LinearRGB{.red = 0.5F, .green = 0.5F, .blue = 0.5F}));
    EXPECT_EQ(debug_uv_color(*interaction), (LinearRGB{.red = -3.0F, .green = 8.0F, .blue = 0.0F}));

    const auto barycentrics = debug_barycentric_color(
        TriangleBarycentrics{.vertex0 = 0.5F, .vertex1 = 0.25F, .vertex2 = 0.25F});
    ASSERT_TRUE(barycentrics.has_value());
    EXPECT_EQ(*barycentrics, (LinearRGB{.red = 0.5F, .green = 0.25F, .blue = 0.25F}));

    const auto edge = debug_barycentric_color(
        TriangleBarycentrics{.vertex0 = 0.0F, .vertex1 = 0.5F, .vertex2 = 0.5F});
    ASSERT_TRUE(edge.has_value());
    EXPECT_EQ(*edge, (LinearRGB{.red = 0.0F, .green = 0.5F, .blue = 0.5F}));
}

TEST(DebugIntegratorsTest, UsesTheUnmodifiedRayParameterForDepth) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    const auto ray =
        Ray::create(Point3{}, Vector3{.x = 2.0F}, 0.0F, 1.0F, 0.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    ASSERT_FLOAT_EQ((**intersection).parameter, 0.5F);
    ASSERT_EQ((**intersection).position, (Point3{.x = 1.0F}));

    const auto depth = debug_depth_color((**intersection).parameter);
    ASSERT_TRUE(depth.has_value());
    EXPECT_EQ(*depth, (LinearRGB{.red = 0.5F, .green = 0.5F, .blue = 0.5F}));
}

TEST(DebugIntegratorsTest, EncodesEverySelectedIdentifierWithoutASentinelOrHash) {
    const auto interaction = make_debug_interaction();
    ASSERT_TRUE(interaction.has_value());
    const auto& identifiers = interaction->identifiers();

    const auto instance =
        debug_identifier_color<TransportScalar>(identifiers, DebugIdentifierKind::instance);
    const auto geometry =
        debug_identifier_color<TransportScalar>(identifiers, DebugIdentifierKind::geometry);
    const auto primitive =
        debug_identifier_color<TransportScalar>(identifiers, DebugIdentifierKind::primitive);
    const auto material =
        debug_identifier_color<TransportScalar>(identifiers, DebugIdentifierKind::material);
    ASSERT_TRUE(instance.has_value());
    ASSERT_TRUE(geometry.has_value());
    ASSERT_TRUE(primitive.has_value());
    ASSERT_TRUE(material.has_value());

    EXPECT_EQ(*instance,
              (LinearRGB{.red = 1.0F / 2048.0F, .green = 1.0F / 2048.0F, .blue = 1.0F / 1024.0F}));
    EXPECT_EQ(*geometry, (LinearRGB{.red = 1.0F, .green = 1.0F, .blue = 1.0F}));
    EXPECT_EQ(*primitive,
              (LinearRGB{.red = 1.0F / 2048.0F, .green = 1.0F / 1024.0F, .blue = 1.0F / 1024.0F}));
    EXPECT_EQ(*material,
              (LinearRGB{.red = 1.0F / 2048.0F, .green = 1.0F / 2048.0F, .blue = 1.0F / 512.0F}));

    constexpr auto round_trip_values = std::array{
        std::uint32_t{0},
        std::uint32_t{1},
        std::uint32_t{2047},
        std::uint32_t{2048},
        std::uint32_t{2049},
        (std::uint32_t{1} << 22U) - 1U,
        std::uint32_t{1} << 22U,
        (std::uint32_t{1} << 22U) + 1U,
        std::numeric_limits<std::uint32_t>::max(),
    };
    for (const auto value : round_trip_values) {
        SCOPED_TRACE(value);
        const auto encoded = debug_identifier_color<TransportScalar>(
            SurfaceIdentifiers{.primitive = {.value = value}}, DebugIdentifierKind::primitive);
        ASSERT_TRUE(encoded.has_value());
        const auto low = static_cast<std::uint32_t>(std::lround(encoded->red * 2048.0F)) - 1U;
        const auto middle = static_cast<std::uint32_t>(std::lround(encoded->green * 2048.0F)) - 1U;
        const auto high = static_cast<std::uint32_t>(std::lround(encoded->blue * 1024.0F)) - 1U;
        EXPECT_EQ(low | (middle << 11U) | (high << 22U), value);
    }
}

TEST(DebugIntegratorsTest, RejectsInvalidStandaloneInputsWithoutFallback) {
    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();

    for (const auto depth : {-1.0F, infinity, nan}) {
        const auto result = debug_depth_color(depth);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    }

    const auto invalid_barycentrics = {
        TriangleBarycentrics{.vertex0 = nan, .vertex1 = 0.5F, .vertex2 = 0.5F},
        TriangleBarycentrics{.vertex0 = -0.25F, .vertex1 = 0.5F, .vertex2 = 0.75F},
        TriangleBarycentrics{.vertex0 = 1.25F, .vertex1 = 0.0F, .vertex2 = 0.0F},
        TriangleBarycentrics{.vertex0 = 0.25F, .vertex1 = 0.25F, .vertex2 = 0.25F},
    };
    for (const auto barycentrics : invalid_barycentrics) {
        const auto result = debug_barycentric_color(barycentrics);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    }

    const auto interaction = make_debug_interaction();
    ASSERT_TRUE(interaction.has_value());
    const auto invalid_identifier = debug_identifier_color<TransportScalar>(
        interaction->identifiers(), static_cast<DebugIdentifierKind>(255));
    ASSERT_FALSE(invalid_identifier.has_value());
    EXPECT_EQ(invalid_identifier.error().code, core::StatusCode::invalid_argument);
}

TEST(DebugIntegratorsTest, SupportsReferencePrecision) {
    const auto interaction = ReferenceSurfaceInteraction::create(
        ReferencePoint3{}, ReferenceNormal3{.x = -1.0}, ReferenceNormal3{.x = -1.0},
        ReferencePoint2{.x = 0.125, .y = 0.875}, ReferenceVector3{.y = 1.0},
        ReferenceVector3{.z = 1.0},
        SurfaceIdentifiers{
            .instance = {.value = 1},
            .geometry = {.value = 2},
            .primitive = {.value = 3},
            .material = {.value = 4},
        },
        -2.0);
    ASSERT_TRUE(interaction.has_value());
    EXPECT_EQ(debug_normal_color(*interaction),
              (ReferenceLinearRGB{.red = 0.0, .green = 0.5, .blue = 0.5}));
    EXPECT_EQ(debug_uv_color(*interaction),
              (ReferenceLinearRGB{.red = 0.125, .green = 0.875, .blue = 0.0}));

    const auto depth = debug_depth_color(0.75);
    ASSERT_TRUE(depth.has_value());
    EXPECT_EQ(*depth, (ReferenceLinearRGB{.red = 0.75, .green = 0.75, .blue = 0.75}));

    const auto identifier = debug_identifier_color<ReferenceScalar>(interaction->identifiers(),
                                                                    DebugIdentifierKind::material);
    ASSERT_TRUE(identifier.has_value());
    EXPECT_EQ(*identifier, (ReferenceLinearRGB{
                               .red = 5.0 / 2048.0, .green = 1.0 / 2048.0, .blue = 1.0 / 1024.0}));
}

} // namespace
} // namespace blackframe::renderer
