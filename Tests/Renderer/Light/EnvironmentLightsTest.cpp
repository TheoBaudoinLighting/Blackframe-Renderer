#include <Blackframe/Renderer/EnvironmentLights.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> using EnvironmentFor = EnvironmentMapLightT<Scalar>;
template <SpectrumScalar Scalar> using SpectrumFor = LightSpectrumT<Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto EnvironmentTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{5.0e-5} : ReferenceScalar{2.0e-12};

template <typename Value> [[nodiscard]] Value require_value(core::Result<Value> result) {
    if (!result.has_value()) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(result).value();
}

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <typename Result> void expect_error(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> [[nodiscard]] SampledWavelengthsT<Scalar> wavelengths() {
    return require_value(sample_uniform_visible_wavelengths(Scalar{0.25}));
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSampleContextT<Scalar> context(const Point3T<Scalar> position = {}) {
    return require_value(LightSampleContextT<Scalar>::create(position, Scalar{0.5}));
}

template <SpectrumScalar Scalar> [[nodiscard]] RayT<Scalar> ray(const Vector3T<Scalar> direction) {
    return require_value(RayT<Scalar>::create(
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{-2}, .z = Scalar{3}}, direction, Scalar{0},
        std::numeric_limits<Scalar>::infinity(), Scalar{0.5}, AllRayVisibility, VacuumMedium));
}

template <SpectrumScalar Scalar> [[nodiscard]] Bounds3T<Scalar> unit_scene_bounds() {
    return require_value(Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{-1}},
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{1}}));
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> gray(const Scalar value) {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(value);
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> lanes(const Scalar lane0, const Scalar lane1, const Scalar lane2,
                                        const Scalar lane3) {
    return SpectrumFor<Scalar>{.values = {lane0, lane1, lane2, lane3}};
}

template <SpectrumScalar Scalar>
void expect_near(const Scalar actual, const ReferenceScalar expected,
                 const ReferenceScalar scale = ReferenceScalar{1}) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected,
                EnvironmentTolerance<Scalar> * std::max(scale, ReferenceScalar{1}));
}

template <SpectrumScalar Scalar>
void expect_direction_near(const Vector3T<Scalar> actual, const Vector3T<Scalar> expected) {
    expect_near(actual.x, static_cast<ReferenceScalar>(expected.x));
    expect_near(actual.y, static_cast<ReferenceScalar>(expected.y));
    expect_near(actual.z, static_cast<ReferenceScalar>(expected.z));
}

template <SpectrumScalar Scalar>
[[nodiscard]] ReferenceScalar texel_solid_angle(const std::uint32_t row, const std::uint32_t width,
                                                const std::uint32_t height) {
    const auto pi = std::numbers::pi_v<ReferenceScalar>;
    const auto theta0 =
        pi * static_cast<ReferenceScalar>(row) / static_cast<ReferenceScalar>(height);
    const auto theta1 =
        pi * static_cast<ReferenceScalar>(row + 1U) / static_cast<ReferenceScalar>(height);
    return (ReferenceScalar{2} * pi / static_cast<ReferenceScalar>(width)) *
           (std::cos(theta0) - std::cos(theta1));
}

template <SpectrumScalar Scalar>
[[nodiscard]] Vector3T<Scalar>
texel_center_direction(const std::uint32_t column, const std::uint32_t row,
                       const std::uint32_t width, const std::uint32_t height) {
    const auto pi = std::numbers::pi_v<Scalar>;
    const auto phi =
        Scalar{2} * pi * (static_cast<Scalar>(column) + Scalar{0.5}) / static_cast<Scalar>(width);
    const auto theta0 = pi * static_cast<Scalar>(row) / static_cast<Scalar>(height);
    const auto theta1 = pi * static_cast<Scalar>(row + 1U) / static_cast<Scalar>(height);
    const auto cosine_theta = (std::cos(theta0) + std::cos(theta1)) / Scalar{2};
    const auto sine_theta =
        std::sqrt(std::max(Scalar{0}, (Scalar{1} - cosine_theta) * (Scalar{1} + cosine_theta)));
    return Vector3T<Scalar>{
        .x = sine_theta * std::cos(phi),
        .y = cosine_theta,
        .z = sine_theta * std::sin(phi),
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] EnvironmentFor<Scalar>
make_environment(const std::uint32_t width, const std::uint32_t height,
                 std::vector<SpectrumFor<Scalar>> texels,
                 const QuaternionT<Scalar> environment_to_world = {}) {
    return require_value(EnvironmentFor<Scalar>::create(
        width, height, std::move(texels), wavelengths<Scalar>(), environment_to_world));
}

TEST(EnvironmentLightsContractTest, ExposesBothPrecisionsThroughTheLightContract) {
    static_assert(LightModelFor<EnvironmentMapLightT<TransportScalar>, TransportScalar>);
    static_assert(LightModelFor<EnvironmentMapLightT<ReferenceScalar>, ReferenceScalar>);
    static_assert(!LightModelFor<EnvironmentMapLightT<TransportScalar>, ReferenceScalar>);
    static_assert(!LightModelFor<EnvironmentMapLightT<ReferenceScalar>, TransportScalar>);
    static_assert(
        std::same_as<decltype(EnvironmentMapLightT<TransportScalar>::create(
                         std::uint32_t{}, std::uint32_t{}, std::vector<TransportSpectrum>{},
                         SampledWavelengths{}, Quaternion{})),
                     core::Result<EnvironmentMapLightT<TransportScalar>>>);

    const auto packet = wavelengths<TransportScalar>();
    const auto rotation = require_value(
        quaternion_from_axis_angle(Vector3{.y = 1.0F}, std::numbers::pi_v<TransportScalar> / 3.0F));
    const auto source = std::vector{gray(1.0F), gray(2.0F)};
    const auto light =
        EnvironmentMapLightT<TransportScalar>::create(2U, 1U, source, packet, rotation);
    ASSERT_TRUE(light.has_value()) << light.error().message;
    EXPECT_EQ(light->width(), 2U);
    EXPECT_EQ(light->height(), 1U);
    ASSERT_EQ(light->texels().size(), source.size());
    EXPECT_EQ(light->texels()[0], source[0]);
    EXPECT_EQ(light->texels()[1], source[1]);
    EXPECT_EQ(light->environment_to_world(), rotation);

    const auto support = light->bounds();
    EXPECT_FALSE(support.is_empty());
    EXPECT_TRUE(std::isinf(support.minimum().x));
    EXPECT_LT(support.minimum().x, 0.0F);
    EXPECT_TRUE(std::isinf(support.maximum().x));
    EXPECT_GT(support.maximum().x, 0.0F);
}

template <SpectrumScalar Scalar> void expect_cardinal_mapping_and_direction_scaling() {
    constexpr auto width = std::uint32_t{4};
    constexpr auto height = std::uint32_t{3};
    auto source = std::vector<SpectrumFor<Scalar>>{};
    source.reserve(static_cast<std::size_t>(width) * height);
    for (auto index = std::uint32_t{}; index < width * height; ++index) {
        source.push_back(gray(static_cast<Scalar>(index + 1U)));
    }
    const auto light = make_environment<Scalar>(width, height, source);
    const auto packet = wavelengths<Scalar>();

    struct DirectionCase final {
        Vector3T<Scalar> direction;
        std::uint32_t texel_index;
    };
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto cases = std::array{
        DirectionCase{.direction = {.x = Scalar{1}}, .texel_index = 4U},
        DirectionCase{.direction = {.z = Scalar{1}}, .texel_index = 5U},
        DirectionCase{.direction = {.x = Scalar{-1}}, .texel_index = 6U},
        DirectionCase{.direction = {.z = Scalar{-1}}, .texel_index = 7U},
        DirectionCase{.direction = {.y = Scalar{1}}, .texel_index = 0U},
        DirectionCase{.direction = {.y = Scalar{-1}}, .texel_index = 8U},
        DirectionCase{.direction = {.x = Scalar{7}}, .texel_index = 4U},
        DirectionCase{.direction = {.x = maximum}, .texel_index = 4U},
        DirectionCase{.direction = {.x = denormal}, .texel_index = 4U},
    };

    for (const auto& test_case : cases) {
        const auto emitted = light.le(ray<Scalar>(test_case.direction), packet);
        ASSERT_TRUE(emitted.has_value()) << emitted.error().message;
        EXPECT_EQ(*emitted, source[test_case.texel_index]);
    }
}

TEST(EnvironmentMapLightTest, MapsCardinalDirectionsAndNormalizesEscapedRays) {
    expect_cardinal_mapping_and_direction_scaling<TransportScalar>();
    expect_cardinal_mapping_and_direction_scaling<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_known_importance_ratio() {
    constexpr auto width = std::uint32_t{4};
    constexpr auto height = std::uint32_t{2};
    auto source = std::vector<SpectrumFor<Scalar>>(width * height);
    source[0] = gray(Scalar{1});
    source[width + 2U] = gray(Scalar{3});
    const auto light = make_environment<Scalar>(width, height, source);
    const auto packet = wavelengths<Scalar>();
    const auto query_context =
        context<Scalar>(Point3T<Scalar>{.x = Scalar{4}, .y = Scalar{-3}, .z = Scalar{2}});

    auto dim_count = std::size_t{};
    auto bright_count = std::size_t{};
    constexpr auto axis_samples = std::size_t{64};
    for (auto y = std::size_t{}; y < axis_samples; ++y) {
        for (auto x = std::size_t{}; x < axis_samples; ++x) {
            const auto canonical = Point2T<Scalar>{
                .x = (static_cast<Scalar>(x) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
                .y = (static_cast<Scalar>(y) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
            };
            const auto sampled = light.sample_li(query_context, canonical, packet);
            ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
            ASSERT_TRUE(sampled->has_value());
            const auto& sample = **sampled;
            ASSERT_EQ(sample.endpoint().kind(), LightEndpointKind::infinite);
            EXPECT_FALSE(sample.endpoint().position().has_value());
            EXPECT_TRUE(std::isinf(sample.distance()));
            EXPECT_EQ(sample.probability().measure, ProbabilityMeasure::solid_angle);

            if (sample.incident_radiance() == gray(Scalar{1})) {
                ++dim_count;
                expect_near(sample.probability().value,
                            ReferenceScalar{1} /
                                (ReferenceScalar{2} * std::numbers::pi_v<ReferenceScalar>));
            } else if (sample.incident_radiance() == gray(Scalar{3})) {
                ++bright_count;
                expect_near(sample.probability().value,
                            ReferenceScalar{3} /
                                (ReferenceScalar{2} * std::numbers::pi_v<ReferenceScalar>));
            } else {
                ADD_FAILURE() << "Importance sampling selected a zero-weight environment texel.";
            }

            const auto queried = light.pdf_li(query_context, sample.direction_to_light(), packet);
            ASSERT_TRUE(queried.has_value()) << queried.error().message;
            EXPECT_EQ(queried->measure(), ProbabilityMeasure::solid_angle);
            expect_near(queried->value(), static_cast<ReferenceScalar>(sample.probability().value));

            const auto escaped = light.le(ray<Scalar>(sample.direction_to_light()), packet);
            ASSERT_TRUE(escaped.has_value()) << escaped.error().message;
            EXPECT_EQ(*escaped, sample.incident_radiance());
        }
    }

    EXPECT_EQ(dim_count, 1024U);
    EXPECT_EQ(bright_count, 3072U);

    for (const auto canonical : std::array{
             Point2T<Scalar>{},
             Point2T<Scalar>{
                 .x = std::nextafter(Scalar{1}, Scalar{0}),
                 .y = std::nextafter(Scalar{1}, Scalar{0}),
             },
         }) {
        SCOPED_TRACE(::testing::Message() << "scalar bytes=" << sizeof(Scalar) << ", canonical=("
                                          << canonical.x << ", " << canonical.y << ")");
        const auto boundary = light.sample_li(query_context, canonical, packet);
        ASSERT_TRUE(boundary.has_value()) << boundary.error().message;
        EXPECT_TRUE(boundary->has_value());
    }

    const auto black_direction = texel_center_direction<Scalar>(1U, 0U, width, height);
    const auto black_pdf = light.pdf_li(query_context, black_direction, packet);
    ASSERT_TRUE(black_pdf.has_value()) << black_pdf.error().message;
    EXPECT_EQ(black_pdf->value(), Scalar{0});
}

TEST(EnvironmentMapLightTest, SamplesTheKnownTwoDimensionalImportanceRatio) {
    expect_known_importance_ratio<TransportScalar>();
    expect_known_importance_ratio<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_pdf_integrates_to_one(const QuaternionT<Scalar> environment_to_world) {
    constexpr auto width = std::uint32_t{8};
    constexpr auto height = std::uint32_t{4};
    auto source = std::vector<SpectrumFor<Scalar>>{};
    source.reserve(static_cast<std::size_t>(width) * height);
    for (auto row = std::uint32_t{}; row < height; ++row) {
        for (auto column = std::uint32_t{}; column < width; ++column) {
            const auto value = Scalar{1} + static_cast<Scalar>((3U * row + 5U * column) % 11U);
            source.push_back(gray(value));
        }
    }
    const auto light =
        make_environment<Scalar>(width, height, std::move(source), environment_to_world);
    const auto transform = require_value(AffineTransformT<Scalar>::rotation(environment_to_world));
    const auto packet = wavelengths<Scalar>();
    const auto query_context = context<Scalar>();

    auto integral = ReferenceScalar{};
    for (auto row = std::uint32_t{}; row < height; ++row) {
        const auto solid_angle = texel_solid_angle<Scalar>(row, width, height);
        for (auto column = std::uint32_t{}; column < width; ++column) {
            const auto local_direction = texel_center_direction<Scalar>(column, row, width, height);
            const auto world_direction =
                require_value(normalized(transform.apply(local_direction)));
            const auto pdf = light.pdf_li(query_context, world_direction, packet);
            ASSERT_TRUE(pdf.has_value()) << pdf.error().message;
            EXPECT_TRUE(std::isfinite(pdf->value()));
            EXPECT_GE(pdf->value(), Scalar{0});
            integral += static_cast<ReferenceScalar>(pdf->value()) * solid_angle;
        }
    }
    EXPECT_NEAR(integral, ReferenceScalar{1}, EnvironmentTolerance<Scalar>);
}

TEST(EnvironmentMapLightTest, IntegratesTheSolidAnglePdfToOneBeforeAndAfterRotation) {
    expect_pdf_integrates_to_one<TransportScalar>(Quaternion{});
    expect_pdf_integrates_to_one<ReferenceScalar>(ReferenceQuaternion{});

    const auto transport_rotation =
        require_value(quaternion_from_axis_angle(Vector3{.x = 1.0F, .y = 2.0F, .z = -1.0F}, 0.7F));
    const auto reference_rotation = require_value(
        quaternion_from_axis_angle(ReferenceVector3{.x = 1.0, .y = 2.0, .z = -1.0}, 0.7));
    expect_pdf_integrates_to_one<TransportScalar>(transport_rotation);
    expect_pdf_integrates_to_one<ReferenceScalar>(reference_rotation);
}

template <SpectrumScalar Scalar> void expect_rotation_covariance() {
    constexpr auto width = std::uint32_t{4};
    constexpr auto height = std::uint32_t{3};
    auto source = std::vector<SpectrumFor<Scalar>>(width * height);
    const auto emitted = lanes(Scalar{1}, Scalar{2}, Scalar{4}, Scalar{8});
    source[width] = emitted;
    const auto packet = wavelengths<Scalar>();
    const auto rotation = require_value(quaternion_from_axis_angle(
        Vector3T<Scalar>{.y = Scalar{1}}, std::numbers::pi_v<Scalar> / Scalar{2}));
    const auto transform = require_value(AffineTransformT<Scalar>::rotation(rotation));
    const auto identity = make_environment<Scalar>(width, height, source);
    const auto rotated = make_environment<Scalar>(width, height, source, rotation);
    const auto query_context = context<Scalar>();
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.375}, .y = Scalar{0.625}};

    const auto identity_sample = identity.sample_li(query_context, canonical, packet);
    const auto rotated_sample = rotated.sample_li(query_context, canonical, packet);
    ASSERT_TRUE(identity_sample.has_value()) << identity_sample.error().message;
    ASSERT_TRUE(rotated_sample.has_value()) << rotated_sample.error().message;
    ASSERT_TRUE(identity_sample->has_value());
    ASSERT_TRUE(rotated_sample->has_value());
    const auto expected_rotated_direction =
        require_value(normalized(transform.apply((**identity_sample).direction_to_light())));
    expect_direction_near((**rotated_sample).direction_to_light(), expected_rotated_direction);
    EXPECT_EQ((**identity_sample).incident_radiance(), emitted);
    EXPECT_EQ((**rotated_sample).incident_radiance(), emitted);
    expect_near((**rotated_sample).probability().value,
                static_cast<ReferenceScalar>((**identity_sample).probability().value));

    const auto identity_pdf =
        identity.pdf_li(query_context, (**identity_sample).direction_to_light(), packet);
    const auto rotated_pdf =
        rotated.pdf_li(query_context, (**rotated_sample).direction_to_light(), packet);
    ASSERT_TRUE(identity_pdf.has_value()) << identity_pdf.error().message;
    ASSERT_TRUE(rotated_pdf.has_value()) << rotated_pdf.error().message;
    expect_near(rotated_pdf->value(), static_cast<ReferenceScalar>(identity_pdf->value()));

    const auto identity_le =
        identity.le(ray<Scalar>((**identity_sample).direction_to_light()), packet);
    const auto rotated_le =
        rotated.le(ray<Scalar>((**rotated_sample).direction_to_light()), packet);
    ASSERT_TRUE(identity_le.has_value()) << identity_le.error().message;
    ASSERT_TRUE(rotated_le.has_value()) << rotated_le.error().message;
    EXPECT_EQ(*identity_le, emitted);
    EXPECT_EQ(*rotated_le, emitted);

    const auto old_world_direction = identity_sample->value().direction_to_light();
    const auto moved = rotated.le(ray<Scalar>(old_world_direction), packet);
    ASSERT_TRUE(moved.has_value()) << moved.error().message;
    EXPECT_EQ(*moved, SpectrumFor<Scalar>{});

    const auto identity_power = identity.power(unit_scene_bounds<Scalar>(), packet);
    const auto rotated_power = rotated.power(unit_scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(identity_power.has_value()) << identity_power.error().message;
    ASSERT_TRUE(rotated_power.has_value()) << rotated_power.error().message;
    EXPECT_EQ(*rotated_power, *identity_power);
}

TEST(EnvironmentMapLightTest, RotatesSamplingLookupAndPdfTogether) {
    expect_rotation_covariance<TransportScalar>();
    expect_rotation_covariance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_rotated_boundary_samples_round_trip() {
    constexpr auto width = std::uint32_t{8};
    constexpr auto height = std::uint32_t{4};
    auto source = std::vector<SpectrumFor<Scalar>>{};
    source.reserve(static_cast<std::size_t>(width) * height);
    for (auto index = std::uint32_t{}; index < width * height; ++index) {
        source.push_back(gray(static_cast<Scalar>(index + 1U)));
    }
    const auto packet = wavelengths<Scalar>();
    const auto rotation = require_value(quaternion_from_axis_angle(
        Vector3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{-1}}, Scalar{0.7}));
    const auto light = make_environment<Scalar>(width, height, source, rotation);
    const auto query_context = context<Scalar>();
    const auto below_one = std::nextafter(Scalar{1}, Scalar{0});

    for (const auto canonical : std::array{
             Point2T<Scalar>{},
             Point2T<Scalar>{.x = below_one},
             Point2T<Scalar>{.y = below_one},
             Point2T<Scalar>{.x = below_one, .y = below_one},
         }) {
        const auto sampled = light.sample_li(query_context, canonical, packet);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        ASSERT_TRUE(sampled->has_value());
        const auto& incident = **sampled;

        const auto queried_pdf = light.pdf_li(query_context, incident.direction_to_light(), packet);
        ASSERT_TRUE(queried_pdf.has_value()) << queried_pdf.error().message;
        expect_near(queried_pdf->value(),
                    static_cast<ReferenceScalar>(incident.probability().value));

        const auto queried_radiance = light.le(ray<Scalar>(incident.direction_to_light()), packet);
        ASSERT_TRUE(queried_radiance.has_value()) << queried_radiance.error().message;
        EXPECT_EQ(*queried_radiance, incident.incident_radiance());
    }
}

TEST(EnvironmentMapLightTest, KeepsRotatedBoundarySamplesInTheirSelectedTexels) {
    expect_rotated_boundary_samples_round_trip<TransportScalar>();
    expect_rotated_boundary_samples_round_trip<ReferenceScalar>();
}

TEST(EnvironmentMapLightTest, KeepsInteriorSamplesInTheirSelectedTexelsAfterRotation) {
    constexpr auto width = std::uint32_t{17};
    constexpr auto height = std::uint32_t{9};
    auto source = std::vector<TransportSpectrum>{};
    source.reserve(static_cast<std::size_t>(width) * height);
    for (auto index = std::uint32_t{}; index < width * height; ++index) {
        source.push_back(gray(static_cast<TransportScalar>(index + 1U)));
    }
    const auto packet = wavelengths<TransportScalar>();
    const auto rotation =
        require_value(quaternion_from_axis_angle(Vector3{.x = 1.0F, .y = 2.0F, .z = -1.0F}, 0.7F));
    const auto light = make_environment<TransportScalar>(width, height, source, rotation);
    const auto query_context = context<TransportScalar>();
    const auto canonical = Point2{
        .x = std::bit_cast<TransportScalar>(std::uint32_t{0x3f126349}),
        .y = std::bit_cast<TransportScalar>(std::uint32_t{0x3f1b6ca9}),
    };

    const auto sampled = light.sample_li(query_context, canonical, packet);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    const auto& incident = **sampled;
    EXPECT_EQ(incident.incident_radiance(), gray(96.0F));

    const auto queried_radiance = light.le(ray(incident.direction_to_light()), packet);
    ASSERT_TRUE(queried_radiance.has_value()) << queried_radiance.error().message;
    EXPECT_EQ(*queried_radiance, incident.incident_radiance());

    const auto queried_pdf = light.pdf_li(query_context, incident.direction_to_light(), packet);
    ASSERT_TRUE(queried_pdf.has_value()) << queried_pdf.error().message;
    EXPECT_EQ(queried_pdf->value(), incident.probability().value);
}

template <SpectrumScalar Scalar> void expect_every_spectral_lane_has_importance_support() {
    constexpr auto width = std::uint32_t{4};
    constexpr auto height = std::uint32_t{2};
    auto source = std::vector<SpectrumFor<Scalar>>(width * height);
    for (auto lane_index = std::size_t{}; lane_index < TransportSpectrumSampleCount; ++lane_index) {
        source[lane_index][lane_index] = Scalar{1};
    }
    const auto light = make_environment<Scalar>(width, height, source);
    const auto packet = wavelengths<Scalar>();
    const auto query_context = context<Scalar>();
    auto lane_counts = std::array<std::size_t, TransportSpectrumSampleCount>{};
    constexpr auto axis_samples = std::size_t{64};
    for (auto y = std::size_t{}; y < axis_samples; ++y) {
        for (auto x = std::size_t{}; x < axis_samples; ++x) {
            const auto sampled = light.sample_li(
                query_context,
                Point2T<Scalar>{
                    .x = (static_cast<Scalar>(x) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
                    .y = (static_cast<Scalar>(y) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
                },
                packet);
            ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
            ASSERT_TRUE(sampled->has_value());
            auto selected_lane = TransportSpectrumSampleCount;
            for (auto lane_index = std::size_t{}; lane_index < TransportSpectrumSampleCount;
                 ++lane_index) {
                const auto value = (**sampled).incident_radiance()[lane_index];
                if (value == Scalar{1}) {
                    ASSERT_EQ(selected_lane, TransportSpectrumSampleCount);
                    selected_lane = lane_index;
                } else {
                    EXPECT_EQ(value, Scalar{0});
                }
            }
            ASSERT_LT(selected_lane, TransportSpectrumSampleCount);
            ++lane_counts[selected_lane];
        }
    }
    for (const auto count : lane_counts) {
        EXPECT_EQ(count, 1024U);
    }
}

TEST(EnvironmentMapLightTest, GivesEverySpectralLaneImportanceSupport) {
    expect_every_spectral_lane_has_importance_support<TransportScalar>();
    expect_every_spectral_lane_has_importance_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_packet_and_black_contracts() {
    const auto packet = wavelengths<Scalar>();
    auto replay_packet = packet;
    for (auto& wavelength : replay_packet.samples) {
        wavelength.probability.value *= Scalar{2};
    }
    const auto different_packet = require_value(sample_uniform_visible_wavelengths(Scalar{0.5}));
    const auto query_context = context<Scalar>();
    const auto query_direction = Vector3T<Scalar>{.x = Scalar{1}};
    const auto escaped_ray = ray<Scalar>(query_direction);
    const auto bounds = unit_scene_bounds<Scalar>();

    const auto radiance = lanes(Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4});
    const auto light = require_value(
        EnvironmentFor<Scalar>::create(1U, 1U, {radiance}, packet, QuaternionT<Scalar>{}));
    const auto replayed_sample = light.sample_li(query_context, Point2T<Scalar>{}, replay_packet);
    ASSERT_TRUE(replayed_sample.has_value()) << replayed_sample.error().message;
    EXPECT_TRUE(replayed_sample->has_value());
    EXPECT_TRUE(light.pdf_li(query_context, query_direction, replay_packet).has_value());
    const auto replayed_le = light.le(escaped_ray, replay_packet);
    ASSERT_TRUE(replayed_le.has_value()) << replayed_le.error().message;
    EXPECT_EQ(*replayed_le, radiance);
    EXPECT_TRUE(light.power(bounds, replay_packet).has_value());

    expect_invalid(light.sample_li(query_context, Point2T<Scalar>{}, different_packet));
    expect_invalid(light.pdf_li(query_context, query_direction, different_packet));
    expect_invalid(light.le(escaped_ray, different_packet));
    expect_invalid(light.power(bounds, different_packet));

    const auto black = make_environment<Scalar>(2U, 2U, std::vector<SpectrumFor<Scalar>>(4U));
    for (const auto canonical : std::array{
             Point2T<Scalar>{},
             Point2T<Scalar>{
                 .x = std::nextafter(Scalar{1}, Scalar{0}),
                 .y = std::nextafter(Scalar{1}, Scalar{0}),
             },
         }) {
        const auto sampled = black.sample_li(query_context, canonical, packet);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        EXPECT_FALSE(sampled->has_value());
    }
    const auto black_pdf = black.pdf_li(query_context, query_direction, packet);
    const auto black_le = black.le(escaped_ray, packet);
    const auto black_power = black.power(bounds, packet);
    ASSERT_TRUE(black_pdf.has_value()) << black_pdf.error().message;
    ASSERT_TRUE(black_le.has_value()) << black_le.error().message;
    ASSERT_TRUE(black_power.has_value()) << black_power.error().message;
    EXPECT_EQ(black_pdf->value(), Scalar{0});
    EXPECT_EQ(*black_le, SpectrumFor<Scalar>{});
    EXPECT_EQ(*black_power, SpectrumFor<Scalar>{});
}

TEST(EnvironmentMapLightTest, ReplaysPacketMetadataAndTreatsBlackAsPhysicalZero) {
    expect_packet_and_black_contracts<TransportScalar>();
    expect_packet_and_black_contracts<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_constant_pdf_and_power() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = lanes(Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4});
    const auto light = require_value(
        EnvironmentFor<Scalar>::create(1U, 1U, {radiance}, packet, QuaternionT<Scalar>{}));
    const auto query_context = context<Scalar>();
    for (const auto direction : std::array{
             Vector3T<Scalar>{.x = Scalar{1}},
             Vector3T<Scalar>{.x = Scalar{-1}},
             Vector3T<Scalar>{.y = Scalar{1}},
             Vector3T<Scalar>{.y = Scalar{-1}},
             Vector3T<Scalar>{.z = Scalar{1}},
             Vector3T<Scalar>{.z = Scalar{-1}},
         }) {
        const auto pdf = light.pdf_li(query_context, direction, packet);
        ASSERT_TRUE(pdf.has_value()) << pdf.error().message;
        expect_near(pdf->value(), ReferenceScalar{1} /
                                      (ReferenceScalar{4} * std::numbers::pi_v<ReferenceScalar>));
    }

    const auto power = light.power(unit_scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(power.has_value()) << power.error().message;
    const auto scale = ReferenceScalar{12} * std::numbers::pi_v<ReferenceScalar> *
                       std::numbers::pi_v<ReferenceScalar>;
    for (auto lane_index = std::size_t{}; lane_index < TransportSpectrumSampleCount; ++lane_index) {
        expect_near((*power)[lane_index],
                    scale * static_cast<ReferenceScalar>(radiance[lane_index]), scale);
    }

    const auto point_bounds =
        require_value(Bounds3T<Scalar>::from_minimum_maximum(Point3T<Scalar>{}, Point3T<Scalar>{}));
    const auto point_power = light.power(point_bounds, packet);
    ASSERT_TRUE(point_power.has_value()) << point_power.error().message;
    EXPECT_EQ(*point_power, SpectrumFor<Scalar>{});
}

TEST(EnvironmentMapLightTest, MakesAConstantMapUniformAndReportsFiniteScenePower) {
    expect_constant_pdf_and_power<TransportScalar>();
    expect_constant_pdf_and_power<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_inputs_rejected() {
    const auto packet = wavelengths<Scalar>();
    const auto valid_texel = gray(Scalar{1});
    expect_invalid(EnvironmentFor<Scalar>::create(0U, 1U, {}, packet, QuaternionT<Scalar>{}));
    expect_invalid(EnvironmentFor<Scalar>::create(1U, 0U, {}, packet, QuaternionT<Scalar>{}));
    expect_invalid(
        EnvironmentFor<Scalar>::create(2U, 2U, {valid_texel}, packet, QuaternionT<Scalar>{}));

    const auto smallest_positive = std::numeric_limits<Scalar>::denorm_min();
    expect_invalid(
        EnvironmentFor<Scalar>::create(3U, 1U, {valid_texel, gray(smallest_positive), valid_texel},
                                       packet, QuaternionT<Scalar>{}));
    const auto unresolvable_tail =
        std::numeric_limits<Scalar>::epsilon() * std::numeric_limits<Scalar>::epsilon();
    expect_invalid(EnvironmentFor<Scalar>::create(5U, 1U,
                                                  {
                                                      gray(Scalar{0.33419472}),
                                                      gray(Scalar{0.55540437}),
                                                      gray(Scalar{0.30724636}),
                                                      gray(Scalar{0.4484264}),
                                                      gray(unresolvable_tail),
                                                  },
                                                  packet, QuaternionT<Scalar>{}));
    expect_error(EnvironmentFor<Scalar>::create(std::numeric_limits<std::uint32_t>::max(),
                                                std::numeric_limits<std::uint32_t>::max(), {},
                                                packet, QuaternionT<Scalar>{}));

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    for (const auto invalid_value : std::array{
             std::nextafter(Scalar{0}, -infinity),
             std::numeric_limits<Scalar>::quiet_NaN(),
             infinity,
             -infinity,
         }) {
        for (auto lane_index = std::size_t{}; lane_index < TransportSpectrumSampleCount;
             ++lane_index) {
            auto malformed = valid_texel;
            malformed[lane_index] = invalid_value;
            expect_invalid(
                EnvironmentFor<Scalar>::create(1U, 1U, {malformed}, packet, QuaternionT<Scalar>{}));
        }
    }

    for (const auto rotation : std::array{
             QuaternionT<Scalar>{.w = Scalar{0}},
             QuaternionT<Scalar>{.w = Scalar{2}},
             QuaternionT<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN()},
             QuaternionT<Scalar>{.y = infinity},
         }) {
        expect_invalid(EnvironmentFor<Scalar>::create(1U, 1U, {valid_texel}, packet, rotation));
    }

    auto malformed_packet = packet;
    malformed_packet[0].probability.value = Scalar{0};
    expect_invalid(EnvironmentFor<Scalar>::create(1U, 1U, {valid_texel}, malformed_packet,
                                                  QuaternionT<Scalar>{}));

    const auto light = make_environment<Scalar>(1U, 1U, {valid_texel});
    const auto query_context = context<Scalar>();
    for (const auto canonical : std::array{
             Point2T<Scalar>{.x = Scalar{-0.25}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{1}},
             Point2T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.5}, .y = infinity},
         }) {
        expect_invalid(light.sample_li(query_context, canonical, packet));
    }

    for (const auto direction : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.x = Scalar{2}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN()},
             Vector3T<Scalar>{.y = infinity},
         }) {
        expect_invalid(light.pdf_li(query_context, direction, packet));
    }
    expect_invalid(light.power(Bounds3T<Scalar>::empty(), packet));
    expect_invalid(light.power(Bounds3T<Scalar>::unbounded(), packet));
}

TEST(EnvironmentMapLightValidationTest, RejectsMalformedInputsWithoutFallbacks) {
    expect_invalid_inputs_rejected<TransportScalar>();
    expect_invalid_inputs_rejected<ReferenceScalar>();

    const auto adjacent =
        std::nextafter(TransportScalar{1}, std::numeric_limits<TransportScalar>::infinity());
    EXPECT_FALSE(
        environment_light_detail::representable_open_interval(TransportScalar{1}, adjacent));
    EXPECT_TRUE(environment_light_detail::representable_open_interval(TransportScalar{1},
                                                                      TransportScalar{1.001}));
}

} // namespace
} // namespace blackframe::renderer
