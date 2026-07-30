#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
inline constexpr auto MappingTolerance =
    std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6} : ReferenceScalar{2.0e-13};

template <GeometryScalar Scalar>
void expect_near(const Scalar actual, const ReferenceScalar expected) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected, MappingTolerance<Scalar>);
}

template <GeometryScalar Scalar> void expect_unit_direction(const Vector3T<Scalar> direction) {
    EXPECT_TRUE(std::isfinite(direction.x));
    EXPECT_TRUE(std::isfinite(direction.y));
    EXPECT_TRUE(std::isfinite(direction.z));
    expect_near(length_squared(direction), 1.0);
}

template <GeometryScalar Scalar> void expect_known_mapping_values() {
    const auto disk_center =
        map_concentric_disk(Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(disk_center.has_value());
    EXPECT_EQ(*disk_center, (Point2T<Scalar>{}));

    const auto disk_axis =
        map_concentric_disk(Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.5}});
    ASSERT_TRUE(disk_axis.has_value());
    expect_near(disk_axis->x, 0.5);
    expect_near(disk_axis->y, 0.0);

    const auto disk_corner = map_concentric_disk(Point2T<Scalar>{});
    ASSERT_TRUE(disk_corner.has_value());
    expect_near(disk_corner->x, -ReferenceScalar{1} / std::numbers::sqrt2_v<ReferenceScalar>);
    expect_near(disk_corner->y, -ReferenceScalar{1} / std::numbers::sqrt2_v<ReferenceScalar>);

    const auto disk_asymmetric =
        map_concentric_disk(Point2T<Scalar>{.x = Scalar{0.875}, .y = Scalar{0.625}});
    ASSERT_TRUE(disk_asymmetric.has_value());
    expect_near(disk_asymmetric->x, 0.7244443697168013);
    expect_near(disk_asymmetric->y, 0.19411428382689055);

    const auto sphere_pole = map_uniform_sphere(Point2T<Scalar>{});
    ASSERT_TRUE(sphere_pole.has_value());
    EXPECT_EQ(*sphere_pole, (Vector3T<Scalar>{.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{1}}));

    const auto sphere_equator =
        map_uniform_sphere(Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.25}});
    ASSERT_TRUE(sphere_equator.has_value());
    expect_near(sphere_equator->x, 0.0);
    expect_near(sphere_equator->y, 1.0);
    expect_near(sphere_equator->z, 0.0);
    expect_unit_direction(*sphere_equator);

    const auto uniform =
        map_uniform_hemisphere(Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.25}});
    ASSERT_TRUE(uniform.has_value());
    expect_near(uniform->x, 0.0);
    expect_near(uniform->y, std::sqrt(0.75));
    expect_near(uniform->z, 0.5);
    expect_unit_direction(*uniform);

    const auto cosine_center =
        map_cosine_hemisphere(Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(cosine_center.has_value());
    EXPECT_EQ(*cosine_center, (Vector3T<Scalar>{.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{1}}));

    const auto cosine_axis =
        map_cosine_hemisphere(Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.5}});
    ASSERT_TRUE(cosine_axis.has_value());
    expect_near(cosine_axis->x, 0.5);
    expect_near(cosine_axis->y, 0.0);
    expect_near(cosine_axis->z, std::sqrt(0.75));
    expect_unit_direction(*cosine_axis);
}

TEST(SamplingMappingsTest, MapsCanonicalValuesInBothPrecisions) {
    static_assert(std::is_same_v<decltype(map_concentric_disk(Point2{})), core::Result<Point2>>);
    static_assert(std::is_same_v<decltype(map_uniform_sphere(Point2{})), core::Result<Vector3>>);
    static_assert(std::is_same_v<decltype(map_uniform_hemisphere(ReferencePoint2{})),
                                 core::Result<ReferenceVector3>>);
    static_assert(std::is_same_v<decltype(map_cosine_hemisphere(ReferencePoint2{})),
                                 core::Result<ReferenceVector3>>);

    expect_known_mapping_values<TransportScalar>();
    expect_known_mapping_values<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_mapping_moments() {
    constexpr auto axis_samples = std::size_t{256};
    constexpr auto sample_count = axis_samples * axis_samples;
    auto disk_radius_squared_sum = 0.0L;
    auto sphere_z_sum = 0.0L;
    auto uniform_z_sum = 0.0L;
    auto cosine_z_sum = 0.0L;
    auto maximum_disk_radius_squared = 0.0L;
    auto maximum_direction_length_error = 0.0L;
    auto domains_are_valid = true;
    auto outputs_are_finite = true;

    for (auto y = std::size_t{0}; y < axis_samples; ++y) {
        for (auto x = std::size_t{0}; x < axis_samples; ++x) {
            const auto sample = Point2T<Scalar>{
                .x = (static_cast<Scalar>(x) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
                .y = (static_cast<Scalar>(y) + Scalar{0.5}) / static_cast<Scalar>(axis_samples),
            };
            const auto disk = map_concentric_disk(sample);
            const auto sphere = map_uniform_sphere(sample);
            const auto uniform = map_uniform_hemisphere(sample);
            const auto cosine = map_cosine_hemisphere(sample);
            if (!disk.has_value() || !sphere.has_value() || !uniform.has_value() ||
                !cosine.has_value()) {
                ADD_FAILURE() << "A valid canonical sample was rejected.";
                return;
            }

            const auto disk_radius_squared =
                static_cast<long double>(disk->x) * static_cast<long double>(disk->x) +
                static_cast<long double>(disk->y) * static_cast<long double>(disk->y);
            maximum_disk_radius_squared =
                std::max(maximum_disk_radius_squared, disk_radius_squared);
            domains_are_valid = domains_are_valid && sphere->z >= Scalar{-1} &&
                                sphere->z <= Scalar{1} && uniform->z >= Scalar{0} &&
                                uniform->z < Scalar{1} && cosine->z >= Scalar{0} &&
                                cosine->z <= Scalar{1};
            outputs_are_finite = outputs_are_finite && std::isfinite(disk->x) &&
                                 std::isfinite(disk->y) && std::isfinite(sphere->x) &&
                                 std::isfinite(sphere->y) && std::isfinite(sphere->z) &&
                                 std::isfinite(uniform->x) && std::isfinite(uniform->y) &&
                                 std::isfinite(uniform->z) && std::isfinite(cosine->x) &&
                                 std::isfinite(cosine->y) && std::isfinite(cosine->z);
            for (const auto direction : std::array{*sphere, *uniform, *cosine}) {
                const auto length_error =
                    std::abs(static_cast<long double>(direction.x) * direction.x +
                             static_cast<long double>(direction.y) * direction.y +
                             static_cast<long double>(direction.z) * direction.z - 1.0L);
                maximum_direction_length_error =
                    std::max(maximum_direction_length_error, length_error);
            }

            disk_radius_squared_sum += disk_radius_squared;
            sphere_z_sum += sphere->z;
            uniform_z_sum += uniform->z;
            cosine_z_sum += cosine->z;
        }
    }

    const auto inverse_count = 1.0L / static_cast<long double>(sample_count);
    EXPECT_TRUE(outputs_are_finite);
    EXPECT_TRUE(domains_are_valid);
    EXPECT_LE(maximum_disk_radius_squared,
              1.0L + 4.0L * static_cast<long double>(MappingTolerance<Scalar>));
    EXPECT_LE(maximum_direction_length_error,
              4.0L * static_cast<long double>(MappingTolerance<Scalar>));
    EXPECT_NEAR(static_cast<double>(disk_radius_squared_sum * inverse_count), 0.5, 2.0e-4);
    EXPECT_NEAR(static_cast<double>(sphere_z_sum * inverse_count), 0.0, 2.0e-6);
    EXPECT_NEAR(static_cast<double>(uniform_z_sum * inverse_count), 0.5, 2.0e-6);
    EXPECT_NEAR(static_cast<double>(cosine_z_sum * inverse_count), 2.0 / 3.0, 2.0e-4);
}

TEST(SamplingMappingsTest, PreservesTheExpectedDeterministicMoments) {
    expect_mapping_moments<TransportScalar>();
    expect_mapping_moments<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_invalid_inputs_rejected() {
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto invalid_samples = std::array{
        Point2T<Scalar>{.x = Scalar{-0.25}, .y = Scalar{0.5}},
        Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
        Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{1}},
        Point2T<Scalar>{.x = nan, .y = Scalar{0.5}},
        Point2T<Scalar>{.x = Scalar{0.5}, .y = infinity},
        Point2T<Scalar>{.x = Scalar{0.5}, .y = -infinity},
    };

    for (const auto sample : invalid_samples) {
        const auto disk = map_concentric_disk(sample);
        const auto sphere = map_uniform_sphere(sample);
        const auto uniform = map_uniform_hemisphere(sample);
        const auto cosine = map_cosine_hemisphere(sample);
        ASSERT_FALSE(disk.has_value());
        ASSERT_FALSE(sphere.has_value());
        ASSERT_FALSE(uniform.has_value());
        ASSERT_FALSE(cosine.has_value());
        EXPECT_EQ(disk.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(sphere.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(uniform.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(cosine.error().code, core::StatusCode::invalid_argument);
    }

    const auto last = std::nextafter(Scalar{1}, Scalar{0});
    const auto last_sample = Point2T<Scalar>{.x = last, .y = last};
    EXPECT_TRUE(map_concentric_disk(last_sample).has_value());
    EXPECT_TRUE(map_uniform_sphere(last_sample).has_value());
    EXPECT_TRUE(map_uniform_hemisphere(last_sample).has_value());
    EXPECT_TRUE(map_cosine_hemisphere(last_sample).has_value());

    const auto signed_zero_sample = Point2T<Scalar>{.x = -Scalar{0}, .y = Scalar{0.5}};
    EXPECT_TRUE(map_concentric_disk(signed_zero_sample).has_value());
    EXPECT_TRUE(map_uniform_sphere(signed_zero_sample).has_value());
    EXPECT_TRUE(map_uniform_hemisphere(signed_zero_sample).has_value());
    EXPECT_TRUE(map_cosine_hemisphere(signed_zero_sample).has_value());
}

TEST(SamplingMappingsTest, RejectsMalformedSquareSamplesWithoutSubstitution) {
    expect_invalid_inputs_rejected<TransportScalar>();
    expect_invalid_inputs_rejected<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_pdf_values_and_errors() {
    static_assert(std::is_same_v<decltype(uniform_disk_pdf<Scalar>()), Scalar>);
    static_assert(std::is_same_v<decltype(uniform_sphere_pdf<Scalar>()), Scalar>);
    static_assert(std::is_same_v<decltype(uniform_hemisphere_pdf<Scalar>()), Scalar>);
    static_assert(std::is_same_v<decltype(cosine_hemisphere_pdf(Scalar{})), core::Result<Scalar>>);

    expect_near(uniform_disk_pdf<Scalar>(), std::numbers::inv_pi_v<ReferenceScalar>);
    expect_near(uniform_sphere_pdf<Scalar>(),
                std::numbers::inv_pi_v<ReferenceScalar> / ReferenceScalar{4});
    expect_near(uniform_hemisphere_pdf<Scalar>(),
                std::numbers::inv_pi_v<ReferenceScalar> / ReferenceScalar{2});

    const auto peak = cosine_hemisphere_pdf(Scalar{1});
    const auto half = cosine_hemisphere_pdf(Scalar{0.5});
    const auto horizon = cosine_hemisphere_pdf(Scalar{0});
    const auto below = cosine_hemisphere_pdf(Scalar{-0.5});
    const auto negative_zero = cosine_hemisphere_pdf(-Scalar{0});
    ASSERT_TRUE(peak.has_value());
    ASSERT_TRUE(half.has_value());
    ASSERT_TRUE(horizon.has_value());
    ASSERT_TRUE(below.has_value());
    ASSERT_TRUE(negative_zero.has_value());
    expect_near(*peak, std::numbers::inv_pi_v<ReferenceScalar>);
    expect_near(*half, std::numbers::inv_pi_v<ReferenceScalar> / ReferenceScalar{2});
    EXPECT_EQ(*horizon, Scalar{0});
    EXPECT_EQ(*below, Scalar{0});
    EXPECT_EQ(*negative_zero, Scalar{0});
    EXPECT_FALSE(std::signbit(*negative_zero));

    for (const auto invalid : std::array{
             std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
             std::nextafter(Scalar{-1}, -std::numeric_limits<Scalar>::infinity()),
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
             -std::numeric_limits<Scalar>::infinity(),
         }) {
        const auto pdf = cosine_hemisphere_pdf(invalid);
        ASSERT_FALSE(pdf.has_value());
        EXPECT_EQ(pdf.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(pdf.error().message,
                  "Cosine hemisphere PDF requires a finite cosine in [-1, 1].");
    }
}

TEST(SamplingMappingsTest, DefinesAnalyticPdfsAndExplicitSupport) {
    expect_pdf_values_and_errors<TransportScalar>();
    expect_pdf_values_and_errors<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_every_pdf_integrates_to_one() {
    constexpr auto radial_steps = std::size_t{64};
    constexpr auto azimuth_steps = std::size_t{128};
    constexpr auto cosine_steps = std::size_t{128};
    constexpr auto pi = std::numbers::pi_v<long double>;
    constexpr auto delta_radius = 1.0L / static_cast<long double>(radial_steps);
    constexpr auto delta_azimuth = 2.0L * pi / static_cast<long double>(azimuth_steps);
    constexpr auto delta_sphere_cosine = 2.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_hemisphere_cosine = 1.0L / static_cast<long double>(cosine_steps);

    auto disk_integral = 0.0L;
    for (auto radial_index = std::size_t{0}; radial_index < radial_steps; ++radial_index) {
        const auto radius = (static_cast<long double>(radial_index) + 0.5L) * delta_radius;
        for (auto azimuth_index = std::size_t{0}; azimuth_index < azimuth_steps; ++azimuth_index) {
            disk_integral += static_cast<long double>(uniform_disk_pdf<Scalar>()) * radius *
                             delta_radius * delta_azimuth;
        }
    }

    auto sphere_integral = 0.0L;
    for (auto cosine_index = std::size_t{0}; cosine_index < cosine_steps; ++cosine_index) {
        for (auto azimuth_index = std::size_t{0}; azimuth_index < azimuth_steps; ++azimuth_index) {
            sphere_integral += static_cast<long double>(uniform_sphere_pdf<Scalar>()) *
                               delta_sphere_cosine * delta_azimuth;
        }
    }

    auto uniform_hemisphere_integral = 0.0L;
    auto cosine_hemisphere_integral = 0.0L;
    for (auto cosine_index = std::size_t{0}; cosine_index < cosine_steps; ++cosine_index) {
        const auto sphere_cosine =
            -1.0L + (static_cast<long double>(cosine_index) + 0.5L) * delta_sphere_cosine;
        const auto cosine_pdf = cosine_hemisphere_pdf(static_cast<Scalar>(sphere_cosine));
        ASSERT_TRUE(cosine_pdf.has_value());
        for (auto azimuth_index = std::size_t{0}; azimuth_index < azimuth_steps; ++azimuth_index) {
            uniform_hemisphere_integral +=
                static_cast<long double>(uniform_hemisphere_pdf<Scalar>()) *
                delta_hemisphere_cosine * delta_azimuth;
            cosine_hemisphere_integral +=
                static_cast<long double>(*cosine_pdf) * delta_sphere_cosine * delta_azimuth;
        }
    }

    const auto tolerance = std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6}
                                                                   : ReferenceScalar{2.0e-12};
    EXPECT_NEAR(static_cast<ReferenceScalar>(disk_integral), 1.0, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(sphere_integral), 1.0, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(uniform_hemisphere_integral), 1.0, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(cosine_hemisphere_integral), 1.0, tolerance);
}

TEST(SamplingMappingsTest, IntegratesEveryPdfToOneInBothPrecisions) {
    expect_every_pdf_integrates_to_one<TransportScalar>();
    expect_every_pdf_integrates_to_one<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
