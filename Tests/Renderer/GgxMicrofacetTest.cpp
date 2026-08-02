#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
using GgxFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, GgxMicrofacet,
                                  ReferenceGgxMicrofacet>;

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <GeometryScalar Scalar> void expect_creation_contract() {
    for (const auto alpha : std::array{Scalar{0.1}, Scalar{1}, Scalar{2}}) {
        const auto distribution = GgxFor<Scalar>::create(alpha);
        ASSERT_TRUE(distribution.has_value());
        EXPECT_EQ(distribution->alpha(), alpha);
    }

    for (const auto alpha : std::array{
             Scalar{0},
             -std::numeric_limits<Scalar>::denorm_min(),
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
         }) {
        expect_invalid(GgxFor<Scalar>::create(alpha));
    }

    const auto representable_limit =
        std::sqrt(std::numeric_limits<Scalar>::max()) * std::sqrt(std::numbers::pi_v<Scalar>);
    for (const auto alpha : std::array{Scalar{1} / representable_limit, representable_limit}) {
        const auto boundary = GgxFor<Scalar>::create(alpha);
        ASSERT_TRUE(boundary.has_value());
        const auto value = boundary->normal_distribution(Normal3T<Scalar>{.z = Scalar{1}});
        ASSERT_TRUE(value.has_value());
        EXPECT_TRUE(std::isfinite(*value));
        EXPECT_GT(*value, Scalar{0});
    }
    const auto widest = GgxFor<Scalar>::create(representable_limit);
    ASSERT_TRUE(widest.has_value());
    const auto nearly_tangent_direction = Vector3T<Scalar>{
        .x = Scalar{1},
        .z = std::numeric_limits<Scalar>::denorm_min(),
    };
    const auto nearly_tangent_normal = Normal3T<Scalar>{
        .x = Scalar{1},
        .z = std::numeric_limits<Scalar>::denorm_min(),
    };
    const auto finite_pdf =
        widest->visible_normal_pdf(nearly_tangent_direction, nearly_tangent_normal);
    ASSERT_TRUE(finite_pdf.has_value());
    EXPECT_TRUE(std::isfinite(finite_pdf->value));
    EXPECT_GT(finite_pdf->value, Scalar{0});
    expect_invalid(widest->sample_visible_normal(
        nearly_tangent_direction, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}));
    expect_invalid(
        GgxFor<Scalar>::create(std::nextafter(Scalar{1} / representable_limit, Scalar{0})));
    expect_invalid(GgxFor<Scalar>::create(
        std::nextafter(representable_limit, std::numeric_limits<Scalar>::infinity())));
}

TEST(GgxMicrofacetTest, CreatesOnlyRepresentablePositiveContinuousWidths) {
    static_assert(std::same_as<decltype(GgxMicrofacet::create(TransportScalar{})),
                               core::Result<GgxMicrofacet>>);
    static_assert(std::same_as<decltype(ReferenceGgxMicrofacet::create(ReferenceScalar{})),
                               core::Result<ReferenceGgxMicrofacet>>);
    static_assert(!std::same_as<GgxMicrofacet, ReferenceGgxMicrofacet>);
    static_assert(!std::same_as<GgxVisibleNormalSample, ReferenceGgxVisibleNormalSample>);

    expect_creation_contract<TransportScalar>();
    expect_creation_contract<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_distribution_values() {
    const auto half_rough = GgxFor<Scalar>::create(Scalar{0.5});
    const auto unit_rough = GgxFor<Scalar>::create(Scalar{1});
    ASSERT_TRUE(half_rough.has_value());
    ASSERT_TRUE(unit_rough.has_value());

    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto tilted_x = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto tilted_y = Normal3T<Scalar>{.y = Scalar{0.6}, .z = Scalar{0.8}};
    const auto below = Normal3T<Scalar>{.z = Scalar{-1}};
    const auto tangent = Normal3T<Scalar>{.x = Scalar{1}};

    const auto normal_value = half_rough->normal_distribution(normal);
    const auto tilted_x_value = unit_rough->normal_distribution(tilted_x);
    const auto tilted_y_value = unit_rough->normal_distribution(tilted_y);
    const auto below_value = half_rough->normal_distribution(below);
    const auto tangent_value = half_rough->normal_distribution(tangent);
    ASSERT_TRUE(normal_value.has_value());
    ASSERT_TRUE(tilted_x_value.has_value());
    ASSERT_TRUE(tilted_y_value.has_value());
    ASSERT_TRUE(below_value.has_value());
    ASSERT_TRUE(tangent_value.has_value());

    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-6 : 2.0e-14;
    EXPECT_NEAR(static_cast<double>(*normal_value),
                static_cast<double>(Scalar{4} * std::numbers::inv_pi_v<Scalar>), tolerance);
    EXPECT_NEAR(static_cast<double>(*tilted_x_value),
                static_cast<double>(std::numbers::inv_pi_v<Scalar>), tolerance);
    EXPECT_EQ(*tilted_x_value, *tilted_y_value);
    EXPECT_EQ(*below_value, Scalar{0});
    EXPECT_EQ(*tangent_value, Scalar{0});

    const auto below_one = GgxFor<Scalar>::create(std::nextafter(Scalar{1}, Scalar{0}));
    const auto above_one =
        GgxFor<Scalar>::create(std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()));
    ASSERT_TRUE(below_one.has_value());
    ASSERT_TRUE(above_one.has_value());
    const auto below_one_value = below_one->normal_distribution(tilted_x);
    const auto above_one_value = above_one->normal_distribution(tilted_x);
    ASSERT_TRUE(below_one_value.has_value());
    ASSERT_TRUE(above_one_value.has_value());
    EXPECT_NEAR(static_cast<double>(*below_one_value), static_cast<double>(*above_one_value),
                tolerance);

    for (const auto malformed : std::array{
             Normal3T<Scalar>{},
             Normal3T<Scalar>{.x = Scalar{2}},
             Normal3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
             Normal3T<Scalar>{.x = std::numeric_limits<Scalar>::infinity(), .z = Scalar{1}},
         }) {
        expect_invalid(half_rough->normal_distribution(malformed));
    }
}

TEST(GgxMicrofacetTest, EvaluatesTheIsotropicProjectedNormalDistribution) {
    expect_distribution_values<TransportScalar>();
    expect_distribution_values<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_smith_values() {
    const auto distribution = GgxFor<Scalar>::create(Scalar{1});
    const auto wide_distribution = GgxFor<Scalar>::create(Scalar{2});
    ASSERT_TRUE(distribution.has_value());
    ASSERT_TRUE(wide_distribution.has_value());
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto tilted_x = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto tilted_y = Vector3T<Scalar>{.y = Scalar{0.6}, .z = Scalar{0.8}};
    const auto below = Vector3T<Scalar>{.z = Scalar{-1}};
    const auto tangent = Vector3T<Scalar>{.x = Scalar{1}};

    const auto lambda_normal = distribution->smith_lambda(normal);
    const auto lambda_x = distribution->smith_lambda(tilted_x);
    const auto lambda_y = distribution->smith_lambda(tilted_y);
    const auto g1_normal = distribution->smith_g1(normal);
    const auto g1_tilted = distribution->smith_g1(tilted_x);
    const auto correlated = distribution->smith_g2(tilted_x, tilted_x);
    const auto mixed_forward = distribution->smith_g2(normal, tilted_x);
    const auto mixed_reverse = distribution->smith_g2(tilted_x, normal);
    ASSERT_TRUE(lambda_normal.has_value());
    ASSERT_TRUE(lambda_x.has_value());
    ASSERT_TRUE(lambda_y.has_value());
    ASSERT_TRUE(g1_normal.has_value());
    ASSERT_TRUE(g1_tilted.has_value());
    ASSERT_TRUE(correlated.has_value());
    ASSERT_TRUE(mixed_forward.has_value());
    ASSERT_TRUE(mixed_reverse.has_value());

    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-6 : 2.0e-14;
    EXPECT_EQ(*lambda_normal, Scalar{0});
    EXPECT_NEAR(static_cast<double>(*lambda_x), 0.125, tolerance);
    EXPECT_EQ(*lambda_x, *lambda_y);
    EXPECT_EQ(*g1_normal, Scalar{1});
    EXPECT_NEAR(static_cast<double>(*g1_tilted), 8.0 / 9.0, tolerance);
    EXPECT_NEAR(static_cast<double>(*correlated), 0.8, tolerance);
    EXPECT_NEAR(static_cast<double>(*mixed_forward), 8.0 / 9.0, tolerance);
    EXPECT_EQ(*mixed_forward, *mixed_reverse);

    const auto wide_lambda = wide_distribution->smith_lambda(tilted_x);
    const auto wide_g1 = wide_distribution->smith_g1(tilted_x);
    const auto wide_g2 = wide_distribution->smith_g2(tilted_x, tilted_x);
    ASSERT_TRUE(wide_lambda.has_value());
    ASSERT_TRUE(wide_g1.has_value());
    ASSERT_TRUE(wide_g2.has_value());
    const auto wide_root = std::sqrt(Scalar{3.25});
    EXPECT_NEAR(static_cast<double>(*wide_lambda),
                static_cast<double>((wide_root - Scalar{1}) / Scalar{2}), tolerance);
    EXPECT_NEAR(static_cast<double>(*wide_g1),
                static_cast<double>(Scalar{2} / (Scalar{1} + wide_root)), tolerance);
    EXPECT_NEAR(static_cast<double>(*wide_g2), static_cast<double>(Scalar{1} / wide_root),
                tolerance);

    const auto below_g1 = distribution->smith_g1(below);
    const auto tangent_g1 = distribution->smith_g1(tangent);
    const auto below_g2 = distribution->smith_g2(normal, below);
    ASSERT_TRUE(below_g1.has_value());
    ASSERT_TRUE(tangent_g1.has_value());
    ASSERT_TRUE(below_g2.has_value());
    EXPECT_EQ(*below_g1, Scalar{0});
    EXPECT_EQ(*tangent_g1, Scalar{0});
    EXPECT_EQ(*below_g2, Scalar{0});
    expect_invalid(distribution->smith_lambda(below));
    expect_invalid(distribution->smith_lambda(tangent));
}

TEST(GgxMicrofacetTest, UsesHeightCorrelatedSmithMasking) {
    expect_smith_values<TransportScalar>();
    expect_smith_values<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_known_visible_normal_sample() {
    const auto distribution = GgxFor<Scalar>::create(Scalar{1});
    ASSERT_TRUE(distribution.has_value());
    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.125}};
    const auto sampled = distribution->sample_visible_normal(outgoing, canonical);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());

    const auto expected_radial = Scalar{0.5} / std::sqrt(Scalar{2});
    const auto expected_z = std::sqrt(Scalar{0.75});
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-6 : 2.0e-14;
    EXPECT_NEAR(static_cast<double>((**sampled).microfacet_normal.x),
                static_cast<double>(expected_radial), tolerance);
    EXPECT_NEAR(static_cast<double>((**sampled).microfacet_normal.y),
                static_cast<double>(expected_radial), tolerance);
    EXPECT_NEAR(static_cast<double>((**sampled).microfacet_normal.z),
                static_cast<double>(expected_z), tolerance);
    EXPECT_EQ((**sampled).probability.measure, ProbabilityMeasure::solid_angle);
    EXPECT_NEAR(static_cast<double>((**sampled).probability.value),
                static_cast<double>(expected_z * std::numbers::inv_pi_v<Scalar>), tolerance);

    const auto queried = distribution->visible_normal_pdf(outgoing, (**sampled).microfacet_normal);
    const auto replay = distribution->sample_visible_normal(outgoing, canonical);
    ASSERT_TRUE(queried.has_value());
    ASSERT_TRUE(replay.has_value());
    ASSERT_TRUE(replay->has_value());
    EXPECT_EQ((**sampled).probability.value, queried->value);
    EXPECT_EQ((**sampled).microfacet_normal, (**replay).microfacet_normal);
    EXPECT_EQ((**sampled).probability.value, (**replay).probability.value);
}

TEST(GgxMicrofacetTest, SamplesTheKnownNormalIncidenceVndfAndReplaysExactly) {
    expect_known_visible_normal_sample<TransportScalar>();
    expect_known_visible_normal_sample<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_visible_normal_domains() {
    const auto distribution = GgxFor<Scalar>::create(Scalar{0.45});
    ASSERT_TRUE(distribution.has_value());
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.36}, .y = Scalar{0.48}, .z = Scalar{0.8}};
    const auto below = Vector3T<Scalar>{.z = Scalar{-1}};
    const auto tangent = Vector3T<Scalar>{.x = Scalar{1}};
    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto invisible = Normal3T<Scalar>{
        .x = -static_cast<Scalar>(std::sqrt(0.96L)),
        .z = Scalar{0.2},
    };
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.37}, .y = Scalar{0.61}};
    const auto sampled = distribution->sample_visible_normal(outgoing, canonical);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    EXPECT_GT((**sampled).microfacet_normal.z, Scalar{0});
    EXPECT_GT(dot(outgoing, (**sampled).microfacet_normal), Scalar{0});
    EXPECT_EQ((**sampled).probability.measure, ProbabilityMeasure::solid_angle);
    EXPECT_GT((**sampled).probability.value, Scalar{0});

    const auto absent = distribution->sample_visible_normal(below, canonical);
    const auto tangent_absent = distribution->sample_visible_normal(tangent, canonical);
    ASSERT_TRUE(absent.has_value());
    ASSERT_TRUE(tangent_absent.has_value());
    EXPECT_FALSE(absent->has_value());
    EXPECT_FALSE(tangent_absent->has_value());
    const auto unsupported_pdf = distribution->visible_normal_pdf(below, normal);
    const auto invisible_pdf = distribution->visible_normal_pdf(outgoing, invisible);
    ASSERT_TRUE(unsupported_pdf.has_value());
    ASSERT_TRUE(invisible_pdf.has_value());
    EXPECT_EQ(unsupported_pdf->value, Scalar{0});
    EXPECT_EQ(unsupported_pdf->measure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ(invisible_pdf->value, Scalar{0});
    EXPECT_EQ(invisible_pdf->measure, ProbabilityMeasure::solid_angle);

    for (const auto malformed : std::array{
             Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = -std::numeric_limits<Scalar>::denorm_min(), .y = Scalar{0.5}},
             Point2T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.5}, .y = std::numeric_limits<Scalar>::infinity()},
         }) {
        expect_invalid(distribution->sample_visible_normal(outgoing, malformed));
    }
    for (const auto malformed : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.x = Scalar{2}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
         }) {
        expect_invalid(distribution->sample_visible_normal(malformed, canonical));
        expect_invalid(distribution->visible_normal_pdf(malformed, normal));
    }
}

TEST(GgxMicrofacetTest, KeepsVisibleSupportAndInvalidInputsExplicit) {
    expect_visible_normal_domains<TransportScalar>();
    expect_visible_normal_domains<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_bounded_roundoff_projection() {
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    const auto accepted =
        ggx_microfacet_detail::nonnegative_roundoff(-Scalar{8} * epsilon, Scalar{1});
    ASSERT_TRUE(accepted.has_value());
    EXPECT_EQ(*accepted, Scalar{0});
    expect_invalid(ggx_microfacet_detail::nonnegative_roundoff(-Scalar{32} * epsilon, Scalar{1}));
}

TEST(GgxMicrofacetTest, BoundsTheAnalyticVndfRoundoffProjection) {
    expect_bounded_roundoff_projection<TransportScalar>();
    expect_bounded_roundoff_projection<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_projected_distribution_normalization() {
    constexpr auto step_count = std::size_t{65'536};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(step_count);
    for (const auto alpha : std::array{Scalar{0.1}, Scalar{0.35}, Scalar{1}, Scalar{2}}) {
        const auto distribution = GgxFor<Scalar>::create(alpha);
        ASSERT_TRUE(distribution.has_value());
        auto integral = 0.0L;
        for (auto index = std::size_t{}; index < step_count; ++index) {
            const auto cosine = (static_cast<long double>(index) + 0.5L) * delta_cosine;
            const auto sine = std::sqrt((1.0L - cosine) * (1.0L + cosine));
            const auto microfacet_normal = Normal3T<Scalar>{
                .x = static_cast<Scalar>(sine),
                .z = static_cast<Scalar>(cosine),
            };
            const auto value = distribution->normal_distribution(microfacet_normal);
            ASSERT_TRUE(value.has_value());
            integral += Scalar{2} * std::numbers::pi_v<Scalar> * static_cast<long double>(*value) *
                        cosine * delta_cosine;
        }
        EXPECT_NEAR(integral, 1.0L, 2.0e-4L);
    }
}

TEST(GgxMicrofacetTest, NormalizesDInProjectedSolidAngle) {
    expect_projected_distribution_normalization<TransportScalar>();
    expect_projected_distribution_normalization<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_visible_normal_pdf_normalization() {
    constexpr auto cosine_steps = std::size_t{256};
    constexpr auto azimuth_steps = std::size_t{512};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    const auto distribution = GgxFor<Scalar>::create(Scalar{2});
    ASSERT_TRUE(distribution.has_value());
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.96}, .z = Scalar{0.28}};
    auto integral = 0.0L;
    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto sine = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto microfacet_normal = Normal3T<Scalar>{
                .x = static_cast<Scalar>(sine * std::cos(azimuth)),
                .y = static_cast<Scalar>(sine * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto probability = distribution->visible_normal_pdf(outgoing, microfacet_normal);
            ASSERT_TRUE(probability.has_value());
            EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
            integral += static_cast<long double>(probability->value) * delta_cosine * delta_azimuth;
        }
    }
    EXPECT_NEAR(integral, 1.0L, 2.0e-3L);
}

TEST(GgxMicrofacetTest, NormalizesTheObliqueVisibleNormalPdf) {
    expect_visible_normal_pdf_normalization<TransportScalar>();
    expect_visible_normal_pdf_normalization<ReferenceScalar>();
}

struct DiskCoordinates final {
    long double radial_cdf{};
    long double azimuth_cdf{};
};

struct LongVector final {
    long double x{};
    long double y{};
    long double z{};
};

struct LongNormal final {
    long double x{};
    long double y{};
    long double z{};
};

[[nodiscard]] std::optional<DiskCoordinates>
invert_visible_normal_mapping(const long double alpha, const LongVector outgoing,
                              const LongNormal microfacet_normal,
                              const long double inverse_tolerance) {
    const auto normalize =
        [](const std::array<long double, 3>& value) -> std::optional<std::array<long double, 3>> {
        const auto length = std::hypot(std::hypot(value[0], value[1]), value[2]);
        if (!std::isfinite(length) || !(length > 0.0L)) {
            return {};
        }
        return std::array{value[0] / length, value[1] / length, value[2] / length};
    };

    const auto view = normalize({alpha * outgoing.x, alpha * outgoing.y, outgoing.z});
    const auto hemisphere_normal =
        normalize({microfacet_normal.x / alpha, microfacet_normal.y / alpha, microfacet_normal.z});
    if (!view || !hemisphere_normal) {
        return {};
    }
    const auto tangent_length = std::hypot((*view)[0], (*view)[1]);
    const auto tangent = tangent_length > 0.0L ? std::array{-(*view)[1] / tangent_length,
                                                            (*view)[0] / tangent_length, 0.0L}
                                               : std::array{1.0L, 0.0L, 0.0L};
    const auto bitangent = std::array{
        -(*view)[2] * tangent[1],
        (*view)[2] * tangent[0],
        (*view)[0] * tangent[1] - (*view)[1] * tangent[0],
    };
    const auto disk_x = (*hemisphere_normal)[0] * tangent[0] +
                        (*hemisphere_normal)[1] * tangent[1] + (*hemisphere_normal)[2] * tangent[2];
    const auto warped_disk_y = (*hemisphere_normal)[0] * bitangent[0] +
                               (*hemisphere_normal)[1] * bitangent[1] +
                               (*hemisphere_normal)[2] * bitangent[2];
    auto disk_y_limit_squared = 1.0L - disk_x * disk_x;
    if (disk_y_limit_squared < -inverse_tolerance) {
        return {};
    }
    disk_y_limit_squared = std::max(0.0L, disk_y_limit_squared);
    const auto projection = 0.5L * (1.0L + (*view)[2]);
    const auto disk_y =
        (warped_disk_y - (1.0L - projection) * std::sqrt(disk_y_limit_squared)) / projection;
    auto radial_cdf = disk_x * disk_x + disk_y * disk_y;
    if (radial_cdf < -inverse_tolerance || radial_cdf > 1.0L + inverse_tolerance) {
        return {};
    }
    radial_cdf = std::clamp(radial_cdf, 0.0L, std::nextafter(1.0L, 0.0L));
    auto azimuth = std::atan2(disk_y, disk_x);
    if (azimuth < 0.0L) {
        azimuth += 2.0L * std::numbers::pi_v<long double>;
    }
    auto azimuth_cdf = azimuth / (2.0L * std::numbers::pi_v<long double>);
    azimuth_cdf = std::min(azimuth_cdf, std::nextafter(1.0L, 0.0L));
    return DiskCoordinates{.radial_cdf = radial_cdf, .azimuth_cdf = azimuth_cdf};
}

[[nodiscard]] long double independent_visible_normal_pdf(const long double alpha,
                                                         const LongVector outgoing,
                                                         const LongNormal microfacet_normal) {
    const auto radial_squared =
        microfacet_normal.x * microfacet_normal.x + microfacet_normal.y * microfacet_normal.y;
    const auto alpha_squared = alpha * alpha;
    const auto distribution_denominator =
        radial_squared + alpha_squared * microfacet_normal.z * microfacet_normal.z;
    const auto distribution = alpha_squared / (std::numbers::pi_v<long double> *
                                               distribution_denominator * distribution_denominator);
    const auto visible_dot = outgoing.x * microfacet_normal.x + outgoing.y * microfacet_normal.y +
                             outgoing.z * microfacet_normal.z;
    const auto outgoing_sine = std::hypot(outgoing.x, outgoing.y);
    const auto projected_radius = std::hypot(outgoing.z, alpha * outgoing_sine);
    return 2.0L * distribution * visible_dot / (outgoing.z + projected_radius);
}

template <GeometryScalar Scalar>
void expect_vndf_chi_square(const Vector3T<Scalar> outgoing, const Scalar alpha,
                            const std::uint64_t case_salt) {
    constexpr auto axis_bins = std::size_t{16};
    constexpr auto bin_count = axis_bins * axis_bins;
    constexpr auto sample_count = std::size_t{131'072};
    constexpr auto expected_per_bin =
        static_cast<long double>(sample_count) / static_cast<long double>(bin_count);
    constexpr auto family_case_count = 4.0L;
    constexpr auto family_false_reject_probability = 1.0e-6L;
    constexpr auto degrees_of_freedom = static_cast<long double>(bin_count - 1U);
    const auto concentration = std::log(family_case_count / family_false_reject_probability);
    const auto upper_threshold = degrees_of_freedom +
                                 2.0L * std::sqrt(degrees_of_freedom * concentration) +
                                 2.0L * concentration;

    const auto distribution = GgxFor<Scalar>::create(alpha);
    ASSERT_TRUE(distribution.has_value());
    const auto sampler = IndependentSamplerT<Scalar>{0x6A09E667F3BCC909ULL ^ case_salt};
    auto histogram = std::array<std::uint32_t, bin_count>{};
    for (auto sample_index = std::size_t{}; sample_index < sample_count; ++sample_index) {
        const auto stream = sampler.make_stream(71U, 29U, sample_index);
        const auto canonical = Point2T<Scalar>{
            .x = stream.sample_1d(0U),
            .y = stream.sample_1d(1U),
        };
        const auto sampled = distribution->sample_visible_normal(outgoing, canonical);
        ASSERT_TRUE(sampled.has_value());
        ASSERT_TRUE(sampled->has_value());

        const auto queried =
            distribution->visible_normal_pdf(outgoing, (**sampled).microfacet_normal);
        ASSERT_TRUE(queried.has_value());
        EXPECT_EQ((**sampled).probability.value, queried->value);
        EXPECT_EQ((**sampled).probability.measure, ProbabilityMeasure::solid_angle);

        if ((sample_index & 1023U) == 0U) {
            const auto expected_pdf = independent_visible_normal_pdf(
                static_cast<long double>(alpha),
                LongVector{.x = static_cast<long double>(outgoing.x),
                           .y = static_cast<long double>(outgoing.y),
                           .z = static_cast<long double>(outgoing.z)},
                LongNormal{
                    .x = static_cast<long double>((**sampled).microfacet_normal.x),
                    .y = static_cast<long double>((**sampled).microfacet_normal.y),
                    .z = static_cast<long double>((**sampled).microfacet_normal.z),
                });
            const auto relative_tolerance =
                std::same_as<Scalar, TransportScalar> ? 2.0e-5L : 2.0e-12L;
            EXPECT_NEAR(static_cast<long double>((**sampled).probability.value), expected_pdf,
                        relative_tolerance * std::max(1.0L, expected_pdf));
        }

        const auto coordinates = invert_visible_normal_mapping(
            static_cast<long double>(alpha),
            LongVector{.x = static_cast<long double>(outgoing.x),
                       .y = static_cast<long double>(outgoing.y),
                       .z = static_cast<long double>(outgoing.z)},
            LongNormal{
                .x = static_cast<long double>((**sampled).microfacet_normal.x),
                .y = static_cast<long double>((**sampled).microfacet_normal.y),
                .z = static_cast<long double>((**sampled).microfacet_normal.z),
            },
            128.0L * static_cast<long double>(std::numeric_limits<Scalar>::epsilon()));
        ASSERT_TRUE(coordinates.has_value());
        const auto radial_bin =
            static_cast<std::size_t>(coordinates->radial_cdf * static_cast<long double>(axis_bins));
        const auto azimuth_bin = static_cast<std::size_t>(coordinates->azimuth_cdf *
                                                          static_cast<long double>(axis_bins));
        ASSERT_LT(radial_bin, axis_bins);
        ASSERT_LT(azimuth_bin, axis_bins);
        ++histogram[azimuth_bin * axis_bins + radial_bin];
    }

    auto chi_squared = 0.0L;
    for (const auto observed : histogram) {
        const auto difference = static_cast<long double>(observed) - expected_per_bin;
        chi_squared += difference * difference / expected_per_bin;
    }
    EXPECT_LT(chi_squared, upper_threshold)
        << "Pearson chi-square=" << static_cast<double>(chi_squared)
        << ", derived upper threshold=" << static_cast<double>(upper_threshold);
}

TEST(GgxMicrofacetTest, PassesTransportNormalIncidenceVndfChiSquare) {
    expect_vndf_chi_square(Vector3{.z = 1.0F}, 0.2F, 0x243F6A8885A308D3ULL);
}

TEST(GgxMicrofacetTest, PassesReferenceNormalIncidenceVndfChiSquare) {
    expect_vndf_chi_square(ReferenceVector3{.z = 1.0}, 0.2, 0x13198A2E03707344ULL);
}

TEST(GgxMicrofacetTest, PassesTransportObliqueVndfChiSquare) {
    expect_vndf_chi_square(Vector3{.x = 0.36F, .y = 0.48F, .z = 0.8F}, 2.0F, 0xA4093822299F31D0ULL);
}

TEST(GgxMicrofacetTest, PassesReferenceObliqueVndfChiSquare) {
    expect_vndf_chi_square(ReferenceVector3{.x = 0.36, .y = 0.48, .z = 0.8}, 2.0,
                           0x082EFA98EC4E6C89ULL);
}

static_assert(std::is_standard_layout_v<GgxVisibleNormalSample>);
static_assert(std::is_trivially_copyable_v<GgxVisibleNormalSample>);
static_assert(std::is_standard_layout_v<ReferenceGgxVisibleNormalSample>);
static_assert(std::is_trivially_copyable_v<ReferenceGgxVisibleNormalSample>);
static_assert(std::is_standard_layout_v<GgxMicrofacet>);
static_assert(std::is_trivially_copyable_v<GgxMicrofacet>);
static_assert(std::is_standard_layout_v<ReferenceGgxMicrofacet>);
static_assert(std::is_trivially_copyable_v<ReferenceGgxMicrofacet>);

} // namespace
} // namespace blackframe::renderer
