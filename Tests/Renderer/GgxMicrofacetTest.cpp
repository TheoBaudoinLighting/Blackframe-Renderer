#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
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
#include <utility>

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
        EXPECT_EQ(distribution->alpha_x(), alpha);
        EXPECT_EQ(distribution->alpha_y(), alpha);
    }

    const auto anisotropic = GgxFor<Scalar>::create(Scalar{0.2}, Scalar{0.8});
    ASSERT_TRUE(anisotropic.has_value());
    EXPECT_EQ(anisotropic->alpha(), Scalar{0.2});
    EXPECT_EQ(anisotropic->alpha_x(), Scalar{0.2});
    EXPECT_EQ(anisotropic->alpha_y(), Scalar{0.8});

    for (const auto alpha : std::array{
             Scalar{0},
             -std::numeric_limits<Scalar>::denorm_min(),
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
         }) {
        expect_invalid(GgxFor<Scalar>::create(alpha));
    }

    for (const auto malformed : std::array{
             std::pair{Scalar{0}, Scalar{0.5}},
             std::pair{Scalar{0.5}, -std::numeric_limits<Scalar>::denorm_min()},
             std::pair{std::numeric_limits<Scalar>::quiet_NaN(), Scalar{0.5}},
             std::pair{Scalar{0.5}, std::numeric_limits<Scalar>::infinity()},
         }) {
        expect_invalid(GgxFor<Scalar>::create(malformed.first, malformed.second));
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
    expect_invalid(GgxFor<Scalar>::create(Scalar{1} / representable_limit, representable_limit));
    EXPECT_FALSE(ggx_microfacet_detail::representable_alpha_pair(Scalar{1} / representable_limit,
                                                                 representable_limit));
}

TEST(GgxMicrofacetTest, CreatesOnlyRepresentablePositiveContinuousWidths) {
    static_assert(std::same_as<decltype(GgxMicrofacet::create(TransportScalar{})),
                               core::Result<GgxMicrofacet>>);
    static_assert(std::same_as<decltype(ReferenceGgxMicrofacet::create(ReferenceScalar{})),
                               core::Result<ReferenceGgxMicrofacet>>);
    static_assert(
        std::same_as<decltype(GgxMicrofacet::create(TransportScalar{}, TransportScalar{})),
                     core::Result<GgxMicrofacet>>);
    static_assert(
        std::same_as<decltype(ReferenceGgxMicrofacet::create(ReferenceScalar{}, ReferenceScalar{})),
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

template <GeometryScalar Scalar> void expect_isotropic_overloads_are_identical() {
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.36}, .y = Scalar{0.48}, .z = Scalar{0.8}};
    const auto incoming = Vector3T<Scalar>{
        .x = Scalar{-0.3},
        .y = Scalar{0.4},
        .z = static_cast<Scalar>(std::sqrt(0.75L)),
    };
    const auto normal = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.37}, .y = Scalar{0.61}};
    for (const auto alpha : std::array{Scalar{0.45}, Scalar{2}}) {
        const auto legacy = GgxFor<Scalar>::create(alpha);
        const auto explicit_axes = GgxFor<Scalar>::create(alpha, alpha);
        ASSERT_TRUE(legacy.has_value());
        ASSERT_TRUE(explicit_axes.has_value());

        const auto legacy_distribution = legacy->normal_distribution(normal);
        const auto explicit_distribution = explicit_axes->normal_distribution(normal);
        const auto legacy_lambda = legacy->smith_lambda(outgoing);
        const auto explicit_lambda = explicit_axes->smith_lambda(outgoing);
        const auto legacy_g1 = legacy->smith_g1(outgoing);
        const auto explicit_g1 = explicit_axes->smith_g1(outgoing);
        const auto legacy_g2 = legacy->smith_g2(outgoing, incoming);
        const auto explicit_g2 = explicit_axes->smith_g2(outgoing, incoming);
        ASSERT_TRUE(legacy_distribution.has_value());
        ASSERT_TRUE(explicit_distribution.has_value());
        ASSERT_TRUE(legacy_lambda.has_value());
        ASSERT_TRUE(explicit_lambda.has_value());
        ASSERT_TRUE(legacy_g1.has_value());
        ASSERT_TRUE(explicit_g1.has_value());
        ASSERT_TRUE(legacy_g2.has_value());
        ASSERT_TRUE(explicit_g2.has_value());
        EXPECT_EQ(*legacy_distribution, *explicit_distribution);
        EXPECT_EQ(*legacy_lambda, *explicit_lambda);
        EXPECT_EQ(*legacy_g1, *explicit_g1);
        EXPECT_EQ(*legacy_g2, *explicit_g2);
        const auto legacy_probability = legacy->visible_normal_pdf(outgoing, normal);
        const auto explicit_probability = explicit_axes->visible_normal_pdf(outgoing, normal);
        ASSERT_TRUE(legacy_probability.has_value());
        ASSERT_TRUE(explicit_probability.has_value());
        EXPECT_EQ(legacy_probability->value, explicit_probability->value);
        EXPECT_EQ(legacy_probability->measure, explicit_probability->measure);

        const auto legacy_sample = legacy->sample_visible_normal(outgoing, canonical);
        const auto explicit_sample = explicit_axes->sample_visible_normal(outgoing, canonical);
        ASSERT_TRUE(legacy_sample.has_value());
        ASSERT_TRUE(legacy_sample->has_value());
        ASSERT_TRUE(explicit_sample.has_value());
        ASSERT_TRUE(explicit_sample->has_value());
        EXPECT_EQ((**legacy_sample).microfacet_normal, (**explicit_sample).microfacet_normal);
        EXPECT_EQ((**legacy_sample).probability.value, (**explicit_sample).probability.value);
        EXPECT_EQ((**legacy_sample).probability.measure, (**explicit_sample).probability.measure);
    }
}

TEST(GgxMicrofacetTest, KeepsLegacyAndExplicitIsotropicPathsBitwiseIdentical) {
    expect_isotropic_overloads_are_identical<TransportScalar>();
    expect_isotropic_overloads_are_identical<ReferenceScalar>();
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

template <GeometryScalar Scalar> void expect_anisotropic_analytic_values() {
    constexpr auto alpha_x = Scalar{0.5};
    constexpr auto alpha_y = Scalar{2};
    const auto distribution = GgxFor<Scalar>::create(alpha_x, alpha_y);
    ASSERT_TRUE(distribution.has_value());

    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto tilted_x = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto tilted_y = Normal3T<Scalar>{.y = Scalar{0.6}, .z = Scalar{0.8}};
    const auto direction_x = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto direction_y = Vector3T<Scalar>{.y = Scalar{0.6}, .z = Scalar{0.8}};
    const auto normal_value = distribution->normal_distribution(normal);
    const auto value_x = distribution->normal_distribution(tilted_x);
    const auto value_y = distribution->normal_distribution(tilted_y);
    const auto lambda_x = distribution->smith_lambda(direction_x);
    const auto lambda_y = distribution->smith_lambda(direction_y);
    ASSERT_TRUE(normal_value.has_value());
    ASSERT_TRUE(value_x.has_value());
    ASSERT_TRUE(value_y.has_value());
    ASSERT_TRUE(lambda_x.has_value());
    ASSERT_TRUE(lambda_y.has_value());

    const auto expected_distribution = [](const long double x, const long double y,
                                          const long double z) {
        const auto root_squared = (x / 0.5L) * (x / 0.5L) + (y / 2.0L) * (y / 2.0L) + z * z;
        return 1.0L / (std::numbers::pi_v<long double> * 0.5L * 2.0L * root_squared * root_squared);
    };
    const auto expected_lambda = [](const long double x, const long double y, const long double z) {
        const auto radius = std::hypot(z, 0.5L * x, 2.0L * y);
        return (radius / z - 1.0L) * 0.5L;
    };
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 3.0e-6L : 3.0e-14L;
    const auto expect_near = [tolerance](const Scalar actual, const long double expected) {
        EXPECT_NEAR(static_cast<long double>(actual), expected,
                    tolerance * std::max(1.0L, std::abs(expected)));
    };
    expect_near(*normal_value, std::numbers::inv_pi_v<long double>);
    expect_near(*value_x, expected_distribution(0.6L, 0.0L, 0.8L));
    expect_near(*value_y, expected_distribution(0.0L, 0.6L, 0.8L));
    expect_near(*lambda_x, expected_lambda(0.6L, 0.0L, 0.8L));
    expect_near(*lambda_y, expected_lambda(0.0L, 0.6L, 0.8L));
    EXPECT_LT(*value_x, *value_y);
    EXPECT_LT(*lambda_x, *lambda_y);
}

TEST(GgxMicrofacetTest, EvaluatesAnisotropicDistributionAndSmithTermsAnalytically) {
    expect_anisotropic_analytic_values<TransportScalar>();
    expect_anisotropic_analytic_values<ReferenceScalar>();
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar>
rotate_tangent_quarter_turn(const Vector3T<Scalar> direction) noexcept {
    return Vector3T<Scalar>{.x = -direction.y, .y = direction.x, .z = direction.z};
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar>
rotate_tangent_quarter_turn(const Normal3T<Scalar> normal) noexcept {
    return Normal3T<Scalar>{.x = -normal.y, .y = normal.x, .z = normal.z};
}

template <GeometryScalar Scalar> void expect_quarter_turn_covariance() {
    constexpr auto alpha_x = Scalar{0.22};
    constexpr auto alpha_y = Scalar{0.78};
    const auto original = GgxFor<Scalar>::create(alpha_x, alpha_y);
    const auto rotated = GgxFor<Scalar>::create(alpha_y, alpha_x);
    ASSERT_TRUE(original.has_value());
    ASSERT_TRUE(rotated.has_value());

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.36}, .y = Scalar{0.48}, .z = Scalar{0.8}};
    const auto rotated_outgoing = rotate_tangent_quarter_turn(outgoing);
    const auto second = Vector3T<Scalar>{
        .x = Scalar{-0.3},
        .y = Scalar{0.4},
        .z = static_cast<Scalar>(std::sqrt(0.75L)),
    };
    const auto rotated_second = rotate_tangent_quarter_turn(second);
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 8.0e-6L : 8.0e-14L;
    const auto expect_near = [tolerance](const Scalar first, const Scalar second_value) {
        const auto scale = std::max({1.0L, std::abs(static_cast<long double>(first)),
                                     std::abs(static_cast<long double>(second_value))});
        EXPECT_NEAR(static_cast<long double>(first), static_cast<long double>(second_value),
                    tolerance * scale);
    };

    const auto lambda = original->smith_lambda(outgoing);
    const auto rotated_lambda = rotated->smith_lambda(rotated_outgoing);
    const auto g1 = original->smith_g1(outgoing);
    const auto rotated_g1 = rotated->smith_g1(rotated_outgoing);
    const auto g2 = original->smith_g2(outgoing, second);
    const auto rotated_g2 = rotated->smith_g2(rotated_outgoing, rotated_second);
    ASSERT_TRUE(lambda.has_value());
    ASSERT_TRUE(rotated_lambda.has_value());
    ASSERT_TRUE(g1.has_value());
    ASSERT_TRUE(rotated_g1.has_value());
    ASSERT_TRUE(g2.has_value());
    ASSERT_TRUE(rotated_g2.has_value());
    expect_near(*lambda, *rotated_lambda);
    expect_near(*g1, *rotated_g1);
    expect_near(*g2, *rotated_g2);

    constexpr auto grid_radius = 16;
    for (auto y_index = -grid_radius; y_index <= grid_radius; ++y_index) {
        for (auto x_index = -grid_radius; x_index <= grid_radius; ++x_index) {
            const auto x =
                Scalar{0.95} * static_cast<Scalar>(x_index) / static_cast<Scalar>(grid_radius);
            const auto y =
                Scalar{0.95} * static_cast<Scalar>(y_index) / static_cast<Scalar>(grid_radius);
            const auto radial_squared = std::fma(x, x, y * y);
            if (!(radial_squared < Scalar{1})) {
                continue;
            }
            const auto normal = Normal3T<Scalar>{
                .x = x,
                .y = y,
                .z = std::sqrt(Scalar{1} - radial_squared),
            };
            const auto rotated_normal = rotate_tangent_quarter_turn(normal);
            const auto value = original->normal_distribution(normal);
            const auto rotated_value = rotated->normal_distribution(rotated_normal);
            const auto probability = original->visible_normal_pdf(outgoing, normal);
            const auto rotated_probability =
                rotated->visible_normal_pdf(rotated_outgoing, rotated_normal);
            ASSERT_TRUE(value.has_value());
            ASSERT_TRUE(rotated_value.has_value());
            ASSERT_TRUE(probability.has_value());
            ASSERT_TRUE(rotated_probability.has_value());
            expect_near(*value, *rotated_value);
            expect_near(probability->value, rotated_probability->value);
        }
    }

    const auto canonical = Point2T<Scalar>{.x = Scalar{0.37}, .y = Scalar{0.61}};
    const auto sample = original->sample_visible_normal(outgoing, canonical);
    const auto rotated_sample = rotated->sample_visible_normal(rotated_outgoing, canonical);
    ASSERT_TRUE(sample.has_value());
    ASSERT_TRUE(sample->has_value());
    ASSERT_TRUE(rotated_sample.has_value());
    ASSERT_TRUE(rotated_sample->has_value());
    const auto expected_normal = rotate_tangent_quarter_turn((**sample).microfacet_normal);
    expect_near(expected_normal.x, (**rotated_sample).microfacet_normal.x);
    expect_near(expected_normal.y, (**rotated_sample).microfacet_normal.y);
    expect_near(expected_normal.z, (**rotated_sample).microfacet_normal.z);
    expect_near((**sample).probability.value, (**rotated_sample).probability.value);
}

TEST(GgxMicrofacetTest, SwapsAnisotropicAxesCovariantlyUnderAQuarterTurn) {
    expect_quarter_turn_covariance<TransportScalar>();
    expect_quarter_turn_covariance<ReferenceScalar>();
}

template <GeometryScalar Scalar>
[[nodiscard]] Vector3T<Scalar> rotate_about_world_normal(const Vector3T<Scalar> direction,
                                                         const Scalar angle) {
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    return Vector3T<Scalar>{
        .x = std::fma(cosine, direction.x, -sine * direction.y),
        .y = std::fma(sine, direction.x, cosine * direction.y),
        .z = direction.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] Normal3T<Scalar> rotate_about_world_normal(const Normal3T<Scalar> normal,
                                                         const Scalar angle) {
    const auto rotated = rotate_about_world_normal(
        Vector3T<Scalar>{.x = normal.x, .y = normal.y, .z = normal.z}, angle);
    return Normal3T<Scalar>{.x = rotated.x, .y = rotated.y, .z = rotated.z};
}

template <GeometryScalar Scalar> void expect_rotated_frame_directional_image() {
    constexpr auto angle = Scalar{0.61};
    const auto distribution = GgxFor<Scalar>::create(Scalar{0.22}, Scalar{0.78});
    const auto frame = OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}});
    ASSERT_TRUE(distribution.has_value()) << distribution.error().message;
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    const auto rotated_frame = frame->rotated_about_normal(angle);
    ASSERT_TRUE(rotated_frame.has_value()) << rotated_frame.error().message;

    const auto outgoing_world =
        Vector3T<Scalar>{.x = Scalar{0.36}, .y = Scalar{0.48}, .z = Scalar{0.8}};
    const auto rotated_outgoing_world = rotate_about_world_normal(outgoing_world, angle);
    const auto outgoing_local = frame->to_local(outgoing_world);
    const auto rotated_outgoing_local = rotated_frame->to_local(rotated_outgoing_world);
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-5L : 2.0e-13L;
    const auto expect_near = [tolerance](const Scalar first, const Scalar second) {
        const auto scale = std::max({1.0L, std::abs(static_cast<long double>(first)),
                                     std::abs(static_cast<long double>(second))});
        EXPECT_NEAR(static_cast<long double>(first), static_cast<long double>(second),
                    tolerance * scale);
    };
    expect_near(outgoing_local.x, rotated_outgoing_local.x);
    expect_near(outgoing_local.y, rotated_outgoing_local.y);
    expect_near(outgoing_local.z, rotated_outgoing_local.z);

    constexpr auto grid_radius = 16;
    for (auto y_index = -grid_radius; y_index <= grid_radius; ++y_index) {
        for (auto x_index = -grid_radius; x_index <= grid_radius; ++x_index) {
            const auto x =
                Scalar{0.95} * static_cast<Scalar>(x_index) / static_cast<Scalar>(grid_radius);
            const auto y =
                Scalar{0.95} * static_cast<Scalar>(y_index) / static_cast<Scalar>(grid_radius);
            const auto radial_squared = std::fma(x, x, y * y);
            if (!(radial_squared < Scalar{1})) {
                continue;
            }
            const auto world_normal = Normal3T<Scalar>{
                .x = x,
                .y = y,
                .z = std::sqrt(Scalar{1} - radial_squared),
            };
            const auto rotated_world_normal = rotate_about_world_normal(world_normal, angle);
            const auto local_normal = frame->to_local(world_normal);
            const auto rotated_local_normal = rotated_frame->to_local(rotated_world_normal);
            const auto value = distribution->normal_distribution(local_normal);
            const auto rotated_value = distribution->normal_distribution(rotated_local_normal);
            const auto probability = distribution->visible_normal_pdf(outgoing_local, local_normal);
            const auto rotated_probability =
                distribution->visible_normal_pdf(rotated_outgoing_local, rotated_local_normal);
            ASSERT_TRUE(value.has_value()) << value.error().message;
            ASSERT_TRUE(rotated_value.has_value()) << rotated_value.error().message;
            ASSERT_TRUE(probability.has_value()) << probability.error().message;
            ASSERT_TRUE(rotated_probability.has_value()) << rotated_probability.error().message;
            expect_near(*value, *rotated_value);
            expect_near(probability->value, rotated_probability->value);
        }
    }

    constexpr auto canonical = Point2T<Scalar>{.x = Scalar{0.37}, .y = Scalar{0.61}};
    const auto sample = distribution->sample_visible_normal(outgoing_local, canonical);
    const auto rotated_sample =
        distribution->sample_visible_normal(rotated_outgoing_local, canonical);
    ASSERT_TRUE(sample.has_value()) << sample.error().message;
    ASSERT_TRUE(sample->has_value());
    ASSERT_TRUE(rotated_sample.has_value()) << rotated_sample.error().message;
    ASSERT_TRUE(rotated_sample->has_value());
    const auto world_sample = frame->to_world((**sample).microfacet_normal);
    const auto rotated_world_sample = rotated_frame->to_world((**rotated_sample).microfacet_normal);
    const auto expected_world_sample = rotate_about_world_normal(world_sample, angle);
    expect_near(expected_world_sample.x, rotated_world_sample.x);
    expect_near(expected_world_sample.y, rotated_world_sample.y);
    expect_near(expected_world_sample.z, rotated_world_sample.z);
    expect_near((**sample).probability.value, (**rotated_sample).probability.value);
}

TEST(GgxMicrofacetTest, RotatesTheDirectionalImageThroughTheCallerTangentFrame) {
    expect_rotated_frame_directional_image<TransportScalar>();
    expect_rotated_frame_directional_image<ReferenceScalar>();
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

template <GeometryScalar Scalar> void expect_anisotropic_distribution_normalization() {
    constexpr auto cosine_steps = std::size_t{256};
    constexpr auto azimuth_steps = std::size_t{512};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    for (const auto widths :
         std::array{std::pair{Scalar{0.2}, Scalar{0.7}}, std::pair{Scalar{0.45}, Scalar{2}}}) {
        const auto distribution = GgxFor<Scalar>::create(widths.first, widths.second);
        ASSERT_TRUE(distribution.has_value());
        auto integral = 0.0L;
        for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
            const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
            const auto sine = std::sqrt((1.0L - cosine) * (1.0L + cosine));
            for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps;
                 ++azimuth_index) {
                const auto azimuth =
                    (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
                const auto microfacet_normal = Normal3T<Scalar>{
                    .x = static_cast<Scalar>(sine * std::cos(azimuth)),
                    .y = static_cast<Scalar>(sine * std::sin(azimuth)),
                    .z = static_cast<Scalar>(cosine),
                };
                const auto value = distribution->normal_distribution(microfacet_normal);
                ASSERT_TRUE(value.has_value());
                integral +=
                    static_cast<long double>(*value) * cosine * delta_cosine * delta_azimuth;
            }
        }
        EXPECT_NEAR(integral, 1.0L, 8.0e-4L);
    }
}

TEST(GgxMicrofacetTest, NormalizesAnisotropicDInProjectedSolidAngle) {
    expect_anisotropic_distribution_normalization<TransportScalar>();
    expect_anisotropic_distribution_normalization<ReferenceScalar>();
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
invert_visible_normal_mapping(const long double alpha_x, const long double alpha_y,
                              const LongVector outgoing, const LongNormal microfacet_normal,
                              const long double inverse_tolerance) {
    const auto normalize =
        [](const std::array<long double, 3>& value) -> std::optional<std::array<long double, 3>> {
        const auto length = std::hypot(std::hypot(value[0], value[1]), value[2]);
        if (!std::isfinite(length) || !(length > 0.0L)) {
            return {};
        }
        return std::array{value[0] / length, value[1] / length, value[2] / length};
    };

    const auto view = normalize({alpha_x * outgoing.x, alpha_y * outgoing.y, outgoing.z});
    const auto hemisphere_normal = normalize(
        {microfacet_normal.x / alpha_x, microfacet_normal.y / alpha_y, microfacet_normal.z});
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

[[nodiscard]] long double independent_visible_normal_pdf(const long double alpha_x,
                                                         const long double alpha_y,
                                                         const LongVector outgoing,
                                                         const LongNormal microfacet_normal) {
    const auto distribution_root = std::hypot(microfacet_normal.x / alpha_x,
                                              microfacet_normal.y / alpha_y, microfacet_normal.z);
    const auto distribution =
        1.0L / (std::numbers::pi_v<long double> * alpha_x * alpha_y * distribution_root *
                distribution_root * distribution_root * distribution_root);
    const auto visible_dot = outgoing.x * microfacet_normal.x + outgoing.y * microfacet_normal.y +
                             outgoing.z * microfacet_normal.z;
    const auto projected_radius =
        std::hypot(outgoing.z, alpha_x * outgoing.x, alpha_y * outgoing.y);
    return 2.0L * distribution * visible_dot / (outgoing.z + projected_radius);
}

template <GeometryScalar Scalar>
void expect_vndf_chi_square(const Vector3T<Scalar> outgoing, const Scalar alpha_x,
                            const Scalar alpha_y, const std::uint64_t case_salt) {
    constexpr auto axis_bins = std::size_t{16};
    constexpr auto bin_count = axis_bins * axis_bins;
    constexpr auto sample_count = std::size_t{131'072};
    constexpr auto expected_per_bin =
        static_cast<long double>(sample_count) / static_cast<long double>(bin_count);
    constexpr auto family_case_count = 6.0L;
    constexpr auto family_false_reject_probability = 1.0e-6L;
    constexpr auto degrees_of_freedom = static_cast<long double>(bin_count - 1U);
    const auto concentration = std::log(family_case_count / family_false_reject_probability);
    const auto upper_threshold = degrees_of_freedom +
                                 2.0L * std::sqrt(degrees_of_freedom * concentration) +
                                 2.0L * concentration;

    const auto distribution = GgxFor<Scalar>::create(alpha_x, alpha_y);
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
                static_cast<long double>(alpha_x), static_cast<long double>(alpha_y),
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
            static_cast<long double>(alpha_x), static_cast<long double>(alpha_y),
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
    expect_vndf_chi_square(Vector3{.z = 1.0F}, 0.2F, 0.2F, 0x243F6A8885A308D3ULL);
}

TEST(GgxMicrofacetTest, PassesReferenceNormalIncidenceVndfChiSquare) {
    expect_vndf_chi_square(ReferenceVector3{.z = 1.0}, 0.2, 0.2, 0x13198A2E03707344ULL);
}

TEST(GgxMicrofacetTest, PassesTransportObliqueVndfChiSquare) {
    expect_vndf_chi_square(Vector3{.x = 0.36F, .y = 0.48F, .z = 0.8F}, 2.0F, 2.0F,
                           0xA4093822299F31D0ULL);
}

TEST(GgxMicrofacetTest, PassesReferenceObliqueVndfChiSquare) {
    expect_vndf_chi_square(ReferenceVector3{.x = 0.36, .y = 0.48, .z = 0.8}, 2.0, 2.0,
                           0x082EFA98EC4E6C89ULL);
}

TEST(GgxMicrofacetTest, PassesTransportAnisotropicObliqueVndfChiSquare) {
    expect_vndf_chi_square(Vector3{.x = 0.36F, .y = 0.48F, .z = 0.8F}, 0.22F, 0.78F,
                           0x452821E638D01377ULL);
}

TEST(GgxMicrofacetTest, PassesReferenceAnisotropicObliqueVndfChiSquare) {
    expect_vndf_chi_square(ReferenceVector3{.x = 0.36, .y = 0.48, .z = 0.8}, 0.22, 0.78,
                           0xBE5466CF34E90C6CULL);
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
