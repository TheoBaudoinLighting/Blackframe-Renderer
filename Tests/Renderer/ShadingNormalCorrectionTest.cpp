#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/ShadingNormalCorrection.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using ReflectionFor = LambertianReflectionT<Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> as_vector(const Normal3T<Scalar> normal) noexcept {
    return {.x = normal.x, .y = normal.y, .z = normal.z};
}

template <SpectrumScalar Scalar>
inline constexpr auto RatioTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6} : ReferenceScalar{2.0e-14};

template <SpectrumScalar Scalar> void expect_identity_and_radiance_contract() {
    constexpr auto geometric_normal = Normal3T<Scalar>{.z = Scalar{1}};
    constexpr auto shading_normal = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};

    static_assert(
        std::same_as<decltype(shading_normal_correction(geometric_normal, shading_normal, outgoing,
                                                        incoming, TransportMode::radiance)),
                     core::Result<Scalar>>);

    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        const auto identity =
            shading_normal_correction(geometric_normal, geometric_normal, outgoing, incoming, mode);
        ASSERT_TRUE(identity.has_value()) << identity.error().message;
        EXPECT_EQ(*identity, Scalar{1});
    }

    const auto tilted_radiance = shading_normal_correction(
        geometric_normal, shading_normal, outgoing, incoming, TransportMode::radiance);
    ASSERT_TRUE(tilted_radiance.has_value()) << tilted_radiance.error().message;
    EXPECT_EQ(*tilted_radiance, Scalar{1});
}

TEST(ShadingNormalCorrectionTest, PreservesIdentityAndRadianceInBothPrecisions) {
    expect_identity_and_radiance_contract<TransportScalar>();
    expect_identity_and_radiance_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_importance_ratio_and_reciprocity() {
    constexpr auto geometric_normal = Normal3T<Scalar>{.z = Scalar{1}};
    constexpr auto shading_normal = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};

    const auto forward = shading_normal_correction(geometric_normal, shading_normal, outgoing,
                                                   incoming, TransportMode::importance);
    const auto reverse = shading_normal_correction(geometric_normal, shading_normal, incoming,
                                                   outgoing, TransportMode::importance);
    ASSERT_TRUE(forward.has_value()) << forward.error().message;
    ASSERT_TRUE(reverse.has_value()) << reverse.error().message;

    EXPECT_NEAR(static_cast<ReferenceScalar>(*forward), ReferenceScalar{0.64},
                RatioTolerance<Scalar>);
    EXPECT_NEAR(static_cast<ReferenceScalar>(*reverse), ReferenceScalar{1.5625},
                RatioTolerance<Scalar>);
    EXPECT_NEAR(static_cast<ReferenceScalar>(*forward) * static_cast<ReferenceScalar>(*reverse),
                ReferenceScalar{1}, RatioTolerance<Scalar>);
}

TEST(ShadingNormalCorrectionTest, AppliesTheVeachAdjointRatioReciprocally) {
    expect_importance_ratio_and_reciprocity<TransportScalar>();
    expect_importance_ratio_and_reciprocity<ReferenceScalar>();
}

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> void expect_invalid_inputs_and_open_support() {
    constexpr auto geometric_normal = Normal3T<Scalar>{.z = Scalar{1}};
    constexpr auto shading_normal = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    constexpr auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    constexpr auto incoming = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();

    expect_invalid(shading_normal_correction(Normal3T<Scalar>{}, shading_normal, outgoing, incoming,
                                             TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, Normal3T<Scalar>{.z = Scalar{2}},
                                             outgoing, incoming, TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, Normal3T<Scalar>{.z = Scalar{-1}},
                                             outgoing, incoming, TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, Normal3T<Scalar>{.x = Scalar{1}},
                                             outgoing, incoming, TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, shading_normal, Vector3T<Scalar>{},
                                             incoming, TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, shading_normal, outgoing,
                                             Vector3T<Scalar>{.x = nan, .z = Scalar{1}},
                                             TransportMode::radiance));
    expect_invalid(shading_normal_correction(geometric_normal, shading_normal, outgoing, incoming,
                                             static_cast<TransportMode>(2U)));

    const auto geometric_horizon =
        shading_normal_correction(geometric_normal, shading_normal, outgoing,
                                  Vector3T<Scalar>{.x = Scalar{1}}, TransportMode::importance);
    ASSERT_TRUE(geometric_horizon.has_value()) << geometric_horizon.error().message;
    EXPECT_EQ(*geometric_horizon, Scalar{0});
    EXPECT_FALSE(std::signbit(*geometric_horizon));

    const auto square_root_three_over_two = std::sqrt(Scalar{0.75});
    const auto geometric_only = Vector3T<Scalar>{
        .x = -square_root_three_over_two,
        .z = Scalar{0.5},
    };
    const auto shading_only = Vector3T<Scalar>{
        .x = square_root_three_over_two,
        .z = Scalar{-0.5},
    };
    for (const auto unsupported : std::array{geometric_only, shading_only}) {
        for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
            const auto correction = shading_normal_correction(geometric_normal, shading_normal,
                                                              outgoing, unsupported, mode);
            ASSERT_TRUE(correction.has_value()) << correction.error().message;
            EXPECT_EQ(*correction, Scalar{0});
            EXPECT_FALSE(std::signbit(*correction));
        }
    }
}

TEST(ShadingNormalCorrectionTest, RejectsMalformedInputsWithoutFacingOrFallback) {
    expect_invalid_inputs_and_open_support<TransportScalar>();
    expect_invalid_inputs_and_open_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_tilted_lambertian_furnace() {
    constexpr auto geometric_normal = Normal3T<Scalar>{.z = Scalar{1}};
    constexpr auto shading_normal = Normal3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto frame = OrthonormalFrameT<Scalar>::from_normal(shading_normal);
    ASSERT_TRUE(frame.has_value()) << frame.error().message;

    auto white = SpectrumFor<Scalar>{};
    white.values.fill(Scalar{1});
    const auto reflection = ReflectionFor<Scalar>::create(white);
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;

    const auto outgoing_world = as_vector(shading_normal);
    const auto outgoing_local = frame->to_local(outgoing_world);
    constexpr auto cosine_steps = std::size_t{256};
    constexpr auto azimuth_steps = std::size_t{256};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = std::array<long double, TransportSpectrumSampleCount>{};

    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming_local = Vector3T<Scalar>{
                .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto incoming_world = frame->to_world(incoming_local);
            const auto correction =
                shading_normal_correction(geometric_normal, shading_normal, outgoing_world,
                                          incoming_world, TransportMode::radiance);
            const auto evaluated = reflection->eval(outgoing_local, incoming_local);
            ASSERT_TRUE(correction.has_value()) << correction.error().message;
            ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
            ASSERT_TRUE(*correction == Scalar{0} || *correction == Scalar{1});

            for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
                integral[lane] += static_cast<long double>((*evaluated)[lane]) *
                                  static_cast<long double>(*correction) * cosine * delta_cosine *
                                  delta_azimuth;
            }
        }
    }

    constexpr auto expected = 0.9L;
    constexpr auto tolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-4L : 1.0e-4L;
    for (const auto lane : integral) {
        EXPECT_NEAR(lane, expected, tolerance);
        EXPECT_LE(lane, 1.0L + tolerance);
    }
}

TEST(ShadingNormalCorrectionTest, MatchesTheAnalyticTiltedLambertianFurnace) {
    expect_tilted_lambertian_furnace<TransportScalar>();
    expect_tilted_lambertian_furnace<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
