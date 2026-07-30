#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using SurfaceFor = BsdfOnlyTriangleSurfaceT<Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar>
spectrum(const std::array<Scalar, TransportSpectrumSampleCount> values) {
    return SpectrumFor<Scalar>{.values = values};
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(value);
    return result;
}

[[nodiscard]] constexpr PathDepthLimits diffuse_limits(const std::uint32_t count) noexcept {
    return PathDepthLimits{.diffuse = count};
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr RussianRoulettePolicyT<Scalar> disabled_roulette() noexcept {
    return RussianRoulettePolicyT<Scalar>::disabled();
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool is_zero_spectrum(const SpectrumFor<Scalar>& value) {
    return std::ranges::all_of(value.values, [](const Scalar lane) { return lane == Scalar{0}; });
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SurfaceFor<Scalar>>
make_surface(const Point3T<Scalar> vertex0, const Point3T<Scalar> vertex1,
             const Point3T<Scalar> vertex2, const SpectrumFor<Scalar>& reflectance,
             const SpectrumFor<Scalar>& radiance, const SampledWavelengthsT<Scalar>& wavelengths,
             const RayMask visibility_mask = AllRayVisibility) {
    const auto triangle = TriangleT<Scalar>::create(vertex0, vertex1, vertex2);
    if (!triangle.has_value()) {
        return std::unexpected(triangle.error());
    }
    const auto reflection = LambertianReflectionT<Scalar>::create(reflectance);
    if (!reflection.has_value()) {
        return std::unexpected(reflection.error());
    }
    const auto emission = OneSidedSurfaceEmissionT<Scalar>::create(radiance);
    if (!emission.has_value()) {
        return std::unexpected(emission.error());
    }
    return SurfaceFor<Scalar>{*triangle, *reflection, *emission, wavelengths, visibility_mask};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BsdfOnlyEnvironmentT<Scalar>>
make_environment(const SpectrumFor<Scalar>& radiance,
                 const SampledWavelengthsT<Scalar>& wavelengths) {
    const auto environment = ConstantEnvironmentT<Scalar>::create(radiance);
    if (!environment.has_value()) {
        return std::unexpected(environment.error());
    }
    return BsdfOnlyEnvironmentT<Scalar>{*environment, wavelengths};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>>
make_ray(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
         const Scalar time = Scalar{0}, const RayMask mask = AllRayVisibility,
         const MediumId medium = VacuumMedium) {
    return RayT<Scalar>::create(origin, direction, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), time, mask, medium);
}

template <SpectrumScalar Scalar> void expect_lambertian_environment_estimator_is_exact() {
    constexpr auto medium = MediumId{.value = 29};
    constexpr auto mask = RayMask{0x52A5U};
    constexpr auto time = Scalar{0.375};
    const auto reflectance = spectrum<Scalar>({Scalar{0.25}, Scalar{0.5}, Scalar{0.75}, Scalar{1}});
    const auto surface_radiance =
        spectrum<Scalar>({Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}});
    const auto environment_radiance =
        spectrum<Scalar>({Scalar{2}, Scalar{1}, Scalar{0.5}, Scalar{0.25}});
    const auto sampler = IndependentSamplerT<Scalar>{0x9E3779B97F4A7C15ULL};
    const auto stream = sampler.make_stream(7, 11, 13);
    const auto wavelengths = sample_visible_wavelengths(stream);
    const auto surface = make_surface(Point3T<Scalar>{.x = Scalar{-10}, .y = Scalar{-10}},
                                      Point3T<Scalar>{.x = Scalar{10}, .y = Scalar{-10}},
                                      Point3T<Scalar>{.y = Scalar{10}}, reflectance,
                                      surface_radiance, wavelengths, mask);
    const auto environment = make_environment(environment_radiance, wavelengths);
    const auto ray = make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-2}},
                              time, mask, medium);
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(ray.has_value());

    auto beta = constant_spectrum(Scalar{1});
    const auto initial = PathStateT<Scalar>::create(beta, SpectrumFor<Scalar>{}, {}, Scalar{1.5},
                                                    wavelengths, PathDeltaFlags::none, medium);
    ASSERT_TRUE(initial.has_value());

    const auto first =
        trace_bsdf_only(*ray, *initial, stream, std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    const auto replay =
        trace_bsdf_only(*ray, *initial, stream, std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_TRUE(first.has_value()) << (first.has_value() ? "" : first.error().message);
    ASSERT_TRUE(replay.has_value());

    const auto expected_radiance = surface_radiance + reflectance * environment_radiance;
    EXPECT_EQ(first->state.beta(), reflectance);
    EXPECT_EQ(first->state.accumulated_radiance(), expected_radiance);
    EXPECT_EQ(first->state.depth(), 1U);
    EXPECT_EQ(first->state.eta_scale(), Scalar{1.5});
    EXPECT_EQ(first->state.wavelengths(), wavelengths);
    EXPECT_EQ(first->state.delta_flags(), PathDeltaFlags::any_non_delta_bounces);
    EXPECT_EQ(first->state.current_medium(), medium);
    EXPECT_EQ(first->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(first->termination, BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(first->blocked_depth_limits, ScatteringLobe::none);
    EXPECT_EQ(first->terminal_ray.time(), time);
    EXPECT_EQ(first->terminal_ray.mask(), mask);
    EXPECT_EQ(first->terminal_ray.current_medium(), medium);
    EXPECT_EQ(first->terminal_ray.t_min(), Scalar{0});
    EXPECT_EQ(first->terminal_ray.t_max(), std::numeric_limits<Scalar>::infinity());
    EXPECT_GT(first->terminal_ray.origin().z, Scalar{0});

    EXPECT_EQ(replay->state.beta(), first->state.beta());
    EXPECT_EQ(replay->state.accumulated_radiance(), first->state.accumulated_radiance());
    EXPECT_EQ(replay->state.depth(), first->state.depth());
    EXPECT_EQ(replay->state.depth_counters(), first->state.depth_counters());
    EXPECT_EQ(replay->terminal_ray.origin(), first->terminal_ray.origin());
    EXPECT_EQ(replay->terminal_ray.direction(), first->terminal_ray.direction());
    EXPECT_EQ(replay->termination, first->termination);
}

TEST(BsdfOnlyPathLoopTest, LambertianContinuationMatchesTheAnalyticEstimator) {
    expect_lambertian_environment_estimator_is_exact<TransportScalar>();
    expect_lambertian_environment_estimator_is_exact<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_explicit_termination_semantics() {
    const auto white = constant_spectrum(Scalar{1});
    const auto black = SpectrumFor<Scalar>{};
    const auto environment_radiance = constant_spectrum(Scalar{4});
    const auto stream = IndependentSamplerT<Scalar>{17}.make_stream(0, 0, 0);
    const auto wavelengths = sample_visible_wavelengths(stream);
    const auto environment = make_environment(environment_radiance, wavelengths);
    const auto state = PathStateT<Scalar>::create_initial(wavelengths, VacuumMedium);
    const auto roulette =
        RussianRoulettePolicyT<Scalar>::create_enabled(1, Scalar{0.5}, Scalar{0.5});
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(state.has_value());
    ASSERT_TRUE(roulette.has_value());

    const auto miss_ray =
        make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{1}});
    ASSERT_TRUE(miss_ray.has_value());
    const auto miss =
        trace_bsdf_only(*miss_ray, *state, stream, std::span<const SurfaceFor<Scalar>>{},
                        *environment, diffuse_limits(0), disabled_roulette<Scalar>());
    ASSERT_TRUE(miss.has_value());
    EXPECT_EQ(miss->state.accumulated_radiance(), environment_radiance);
    EXPECT_EQ(miss->state.depth(), 0U);
    EXPECT_EQ(miss->state.depth_counters(), PathDepthCounters{});
    EXPECT_EQ(miss->termination, BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(miss->blocked_depth_limits, ScatteringLobe::none);

    const auto front = make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                                    Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                                    Point3T<Scalar>{.y = Scalar{2}}, white, white, wavelengths);
    ASSERT_TRUE(front.has_value());
    const auto front_ray =
        make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}});
    ASSERT_TRUE(front_ray.has_value());
    const auto cutoff =
        trace_bsdf_only(*front_ray, *state, stream, std::span<const SurfaceFor<Scalar>>{&*front, 1},
                        *environment, diffuse_limits(0), disabled_roulette<Scalar>());
    ASSERT_TRUE(cutoff.has_value());
    EXPECT_EQ(cutoff->state.accumulated_radiance(), white);
    EXPECT_EQ(cutoff->state.depth(), 0U);
    EXPECT_EQ(cutoff->state.depth_counters(), PathDepthCounters{});
    EXPECT_EQ(cutoff->termination, BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(cutoff->blocked_depth_limits, ScatteringLobe::diffuse);

    const auto absorber = make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                                       Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                                       Point3T<Scalar>{.y = Scalar{2}}, black, black, wavelengths);
    ASSERT_TRUE(absorber.has_value());
    const auto absorbed = trace_bsdf_only(*front_ray, *state, stream,
                                          std::span<const SurfaceFor<Scalar>>{&*absorber, 1},
                                          *environment, diffuse_limits(2), *roulette);
    ASSERT_TRUE(absorbed.has_value());
    EXPECT_EQ(absorbed->state.beta(), black);
    EXPECT_EQ(absorbed->state.accumulated_radiance(), black);
    EXPECT_EQ(absorbed->state.depth(), 1U);
    EXPECT_EQ(absorbed->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(absorbed->state.delta_flags(), PathDeltaFlags::any_non_delta_bounces);
    EXPECT_EQ(absorbed->termination, BsdfOnlyPathTermination::zero_throughput);
    EXPECT_EQ(absorbed->blocked_depth_limits, ScatteringLobe::none);
    EXPECT_GT(absorbed->terminal_ray.origin().z, Scalar{0});
    EXPECT_GT(absorbed->terminal_ray.direction().z, Scalar{0});
    EXPECT_EQ(absorbed->terminal_ray.t_min(), Scalar{0});
    EXPECT_EQ(absorbed->terminal_ray.t_max(), std::numeric_limits<Scalar>::infinity());

    const auto back = make_surface(
        Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}}, Point3T<Scalar>{.y = Scalar{2}},
        Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}}, white, white, wavelengths);
    ASSERT_TRUE(back.has_value());
    const auto occluded =
        trace_bsdf_only(*front_ray, *state, stream, std::span<const SurfaceFor<Scalar>>{&*back, 1},
                        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_TRUE(occluded.has_value());
    EXPECT_EQ(occluded->state.accumulated_radiance(), black);
    EXPECT_EQ(occluded->state.depth(), 0U);
    EXPECT_EQ(occluded->state.depth_counters(), PathDepthCounters{});
    EXPECT_EQ(occluded->termination, BsdfOnlyPathTermination::outside_bsdf_support);
    EXPECT_EQ(occluded->blocked_depth_limits, ScatteringLobe::none);
}

TEST(BsdfOnlyPathLoopTest, MissCutoffAbsorptionAndBackfacesTerminateExplicitly) {
    expect_explicit_termination_semantics<TransportScalar>();
    expect_explicit_termination_semantics<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_exact_diffuse_budget_resume() {
    const auto reflectance = constant_spectrum(Scalar{0.5});
    const auto white = constant_spectrum(Scalar{1});
    const auto black = SpectrumFor<Scalar>{};
    const auto stream = IndependentSamplerT<Scalar>{0xA4093822299F31D0ULL}.make_stream(3, 5, 7);
    const auto wavelengths = sample_visible_wavelengths(stream);
    const auto diffuse =
        make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.y = Scalar{2}}, reflectance, black, wavelengths);
    const auto emitter =
        make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.y = Scalar{2}}, reflectance, white, wavelengths);
    const auto environment = make_environment(black, wavelengths);
    const auto initial = PathStateT<Scalar>::create_initial(wavelengths, VacuumMedium);
    const auto ray = make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}});
    ASSERT_TRUE(diffuse.has_value());
    ASSERT_TRUE(emitter.has_value());
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(initial.has_value());
    ASSERT_TRUE(ray.has_value());
    const auto limits = diffuse_limits(2);

    const auto first =
        trace_bsdf_only(*ray, *initial, stream, std::span<const SurfaceFor<Scalar>>{&*diffuse, 1},
                        *environment, limits, disabled_roulette<Scalar>());
    ASSERT_TRUE(first.has_value()) << (first.has_value() ? "" : first.error().message);
    EXPECT_EQ(first->state.depth(), 1U);
    EXPECT_EQ(first->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(first->state.beta(), reflectance);
    EXPECT_EQ(first->termination, BsdfOnlyPathTermination::escaped_environment);

    const auto second = trace_bsdf_only(*ray, first->state, stream,
                                        std::span<const SurfaceFor<Scalar>>{&*diffuse, 1},
                                        *environment, limits, disabled_roulette<Scalar>());
    ASSERT_TRUE(second.has_value()) << (second.has_value() ? "" : second.error().message);
    EXPECT_EQ(second->state.depth(), 2U);
    EXPECT_EQ(second->state.depth_counters(), (PathDepthCounters{.diffuse = 2}));
    EXPECT_EQ(second->state.beta(), constant_spectrum(Scalar{0.25}));
    EXPECT_EQ(second->termination, BsdfOnlyPathTermination::escaped_environment);

    const auto cutoff = trace_bsdf_only(*ray, second->state, stream,
                                        std::span<const SurfaceFor<Scalar>>{&*emitter, 1},
                                        *environment, limits, disabled_roulette<Scalar>());
    ASSERT_TRUE(cutoff.has_value()) << (cutoff.has_value() ? "" : cutoff.error().message);
    EXPECT_EQ(cutoff->state.depth(), 2U);
    EXPECT_EQ(cutoff->state.depth_counters(), second->state.depth_counters());
    EXPECT_EQ(cutoff->state.beta(), second->state.beta());
    EXPECT_EQ(cutoff->state.accumulated_radiance(), constant_spectrum(Scalar{0.25}));
    EXPECT_EQ(cutoff->terminal_ray.origin(), ray->origin());
    EXPECT_EQ(cutoff->terminal_ray.direction(), ray->direction());
    EXPECT_EQ(cutoff->termination, BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(cutoff->blocked_depth_limits, ScatteringLobe::diffuse);

    const auto lowered = trace_bsdf_only(
        *ray, first->state, stream, std::span<const SurfaceFor<Scalar>>{&*emitter, 1}, *environment,
        diffuse_limits(0), disabled_roulette<Scalar>());
    ASSERT_FALSE(lowered.has_value());
    EXPECT_EQ(lowered.error().code, core::StatusCode::invalid_argument);

    const auto mixed_counters = PathDepthCounters{.glossy = 1};
    const auto mixed_state =
        PathStateT<Scalar>::create(white, black, mixed_counters, Scalar{1}, wavelengths,
                                   PathDeltaFlags::any_non_delta_bounces, VacuumMedium);
    ASSERT_TRUE(mixed_state.has_value());
    const auto mixed_limits = PathDepthLimits{
        .diffuse = 1,
        .glossy = 1,
    };
    const auto mixed = trace_bsdf_only(*ray, *mixed_state, stream,
                                       std::span<const SurfaceFor<Scalar>>{&*diffuse, 1},
                                       *environment, mixed_limits, disabled_roulette<Scalar>());
    ASSERT_TRUE(mixed.has_value()) << (mixed.has_value() ? "" : mixed.error().message);
    EXPECT_EQ(mixed->state.depth(), 2U);
    EXPECT_EQ(mixed->state.depth_counters(), (PathDepthCounters{.diffuse = 1, .glossy = 1}));
}

TEST(BsdfOnlyPathLoopTest, ContinuesExactBoundDepthCountersWithoutReclassification) {
    expect_exact_diffuse_budget_resume<TransportScalar>();
    expect_exact_diffuse_budget_resume<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_russian_roulette_uses_its_reserved_dimension() {
    constexpr auto seed = std::uint64_t{0xA4093822299F31D0ULL};
    const auto sampler = IndependentSamplerT<Scalar>{seed};
    const auto wavelength_stream = sampler.make_stream(3, 5, 0);
    const auto surviving_stream = sampler.make_stream(3, 5, 10);
    const auto terminating_stream = sampler.make_stream(3, 5, 45);
    const auto dimensions = sample_dimensions_for_bounce(0);
    ASSERT_TRUE(dimensions.has_value());
    EXPECT_LT(surviving_stream.sample_1d(dimensions->russian_roulette), Scalar{0.5});
    EXPECT_GT(surviving_stream.sample_1d(dimensions->bsdf_u), Scalar{0.5});
    EXPECT_GE(terminating_stream.sample_1d(dimensions->russian_roulette), Scalar{0.5});
    EXPECT_LT(terminating_stream.sample_1d(dimensions->bsdf_u), Scalar{0.5});

    const auto wavelengths = sample_visible_wavelengths(wavelength_stream);
    const auto reflectance = constant_spectrum(Scalar{0.5});
    const auto surface_radiance = constant_spectrum(Scalar{0.25});
    const auto environment_radiance = constant_spectrum(Scalar{1});
    const auto surface =
        make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                     Point3T<Scalar>{.y = Scalar{2}}, reflectance, surface_radiance, wavelengths);
    const auto environment = make_environment(environment_radiance, wavelengths);
    const auto state = PathStateT<Scalar>::create_initial(wavelengths, VacuumMedium);
    const auto ray = make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}});
    const auto policy = RussianRoulettePolicyT<Scalar>::create_enabled(1, Scalar{0.5}, Scalar{0.5});
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(state.has_value());
    ASSERT_TRUE(ray.has_value());
    ASSERT_TRUE(policy.has_value());

    const auto survived = trace_bsdf_only(*ray, *state, surviving_stream,
                                          std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                                          *environment, diffuse_limits(1), *policy);
    ASSERT_TRUE(survived.has_value()) << (survived.has_value() ? "" : survived.error().message);
    EXPECT_EQ(survived->termination, BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(survived->state.depth(), 1U);
    EXPECT_EQ(survived->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(survived->state.beta(), constant_spectrum(Scalar{1}));
    EXPECT_EQ(survived->state.accumulated_radiance(), constant_spectrum(Scalar{1.25}));
    EXPECT_EQ(survived->blocked_depth_limits, ScatteringLobe::none);

    const auto terminated = trace_bsdf_only(*ray, *state, terminating_stream,
                                            std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                                            *environment, diffuse_limits(1), *policy);
    ASSERT_TRUE(terminated.has_value())
        << (terminated.has_value() ? "" : terminated.error().message);
    EXPECT_EQ(terminated->termination, BsdfOnlyPathTermination::russian_roulette);
    EXPECT_EQ(terminated->state.depth(), 1U);
    EXPECT_EQ(terminated->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(terminated->state.beta(), reflectance);
    EXPECT_EQ(terminated->state.accumulated_radiance(), surface_radiance);
    EXPECT_EQ(terminated->blocked_depth_limits, ScatteringLobe::none);

    const auto delayed_policy =
        RussianRoulettePolicyT<Scalar>::create_enabled(2, Scalar{0.5}, Scalar{0.5});
    ASSERT_TRUE(delayed_policy.has_value());
    const auto delayed = trace_bsdf_only(*ray, *state, terminating_stream,
                                         std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                                         *environment, diffuse_limits(1), *delayed_policy);
    ASSERT_TRUE(delayed.has_value());
    EXPECT_EQ(delayed->termination, BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(delayed->state.beta(), reflectance);
    EXPECT_EQ(delayed->state.accumulated_radiance(), constant_spectrum(Scalar{0.75}));

    const auto limited = trace_bsdf_only(*ray, *state, terminating_stream,
                                         std::span<const SurfaceFor<Scalar>>{&*surface, 1},
                                         *environment, diffuse_limits(0), *policy);
    ASSERT_TRUE(limited.has_value());
    EXPECT_EQ(limited->termination, BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(limited->state.depth(), 0U);
    EXPECT_EQ(limited->state.depth_counters(), PathDepthCounters{});
    EXPECT_EQ(limited->state.accumulated_radiance(), surface_radiance);
    EXPECT_EQ(limited->blocked_depth_limits, ScatteringLobe::diffuse);
}

TEST(BsdfOnlyPathLoopTest, RussianRouletteUsesReservedDimensionAfterAcceptedBounce) {
    expect_russian_roulette_uses_its_reserved_dimension<TransportScalar>();
    expect_russian_roulette_uses_its_reserved_dimension<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_paths_fail_explicitly() {
    const auto white = constant_spectrum(Scalar{1});
    const auto black = SpectrumFor<Scalar>{};
    const auto stream = IndependentSamplerT<Scalar>{23}.make_stream(0, 0, 0);
    const auto wavelengths = sample_visible_wavelengths(stream);
    const auto surface = make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                                      Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                                      Point3T<Scalar>{.y = Scalar{2}}, white, black, wavelengths);
    const auto environment = make_environment(black, wavelengths);
    const auto state = PathStateT<Scalar>::create_initial(wavelengths, MediumId{.value = 1});
    const auto wrong_medium_ray =
        make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}}, Scalar{0},
                 AllRayVisibility, MediumId{.value = 2});
    ASSERT_TRUE(surface.has_value());
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(state.has_value());
    ASSERT_TRUE(wrong_medium_ray.has_value());

    const auto wrong_medium = trace_bsdf_only(
        *wrong_medium_ray, *state, stream, std::span<const SurfaceFor<Scalar>>{&*surface, 1},
        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(wrong_medium.has_value());
    EXPECT_EQ(wrong_medium.error().code, core::StatusCode::invalid_argument);

    const auto coplanar_ray = make_ray(Point3T<Scalar>{}, Vector3T<Scalar>{.x = Scalar{1}},
                                       Scalar{0}, AllRayVisibility, MediumId{.value = 1});
    ASSERT_TRUE(coplanar_ray.has_value());
    const auto coplanar = trace_bsdf_only(
        *coplanar_ray, *state, stream, std::span<const SurfaceFor<Scalar>>{&*surface, 1},
        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(coplanar.has_value());
    EXPECT_EQ(coplanar.error().code, core::StatusCode::invalid_argument);

    auto negative_beta = white;
    negative_beta[0] = Scalar{-1};
    const auto negative_state =
        PathStateT<Scalar>::create(negative_beta, black, {}, Scalar{1}, wavelengths,
                                   PathDeltaFlags::none, MediumId{.value = 1});
    const auto valid_ray =
        make_ray(Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}}, Scalar{0},
                 AllRayVisibility, MediumId{.value = 1});
    ASSERT_TRUE(negative_state.has_value());
    ASSERT_TRUE(valid_ray.has_value());
    const auto negative = trace_bsdf_only(
        *valid_ray, *negative_state, stream, std::span<const SurfaceFor<Scalar>>{&*surface, 1},
        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(negative.has_value());
    EXPECT_EQ(negative.error().code, core::StatusCode::invalid_argument);

    const auto mismatched_stream = IndependentSamplerT<Scalar>{23}.make_stream(0, 0, 1);
    const auto mismatched_wavelengths = sample_visible_wavelengths(mismatched_stream);
    const auto mismatched_state =
        PathStateT<Scalar>::create_initial(mismatched_wavelengths, MediumId{.value = 1});
    ASSERT_TRUE(mismatched_state.has_value());
    ASSERT_NE(mismatched_wavelengths, wavelengths);
    const auto mismatch =
        trace_bsdf_only(*valid_ray, *mismatched_state, mismatched_stream,
                        std::span<const SurfaceFor<Scalar>>{&*surface, 1}, *environment,
                        diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(mismatch.has_value());
    EXPECT_EQ(mismatch.error().code, core::StatusCode::invalid_argument);

    const auto mismatched_environment = make_environment(black, mismatched_wavelengths);
    ASSERT_TRUE(mismatched_environment.has_value());
    const auto surface_mismatch =
        trace_bsdf_only(*valid_ray, *mismatched_state, mismatched_stream,
                        std::span<const SurfaceFor<Scalar>>{&*surface, 1}, *mismatched_environment,
                        diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(surface_mismatch.has_value());
    EXPECT_EQ(surface_mismatch.error().code, core::StatusCode::invalid_argument);

    auto tiny_beta = black;
    tiny_beta.values.fill(std::numeric_limits<Scalar>::denorm_min());
    const auto half_surface = make_surface(Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}},
                                           Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{-2}},
                                           Point3T<Scalar>{.y = Scalar{2}},
                                           constant_spectrum(Scalar{0.5}), black, wavelengths);
    const auto tiny_state = PathStateT<Scalar>::create(tiny_beta, black, {}, Scalar{1}, wavelengths,
                                                       PathDeltaFlags::none, MediumId{.value = 1});
    ASSERT_TRUE(half_surface.has_value());
    ASSERT_TRUE(tiny_state.has_value());
    const auto underflow = trace_bsdf_only(
        *valid_ray, *tiny_state, stream, std::span<const SurfaceFor<Scalar>>{&*half_surface, 1},
        *environment, diffuse_limits(1), disabled_roulette<Scalar>());
    ASSERT_FALSE(underflow.has_value());
    EXPECT_EQ(underflow.error().code, core::StatusCode::invalid_argument);
}

TEST(BsdfOnlyPathLoopTest, InvalidGeometryStateAndUnderflowAreNeverHidden) {
    expect_invalid_paths_fail_explicitly<TransportScalar>();
    expect_invalid_paths_fail_explicitly<ReferenceScalar>();
}

TEST(BsdfOnlyPathLoopTest, DerivedHitErrorDoesNotSkipNearbyGeometry) {
    constexpr auto base_height = TransportScalar{1024};
    constexpr auto distant_origin_height = TransportScalar{16'777'216};
    auto nearby_height = base_height;
    for (auto step = 0; step < 16; ++step) {
        nearby_height =
            std::nextafter(nearby_height, std::numeric_limits<TransportScalar>::infinity());
    }
    const auto nearby_distance = nearby_height - base_height;
    const auto sampler = IndependentSampler{0xD1B54A32D192ED03ULL};
    const auto dimensions = sample_dimensions_for_bounce(0);
    const auto frame = OrthonormalFrame::from_normal(Normal3{.z = 1.0F});
    ASSERT_TRUE(dimensions.has_value());
    ASSERT_TRUE(frame.has_value());

    auto selected_sample = std::uint64_t{0};
    auto target = Point3{};
    for (; selected_sample < 128; ++selected_sample) {
        const auto candidate_stream = sampler.make_stream(0, 0, selected_sample);
        const auto local = map_cosine_hemisphere(Point2{
            .x = candidate_stream.sample_1d(dimensions->bsdf_u),
            .y = candidate_stream.sample_1d(dimensions->bsdf_v),
        });
        ASSERT_TRUE(local.has_value());
        const auto world = frame->to_world(*local);
        if (world.z > 0.2F &&
            (std::abs(world.x / world.z) > 1.0F || std::abs(world.y / world.z) > 1.0F)) {
            target = Point3{
                .x = nearby_distance * world.x / world.z,
                .y = nearby_distance * world.y / world.z,
                .z = nearby_height,
            };
            break;
        }
    }
    ASSERT_LT(selected_sample, 128U);

    const auto stream = sampler.make_stream(0, 0, selected_sample);
    const auto wavelengths = sample_visible_wavelengths(stream);
    const auto white = constant_spectrum(1.0F);
    const auto black = TransportSpectrum{};
    const auto base =
        make_surface(Point3{.x = -100.0F, .y = -100.0F, .z = base_height},
                     Point3{.x = 100.0F, .y = -100.0F, .z = base_height},
                     Point3{.y = 100.0F, .z = base_height}, white, black, wavelengths);
    const auto half_extent =
        TransportScalar{0.75} * std::max(std::abs(target.x), std::abs(target.y));
    const auto nearby_first = make_surface(
        Point3{.x = target.x - half_extent, .y = target.y - half_extent, .z = nearby_height},
        Point3{.x = target.x - half_extent, .y = target.y + half_extent, .z = nearby_height},
        Point3{.x = target.x + half_extent, .y = target.y - half_extent, .z = nearby_height}, black,
        white, wavelengths);
    const auto nearby_second = make_surface(
        Point3{.x = target.x - half_extent, .y = target.y + half_extent, .z = nearby_height},
        Point3{.x = target.x + half_extent, .y = target.y + half_extent, .z = nearby_height},
        Point3{.x = target.x + half_extent, .y = target.y - half_extent, .z = nearby_height}, black,
        white, wavelengths);
    const auto environment = make_environment(black, wavelengths);
    const auto state = PathState::create_initial(wavelengths, VacuumMedium);
    const auto ray = make_ray(Point3{.z = distant_origin_height},
                              Vector3{.z = base_height - distant_origin_height});
    ASSERT_TRUE(base.has_value());
    ASSERT_TRUE(nearby_first.has_value());
    ASSERT_TRUE(nearby_second.has_value());
    ASSERT_TRUE(environment.has_value());
    ASSERT_TRUE(state.has_value());
    ASSERT_TRUE(ray.has_value());
    const auto surfaces = std::array{*base, *nearby_first, *nearby_second};

    const auto traced = trace_bsdf_only(*ray, *state, stream, surfaces, *environment,
                                        diffuse_limits(1), RussianRoulettePolicy::disabled());
    ASSERT_TRUE(traced.has_value());
    EXPECT_EQ(traced->state.accumulated_radiance(), white);
    EXPECT_EQ(traced->state.depth(), 1U);
    EXPECT_EQ(traced->state.depth_counters(), (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(traced->termination, BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(traced->blocked_depth_limits, ScatteringLobe::diffuse);
    EXPECT_GT(traced->terminal_ray.origin().z, base_height);
    EXPECT_LT(traced->terminal_ray.origin().z, nearby_height);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
append_quad(std::vector<SurfaceFor<Scalar>>& surfaces, const Point3T<Scalar> vertex0,
            const Point3T<Scalar> vertex1, const Point3T<Scalar> vertex2,
            const Point3T<Scalar> vertex3, const SpectrumFor<Scalar>& reflectance,
            const SpectrumFor<Scalar>& radiance, const SampledWavelengthsT<Scalar>& wavelengths) {
    const auto first = make_surface(vertex0, vertex1, vertex2, reflectance, radiance, wavelengths);
    if (!first.has_value()) {
        return std::unexpected(first.error());
    }
    const auto second = make_surface(vertex0, vertex2, vertex3, reflectance, radiance, wavelengths);
    if (!second.has_value()) {
        return std::unexpected(second.error());
    }
    surfaces.push_back(*first);
    surfaces.push_back(*second);
    return {};
}

[[nodiscard]] core::Result<std::vector<ReferenceBsdfOnlyTriangleSurface>>
make_cornell_diffuse_surfaces(const ReferenceSampledWavelengths& wavelengths) {
    const auto white = constant_spectrum<ReferenceScalar>(0.75);
    const auto red = spectrum<ReferenceScalar>({0.75, 0.25, 0.25, 0.2});
    const auto green = spectrum<ReferenceScalar>({0.2, 0.75, 0.25, 0.2});
    const auto black = ReferenceSpectrum{};
    const auto light = spectrum<ReferenceScalar>({8.0, 6.0, 4.0, 2.0});

    auto surfaces = std::vector<ReferenceBsdfOnlyTriangleSurface>{};
    surfaces.reserve(10);
    for (const auto& status : std::array{
             append_quad(surfaces, ReferencePoint3{.x = -1.0, .y = -1.0},
                         ReferencePoint3{.x = 1.0, .y = -1.0}, ReferencePoint3{.x = 1.0, .y = 1.0},
                         ReferencePoint3{.x = -1.0, .y = 1.0}, white, black, wavelengths),
             append_quad(surfaces, ReferencePoint3{.x = -1.0, .y = -1.0},
                         ReferencePoint3{.x = -1.0, .y = -1.0, .z = 2.0},
                         ReferencePoint3{.x = 1.0, .y = -1.0, .z = 2.0},
                         ReferencePoint3{.x = 1.0, .y = -1.0}, white, black, wavelengths),
             append_quad(surfaces, ReferencePoint3{.x = -1.0, .y = 1.0},
                         ReferencePoint3{.x = 1.0, .y = 1.0},
                         ReferencePoint3{.x = 1.0, .y = 1.0, .z = 2.0},
                         ReferencePoint3{.x = -1.0, .y = 1.0, .z = 2.0}, black, light, wavelengths),
             append_quad(surfaces, ReferencePoint3{.x = -1.0, .y = -1.0},
                         ReferencePoint3{.x = -1.0, .y = 1.0},
                         ReferencePoint3{.x = -1.0, .y = 1.0, .z = 2.0},
                         ReferencePoint3{.x = -1.0, .y = -1.0, .z = 2.0}, red, black, wavelengths),
             append_quad(surfaces, ReferencePoint3{.x = 1.0, .y = -1.0},
                         ReferencePoint3{.x = 1.0, .y = -1.0, .z = 2.0},
                         ReferencePoint3{.x = 1.0, .y = 1.0, .z = 2.0},
                         ReferencePoint3{.x = 1.0, .y = 1.0}, green, black, wavelengths),
         }) {
        if (!status.has_value()) {
            return std::unexpected(status.error());
        }
    }
    return surfaces;
}

struct CornellEstimate final {
    std::array<ReferenceSpectrum, 3> probe_radiance{};
    std::uint64_t first_bounce_light_paths{};
    std::uint64_t multiple_bounce_light_paths{};
};

[[nodiscard]] core::Result<CornellEstimate>
estimate_cornell(const std::span<const ReferenceBsdfOnlyTriangleSurface> surfaces,
                 const ReferenceSampledWavelengths& wavelengths, const std::uint64_t seed,
                 const std::uint32_t sample_count) {
    constexpr auto probe_directions = std::array{
        ReferenceVector3{.z = -3.0},
        ReferenceVector3{.y = -1.0, .z = -2.0},
        ReferenceVector3{.x = -1.0, .z = -2.0},
    };
    const auto environment = make_environment(ReferenceSpectrum{}, wavelengths);
    if (!environment.has_value()) {
        return std::unexpected(environment.error());
    }

    auto estimate = CornellEstimate{};
    const auto sampler = ReferenceIndependentSampler{seed};
    for (auto probe = std::size_t{0}; probe < probe_directions.size(); ++probe) {
        const auto ray = make_ray(ReferencePoint3{.z = 3.0}, probe_directions[probe]);
        if (!ray.has_value()) {
            return std::unexpected(ray.error());
        }

        for (auto sample_index = std::uint64_t{0}; sample_index < sample_count; ++sample_index) {
            const auto stream =
                sampler.make_stream(static_cast<std::uint32_t>(probe), 0, sample_index);
            const auto state = ReferencePathState::create_initial(wavelengths, VacuumMedium);
            if (!state.has_value()) {
                return std::unexpected(state.error());
            }
            const auto traced =
                trace_bsdf_only(*ray, *state, stream, surfaces, *environment, diffuse_limits(3),
                                disabled_roulette<ReferenceScalar>());
            if (!traced.has_value()) {
                return std::unexpected(traced.error());
            }

            const auto& radiance = traced->state.accumulated_radiance();
            const auto& counters = traced->state.depth_counters();
            if (counters.diffuse != traced->state.depth() || counters.glossy != 0 ||
                counters.specular != 0 || counters.transmission != 0 || counters.volume != 0) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::invalid_argument,
                    .message = "Cornell BSDF-only depth counters are inconsistent.",
                });
            }
            for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                estimate.probe_radiance[probe][lane] += radiance[lane];
            }
            if (!is_zero_spectrum(radiance)) {
                if (traced->state.depth() <= 2) {
                    ++estimate.first_bounce_light_paths;
                } else {
                    ++estimate.multiple_bounce_light_paths;
                }
            }
        }

        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            estimate.probe_radiance[probe][lane] /= static_cast<ReferenceScalar>(sample_count);
        }
    }
    return estimate;
}

[[nodiscard]] std::array<ReferenceSpectrum, 3>
mean_estimate(const std::span<const CornellEstimate> estimates) {
    auto mean = std::array<ReferenceSpectrum, 3>{};
    for (const auto& estimate : estimates) {
        for (auto probe = std::size_t{0}; probe < mean.size(); ++probe) {
            for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                mean[probe][lane] += estimate.probe_radiance[probe][lane];
            }
        }
    }
    for (auto& probe : mean) {
        for (auto& lane : probe.values) {
            lane /= static_cast<ReferenceScalar>(estimates.size());
        }
    }
    return mean;
}

[[nodiscard]] ReferenceScalar relative_rmse(const std::array<ReferenceSpectrum, 3>& estimate,
                                            const std::array<ReferenceSpectrum, 3>& reference) {
    auto squared_error = ReferenceScalar{0};
    auto squared_reference = ReferenceScalar{0};
    for (auto probe = std::size_t{0}; probe < estimate.size(); ++probe) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto difference = estimate[probe][lane] - reference[probe][lane];
            squared_error = std::fma(difference, difference, squared_error);
            squared_reference =
                std::fma(reference[probe][lane], reference[probe][lane], squared_reference);
        }
    }
    return std::sqrt(squared_error / squared_reference);
}

TEST(BsdfOnlyPathLoopTest, CornellDiffuseReplayIsDeterministic) {
    const auto wavelength_stream = ReferenceIndependentSampler{0}.make_stream(0, 0, 0);
    const auto wavelengths = sample_visible_wavelengths(wavelength_stream);
    const auto surfaces = make_cornell_diffuse_surfaces(wavelengths);
    ASSERT_TRUE(surfaces.has_value());
    ASSERT_EQ(surfaces->size(), 10U);

    const auto first = estimate_cornell(*surfaces, wavelengths, 0x243F6A8885A308D3ULL, 128);
    const auto replay = estimate_cornell(*surfaces, wavelengths, 0x243F6A8885A308D3ULL, 128);
    ASSERT_TRUE(first.has_value()) << (first.has_value() ? "" : first.error().message);
    ASSERT_TRUE(replay.has_value()) << (replay.has_value() ? "" : replay.error().message);
    EXPECT_EQ(first->probe_radiance, replay->probe_radiance);
    EXPECT_EQ(first->first_bounce_light_paths, replay->first_bounce_light_paths);
    EXPECT_EQ(first->multiple_bounce_light_paths, replay->multiple_bounce_light_paths);
    EXPECT_GT(first->first_bounce_light_paths, 0U);
    EXPECT_GT(first->multiple_bounce_light_paths, 0U);
    EXPECT_EQ(relative_rmse(first->probe_radiance, replay->probe_radiance), 0.0);
    const auto identical = std::array{*first, *replay};
    EXPECT_EQ(mean_estimate(identical), first->probe_radiance);
}

} // namespace
} // namespace blackframe::renderer
