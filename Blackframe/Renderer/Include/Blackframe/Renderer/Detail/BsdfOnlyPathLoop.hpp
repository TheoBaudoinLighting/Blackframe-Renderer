#pragma once

#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/ClosureMixture.hpp>
#include <Blackframe/Renderer/Detail/DepthFilteredClosureMixture.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/RayOriginOffset.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/ShadingNormalCorrection.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace blackframe::renderer::bsdf_only_path_loop_detail {

[[nodiscard]] inline core::Error path_loop_error(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
using PathSpectrum = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_non_negative(const PathSpectrum<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool zero_spectrum(const PathSpectrum<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (value != Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>> checked_product(const PathSpectrum<Scalar>& left,
                                                                 const PathSpectrum<Scalar>& right,
                                                                 const char* const error_message) {
    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = left[lane] * right[lane];
        if (!std::isfinite(result[lane]) ||
            (left[lane] != Scalar{0} && right[lane] != Scalar{0} && result[lane] == Scalar{0})) {
            return std::unexpected(path_loop_error(error_message));
        }
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>>
update_closure_throughput(const PathSpectrum<Scalar>& beta,
                          const ClosureMixtureSampleT<Scalar>& sample,
                          const Scalar shading_normal_correction,
                          const detail::DepthFilteredClosureMixtureT<Scalar>& closures) {
    if (!is_valid_surface_scattering_event(sample.lobes) || !finite_non_negative(sample.value) ||
        !std::isfinite(sample.probability.value) || !(sample.probability.value > Scalar{0}) ||
        !std::isfinite(shading_normal_correction) || !(shading_normal_correction > Scalar{0})) {
        return std::unexpected(path_loop_error("A closure returned an invalid transport sample."));
    }
    const auto delta = is_delta_surface_scattering_event(sample.lobes);
    const auto expected_measure =
        delta ? DeltaBsdfProbabilityMeasure : ContinuousBsdfProbabilityMeasure;
    if (sample.probability.measure != expected_measure ||
        (delta && sample.probability.value > Scalar{1})) {
        return std::unexpected(
            path_loop_error("A closure sample uses an incompatible probability measure."));
    }
    const auto cosine = std::abs(sample.incoming_local.z);
    if (!std::isfinite(cosine) || !(cosine > Scalar{0})) {
        return std::unexpected(
            path_loop_error("A closure direction has no representable projected measure."));
    }

    if (closures.size() == 1U &&
        closures.active_closure(0U).kind == ClosureKind::lambertian_reflection &&
        shading_normal_correction == Scalar{1}) {
        return checked_product(beta, closures.active_closure(0U).weight,
                               "Lambertian path throughput is not representable.");
    }

    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto numerators =
            std::array{beta[lane], sample.value[lane], cosine, shading_normal_correction};
        auto significand = Scalar{1};
        auto exponent = 0;
        auto zero = false;
        for (const auto factor : numerators) {
            if (!std::isfinite(factor) || factor < Scalar{0}) {
                return std::unexpected(
                    path_loop_error("Closure factors must be finite and non-negative."));
            }
            if (factor == Scalar{0}) {
                zero = true;
                break;
            }
            auto factor_exponent = 0;
            significand *= std::frexp(factor, &factor_exponent);
            exponent += factor_exponent;
        }
        if (zero) {
            result[lane] = Scalar{0};
            continue;
        }
        auto probability_exponent = 0;
        significand /= std::frexp(sample.probability.value, &probability_exponent);
        exponent -= probability_exponent;
        auto normalization_exponent = 0;
        significand = std::frexp(significand, &normalization_exponent);
        const auto value = std::scalbn(significand, exponent + normalization_exponent);
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return std::unexpected(path_loop_error("Closure throughput is not representable."));
        }
        result[lane] = value;
    }
    return result;
}

[[nodiscard]] constexpr PathDeltaFlags updated_delta_flags(const PathDeltaFlags current,
                                                           const ScatteringLobe event) noexcept {
    auto flags = has_path_delta_flag(current, PathDeltaFlags::any_non_delta_bounces)
                     ? PathDeltaFlags::any_non_delta_bounces
                     : PathDeltaFlags::none;
    if (is_delta_surface_scattering_event(event)) {
        flags = flags | PathDeltaFlags::previous_bounce_was_delta;
    } else {
        flags = flags | PathDeltaFlags::any_non_delta_bounces;
    }
    return flags;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>>
accumulate_emission(const PathSpectrum<Scalar>& accumulated, const PathSpectrum<Scalar>& throughput,
                    const PathSpectrum<Scalar>& emitted_radiance) {
    const auto contribution =
        checked_product(throughput, emitted_radiance,
                        "BSDF-only emitted-radiance multiplication is not representable.");
    if (!contribution) {
        return std::unexpected(contribution.error());
    }

    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = accumulated[lane] + (*contribution)[lane];
        if (!std::isfinite(result[lane])) {
            return std::unexpected(
                path_loop_error("BSDF-only radiance accumulation is not representable."));
        }
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PathSpectrum<Scalar>>
accumulate_direct_lighting(const PathSpectrum<Scalar>& accumulated,
                           const PathSpectrum<Scalar>& contribution) {
    if (!finite_non_negative(contribution)) {
        return std::unexpected(
            path_loop_error("A direct-lighting contribution must be finite and non-negative."));
    }

    auto result = PathSpectrum<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = accumulated[lane] + contribution[lane];
        if (!std::isfinite(result[lane])) {
            return std::unexpected(
                path_loop_error("Direct-lighting accumulation is not representable."));
        }
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>>
robust_unit_direction(const Vector3T<Scalar> direction) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return std::unexpected(path_loop_error("A BSDF-only path direction must remain finite."));
    }
    const auto maximum_component =
        std::max({std::abs(direction.x), std::abs(direction.y), std::abs(direction.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(path_loop_error("A BSDF-only path direction must remain non-zero."));
    }
    const auto scaled = direction / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    const auto result = scaled / magnitude;
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        return std::unexpected(
            path_loop_error("BSDF-only path normalization is not representable."));
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>>
make_result(const PathSpectrum<Scalar>& beta, const PathSpectrum<Scalar>& radiance,
            const std::uint32_t depth, const Scalar eta_scale,
            const SampledWavelengthsT<Scalar>& wavelengths, const PathDeltaFlags delta_flags,
            const MediumId current_medium, const PathDepthLimits& depth_limits,
            const PathDepthCounters& depth_counters, const RayT<Scalar>& terminal_ray,
            const BsdfOnlyPathTermination termination, const ScatteringLobe blocked_depth_limits) {
    const auto depth_status = validate_path_depth_state(depth_limits, depth_counters, depth);
    if (!depth_status) {
        return std::unexpected(depth_status.error());
    }
    const auto state = PathStateT<Scalar>::create(beta, radiance, depth_counters, eta_scale,
                                                  wavelengths, delta_flags, current_medium);
    if (!state) {
        return std::unexpected(state.error());
    }
    return BsdfOnlyPathResultT<Scalar>{
        .state = *state,
        .terminal_ray = terminal_ray,
        .termination = termination,
        .blocked_depth_limits = blocked_depth_limits,
    };
}

// Internal compile-time seam between the scalar transport kernel and a scene
// query. It is deliberately not a runtime transport interface: Renderer uses a
// linear triangle query, while SceneGeometry supplies AccelBackend-resolved
// surfaces. Both execute this exact bounce, depth, sampling, and roulette code.
template <SpectrumScalar Scalar, typename SurfaceQuery, typename DirectLightingPolicy>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>> trace_lambertian_with_query(
    const RayT<Scalar>& initial_ray, const PathStateT<Scalar>& initial_state,
    const SampleStreamT<Scalar>& sample_stream, SurfaceQuery& surface_query,
    const BsdfOnlyEnvironmentT<Scalar>& environment, const PathDepthLimits& depth_limits,
    const RussianRoulettePolicyT<Scalar>& roulette_policy, DirectLightingPolicy& direct_lighting) {
    const auto roulette_status = validate_russian_roulette_policy(roulette_policy);
    if (!roulette_status) {
        return std::unexpected(roulette_status.error());
    }
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(path_loop_error(
            "The BSDF-only ray and path state must carry the same current medium."));
    }
    if (!finite_non_negative(initial_state.beta())) {
        return std::unexpected(
            path_loop_error("BSDF-only path throughput must be finite and non-negative."));
    }
    const auto initial_depth_status = validate_path_depth_state(
        depth_limits, initial_state.depth_counters(), initial_state.depth());
    if (!initial_depth_status) {
        return std::unexpected(initial_depth_status.error());
    }
    if (environment.wavelengths() != initial_state.wavelengths()) {
        return std::unexpected(
            path_loop_error("The BSDF-only environment was not resolved at the path wavelengths."));
    }
    const auto query_status = surface_query.validate(initial_state.wavelengths());
    if (!query_status) {
        return std::unexpected(query_status.error());
    }

    auto beta = initial_state.beta();
    auto radiance = initial_state.accumulated_radiance();
    auto depth = initial_state.depth();
    auto depth_counters = initial_state.depth_counters();
    const auto eta_scale = initial_state.eta_scale();
    const auto wavelengths = initial_state.wavelengths();
    auto delta_flags = initial_state.delta_flags();
    const auto current_medium = initial_state.current_medium();
    auto ray = initial_ray;

    const auto finish = [&](const BsdfOnlyPathTermination termination,
                            const ScatteringLobe blocked_depth_limits)
        -> core::Result<BsdfOnlyPathResultT<Scalar>> {
        return make_result(beta, radiance, depth, eta_scale, wavelengths, delta_flags,
                           current_medium, depth_limits, depth_counters, ray, termination,
                           blocked_depth_limits);
    };

    while (true) {
        const auto resolved = surface_query.closest_hit(ray);
        if (!resolved) {
            return std::unexpected(resolved.error());
        }
        if (!*resolved) {
            const auto emitted = environment.environment().eval(ray.direction());
            if (!emitted) {
                return std::unexpected(emitted.error());
            }
            const auto accumulated = accumulate_emission(radiance, beta, *emitted);
            if (!accumulated) {
                return std::unexpected(accumulated.error());
            }
            radiance = *accumulated;
            return finish(BsdfOnlyPathTermination::escaped_environment, ScatteringLobe::none);
        }

        const auto& surface = **resolved;
        const auto emitted = surface.emission().eval(surface.geometric_normal(), -ray.direction());
        if (!emitted) {
            return std::unexpected(emitted.error());
        }
        const auto accumulated = accumulate_emission(radiance, beta, *emitted);
        if (!accumulated) {
            return std::unexpected(accumulated.error());
        }
        radiance = *accumulated;

        constexpr auto diffuse_reflection = ScatteringLobe::diffuse | ScatteringLobe::reflection;
        const auto depth_event =
            evaluate_path_depth_event(depth_limits, depth_counters, diffuse_reflection);
        if (!depth_event) {
            return std::unexpected(depth_event.error());
        }
        if (!depth_event->accepted()) {
            return finish(BsdfOnlyPathTermination::depth_limit, depth_event->blocked_limits);
        }
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }

        const auto frame = OrthonormalFrameT<Scalar>::from_normal(surface.shading_normal());
        if (!frame) {
            return std::unexpected(frame.error());
        }
        const auto outgoing_world = robust_unit_direction(-ray.direction());
        if (!outgoing_world) {
            return std::unexpected(outgoing_world.error());
        }
        const auto outgoing_local = frame->to_local(*outgoing_world);

        const auto dimensions = sample_dimensions_for_bounce(depth);
        if (!dimensions) {
            return std::unexpected(dimensions.error());
        }
        if constexpr (DirectLightingPolicy::enabled) {
            const auto direct = direct_lighting.estimate(surface, beta, *frame, *outgoing_world,
                                                         *dimensions, sample_stream, ray);
            if (!direct) {
                return std::unexpected(direct.error());
            }
            const auto accumulated_direct = accumulate_direct_lighting(radiance, *direct);
            if (!accumulated_direct) {
                return std::unexpected(accumulated_direct.error());
            }
            radiance = *accumulated_direct;
        }
        const auto canonical_sample = Point2T<Scalar>{
            .x = sample_stream.sample_1d(dimensions->bsdf_u),
            .y = sample_stream.sample_1d(dimensions->bsdf_v),
        };
        const auto sampled = surface.reflection().sample(outgoing_local, canonical_sample);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        if (!*sampled) {
            return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
        }

        const auto& bsdf_sample = **sampled;
        if (!finite_non_negative(bsdf_sample.value) ||
            bsdf_sample.probability.measure != ProbabilityMeasure::solid_angle ||
            !std::isfinite(bsdf_sample.probability.value) ||
            !(bsdf_sample.probability.value > Scalar{0}) ||
            !(bsdf_sample.incoming_local.z > Scalar{0})) {
            return std::unexpected(
                path_loop_error("The Lambertian BSDF returned an invalid continuation sample."));
        }
        const auto incoming_world =
            robust_unit_direction(frame->to_world(bsdf_sample.incoming_local));
        if (!incoming_world) {
            return std::unexpected(incoming_world.error());
        }
        const auto correction =
            shading_normal_correction(surface.geometric_normal(), surface.shading_normal(),
                                      *outgoing_world, *incoming_world, TransportMode::radiance);
        if (!correction) {
            return std::unexpected(correction.error());
        }
        if (*correction == Scalar{0}) {
            return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
        }
        if (*correction != Scalar{1}) {
            return std::unexpected(
                path_loop_error("A radiance path received a non-unit shading-normal correction."));
        }

        // Ns defines both the Lambertian cosine and its sampling PDF, so f * cos(Ns) / pdf
        // cancels exactly to reflectance. The radiance-mode Veach correction above is one.
        const auto updated_beta =
            checked_product(beta, surface.reflection().reflectance(),
                            "BSDF-only Lambertian throughput is not representable.");
        if (!updated_beta) {
            return std::unexpected(updated_beta.error());
        }

        const auto next_depth = path_depth_total(depth_event->counters);
        if (!next_depth) {
            return std::unexpected(next_depth.error());
        }
        constexpr auto next_delta_flags = PathDeltaFlags::any_non_delta_bounces;
        const auto origin = offset_ray_origin(surface.position(), surface.position_error(),
                                              surface.geometric_normal(), *incoming_world);
        if (!origin) {
            return std::unexpected(origin.error());
        }
        if (*origin == surface.position()) {
            return std::unexpected(path_loop_error(
                "The derived triangle error did not move the continuation-ray origin."));
        }

        const auto next_ray = RayT<Scalar>::create(*origin, *incoming_world, Scalar{0},
                                                   std::numeric_limits<Scalar>::infinity(),
                                                   ray.time(), ray.mask(), ray.current_medium());
        if (!next_ray) {
            return std::unexpected(next_ray.error());
        }

        beta = *updated_beta;
        depth = *next_depth;
        depth_counters = depth_event->counters;
        delta_flags = next_delta_flags;
        ray = *next_ray;
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }
        if (roulette_policy.is_enabled() && depth >= roulette_policy.first_eligible_depth()) {
            const auto roulette = evaluate_russian_roulette(
                beta, eta_scale, depth, sample_stream.sample_1d(dimensions->russian_roulette),
                roulette_policy);
            if (!roulette) {
                return std::unexpected(roulette.error());
            }
            switch (roulette->outcome) {
            case RussianRouletteOutcome::survived:
                beta = roulette->throughput;
                break;
            case RussianRouletteOutcome::terminated:
                beta = roulette->throughput;
                return finish(BsdfOnlyPathTermination::russian_roulette, ScatteringLobe::none);
            case RussianRouletteOutcome::not_evaluated:
                return std::unexpected(
                    path_loop_error("An eligible Russian roulette decision was not evaluated."));
            default:
                return std::unexpected(
                    path_loop_error("Russian roulette returned an unsupported outcome."));
            }
        }
    }
}

// Frame-scene closure transport keeps the same scalar path contract while selecting a bounded
// mixture and event-specific depth counter. It is separate from the historical Lambertian oracle
// above so that extending scene materials cannot alter that independent reference path.
template <typename SurfaceQuery, typename DirectLightingPolicy>
[[nodiscard]] core::Result<BsdfOnlyPathResult> trace_closure_mixture_with_query(
    const Ray& initial_ray, const PathState& initial_state, const SampleStream& sample_stream,
    SurfaceQuery& surface_query, const BsdfOnlyEnvironment& environment,
    const PathDepthLimits& depth_limits, const RussianRoulettePolicy& roulette_policy,
    DirectLightingPolicy& direct_lighting) {
    if (const auto status = validate_russian_roulette_policy(roulette_policy); !status) {
        return std::unexpected(status.error());
    }
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(
            path_loop_error("The closure ray and path state must carry the same current medium."));
    }
    if (!finite_non_negative(initial_state.beta())) {
        return std::unexpected(
            path_loop_error("Closure path throughput must be finite and non-negative."));
    }
    if (const auto status = validate_path_depth_state(depth_limits, initial_state.depth_counters(),
                                                      initial_state.depth());
        !status) {
        return std::unexpected(status.error());
    }
    if (environment.wavelengths() != initial_state.wavelengths()) {
        return std::unexpected(
            path_loop_error("The closure environment was not resolved at the path wavelengths."));
    }
    if (const auto status = surface_query.validate(initial_state.wavelengths()); !status) {
        return std::unexpected(status.error());
    }

    auto beta = initial_state.beta();
    auto radiance = initial_state.accumulated_radiance();
    auto depth = initial_state.depth();
    auto depth_counters = initial_state.depth_counters();
    auto eta_scale = initial_state.eta_scale();
    const auto wavelengths = initial_state.wavelengths();
    auto delta_flags = initial_state.delta_flags();
    const auto current_medium = initial_state.current_medium();
    auto ray = initial_ray;

    const auto finish =
        [&](const BsdfOnlyPathTermination termination,
            const ScatteringLobe blocked_depth_limits) -> core::Result<BsdfOnlyPathResult> {
        return make_result(beta, radiance, depth, eta_scale, wavelengths, delta_flags,
                           current_medium, depth_limits, depth_counters, ray, termination,
                           blocked_depth_limits);
    };

    while (true) {
        const auto resolved = surface_query.closest_hit(ray);
        if (!resolved) {
            return std::unexpected(resolved.error());
        }
        if (!*resolved) {
            const auto emitted = environment.environment().eval(ray.direction());
            if (!emitted) {
                return std::unexpected(emitted.error());
            }
            const auto accumulated = accumulate_emission(radiance, beta, *emitted);
            if (!accumulated) {
                return std::unexpected(accumulated.error());
            }
            radiance = *accumulated;
            return finish(BsdfOnlyPathTermination::escaped_environment, ScatteringLobe::none);
        }

        const auto& surface = **resolved;
        const auto emitted = surface.emission().eval(surface.geometric_normal(), -ray.direction());
        if (!emitted) {
            return std::unexpected(emitted.error());
        }
        const auto accumulated = accumulate_emission(radiance, beta, *emitted);
        if (!accumulated) {
            return std::unexpected(accumulated.error());
        }
        radiance = *accumulated;
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }

        const auto filtered_closures = detail::DepthFilteredClosureMixture::create(
            surface.closures(), depth_limits, depth_counters);
        if (!filtered_closures) {
            return std::unexpected(filtered_closures.error());
        }
        if (filtered_closures->empty()) {
            if (filtered_closures->source_empty()) {
                return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
            }
            return finish(BsdfOnlyPathTermination::depth_limit, filtered_closures->blocked_lobes());
        }

        const auto& frame = surface.closure_frame();
        const auto outgoing_world = robust_unit_direction(-ray.direction());
        if (!outgoing_world) {
            return std::unexpected(outgoing_world.error());
        }
        const auto outgoing_local = frame.to_local(*outgoing_world);
        const auto dimensions = sample_dimensions_for_bounce(depth);
        if (!dimensions) {
            return std::unexpected(dimensions.error());
        }
        const auto sampled = filtered_closures->sample(
            outgoing_local, sample_stream.sample_1d(dimensions->bsdf_component),
            Point2{
                .x = sample_stream.sample_1d(dimensions->bsdf_u),
                .y = sample_stream.sample_1d(dimensions->bsdf_v),
            },
            TransportMode::radiance);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }

        auto depth_event = PathDepthEventResult{};
        if (*sampled) {
            const auto& candidate = **sampled;
            if (!is_valid_surface_scattering_event(candidate.lobes) ||
                !finite_non_negative(candidate.value) ||
                !std::isfinite(candidate.probability.value) ||
                !(candidate.probability.value > TransportScalar{0}) ||
                candidate.incoming_local.z == TransportScalar{0}) {
                return std::unexpected(
                    path_loop_error("A scene closure returned an invalid continuation sample."));
            }
            const auto evaluated_depth =
                evaluate_path_depth_event(depth_limits, depth_counters, candidate.lobes);
            if (!evaluated_depth) {
                return std::unexpected(evaluated_depth.error());
            }
            if (!evaluated_depth->accepted()) {
                return std::unexpected(path_loop_error(
                    "A depth-filtered closure selected a blocked continuation event."));
            }
            depth_event = *evaluated_depth;
        }

        if constexpr (DirectLightingPolicy::enabled) {
            const auto direct =
                direct_lighting.estimate(surface, *filtered_closures, beta, frame, *outgoing_world,
                                         *dimensions, sample_stream, ray);
            if (!direct) {
                return std::unexpected(direct.error());
            }
            const auto accumulated_direct = accumulate_direct_lighting(radiance, *direct);
            if (!accumulated_direct) {
                return std::unexpected(accumulated_direct.error());
            }
            radiance = *accumulated_direct;
        }
        if (!*sampled) {
            return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
        }

        const auto& closure_sample = **sampled;
        const auto incoming_world =
            robust_unit_direction(frame.to_world(closure_sample.incoming_local));
        if (!incoming_world) {
            return std::unexpected(incoming_world.error());
        }
        const auto correction =
            shading_normal_correction(surface.geometric_normal(), surface.shading_normal(),
                                      *outgoing_world, *incoming_world, TransportMode::radiance);
        if (!correction) {
            return std::unexpected(correction.error());
        }
        if (*correction == TransportScalar{0}) {
            return finish(BsdfOnlyPathTermination::outside_bsdf_support, ScatteringLobe::none);
        }
        const auto updated_beta =
            update_closure_throughput(beta, closure_sample, *correction, *filtered_closures);
        if (!updated_beta) {
            return std::unexpected(updated_beta.error());
        }
        const auto next_depth = path_depth_total(depth_event.counters);
        if (!next_depth) {
            return std::unexpected(next_depth.error());
        }
        const auto next_eta_scale = eta_scale * closure_sample.eta_scale_multiplier;
        if (!std::isfinite(next_eta_scale) || !(next_eta_scale > TransportScalar{0})) {
            return std::unexpected(
                path_loop_error("A scene closure eta-scale update is not representable."));
        }
        const auto origin = offset_ray_origin(surface.position(), surface.position_error(),
                                              surface.geometric_normal(), *incoming_world);
        if (!origin) {
            return std::unexpected(origin.error());
        }
        if (*origin == surface.position()) {
            return std::unexpected(path_loop_error(
                "The derived triangle error did not move the closure continuation origin."));
        }
        const auto next_ray = Ray::create(*origin, *incoming_world, TransportScalar{0},
                                          std::numeric_limits<TransportScalar>::infinity(),
                                          ray.time(), ray.mask(), ray.current_medium());
        if (!next_ray) {
            return std::unexpected(next_ray.error());
        }

        beta = *updated_beta;
        depth = *next_depth;
        depth_counters = depth_event.counters;
        eta_scale = next_eta_scale;
        delta_flags = updated_delta_flags(delta_flags, closure_sample.lobes);
        ray = *next_ray;
        if (zero_spectrum(beta)) {
            return finish(BsdfOnlyPathTermination::zero_throughput, ScatteringLobe::none);
        }
        if (roulette_policy.is_enabled() && depth >= roulette_policy.first_eligible_depth()) {
            const auto roulette = evaluate_russian_roulette(
                beta, eta_scale, depth, sample_stream.sample_1d(dimensions->russian_roulette),
                roulette_policy);
            if (!roulette) {
                return std::unexpected(roulette.error());
            }
            switch (roulette->outcome) {
            case RussianRouletteOutcome::survived:
                beta = roulette->throughput;
                break;
            case RussianRouletteOutcome::terminated:
                beta = roulette->throughput;
                return finish(BsdfOnlyPathTermination::russian_roulette, ScatteringLobe::none);
            case RussianRouletteOutcome::not_evaluated:
                return std::unexpected(
                    path_loop_error("An eligible closure roulette decision was not evaluated."));
            default:
                return std::unexpected(
                    path_loop_error("Closure roulette returned an unsupported outcome."));
            }
        }
    }
}

struct NoDirectLighting final {
    static constexpr bool enabled = false;
};

template <SpectrumScalar Scalar, typename SurfaceQuery>
[[nodiscard]] core::Result<BsdfOnlyPathResultT<Scalar>>
trace_bsdf_only_with_query(const RayT<Scalar>& initial_ray, const PathStateT<Scalar>& initial_state,
                           const SampleStreamT<Scalar>& sample_stream, SurfaceQuery& surface_query,
                           const BsdfOnlyEnvironmentT<Scalar>& environment,
                           const PathDepthLimits& depth_limits,
                           const RussianRoulettePolicyT<Scalar>& roulette_policy) {
    auto direct_lighting = NoDirectLighting{};
    return trace_lambertian_with_query(initial_ray, initial_state, sample_stream, surface_query,
                                       environment, depth_limits, roulette_policy, direct_lighting);
}

template <typename SurfaceQuery>
[[nodiscard]] core::Result<BsdfOnlyPathResult> trace_closure_bsdf_only_with_query(
    const Ray& initial_ray, const PathState& initial_state, const SampleStream& sample_stream,
    SurfaceQuery& surface_query, const BsdfOnlyEnvironment& environment,
    const PathDepthLimits& depth_limits, const RussianRoulettePolicy& roulette_policy) {
    auto direct_lighting = NoDirectLighting{};
    return trace_closure_mixture_with_query(initial_ray, initial_state, sample_stream,
                                            surface_query, environment, depth_limits,
                                            roulette_policy, direct_lighting);
}

} // namespace blackframe::renderer::bsdf_only_path_loop_detail
