#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace blackframe::renderer {
namespace direct_lighting_detail {

[[nodiscard]] inline core::Error invalid_direct_lighting(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_non_negative(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

// Combines the positive factors in exponent space. The significand remains
// near one, so representable final values are not lost to an intermediate
// product or quotient. A positive result outside the requested precision is
// an error rather than zero or infinity.
template <SpectrumScalar Scalar, std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] core::Result<Scalar>
checked_product_quotient(const std::array<Scalar, NumeratorCount>& numerators,
                         const std::array<Scalar, DenominatorCount>& denominators) {
    auto significand = Scalar{1};
    auto exponent = 0;

    for (const auto value : numerators) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return std::unexpected(invalid_direct_lighting(
                "Direct-lighting numerator factors must be finite and non-negative."));
        }
        if (value == Scalar{0}) {
            return Scalar{0};
        }
        auto factor_exponent = 0;
        const auto factor_significand = std::frexp(value, &factor_exponent);
        significand *= factor_significand;
        exponent += factor_exponent;
    }

    for (const auto value : denominators) {
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return std::unexpected(invalid_direct_lighting(
                "Direct-lighting PDF factors must be finite and strictly positive."));
        }
        auto factor_exponent = 0;
        const auto factor_significand = std::frexp(value, &factor_exponent);
        significand /= factor_significand;
        exponent -= factor_exponent;
    }

    auto normalization_exponent = 0;
    significand = std::frexp(significand, &normalization_exponent);
    exponent += normalization_exponent;
    const auto result = std::scalbn(significand, exponent);
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(invalid_direct_lighting(
            "Direct-lighting contribution is not representable in the requested precision."));
    }
    return result;
}

} // namespace direct_lighting_detail

// Evaluates one sampled-light estimator from an already evaluated BSDF. This is the generic
// closure path used by bounded mixtures; unlike the Lambertian convenience wrapper below it does
// not assume a particular lobe or reconstruct its coefficient. The BSDF value excludes cosine and
// probability terms, following TransportConventions.hpp.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LightSpectrumT<Scalar>> evaluate_bsdf_direct_lighting(
    const LightSpectrumT<Scalar>& beta, const LightSpectrumT<Scalar>& bsdf_value,
    const Scalar absolute_incoming_cosine,
    const LightSelectionProbabilityT<Scalar> selection_probability,
    const IncidentLightSampleT<Scalar>& incident_light, const LightSpectrumT<Scalar>& transmittance,
    const Scalar estimator_weight = Scalar{1}) {
    if (!direct_lighting_detail::finite_non_negative(beta) ||
        !direct_lighting_detail::finite_non_negative(bsdf_value)) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct-lighting throughput and BSDF values must be finite and non-negative."));
    }
    if (!direct_lighting_detail::finite_non_negative(transmittance)) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct-lighting transmittance must be finite and lie in [0, 1]."));
    }
    for (const auto value : transmittance.values) {
        if (value > Scalar{1}) {
            return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
                "Direct-lighting transmittance must be finite and lie in [0, 1]."));
        }
    }
    if (!std::isfinite(absolute_incoming_cosine) || absolute_incoming_cosine < Scalar{0} ||
        absolute_incoming_cosine > Scalar{1}) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting BSDF cosine must lie in [0, 1]."));
    }
    if (!std::isfinite(selection_probability.value()) ||
        !(selection_probability.value() > Scalar{0}) || selection_probability.value() > Scalar{1}) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting selection probability must lie in (0, 1]."));
    }
    if (!std::isfinite(estimator_weight) || estimator_weight < Scalar{0} ||
        estimator_weight > Scalar{1}) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting estimator weight must lie in [0, 1]."));
    }

    const auto conditional_probability = incident_light.probability();
    if ((conditional_probability.measure != ProbabilityMeasure::solid_angle &&
         conditional_probability.measure != ProbabilityMeasure::discrete) ||
        !std::isfinite(conditional_probability.value) ||
        !(conditional_probability.value > Scalar{0}) ||
        (conditional_probability.measure == ProbabilityMeasure::discrete &&
         conditional_probability.value > Scalar{1})) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting conditional PDF must be positive and use solid-angle or discrete "
            "measure."));
    }
    if (!direct_lighting_detail::finite_non_negative(incident_light.incident_radiance())) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct incident radiance must be finite and non-negative."));
    }

    auto result = LightSpectrumT<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto contribution =
            estimator_weight == Scalar{1}
                ? direct_lighting_detail::checked_product_quotient(
                      std::array{beta[lane], bsdf_value[lane],
                                 incident_light.incident_radiance()[lane], absolute_incoming_cosine,
                                 transmittance[lane]},
                      std::array{selection_probability.value(), conditional_probability.value})
                : direct_lighting_detail::checked_product_quotient(
                      std::array{beta[lane], bsdf_value[lane],
                                 incident_light.incident_radiance()[lane], absolute_incoming_cosine,
                                 transmittance[lane], estimator_weight},
                      std::array{selection_probability.value(), conditional_probability.value});
        if (!contribution) {
            return std::unexpected(contribution.error());
        }
        result[lane] = *contribution;
    }
    return result;
}

// Evaluates one sampled-light estimator. The light selection probability is
// discrete and the incident sample probability remains conditional on that
// selection; both therefore appear exactly once in the denominator. The
// Lambertian value is an unweighted BRDF, so the +Z local cosine is explicit.
// An optional estimator weight, such as a same-measure MIS weight, appears
// exactly once in the numerator; delta callers leave it at one.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LightSpectrumT<Scalar>> evaluate_lambertian_direct_lighting(
    const LightSpectrumT<Scalar>& beta, const LambertianReflectionT<Scalar>& reflection,
    const OrthonormalFrameT<Scalar>& frame, const Vector3T<Scalar> outgoing_world,
    const LightSelectionProbabilityT<Scalar> selection_probability,
    const IncidentLightSampleT<Scalar>& incident_light, const LightSpectrumT<Scalar>& transmittance,
    const Scalar estimator_weight = Scalar{1}) {
    if (!direct_lighting_detail::finite_non_negative(beta)) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct-lighting throughput must be finite and non-negative."));
    }
    if (!direct_lighting_detail::finite_non_negative(transmittance)) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct-lighting transmittance must be finite and lie in [0, 1]."));
    }
    for (const auto value : transmittance.values) {
        if (value > Scalar{1}) {
            return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
                "Direct-lighting transmittance must be finite and lie in [0, 1]."));
        }
    }
    if (!std::isfinite(selection_probability.value()) ||
        !(selection_probability.value() > Scalar{0}) || selection_probability.value() > Scalar{1}) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting selection probability must lie in (0, 1]."));
    }
    if (!std::isfinite(estimator_weight) || estimator_weight < Scalar{0} ||
        estimator_weight > Scalar{1}) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting estimator weight must lie in [0, 1]."));
    }

    const auto conditional_probability = incident_light.probability();
    if ((conditional_probability.measure != ProbabilityMeasure::solid_angle &&
         conditional_probability.measure != ProbabilityMeasure::discrete) ||
        !std::isfinite(conditional_probability.value) ||
        !(conditional_probability.value > Scalar{0}) ||
        (conditional_probability.measure == ProbabilityMeasure::discrete &&
         conditional_probability.value > Scalar{1})) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "A direct-lighting conditional PDF must be positive and use solid-angle or discrete "
            "measure."));
    }
    if (!direct_lighting_detail::finite_non_negative(incident_light.incident_radiance())) {
        return std::unexpected(direct_lighting_detail::invalid_direct_lighting(
            "Direct incident radiance must be finite and non-negative."));
    }

    const auto outgoing_local = frame.to_local(outgoing_world);
    const auto incoming_local = frame.to_local(incident_light.direction_to_light());
    const auto validated_directions = reflection.eval(outgoing_local, incoming_local);
    if (!validated_directions) {
        return std::unexpected(validated_directions.error());
    }
    if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
        return LightSpectrumT<Scalar>{};
    }

    auto result = LightSpectrumT<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto contribution =
            estimator_weight == Scalar{1}
                ? direct_lighting_detail::checked_product_quotient(
                      std::array{beta[lane], reflection.reflectance()[lane],
                                 std::numbers::inv_pi_v<Scalar>,
                                 incident_light.incident_radiance()[lane], incoming_local.z,
                                 transmittance[lane]},
                      std::array{selection_probability.value(), conditional_probability.value})
                : direct_lighting_detail::checked_product_quotient(
                      std::array{beta[lane], reflection.reflectance()[lane],
                                 std::numbers::inv_pi_v<Scalar>,
                                 incident_light.incident_radiance()[lane], incoming_local.z,
                                 transmittance[lane], estimator_weight},
                      std::array{selection_probability.value(), conditional_probability.value});
        if (!contribution) {
            return std::unexpected(contribution.error());
        }
        result[lane] = *contribution;
    }
    return result;
}

} // namespace blackframe::renderer
