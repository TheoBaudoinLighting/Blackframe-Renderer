#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Fresnel.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using RoughDielectricProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct RoughDielectricSampleT final {
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    RoughDielectricProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
    ScatteringLobe lobes{ScatteringLobe::none};
    Scalar eta_scale_multiplier{Scalar{1}};
};

using RoughDielectricSample = RoughDielectricSampleT<TransportScalar>;
using ReferenceRoughDielectricSample = RoughDielectricSampleT<ReferenceScalar>;

namespace rough_dielectric_detail {

[[nodiscard]] inline core::Error invalid_rough_dielectric(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_coefficient(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& coefficient) noexcept {
    for (const auto value : coefficient.values) {
        if (!std::isfinite(value) || value < Scalar{0} || value > Scalar{1}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool unit_local_direction(const Vector3T<Scalar> direction) noexcept {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return false;
    }
    const auto squared_length = std::fma(
        direction.x, direction.x, std::fma(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool canonical_event_sample(const Scalar sample) noexcept {
    return std::isfinite(sample) && sample >= Scalar{0} && sample < Scalar{1};
}

template <SpectrumScalar Scalar, std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] core::Result<Scalar>
checked_product_ratio(const std::array<Scalar, NumeratorCount>& numerators,
                      const std::array<Scalar, DenominatorCount>& denominators,
                      const char* const message) {
    auto has_zero = false;
    for (const auto value : numerators) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return std::unexpected(invalid_rough_dielectric(message));
        }
        has_zero = has_zero || value == Scalar{0};
    }
    for (const auto value : denominators) {
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return std::unexpected(invalid_rough_dielectric(message));
        }
    }
    if (has_zero) {
        return Scalar{0};
    }

    auto significand = Scalar{1};
    auto exponent = 0;
    for (const auto value : numerators) {
        auto value_exponent = 0;
        const auto value_significand = std::frexp(value, &value_exponent);
        auto normalization_exponent = 0;
        significand = std::frexp(significand * value_significand, &normalization_exponent);
        exponent += value_exponent + normalization_exponent;
    }
    for (const auto value : denominators) {
        auto value_exponent = 0;
        const auto value_significand = std::frexp(value, &value_exponent);
        auto normalization_exponent = 0;
        significand = std::frexp(significand / value_significand, &normalization_exponent);
        exponent += normalization_exponent - value_exponent;
    }
    const auto result = std::scalbn(significand, exponent);
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(invalid_rough_dielectric(message));
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
scale_spectrum(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& coefficient,
               const Scalar scale) {
    auto result = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto value =
            checked_product_ratio(std::array{coefficient[lane], scale}, std::array{Scalar{1}},
                                  "A rough-dielectric spectral BSDF value is not representable.");
        if (!value) {
            return std::unexpected(value.error());
        }
        result[lane] = *value;
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> scaled_direction(const Vector3T<Scalar> direction,
                                                          const Scalar scale) noexcept {
    return {
        .x = scale * direction.x,
        .y = scale * direction.y,
        .z = scale * direction.z,
    };
}

template <SpectrumScalar Scalar> struct InterfaceT final {
    Scalar face_sign{};
    Scalar incident_eta{};
    Scalar transmitted_eta{};
    Scalar normalized_incident_eta{};
    Scalar normalized_transmitted_eta{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] InterfaceT<Scalar> interface_for(const Vector3T<Scalar> outgoing_local,
                                               const Scalar exterior_eta,
                                               const Scalar interior_eta) noexcept {
    const auto face_sign = outgoing_local.z > Scalar{0} ? Scalar{1} : Scalar{-1};
    const auto incident_eta = face_sign > Scalar{0} ? exterior_eta : interior_eta;
    const auto transmitted_eta = face_sign > Scalar{0} ? interior_eta : exterior_eta;
    const auto maximum_eta = std::max(incident_eta, transmitted_eta);
    return {
        .face_sign = face_sign,
        .incident_eta = incident_eta,
        .transmitted_eta = transmitted_eta,
        .normalized_incident_eta = incident_eta / maximum_eta,
        .normalized_transmitted_eta = transmitted_eta / maximum_eta,
    };
}

template <SpectrumScalar Scalar> struct ReflectionGeometryT final {
    Normal3T<Scalar> microfacet_normal{};
    Scalar outgoing_dot_microfacet{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ReflectionGeometryT<Scalar>>
reflection_geometry(const Vector3T<Scalar> outgoing_face, const Vector3T<Scalar> incoming_face) {
    const auto half_vector = Vector3T<Scalar>{
        .x = outgoing_face.x + incoming_face.x,
        .y = outgoing_face.y + incoming_face.y,
        .z = outgoing_face.z + incoming_face.z,
    };
    const auto length = std::hypot(std::hypot(half_vector.x, half_vector.y), half_vector.z);
    if (!std::isfinite(length) || !(length > Scalar{0})) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric reflection half-vector is not normalizable."));
    }
    const auto microfacet_normal = Normal3T<Scalar>{
        .x = half_vector.x / length,
        .y = half_vector.y / length,
        .z = half_vector.z / length,
    };
    if (!unit_local_direction(Vector3T<Scalar>{
            .x = microfacet_normal.x, .y = microfacet_normal.y, .z = microfacet_normal.z}) ||
        !(microfacet_normal.z > Scalar{0})) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric reflection half-vector is not representable."));
    }
    const auto outgoing_dot_microfacet = Scalar{0.5} * length;
    if (!std::isfinite(outgoing_dot_microfacet) || !(outgoing_dot_microfacet > Scalar{0}) ||
        outgoing_dot_microfacet > Scalar{1}) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric reflection half-angle cosine is outside [0, 1]."));
    }
    return ReflectionGeometryT<Scalar>{
        .microfacet_normal = microfacet_normal,
        .outgoing_dot_microfacet = outgoing_dot_microfacet,
    };
}

template <SpectrumScalar Scalar> struct TransmissionGeometryT final {
    Normal3T<Scalar> microfacet_normal{};
    Scalar outgoing_dot_microfacet{};
    Scalar incoming_dot_microfacet{};
    Scalar normalized_half_vector_length{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<std::optional<TransmissionGeometryT<Scalar>>>
transmission_geometry(const Vector3T<Scalar> outgoing_face, const Vector3T<Scalar> incoming_face,
                      const InterfaceT<Scalar> interface) {
    auto half_vector = Vector3T<Scalar>{
        .x = std::fma(interface.normalized_incident_eta, outgoing_face.x,
                      interface.normalized_transmitted_eta * incoming_face.x),
        .y = std::fma(interface.normalized_incident_eta, outgoing_face.y,
                      interface.normalized_transmitted_eta * incoming_face.y),
        .z = std::fma(interface.normalized_incident_eta, outgoing_face.z,
                      interface.normalized_transmitted_eta * incoming_face.z),
    };
    const auto length = std::hypot(std::hypot(half_vector.x, half_vector.y), half_vector.z);
    if (!std::isfinite(length) || !(length > Scalar{0})) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric transmission half-vector is not normalizable."));
    }
    half_vector = scaled_direction(half_vector, Scalar{1} / length);
    if (half_vector.z < Scalar{0}) {
        half_vector = scaled_direction(half_vector, Scalar{-1});
    }
    if (half_vector.z == Scalar{0}) {
        return std::optional<TransmissionGeometryT<Scalar>>{};
    }
    const auto microfacet_normal = Normal3T<Scalar>{
        .x = half_vector.x,
        .y = half_vector.y,
        .z = half_vector.z,
    };
    if (!unit_local_direction(half_vector)) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric transmission half-vector is not representable."));
    }
    const auto outgoing_dot_microfacet = dot(outgoing_face, microfacet_normal);
    const auto incoming_dot_microfacet = dot(incoming_face, microfacet_normal);
    if (!std::isfinite(outgoing_dot_microfacet) || !std::isfinite(incoming_dot_microfacet)) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric transmission half-angle cosines are not representable."));
    }
    if (!(outgoing_dot_microfacet > Scalar{0}) || !(incoming_dot_microfacet < Scalar{0})) {
        return std::optional<TransmissionGeometryT<Scalar>>{};
    }
    if (outgoing_dot_microfacet > Scalar{1} || incoming_dot_microfacet < Scalar{-1}) {
        return std::unexpected(invalid_rough_dielectric(
            "The rough-dielectric transmission half-angle cosines are outside [-1, 1]."));
    }
    return std::optional<TransmissionGeometryT<Scalar>>{TransmissionGeometryT<Scalar>{
        .microfacet_normal = microfacet_normal,
        .outgoing_dot_microfacet = outgoing_dot_microfacet,
        .incoming_dot_microfacet = incoming_dot_microfacet,
        .normalized_half_vector_length = length,
    }};
}

} // namespace rough_dielectric_detail

// Anisotropic single-scattering GGX dielectric interface. Directions point away from the surface;
// the exterior occupies +Z and the interior -Z in the caller's local closure frame. AlphaX and
// alphaY are the mathematical slope widths along the frame's tangent and bitangent; rotating those
// axes belongs to the caller's frame construction. The same spectral coefficient scales reflection
// and transmission while exact achromatic Fresnel selects their physical split. Equal indices are
// rejected because their straight-through support is a Dirac event owned by SpecularTransmission,
// not this continuous closure.
template <SpectrumScalar Scalar> class RoughDielectricT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = RoughDielectricProbabilityDensityT<Scalar>;
    using sample_type = RoughDielectricSampleT<Scalar>;

    [[nodiscard]] static core::Result<RoughDielectricT> create(const spectrum_type coefficient,
                                                               const Scalar exterior_eta,
                                                               const Scalar interior_eta,
                                                               const Scalar alpha) {
        return create(coefficient, exterior_eta, interior_eta, alpha, alpha);
    }

    [[nodiscard]] static core::Result<RoughDielectricT>
    create(const spectrum_type coefficient, const Scalar exterior_eta, const Scalar interior_eta,
           const Scalar alpha_x, const Scalar alpha_y) {
        if (!rough_dielectric_detail::valid_coefficient(coefficient)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric coefficients require every spectral lane to be finite and in "
                "[0, 1]."));
        }
        if (!std::isfinite(exterior_eta) || !(exterior_eta > Scalar{0}) ||
            !std::isfinite(interior_eta) || !(interior_eta > Scalar{0})) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric indices must be finite and strictly positive."));
        }
        if (exterior_eta == interior_eta) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Equal dielectric indices have delta transmission support, not rough support."));
        }
        const auto microfacet = GgxMicrofacetT<Scalar>::create(alpha_x, alpha_y);
        if (!microfacet) {
            return std::unexpected(microfacet.error());
        }
        return RoughDielectricT{coefficient, exterior_eta, interior_eta, *microfacet};
    }

    [[nodiscard]] constexpr const spectrum_type& coefficient() const noexcept {
        return coefficient_;
    }

    [[nodiscard]] constexpr Scalar exterior_eta() const noexcept {
        return exterior_eta_;
    }

    [[nodiscard]] constexpr Scalar interior_eta() const noexcept {
        return interior_eta_;
    }

    [[nodiscard]] constexpr Scalar alpha() const noexcept {
        return microfacet_.alpha();
    }

    [[nodiscard]] constexpr Scalar alpha_x() const noexcept {
        return microfacet_.alpha_x();
    }

    [[nodiscard]] constexpr Scalar alpha_y() const noexcept {
        return microfacet_.alpha_y();
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local,
                                                   const TransportMode mode) const {
        const auto status = validate_query(outgoing_local, incoming_local, mode);
        if (!status) {
            return std::unexpected(status.error());
        }
        auto result = spectrum_type{};
        if (outgoing_local.z == Scalar{0} || incoming_local.z == Scalar{0}) {
            return result;
        }

        const auto interface =
            rough_dielectric_detail::interface_for(outgoing_local, exterior_eta_, interior_eta_);
        const auto outgoing_face =
            rough_dielectric_detail::scaled_direction(outgoing_local, interface.face_sign);
        const auto incoming_face =
            rough_dielectric_detail::scaled_direction(incoming_local, interface.face_sign);
        const auto reflection = incoming_face.z > Scalar{0};
        if (reflection) {
            const auto geometry =
                rough_dielectric_detail::reflection_geometry(outgoing_face, incoming_face);
            if (!geometry) {
                return std::unexpected(geometry.error());
            }
            const auto fresnel =
                dielectric_fresnel(geometry->outgoing_dot_microfacet, interface.incident_eta,
                                   interface.transmitted_eta);
            const auto distribution = microfacet_.normal_distribution(geometry->microfacet_normal);
            const auto masking = microfacet_.smith_g2(outgoing_face, incoming_face);
            if (!fresnel) {
                return std::unexpected(fresnel.error());
            }
            if (!distribution) {
                return std::unexpected(distribution.error());
            }
            if (!masking) {
                return std::unexpected(masking.error());
            }
            const auto scale = rough_dielectric_detail::checked_product_ratio(
                std::array{*fresnel, *distribution, *masking},
                std::array{Scalar{4}, outgoing_face.z, incoming_face.z},
                "The rough-dielectric GGX reflection value is not representable.");
            if (!scale) {
                return std::unexpected(scale.error());
            }
            return rough_dielectric_detail::scale_spectrum(coefficient_, *scale);
        }

        const auto geometry =
            rough_dielectric_detail::transmission_geometry(outgoing_face, incoming_face, interface);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        if (!geometry->has_value()) {
            return result;
        }
        const auto fresnel = dielectric_fresnel((**geometry).outgoing_dot_microfacet,
                                                interface.incident_eta, interface.transmitted_eta);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }
        if (*fresnel == Scalar{1}) {
            return result;
        }
        if (!((**geometry).normalized_half_vector_length > Scalar{0})) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "The rough-dielectric transmission Jacobian denominator is zero."));
        }
        const auto distribution = microfacet_.normal_distribution((**geometry).microfacet_normal);
        const auto incoming_mask_direction =
            rough_dielectric_detail::scaled_direction(incoming_face, Scalar{-1});
        const auto masking = microfacet_.smith_g2(outgoing_face, incoming_mask_direction);
        if (!distribution) {
            return std::unexpected(distribution.error());
        }
        if (!masking) {
            return std::unexpected(masking.error());
        }
        const auto transport_eta = mode == TransportMode::radiance
                                       ? interface.normalized_incident_eta
                                       : interface.normalized_transmitted_eta;
        const auto scale = rough_dielectric_detail::checked_product_ratio(
            std::array{Scalar{1} - *fresnel, *distribution, *masking, transport_eta, transport_eta,
                       (**geometry).outgoing_dot_microfacet, -(**geometry).incoming_dot_microfacet},
            std::array{outgoing_face.z, -incoming_face.z,
                       (**geometry).normalized_half_vector_length,
                       (**geometry).normalized_half_vector_length},
            "The rough-dielectric GGX transmission value is not representable.");
        if (!scale) {
            return std::unexpected(scale.error());
        }
        return rough_dielectric_detail::scale_spectrum(coefficient_, *scale);
    }

    [[nodiscard]] core::Result<probability_density_type> pdf(const Vector3T<Scalar> outgoing_local,
                                                             const Vector3T<Scalar> incoming_local,
                                                             const TransportMode mode) const {
        const auto status = validate_query(outgoing_local, incoming_local, mode);
        if (!status) {
            return std::unexpected(status.error());
        }
        auto result = probability_density_type{
            .value = Scalar{0},
            .measure = ContinuousBsdfProbabilityMeasure,
        };
        if (outgoing_local.z == Scalar{0} || incoming_local.z == Scalar{0}) {
            return result;
        }

        const auto interface =
            rough_dielectric_detail::interface_for(outgoing_local, exterior_eta_, interior_eta_);
        const auto outgoing_face =
            rough_dielectric_detail::scaled_direction(outgoing_local, interface.face_sign);
        const auto incoming_face =
            rough_dielectric_detail::scaled_direction(incoming_local, interface.face_sign);
        if (incoming_face.z > Scalar{0}) {
            const auto geometry =
                rough_dielectric_detail::reflection_geometry(outgoing_face, incoming_face);
            if (!geometry) {
                return std::unexpected(geometry.error());
            }
            const auto fresnel =
                dielectric_fresnel(geometry->outgoing_dot_microfacet, interface.incident_eta,
                                   interface.transmitted_eta);
            const auto visible_probability =
                microfacet_.visible_normal_pdf(outgoing_face, geometry->microfacet_normal);
            if (!fresnel) {
                return std::unexpected(fresnel.error());
            }
            if (!visible_probability) {
                return std::unexpected(visible_probability.error());
            }
            if (visible_probability->measure != ProbabilityMeasure::solid_angle) {
                return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                    "The GGX visible-normal PDF uses an incompatible measure."));
            }
            const auto probability = rough_dielectric_detail::checked_product_ratio(
                std::array{*fresnel, visible_probability->value},
                std::array{Scalar{4}, geometry->outgoing_dot_microfacet},
                "The rough-dielectric reflected-direction PDF is not representable.");
            if (!probability) {
                return std::unexpected(probability.error());
            }
            result.value = *probability;
            return result;
        }

        const auto geometry =
            rough_dielectric_detail::transmission_geometry(outgoing_face, incoming_face, interface);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        if (!geometry->has_value()) {
            return result;
        }
        const auto fresnel = dielectric_fresnel((**geometry).outgoing_dot_microfacet,
                                                interface.incident_eta, interface.transmitted_eta);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }
        if (*fresnel == Scalar{1}) {
            return result;
        }
        if (!((**geometry).normalized_half_vector_length > Scalar{0})) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "The rough-dielectric transmission Jacobian denominator is zero."));
        }
        const auto visible_probability =
            microfacet_.visible_normal_pdf(outgoing_face, (**geometry).microfacet_normal);
        if (!visible_probability) {
            return std::unexpected(visible_probability.error());
        }
        if (visible_probability->measure != ProbabilityMeasure::solid_angle) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "The GGX visible-normal PDF uses an incompatible measure."));
        }
        const auto probability = rough_dielectric_detail::checked_product_ratio(
            std::array{Scalar{1} - *fresnel, visible_probability->value,
                       interface.normalized_transmitted_eta, interface.normalized_transmitted_eta,
                       -(**geometry).incoming_dot_microfacet},
            std::array{(**geometry).normalized_half_vector_length,
                       (**geometry).normalized_half_vector_length},
            "The rough-dielectric transmitted-direction PDF is not representable.");
        if (!probability) {
            return std::unexpected(probability.error());
        }
        result.value = *probability;
        return result;
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Scalar event_sample,
           const Point2T<Scalar> visible_normal_sample, const TransportMode mode) const {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric sampling requires a supported transport mode."));
        }
        if (!rough_dielectric_detail::unit_local_direction(outgoing_local)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric directions must be finite unit vectors."));
        }
        if (!rough_dielectric_detail::canonical_event_sample(event_sample)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric event selection requires a finite sample in [0, 1)."));
        }
        if (outgoing_local.z == Scalar{0}) {
            return std::optional<sample_type>{};
        }

        const auto interface =
            rough_dielectric_detail::interface_for(outgoing_local, exterior_eta_, interior_eta_);
        const auto outgoing_face =
            rough_dielectric_detail::scaled_direction(outgoing_local, interface.face_sign);
        const auto sampled_normal =
            microfacet_.sample_visible_normal(outgoing_face, visible_normal_sample);
        if (!sampled_normal) {
            return std::unexpected(sampled_normal.error());
        }
        if (!sampled_normal->has_value()) {
            return std::optional<sample_type>{};
        }
        const auto outgoing_dot_microfacet =
            dot(outgoing_face, (**sampled_normal).microfacet_normal);
        if (!std::isfinite(outgoing_dot_microfacet) || !(outgoing_dot_microfacet > Scalar{0}) ||
            outgoing_dot_microfacet > Scalar{1}) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "The sampled rough-dielectric half-angle cosine is outside [0, 1]."));
        }
        const auto fresnel = dielectric_fresnel(outgoing_dot_microfacet, interface.incident_eta,
                                                interface.transmitted_eta);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }

        auto incoming_face = Vector3T<Scalar>{};
        auto lobes = ScatteringLobe::none;
        auto eta_scale_multiplier = Scalar{1};
        if (event_sample < *fresnel) {
            incoming_face = Vector3T<Scalar>{
                .x = Scalar{2} * outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.x -
                     outgoing_face.x,
                .y = Scalar{2} * outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.y -
                     outgoing_face.y,
                .z = Scalar{2} * outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.z -
                     outgoing_face.z,
            };
            lobes = ScatteringLobe::glossy | ScatteringLobe::reflection;
            if (!rough_dielectric_detail::unit_local_direction(incoming_face)) {
                return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                    "The sampled rough-dielectric reflection direction is not representable."));
            }
            if (!(incoming_face.z > Scalar{0})) {
                return std::optional<sample_type>{};
            }
        } else {
            const auto tangent = Vector3T<Scalar>{
                .x = outgoing_face.x -
                     outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.x,
                .y = outgoing_face.y -
                     outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.y,
                .z = outgoing_face.z -
                     outgoing_dot_microfacet * (**sampled_normal).microfacet_normal.z,
            };
            const auto incident_sine = std::hypot(std::hypot(tangent.x, tangent.y), tangent.z);
            if (!std::isfinite(incident_sine) || incident_sine > Scalar{1}) {
                return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                    "The sampled rough-dielectric incident sine is not representable."));
            }

            auto incoming_cosine = Scalar{1};
            auto tangent_scale = Scalar{0};
            if (incident_sine > Scalar{0}) {
                auto transmitted_sine = Scalar{};
                if (interface.incident_eta > interface.transmitted_eta) {
                    const auto critical_sine = interface.transmitted_eta / interface.incident_eta;
                    if (critical_sine == Scalar{0} || incident_sine >= critical_sine) {
                        return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                            "Fresnel selected transmission at total internal reflection."));
                    }
                    transmitted_sine = incident_sine / critical_sine;
                } else {
                    const auto eta_ratio = interface.incident_eta / interface.transmitted_eta;
                    if (eta_ratio == Scalar{0}) {
                        return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                            "The rough-dielectric refraction ratio is not representable."));
                    }
                    transmitted_sine = eta_ratio * incident_sine;
                }
                if (!std::isfinite(transmitted_sine) || !(transmitted_sine < Scalar{1})) {
                    return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                        "A subcritical rough-dielectric transmission sine is not representable."));
                }
                incoming_cosine =
                    std::sqrt((Scalar{1} - transmitted_sine) * (Scalar{1} + transmitted_sine));
                tangent_scale = transmitted_sine / incident_sine;
                if (!std::isfinite(incoming_cosine) || !(incoming_cosine > Scalar{0}) ||
                    !std::isfinite(tangent_scale) || !(tangent_scale > Scalar{0})) {
                    return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                        "The sampled rough-dielectric transmission direction is not "
                        "representable."));
                }
            }
            incoming_face = Vector3T<Scalar>{
                .x = std::fma(-tangent_scale, tangent.x,
                              -incoming_cosine * (**sampled_normal).microfacet_normal.x),
                .y = std::fma(-tangent_scale, tangent.y,
                              -incoming_cosine * (**sampled_normal).microfacet_normal.y),
                .z = std::fma(-tangent_scale, tangent.z,
                              -incoming_cosine * (**sampled_normal).microfacet_normal.z),
            };
            lobes = ScatteringLobe::glossy | ScatteringLobe::transmission;
            if (!rough_dielectric_detail::unit_local_direction(incoming_face)) {
                return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                    "The sampled rough-dielectric transmission direction is not a representable "
                    "unit vector."));
            }
            if (!(incoming_face.z < Scalar{0})) {
                return std::optional<sample_type>{};
            }
            if (mode == TransportMode::radiance) {
                const auto eta_scale = rough_dielectric_detail::checked_product_ratio(
                    std::array{interface.normalized_transmitted_eta,
                               interface.normalized_transmitted_eta},
                    std::array{interface.normalized_incident_eta,
                               interface.normalized_incident_eta},
                    "The rough-dielectric etaScale multiplier is not representable.");
                if (!eta_scale) {
                    return std::unexpected(eta_scale.error());
                }
                eta_scale_multiplier = *eta_scale;
            }
        }

        const auto incoming_local =
            rough_dielectric_detail::scaled_direction(incoming_face, interface.face_sign);
        const auto value = eval(outgoing_local, incoming_local, mode);
        if (!value) {
            return std::unexpected(value.error());
        }
        const auto probability = pdf(outgoing_local, incoming_local, mode);
        if (!probability) {
            return std::unexpected(probability.error());
        }
        return std::optional<sample_type>{sample_type{
            .incoming_local = incoming_local,
            .value = *value,
            .probability = *probability,
            .lobes = lobes,
            .eta_scale_multiplier = eta_scale_multiplier,
        }};
    }

  private:
    constexpr RoughDielectricT(const spectrum_type coefficient, const Scalar exterior_eta,
                               const Scalar interior_eta,
                               const GgxMicrofacetT<Scalar> microfacet) noexcept
        : coefficient_{coefficient}, exterior_eta_{exterior_eta}, interior_eta_{interior_eta},
          microfacet_{microfacet} {}

    [[nodiscard]] static core::Status validate_query(const Vector3T<Scalar> outgoing_local,
                                                     const Vector3T<Scalar> incoming_local,
                                                     const TransportMode mode) {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric queries require a supported transport mode."));
        }
        if (!rough_dielectric_detail::unit_local_direction(outgoing_local) ||
            !rough_dielectric_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(rough_dielectric_detail::invalid_rough_dielectric(
                "Rough-dielectric directions must be finite unit vectors."));
        }
        return {};
    }

    spectrum_type coefficient_;
    Scalar exterior_eta_{};
    Scalar interior_eta_{};
    GgxMicrofacetT<Scalar> microfacet_;
};

using RoughDielectric = RoughDielectricT<TransportScalar>;
using ReferenceRoughDielectric = RoughDielectricT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<RoughDielectricSample>);
static_assert(std::is_trivially_copyable_v<RoughDielectricSample>);
static_assert(std::is_standard_layout_v<ReferenceRoughDielectricSample>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDielectricSample>);
static_assert(std::is_standard_layout_v<RoughDielectric>);
static_assert(std::is_trivially_copyable_v<RoughDielectric>);
static_assert(std::is_standard_layout_v<ReferenceRoughDielectric>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDielectric>);

} // namespace blackframe::renderer
