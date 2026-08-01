#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using SpecularDeltaProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct SpecularDeltaSampleT final {
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    SpecularDeltaProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = DeltaBsdfProbabilityMeasure,
    };
    ScatteringLobe lobes{ScatteringLobe::none};
    Scalar eta_scale_multiplier{Scalar{1}};
};

using SpecularDeltaSample = SpecularDeltaSampleT<TransportScalar>;
using ReferenceSpecularDeltaSample = SpecularDeltaSampleT<ReferenceScalar>;

namespace specular_delta_detail {

[[nodiscard]] inline core::Error invalid_specular_delta(const char* const message) {
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
[[nodiscard]] core::Status validate_direction(const Vector3T<Scalar> direction) {
    if (!unit_local_direction(direction)) {
        return std::unexpected(
            invalid_specular_delta("Specular delta directions must be finite unit vectors."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_directions(const Vector3T<Scalar> outgoing_local,
                                               const Vector3T<Scalar> incoming_local) {
    if (!unit_local_direction(outgoing_local) || !unit_local_direction(incoming_local)) {
        return std::unexpected(
            invalid_specular_delta("Specular delta directions must be finite unit vectors."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar>
checked_positive_product_quotient(const Scalar left, const Scalar right, const Scalar denominator,
                                  const char* const message) {
    if (left == Scalar{0}) {
        return Scalar{0};
    }
    if (!std::isfinite(left) || !(left > Scalar{0}) || !std::isfinite(right) ||
        !(right > Scalar{0}) || !std::isfinite(denominator) || !(denominator > Scalar{0})) {
        return std::unexpected(invalid_specular_delta(message));
    }

    auto left_exponent = 0;
    auto right_exponent = 0;
    auto denominator_exponent = 0;
    const auto left_significand = std::frexp(left, &left_exponent);
    const auto right_significand = std::frexp(right, &right_exponent);
    const auto denominator_significand = std::frexp(denominator, &denominator_exponent);
    auto normalization_exponent = 0;
    const auto normalized_significand = std::frexp(
        (left_significand * right_significand) / denominator_significand, &normalization_exponent);
    const auto result =
        std::scalbn(normalized_significand,
                    left_exponent + right_exponent - denominator_exponent + normalization_exponent);
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(invalid_specular_delta(message));
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar>
checked_squared_ratio(const Scalar numerator, const Scalar denominator, const char* const message) {
    if (!std::isfinite(numerator) || !(numerator > Scalar{0}) || !std::isfinite(denominator) ||
        !(denominator > Scalar{0})) {
        return std::unexpected(invalid_specular_delta(message));
    }

    auto numerator_exponent = 0;
    auto denominator_exponent = 0;
    const auto numerator_significand = std::frexp(numerator, &numerator_exponent);
    const auto denominator_significand = std::frexp(denominator, &denominator_exponent);
    const auto ratio_significand = numerator_significand / denominator_significand;
    auto normalization_exponent = 0;
    const auto normalized_significand =
        std::frexp(ratio_significand * ratio_significand, &normalization_exponent);
    const auto result =
        std::scalbn(normalized_significand,
                    2 * (numerator_exponent - denominator_exponent) + normalization_exponent);
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(invalid_specular_delta(message));
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
scaled_delta_value(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& coefficient,
                   const Scalar transport_factor, const Scalar absolute_incoming_cosine) {
    auto result = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto value = checked_positive_product_quotient(
            coefficient[lane], transport_factor, absolute_incoming_cosine,
            "A specular delta value is not representable in the requested precision.");
        if (!value) {
            return std::unexpected(value.error());
        }
        result[lane] = *value;
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr SpecularDeltaProbabilityDensityT<Scalar>
zero_directional_probability() noexcept {
    return {
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr SpecularDeltaProbabilityDensityT<Scalar>
unit_discrete_probability() noexcept {
    return {
        .value = Scalar{1},
        .measure = DeltaBsdfProbabilityMeasure,
    };
}

} // namespace specular_delta_detail

// An ideal two-sided mirror. Directional eval/pdf queries exclude the Dirac atom and return zero;
// sample() is the only operation that exposes its positive discrete mass.
template <SpectrumScalar Scalar> class SpecularReflectionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = SpecularDeltaProbabilityDensityT<Scalar>;
    using sample_type = SpecularDeltaSampleT<Scalar>;

    [[nodiscard]] static core::Result<SpecularReflectionT> create(const spectrum_type reflectance) {
        if (!specular_delta_detail::valid_coefficient(reflectance)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "Specular mirror reflectance requires every spectral lane to be finite and in "
                "[0, 1]."));
        }
        return SpecularReflectionT{reflectance};
    }

    [[nodiscard]] constexpr const spectrum_type& reflectance() const noexcept {
        return reflectance_;
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local) const {
        const auto status =
            specular_delta_detail::validate_directions(outgoing_local, incoming_local);
        if (!status) {
            return std::unexpected(status.error());
        }
        return spectrum_type{};
    }

    [[nodiscard]] core::Result<probability_density_type>
    pdf(const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local) const {
        const auto status =
            specular_delta_detail::validate_directions(outgoing_local, incoming_local);
        if (!status) {
            return std::unexpected(status.error());
        }
        return specular_delta_detail::zero_directional_probability<Scalar>();
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local) const {
        const auto status = specular_delta_detail::validate_direction(outgoing_local);
        if (!status) {
            return std::unexpected(status.error());
        }
        if (outgoing_local.z == Scalar{0}) {
            return std::optional<sample_type>{};
        }

        const auto incoming_local = Vector3T<Scalar>{
            .x = -outgoing_local.x,
            .y = -outgoing_local.y,
            .z = outgoing_local.z,
        };
        const auto value = specular_delta_detail::scaled_delta_value(reflectance_, Scalar{1},
                                                                     std::abs(incoming_local.z));
        if (!value) {
            return std::unexpected(value.error());
        }
        return std::optional<sample_type>{sample_type{
            .incoming_local = incoming_local,
            .value = *value,
            .probability = specular_delta_detail::unit_discrete_probability<Scalar>(),
            .lobes = ScatteringLobe::specular | ScatteringLobe::reflection,
            .eta_scale_multiplier = Scalar{1},
        }};
    }

  private:
    constexpr explicit SpecularReflectionT(const spectrum_type reflectance) noexcept
        : reflectance_{reflectance} {}

    spectrum_type reflectance_;
};

using SpecularReflection = SpecularReflectionT<TransportScalar>;
using ReferenceSpecularReflection = SpecularReflectionT<ReferenceScalar>;

// An ideal two-sided transmission lobe with an achromatic interface IOR. It never turns total
// internal reflection into another lobe: TIR has no transmission support and returns an empty
// sample. Radiance samples expose the reciprocal etaScale multiplier needed by path roulette.
template <SpectrumScalar Scalar> class SpecularTransmissionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = SpecularDeltaProbabilityDensityT<Scalar>;
    using sample_type = SpecularDeltaSampleT<Scalar>;

    [[nodiscard]] static core::Result<SpecularTransmissionT>
    create(const spectrum_type transmittance, const Scalar exterior_eta,
           const Scalar interior_eta) {
        if (!specular_delta_detail::valid_coefficient(transmittance)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "Specular transmittance requires every spectral lane to be finite and in [0, "
                "1]."));
        }
        if (!std::isfinite(exterior_eta) || !(exterior_eta > Scalar{0}) ||
            !std::isfinite(interior_eta) || !(interior_eta > Scalar{0})) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "Specular transmission indices must be finite and strictly positive."));
        }
        return SpecularTransmissionT{transmittance, exterior_eta, interior_eta};
    }

    [[nodiscard]] constexpr const spectrum_type& transmittance() const noexcept {
        return transmittance_;
    }

    [[nodiscard]] constexpr Scalar exterior_eta() const noexcept {
        return exterior_eta_;
    }

    [[nodiscard]] constexpr Scalar interior_eta() const noexcept {
        return interior_eta_;
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local,
                                                   const TransportMode mode) const {
        const auto status = validate_query(outgoing_local, incoming_local, mode);
        if (!status) {
            return std::unexpected(status.error());
        }
        return spectrum_type{};
    }

    [[nodiscard]] core::Result<probability_density_type> pdf(const Vector3T<Scalar> outgoing_local,
                                                             const Vector3T<Scalar> incoming_local,
                                                             const TransportMode mode) const {
        const auto status = validate_query(outgoing_local, incoming_local, mode);
        if (!status) {
            return std::unexpected(status.error());
        }
        return specular_delta_detail::zero_directional_probability<Scalar>();
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const TransportMode mode) const {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "Specular transmission requires a supported transport mode."));
        }
        const auto direction_status = specular_delta_detail::validate_direction(outgoing_local);
        if (!direction_status) {
            return std::unexpected(direction_status.error());
        }
        if (outgoing_local.z == Scalar{0}) {
            return std::optional<sample_type>{};
        }

        const auto entering = outgoing_local.z > Scalar{0};
        const auto incident_eta = entering ? exterior_eta_ : interior_eta_;
        const auto transmitted_eta = entering ? interior_eta_ : exterior_eta_;
        if (incident_eta == transmitted_eta) {
            const auto incoming_local = Vector3T<Scalar>{
                .x = -outgoing_local.x,
                .y = -outgoing_local.y,
                .z = -outgoing_local.z,
            };
            const auto value = specular_delta_detail::scaled_delta_value(
                transmittance_, Scalar{1}, std::abs(incoming_local.z));
            if (!value) {
                return std::unexpected(value.error());
            }
            return std::optional<sample_type>{sample_type{
                .incoming_local = incoming_local,
                .value = *value,
                .probability = specular_delta_detail::unit_discrete_probability<Scalar>(),
                .lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
                .eta_scale_multiplier = Scalar{1},
            }};
        }
        const auto raw_incident_sine = std::hypot(outgoing_local.x, outgoing_local.y);
        if (!std::isfinite(raw_incident_sine)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The specular transmission incident sine is not representable."));
        }
        auto incident_sine = raw_incident_sine;
        auto tangent_normalization = Scalar{1};
        if (raw_incident_sine > Scalar{1}) {
            // A direction accepted by the unit-vector tolerance can cross one after an
            // orthonormal-frame dot product. Reconstruct only its tangent magnitude from z while
            // retaining the measured azimuth; larger inconsistencies remain errors.
            const auto absolute_incident_cosine = std::abs(outgoing_local.z);
            if (absolute_incident_cosine > Scalar{1}) {
                return std::unexpected(specular_delta_detail::invalid_specular_delta(
                    "The specular transmission incident direction is inconsistent."));
            }
            incident_sine = std::sqrt((Scalar{1} - absolute_incident_cosine) *
                                      (Scalar{1} + absolute_incident_cosine));
            tangent_normalization = incident_sine / raw_incident_sine;
            if (!std::isfinite(incident_sine) || !std::isfinite(tangent_normalization) ||
                !(tangent_normalization > Scalar{0})) {
                return std::unexpected(specular_delta_detail::invalid_specular_delta(
                    "The specular transmission incident sine is not representable."));
            }
        }

        // Equality is the critical tangent, which has zero BSDF support under the engine's
        // direction convention. Comparing against eta_t / eta_i avoids overflow in eta_i*sin_i.
        if (incident_sine > Scalar{0}) {
            const auto critical_sine = transmitted_eta / incident_eta;
            if (critical_sine <= Scalar{1} && incident_sine >= critical_sine) {
                return std::optional<sample_type>{};
            }
        }

        const auto incident_over_transmitted_eta = incident_eta / transmitted_eta;
        if (!std::isfinite(incident_over_transmitted_eta) ||
            !(incident_over_transmitted_eta > Scalar{0})) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The specular transmission relative index is not representable."));
        }
        const auto incoming_x =
            -incident_over_transmitted_eta * tangent_normalization * outgoing_local.x;
        const auto incoming_y =
            -incident_over_transmitted_eta * tangent_normalization * outgoing_local.y;
        if (!std::isfinite(incoming_x) || !std::isfinite(incoming_y) ||
            (outgoing_local.x != Scalar{0} && incoming_x == Scalar{0}) ||
            (outgoing_local.y != Scalar{0} && incoming_y == Scalar{0})) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The specular transmission tangent is not representable."));
        }
        const auto transmitted_sine = std::hypot(incoming_x, incoming_y);
        if (!std::isfinite(transmitted_sine)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The specular transmission sine is not representable."));
        }
        if (transmitted_sine >= Scalar{1}) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "A subcritical specular transmission direction is not representable."));
        }
        const auto transmitted_cosine =
            std::sqrt((Scalar{1} - transmitted_sine) * (Scalar{1} + transmitted_sine));
        if (!std::isfinite(transmitted_cosine) || !(transmitted_cosine > Scalar{0})) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The specular transmission cosine is not representable."));
        }
        const auto incoming_local = Vector3T<Scalar>{
            .x = incoming_x,
            .y = incoming_y,
            .z = entering ? -transmitted_cosine : transmitted_cosine,
        };
        if (!specular_delta_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "The refracted specular direction is not a representable unit vector."));
        }

        auto transport_factor = Scalar{1};
        auto eta_scale_multiplier = Scalar{1};
        if (mode == TransportMode::radiance) {
            const auto adjoint = specular_delta_detail::checked_squared_ratio(
                incident_eta, transmitted_eta,
                "The specular transmission radiance factor is not representable.");
            if (!adjoint) {
                return std::unexpected(adjoint.error());
            }
            const auto eta_scale = specular_delta_detail::checked_squared_ratio(
                transmitted_eta, incident_eta,
                "The specular transmission etaScale multiplier is not representable.");
            if (!eta_scale) {
                return std::unexpected(eta_scale.error());
            }
            transport_factor = *adjoint;
            eta_scale_multiplier = *eta_scale;
        }

        const auto value = specular_delta_detail::scaled_delta_value(
            transmittance_, transport_factor, transmitted_cosine);
        if (!value) {
            return std::unexpected(value.error());
        }
        return std::optional<sample_type>{sample_type{
            .incoming_local = incoming_local,
            .value = *value,
            .probability = specular_delta_detail::unit_discrete_probability<Scalar>(),
            .lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
            .eta_scale_multiplier = eta_scale_multiplier,
        }};
    }

  private:
    constexpr SpecularTransmissionT(const spectrum_type transmittance, const Scalar exterior_eta,
                                    const Scalar interior_eta) noexcept
        : transmittance_{transmittance}, exterior_eta_{exterior_eta}, interior_eta_{interior_eta} {}

    [[nodiscard]] static core::Status validate_query(const Vector3T<Scalar> outgoing_local,
                                                     const Vector3T<Scalar> incoming_local,
                                                     const TransportMode mode) {
        if (!is_known_transport_mode(mode)) {
            return std::unexpected(specular_delta_detail::invalid_specular_delta(
                "Specular transmission requires a supported transport mode."));
        }
        return specular_delta_detail::validate_directions(outgoing_local, incoming_local);
    }

    spectrum_type transmittance_;
    Scalar exterior_eta_;
    Scalar interior_eta_;
};

using SpecularTransmission = SpecularTransmissionT<TransportScalar>;
using ReferenceSpecularTransmission = SpecularTransmissionT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<SpecularDeltaSample>);
static_assert(std::is_trivially_copyable_v<SpecularDeltaSample>);
static_assert(std::is_standard_layout_v<ReferenceSpecularDeltaSample>);
static_assert(std::is_trivially_copyable_v<ReferenceSpecularDeltaSample>);
static_assert(std::is_standard_layout_v<SpecularReflection>);
static_assert(std::is_trivially_copyable_v<SpecularReflection>);
static_assert(std::is_standard_layout_v<ReferenceSpecularReflection>);
static_assert(std::is_trivially_copyable_v<ReferenceSpecularReflection>);
static_assert(std::is_standard_layout_v<SpecularTransmission>);
static_assert(std::is_trivially_copyable_v<SpecularTransmission>);
static_assert(std::is_standard_layout_v<ReferenceSpecularTransmission>);
static_assert(std::is_trivially_copyable_v<ReferenceSpecularTransmission>);

} // namespace blackframe::renderer
