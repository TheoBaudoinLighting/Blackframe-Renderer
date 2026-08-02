#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Fresnel.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/GgxMicrofacet.hpp>
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
using RoughConductorProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct RoughConductorSampleT final {
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    RoughConductorProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
};

using RoughConductorSample = RoughConductorSampleT<TransportScalar>;
using ReferenceRoughConductorSample = RoughConductorSampleT<ReferenceScalar>;

namespace rough_conductor_reflection_detail {

[[nodiscard]] inline core::Error invalid_rough_conductor(const char* const message) {
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
[[nodiscard]] bool valid_relative_eta(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_eta) noexcept {
    for (const auto value : relative_eta.values) {
        if (!std::isfinite(value) || !(value > Scalar{0})) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool
valid_relative_k(const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& relative_k) noexcept {
    for (const auto value : relative_k.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
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

template <SpectrumScalar Scalar> struct ReflectionGeometryT final {
    Normal3T<Scalar> microfacet_normal{};
    Scalar half_angle_cosine{};
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<ReflectionGeometryT<Scalar>>
reflection_geometry(const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local) {
    const auto half_vector = Vector3T<Scalar>{
        .x = outgoing_local.x + incoming_local.x,
        .y = outgoing_local.y + incoming_local.y,
        .z = outgoing_local.z + incoming_local.z,
    };
    const auto length = std::hypot(std::hypot(half_vector.x, half_vector.y), half_vector.z);
    if (!std::isfinite(length) || !(length > Scalar{0})) {
        return std::unexpected(invalid_rough_conductor(
            "The rough-conductor reflection half-vector is not normalizable."));
    }

    const auto microfacet_normal = Normal3T<Scalar>{
        .x = half_vector.x / length,
        .y = half_vector.y / length,
        .z = half_vector.z / length,
    };
    if (!unit_local_direction(Vector3T<Scalar>{
            .x = microfacet_normal.x, .y = microfacet_normal.y, .z = microfacet_normal.z}) ||
        !(microfacet_normal.z > Scalar{0})) {
        return std::unexpected(invalid_rough_conductor(
            "The rough-conductor reflection half-vector is not representable."));
    }

    // For unit reflection directions |wo + wi| / 2 equals both wo dot wm and wi dot wm. Using
    // this symmetric form keeps reciprocal eval queries on the same numerical path.
    const auto half_angle_cosine = Scalar{0.5} * length;
    if (!std::isfinite(half_angle_cosine) || !(half_angle_cosine > Scalar{0}) ||
        half_angle_cosine > Scalar{1}) {
        return std::unexpected(
            invalid_rough_conductor("The rough-conductor half-angle cosine is outside [0, 1]."));
    }
    return ReflectionGeometryT<Scalar>{
        .microfacet_normal = microfacet_normal,
        .half_angle_cosine = half_angle_cosine,
    };
}

} // namespace rough_conductor_reflection_detail

// Isotropic single-scattering GGX conductor reflection. Directions are finite unit vectors in a
// caller-supplied local closure frame, point away from the surface, and use +Z as the open
// reflection hemisphere. Alpha is the mathematical GGX slope width and is neither remapped nor
// clamped. Eta and k are relative to the non-absorbing incident medium. The coefficient scales the
// physical closure and is kept separate from the exact spectral Fresnel term.
template <SpectrumScalar Scalar> class RoughConductorReflectionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = RoughConductorProbabilityDensityT<Scalar>;
    using sample_type = RoughConductorSampleT<Scalar>;

    [[nodiscard]] static core::Result<RoughConductorReflectionT>
    create(const spectrum_type coefficient, const spectrum_type relative_eta,
           const spectrum_type relative_k, const Scalar alpha) {
        if (!rough_conductor_reflection_detail::valid_coefficient(coefficient)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "Rough-conductor coefficients require every spectral lane to be finite and in "
                "[0, 1]."));
        }
        if (!rough_conductor_reflection_detail::valid_relative_eta(relative_eta)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "Rough-conductor eta requires every spectral lane to be finite and strictly "
                "positive."));
        }
        if (!rough_conductor_reflection_detail::valid_relative_k(relative_k)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "Rough-conductor k requires every spectral lane to be finite and non-negative."));
        }
        const auto microfacet = GgxMicrofacetT<Scalar>::create(alpha);
        if (!microfacet) {
            return std::unexpected(microfacet.error());
        }
        return RoughConductorReflectionT{coefficient, relative_eta, relative_k, *microfacet};
    }

    [[nodiscard]] constexpr const spectrum_type& coefficient() const noexcept {
        return coefficient_;
    }

    [[nodiscard]] constexpr const spectrum_type& relative_eta() const noexcept {
        return relative_eta_;
    }

    [[nodiscard]] constexpr const spectrum_type& relative_k() const noexcept {
        return relative_k_;
    }

    [[nodiscard]] constexpr Scalar alpha() const noexcept {
        return microfacet_.alpha();
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local) const {
        const auto directions = validate_directions(outgoing_local, incoming_local);
        if (!directions) {
            return std::unexpected(directions.error());
        }
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return spectrum_type{};
        }

        auto is_black = true;
        for (const auto value : coefficient_.values) {
            is_black = is_black && value == Scalar{0};
        }
        if (is_black) {
            return spectrum_type{};
        }

        const auto geometry =
            rough_conductor_reflection_detail::reflection_geometry(outgoing_local, incoming_local);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        const auto distribution = microfacet_.normal_distribution(geometry->microfacet_normal);
        if (!distribution) {
            return std::unexpected(distribution.error());
        }
        const auto masking = microfacet_.smith_g2(outgoing_local, incoming_local);
        if (!masking) {
            return std::unexpected(masking.error());
        }
        const auto fresnel =
            conductor_fresnel(geometry->half_angle_cosine, relative_eta_, relative_k_);
        if (!fresnel) {
            return std::unexpected(fresnel.error());
        }

        const auto denominator = Scalar{4} * outgoing_local.z * incoming_local.z;
        const auto scale = (*distribution * *masking) / denominator;
        if (!std::isfinite(denominator) || !(denominator > Scalar{0}) || !std::isfinite(scale) ||
            !(scale > Scalar{0})) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "The rough-conductor GGX reflection value is not representable."));
        }

        auto result = spectrum_type{};
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto value = coefficient_[lane] * (*fresnel)[lane] * scale;
            if (!std::isfinite(value) || value < Scalar{0}) {
                return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                    "A rough-conductor spectral BRDF value is not representable."));
            }
            result[lane] = value;
        }
        return result;
    }

    [[nodiscard]] core::Result<probability_density_type>
    pdf(const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local) const {
        const auto directions = validate_directions(outgoing_local, incoming_local);
        if (!directions) {
            return std::unexpected(directions.error());
        }

        auto result = probability_density_type{
            .value = Scalar{0},
            .measure = ContinuousBsdfProbabilityMeasure,
        };
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return result;
        }

        const auto geometry =
            rough_conductor_reflection_detail::reflection_geometry(outgoing_local, incoming_local);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        const auto visible_probability =
            microfacet_.visible_normal_pdf(outgoing_local, geometry->microfacet_normal);
        if (!visible_probability) {
            return std::unexpected(visible_probability.error());
        }
        if (visible_probability->measure != ProbabilityMeasure::solid_angle) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "The GGX visible-normal PDF uses an incompatible measure."));
        }

        result.value = visible_probability->value / (Scalar{4} * geometry->half_angle_cosine);
        if (!std::isfinite(result.value) || !(result.value > Scalar{0})) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "The rough-conductor reflected-direction PDF is not representable."));
        }
        return result;
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Point2T<Scalar> canonical_sample) const {
        if (!rough_conductor_reflection_detail::unit_local_direction(outgoing_local)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "Rough-conductor directions must be finite unit vectors."));
        }

        const auto visible_normal =
            microfacet_.sample_visible_normal(outgoing_local, canonical_sample);
        if (!visible_normal) {
            return std::unexpected(visible_normal.error());
        }
        if (!visible_normal->has_value()) {
            return std::optional<sample_type>{};
        }

        const auto half_angle_cosine = dot(outgoing_local, (**visible_normal).microfacet_normal);
        if (!std::isfinite(half_angle_cosine) || !(half_angle_cosine > Scalar{0}) ||
            half_angle_cosine > Scalar{1}) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "The sampled rough-conductor half-angle cosine is outside [0, 1]."));
        }
        const auto incoming_local = Vector3T<Scalar>{
            .x = Scalar{2} * half_angle_cosine * (**visible_normal).microfacet_normal.x -
                 outgoing_local.x,
            .y = Scalar{2} * half_angle_cosine * (**visible_normal).microfacet_normal.y -
                 outgoing_local.y,
            .z = Scalar{2} * half_angle_cosine * (**visible_normal).microfacet_normal.z -
                 outgoing_local.z,
        };
        if (!rough_conductor_reflection_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "The sampled rough-conductor reflection direction is not representable."));
        }
        if (!(incoming_local.z > Scalar{0})) {
            return std::optional<sample_type>{};
        }

        const auto value = eval(outgoing_local, incoming_local);
        if (!value) {
            return std::unexpected(value.error());
        }
        const auto probability = pdf(outgoing_local, incoming_local);
        if (!probability) {
            return std::unexpected(probability.error());
        }
        return std::optional<sample_type>{sample_type{
            .incoming_local = incoming_local,
            .value = *value,
            .probability = *probability,
        }};
    }

  private:
    constexpr RoughConductorReflectionT(const spectrum_type coefficient,
                                        const spectrum_type relative_eta,
                                        const spectrum_type relative_k,
                                        const GgxMicrofacetT<Scalar> microfacet) noexcept
        : coefficient_{coefficient}, relative_eta_{relative_eta}, relative_k_{relative_k},
          microfacet_{microfacet} {}

    [[nodiscard]] static core::Status validate_directions(const Vector3T<Scalar> outgoing_local,
                                                          const Vector3T<Scalar> incoming_local) {
        if (!rough_conductor_reflection_detail::unit_local_direction(outgoing_local) ||
            !rough_conductor_reflection_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(rough_conductor_reflection_detail::invalid_rough_conductor(
                "Rough-conductor directions must be finite unit vectors."));
        }
        return {};
    }

    spectrum_type coefficient_;
    spectrum_type relative_eta_;
    spectrum_type relative_k_;
    GgxMicrofacetT<Scalar> microfacet_;
};

using RoughConductorReflection = RoughConductorReflectionT<TransportScalar>;
using ReferenceRoughConductorReflection = RoughConductorReflectionT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<RoughConductorSample>);
static_assert(std::is_trivially_copyable_v<RoughConductorSample>);
static_assert(std::is_standard_layout_v<ReferenceRoughConductorSample>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughConductorSample>);
static_assert(std::is_standard_layout_v<RoughConductorReflection>);
static_assert(std::is_trivially_copyable_v<RoughConductorReflection>);
static_assert(std::is_standard_layout_v<ReferenceRoughConductorReflection>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughConductorReflection>);

} // namespace blackframe::renderer
