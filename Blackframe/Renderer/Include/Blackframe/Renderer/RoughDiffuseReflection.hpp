#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using RoughDiffuseProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct RoughDiffuseSampleT final {
    Vector3T<Scalar> incoming_local{};
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    RoughDiffuseProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ContinuousBsdfProbabilityMeasure,
    };
};

using RoughDiffuseSample = RoughDiffuseSampleT<TransportScalar>;
using ReferenceRoughDiffuseSample = RoughDiffuseSampleT<ReferenceScalar>;

namespace rough_diffuse_reflection_detail {

[[nodiscard]] inline core::Error invalid_rough_diffuse(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool valid_reflectance(
    const SampledSpectrum<TransportSpectrumSampleCount, Scalar>& reflectance) noexcept {
    for (const auto value : reflectance.values) {
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
inline constexpr auto FonCoefficient =
    Scalar{0.5} - Scalar{2} * std::numbers::inv_pi_v<Scalar> / Scalar{3};

template <SpectrumScalar Scalar>
inline constexpr auto FonAverageCoefficient =
    Scalar{2} / Scalar{3} - Scalar{28} * std::numbers::inv_pi_v<Scalar> / Scalar{15};

template <SpectrumScalar Scalar>
inline constexpr auto FonAverageLossCoefficient =
    FonCoefficient<Scalar> - FonAverageCoefficient<Scalar>;

// This is 1 - E_FON for unit albedo with the common positive factor A_F * r removed.
// Factoring 1 - sin(theta)^3 avoids the cancellation and division by cos(theta) in the
// published exact directional-albedo expression at grazing angles.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> directional_loss_shape(const Vector3T<Scalar> direction) {
    const auto sine = std::hypot(direction.x, direction.y);
    const auto cosine = direction.z;
    const auto angle = std::atan2(sine, cosine);
    const auto stable_ratio = sine * cosine * (Scalar{1} + sine + sine * sine) / (Scalar{1} + sine);
    const auto g = sine * (angle - sine * cosine) + (Scalar{2} / Scalar{3}) * (stable_ratio - sine);
    const auto loss_shape = FonCoefficient<Scalar> - g * std::numbers::inv_pi_v<Scalar>;
    if (!std::isfinite(loss_shape) || loss_shape < Scalar{0}) {
        return std::unexpected(invalid_rough_diffuse(
            "The rough-diffuse directional albedo is not representable for the supplied "
            "direction."));
    }
    return loss_shape;
}

} // namespace rough_diffuse_reflection_detail

// Energy-preserving Oren--Nayar (EON) reflection. Directions are finite unit vectors in a local
// closure frame, point away from the surface, and use +Z as the open reflection hemisphere. The
// scalar roughness r is the Fujii/EON interpolation parameter in [0, 1], not an angle. The BRDF
// uses the exact FON directional albedo and its reciprocal analytical multiple-scattering term.
// Cosine-hemisphere sampling is deliberately retained as the complete conditional distribution.
template <SpectrumScalar Scalar> class RoughDiffuseReflectionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = RoughDiffuseProbabilityDensityT<Scalar>;
    using sample_type = RoughDiffuseSampleT<Scalar>;

    [[nodiscard]] static core::Result<RoughDiffuseReflectionT>
    create(const spectrum_type reflectance, const Scalar roughness) {
        if (!rough_diffuse_reflection_detail::valid_reflectance(reflectance)) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "Rough-diffuse reflectance requires every spectral lane to be finite and in [0, "
                "1]."));
        }
        if (!std::isfinite(roughness) || roughness < Scalar{0} || roughness > Scalar{1}) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "Rough-diffuse roughness requires a finite value in [0, 1]."));
        }
        return RoughDiffuseReflectionT{reflectance, roughness};
    }

    [[nodiscard]] constexpr const spectrum_type& reflectance() const noexcept {
        return reflectance_;
    }

    [[nodiscard]] constexpr Scalar roughness() const noexcept {
        return roughness_;
    }

    [[nodiscard]] core::Result<spectrum_type> eval(const Vector3T<Scalar> outgoing_local,
                                                   const Vector3T<Scalar> incoming_local) const {
        const auto directions = validate_directions(outgoing_local, incoming_local);
        if (!directions.has_value()) {
            return std::unexpected(directions.error());
        }
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return spectrum_type{};
        }
        auto is_black = true;
        for (const auto value : reflectance_.values) {
            is_black = is_black && value == Scalar{0};
        }
        if (is_black) {
            return spectrum_type{};
        }
        if (roughness_ == Scalar{0}) {
            return reflectance_ * std::numbers::inv_pi_v<Scalar>;
        }

        const auto tangent_dot =
            std::fma(outgoing_local.x, incoming_local.x, outgoing_local.y * incoming_local.y);
        const auto s_over_t = tangent_dot > Scalar{0}
                                  ? tangent_dot / std::max(outgoing_local.z, incoming_local.z)
                                  : tangent_dot;
        const auto a =
            Scalar{1} /
            (Scalar{1} + rough_diffuse_reflection_detail::FonCoefficient<Scalar> * roughness_);
        const auto single_scatter =
            std::numbers::inv_pi_v<Scalar> * a * (Scalar{1} + roughness_ * s_over_t);
        if (!std::isfinite(single_scatter) || single_scatter < Scalar{0}) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "The rough-diffuse single-scattering value is not representable."));
        }

        const auto outgoing_loss =
            rough_diffuse_reflection_detail::directional_loss_shape(outgoing_local);
        if (!outgoing_loss.has_value()) {
            return std::unexpected(outgoing_loss.error());
        }
        const auto incoming_loss =
            rough_diffuse_reflection_detail::directional_loss_shape(incoming_local);
        if (!incoming_loss.has_value()) {
            return std::unexpected(incoming_loss.error());
        }

        const auto average_albedo =
            a * (Scalar{1} +
                 rough_diffuse_reflection_detail::FonAverageCoefficient<Scalar> * roughness_);
        const auto loss_product_over_average =
            a * roughness_ * *incoming_loss * *outgoing_loss /
            rough_diffuse_reflection_detail::FonAverageLossCoefficient<Scalar>;
        if (!std::isfinite(average_albedo) || !(average_albedo > Scalar{0}) ||
            average_albedo > Scalar{1} || !std::isfinite(loss_product_over_average) ||
            loss_product_over_average < Scalar{0}) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "The rough-diffuse energy compensation is not representable."));
        }

        auto result = spectrum_type{};
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto rho = reflectance_[lane];
            const auto multiple_scatter_denominator = (Scalar{1} - rho) + rho * average_albedo;
            if (!std::isfinite(multiple_scatter_denominator) ||
                !(multiple_scatter_denominator > Scalar{0})) {
                return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                    "The rough-diffuse multiple-scattering albedo is not representable."));
            }
            const auto multiple_scatter_albedo =
                rho * rho * average_albedo / multiple_scatter_denominator;
            const auto value = rho * single_scatter + multiple_scatter_albedo *
                                                          std::numbers::inv_pi_v<Scalar> *
                                                          loss_product_over_average;
            if (!std::isfinite(value) || value < Scalar{0}) {
                return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                    "The rough-diffuse BRDF value is not representable."));
            }
            result[lane] = value;
        }
        return result;
    }

    [[nodiscard]] core::Result<probability_density_type>
    pdf(const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local) const {
        const auto directions = validate_directions(outgoing_local, incoming_local);
        if (!directions.has_value()) {
            return std::unexpected(directions.error());
        }

        auto result = probability_density_type{
            .value = Scalar{0},
            .measure = ContinuousBsdfProbabilityMeasure,
        };
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return result;
        }

        result.value = incoming_local.z * std::numbers::inv_pi_v<Scalar>;
        if (!(result.value > Scalar{0}) || !std::isfinite(result.value)) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "Rough-diffuse PDF is not representable for the supplied direction."));
        }
        return result;
    }

    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Point2T<Scalar> canonical_sample) const {
        if (!rough_diffuse_reflection_detail::unit_local_direction(outgoing_local)) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "Rough-diffuse directions must be finite unit vectors."));
        }

        const auto incoming_local = map_cosine_hemisphere(canonical_sample);
        if (!incoming_local.has_value()) {
            return std::unexpected(incoming_local.error());
        }
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local->z > Scalar{0})) {
            return std::optional<sample_type>{};
        }

        const auto value = eval(outgoing_local, *incoming_local);
        if (!value.has_value()) {
            return std::unexpected(value.error());
        }
        const auto probability = pdf(outgoing_local, *incoming_local);
        if (!probability.has_value()) {
            return std::unexpected(probability.error());
        }
        return std::optional<sample_type>{sample_type{
            .incoming_local = *incoming_local,
            .value = *value,
            .probability = *probability,
        }};
    }

  private:
    constexpr RoughDiffuseReflectionT(const spectrum_type reflectance,
                                      const Scalar roughness) noexcept
        : reflectance_{reflectance}, roughness_{roughness} {}

    [[nodiscard]] static core::Status validate_directions(const Vector3T<Scalar> outgoing_local,
                                                          const Vector3T<Scalar> incoming_local) {
        if (!rough_diffuse_reflection_detail::unit_local_direction(outgoing_local) ||
            !rough_diffuse_reflection_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(rough_diffuse_reflection_detail::invalid_rough_diffuse(
                "Rough-diffuse directions must be finite unit vectors."));
        }
        return {};
    }

    spectrum_type reflectance_;
    Scalar roughness_{};
};

using RoughDiffuseReflection = RoughDiffuseReflectionT<TransportScalar>;
using ReferenceRoughDiffuseReflection = RoughDiffuseReflectionT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<RoughDiffuseSample>);
static_assert(std::is_trivially_copyable_v<RoughDiffuseSample>);
static_assert(std::is_standard_layout_v<ReferenceRoughDiffuseSample>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDiffuseSample>);
static_assert(std::is_standard_layout_v<RoughDiffuseReflection>);
static_assert(std::is_trivially_copyable_v<RoughDiffuseReflection>);
static_assert(std::is_standard_layout_v<ReferenceRoughDiffuseReflection>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDiffuseReflection>);

} // namespace blackframe::renderer
