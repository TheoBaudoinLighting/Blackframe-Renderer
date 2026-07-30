#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/SamplingMappings.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using LambertianProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar> struct LambertianSampleT final {
    // Both directions use the local convention documented by LambertianReflectionT.
    Vector3T<Scalar> incoming_local{};
    // This is the BRDF value f, not a cosine- or PDF-weighted throughput.
    SampledSpectrum<TransportSpectrumSampleCount, Scalar> value{};
    LambertianProbabilityDensityT<Scalar> probability{
        .value = Scalar{0},
        .measure = ProbabilityMeasure::solid_angle,
    };
};

using LambertianSample = LambertianSampleT<TransportScalar>;
using ReferenceLambertianSample = LambertianSampleT<ReferenceScalar>;

namespace lambertian_reflection_detail {

[[nodiscard]] inline core::Error invalid_lambertian(const char* const message) {
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

} // namespace lambertian_reflection_detail

// Directions are unit vectors in the local shading frame and point away from the surface. +Z is
// the oriented surface hemisphere. The closure is one-sided: valid directions outside its open
// reflection support return zero rather than being flipped or face-forwarded.
template <SpectrumScalar Scalar> class LambertianReflectionT final {
  public:
    using spectrum_type = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;
    using probability_density_type = LambertianProbabilityDensityT<Scalar>;
    using sample_type = LambertianSampleT<Scalar>;

    [[nodiscard]] static core::Result<LambertianReflectionT>
    create(const spectrum_type reflectance) {
        if (!lambertian_reflection_detail::valid_reflectance(reflectance)) {
            return std::unexpected(lambertian_reflection_detail::invalid_lambertian(
                "Lambertian reflectance requires every spectral lane to be finite and in [0, 1]."));
        }
        return LambertianReflectionT{reflectance};
    }

    [[nodiscard]] constexpr const spectrum_type& reflectance() const noexcept {
        return reflectance_;
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
        return reflectance_ * std::numbers::inv_pi_v<Scalar>;
    }

    [[nodiscard]] core::Result<probability_density_type>
    pdf(const Vector3T<Scalar> outgoing_local, const Vector3T<Scalar> incoming_local) const {
        const auto directions = validate_directions(outgoing_local, incoming_local);
        if (!directions.has_value()) {
            return std::unexpected(directions.error());
        }

        auto result = probability_density_type{
            .value = Scalar{0},
            .measure = ProbabilityMeasure::solid_angle,
        };
        if (!(outgoing_local.z > Scalar{0}) || !(incoming_local.z > Scalar{0})) {
            return result;
        }

        // Direction validation already applies the engine's unit-length tolerance. Requiring the
        // cosine to lie exactly in [-1, 1] here would reject valid round-trips through a numerical
        // frame when z differs from one by a few ULPs.
        result.value = incoming_local.z * std::numbers::inv_pi_v<Scalar>;
        if (!(result.value > Scalar{0}) || !std::isfinite(result.value)) {
            return std::unexpected(lambertian_reflection_detail::invalid_lambertian(
                "Lambertian PDF is not representable for the supplied direction."));
        }
        return result;
    }

    // A valid input outside the reflection support, including a canonical sample mapped exactly
    // onto the horizon, returns an empty optional. Invalid inputs remain explicit errors.
    [[nodiscard]] core::Result<std::optional<sample_type>>
    sample(const Vector3T<Scalar> outgoing_local, const Point2T<Scalar> canonical_sample) const {
        if (!lambertian_reflection_detail::unit_local_direction(outgoing_local)) {
            return std::unexpected(lambertian_reflection_detail::invalid_lambertian(
                "Lambertian directions must be finite unit vectors."));
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
    constexpr explicit LambertianReflectionT(const spectrum_type reflectance) noexcept
        : reflectance_{reflectance} {}

    [[nodiscard]] static core::Status validate_directions(const Vector3T<Scalar> outgoing_local,
                                                          const Vector3T<Scalar> incoming_local) {
        if (!lambertian_reflection_detail::unit_local_direction(outgoing_local) ||
            !lambertian_reflection_detail::unit_local_direction(incoming_local)) {
            return std::unexpected(lambertian_reflection_detail::invalid_lambertian(
                "Lambertian directions must be finite unit vectors."));
        }
        return {};
    }

    spectrum_type reflectance_;
};

using LambertianReflection = LambertianReflectionT<TransportScalar>;
using ReferenceLambertianReflection = LambertianReflectionT<ReferenceScalar>;

static_assert(std::is_standard_layout_v<LambertianSample>);
static_assert(std::is_trivially_copyable_v<LambertianSample>);
static_assert(std::is_standard_layout_v<ReferenceLambertianSample>);
static_assert(std::is_trivially_copyable_v<ReferenceLambertianSample>);
static_assert(std::is_standard_layout_v<LambertianReflection>);
static_assert(std::is_trivially_copyable_v<LambertianReflection>);
static_assert(std::is_standard_layout_v<ReferenceLambertianReflection>);
static_assert(std::is_trivially_copyable_v<ReferenceLambertianReflection>);

} // namespace blackframe::renderer
