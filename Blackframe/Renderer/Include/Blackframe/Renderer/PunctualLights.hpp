#pragma once

#include <Blackframe/Renderer/Light.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace punctual_light_detail {

[[nodiscard]] inline core::Error invalid_punctual_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
validate_wavelength_packet(const SampledWavelengthsT<Scalar>& wavelengths) {
    for (const auto& sample : wavelengths.samples) {
        if (!std::isfinite(sample.nanometers) ||
            sample.nanometers < Scalar{VisibleWavelengthMinimumNanometers} ||
            sample.nanometers > Scalar{VisibleWavelengthMaximumNanometers} ||
            sample.probability.measure != ProbabilityMeasure::wavelength ||
            !std::isfinite(sample.probability.value) || !(sample.probability.value > Scalar{0})) {
            return std::unexpected(invalid_punctual_light(
                "Punctual lights require finite visible wavelengths with positive wavelength "
                "densities."));
        }
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_canonical_sample(const Point2T<Scalar> sample) {
    if (!std::isfinite(sample.x) || !std::isfinite(sample.y) || sample.x < Scalar{0} ||
        sample.x >= Scalar{1} || sample.y < Scalar{0} || sample.y >= Scalar{1}) {
        return std::unexpected(invalid_punctual_light(
            "Punctual light sampling requires a finite canonical sample in [0, 1)^2."));
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_finite_scene_bounds(const Bounds3T<Scalar>& bounds) {
    if (bounds.is_empty()) {
        return std::unexpected(invalid_punctual_light(
            "Punctual light power requires explicit non-empty finite scene bounds."));
    }
    const auto& minimum = bounds.minimum();
    const auto& maximum = bounds.maximum();
    if (!light_detail::finite(minimum) || !light_detail::finite(maximum)) {
        return std::unexpected(invalid_punctual_light(
            "Punctual light power requires explicit non-empty finite scene bounds."));
    }
    return {};
}

template <SpectrumScalar Scalar> class PacketSpectrumT final {
  public:
    [[nodiscard]] static core::Result<PacketSpectrumT>
    create(const SampledWavelengthsT<Scalar>& wavelengths, const LightSpectrumT<Scalar>& values) {
        const auto wavelength_status = validate_wavelength_packet(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        if (!light_detail::finite_non_negative(values)) {
            return std::unexpected(invalid_punctual_light(
                "Punctual light spectra require finite non-negative lanes."));
        }

        auto nanometers = std::array<Scalar, TransportSpectrumSampleCount>{};
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            nanometers[lane] = wavelengths[lane].nanometers;
        }
        return PacketSpectrumT{nanometers, values};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    evaluate(const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto wavelength_status = validate_wavelength_packet(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            if (wavelengths[lane].nanometers != nanometers_[lane]) {
                return std::unexpected(invalid_punctual_light(
                    "A packet-bound punctual light cannot be evaluated at different "
                    "wavelengths."));
            }
        }
        return values_;
    }

    [[nodiscard]] constexpr bool is_black() const noexcept {
        for (const auto value : values_.values) {
            if (value != Scalar{0}) {
                return false;
            }
        }
        return true;
    }

  private:
    constexpr PacketSpectrumT(const std::array<Scalar, TransportSpectrumSampleCount> nanometers,
                              const LightSpectrumT<Scalar> values) noexcept
        : nanometers_{nanometers}, values_{values} {}

    std::array<Scalar, TransportSpectrumSampleCount> nanometers_;
    LightSpectrumT<Scalar> values_;
};

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr LightSpectrumT<Scalar> black_spectrum() noexcept {
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr LightProbabilityDensityT<Scalar> delta_probability() noexcept {
    return LightProbabilityDensityT<Scalar>{
        .value = Scalar{1},
        .measure = ProbabilityMeasure::discrete,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Bounds3T<Scalar>>
finite_emitter_bounds(const Point3T<Scalar> position,
                      const Vector3T<Scalar> absolute_position_error) {
    if (!light_detail::finite(position)) {
        return std::unexpected(invalid_punctual_light(
            "A finite punctual light requires a finite world-space position."));
    }
    if (!light_detail::finite_non_negative(absolute_position_error)) {
        return std::unexpected(invalid_punctual_light(
            "A finite punctual light requires finite non-negative position error."));
    }

    const auto rounded_sum = [](const Scalar left, const Scalar right) noexcept {
        const auto rounded = left + right;
        if (!std::isfinite(rounded)) {
            return std::pair{rounded, Scalar{0}};
        }
        const auto recovered_right = rounded - left;
        const auto residual = (left - (rounded - recovered_right)) + (right - recovered_right);
        return std::pair{rounded, residual};
    };
    const auto outward_lower = [&rounded_sum](const Scalar coordinate,
                                              const Scalar error) noexcept {
        const auto [rounded, residual] = rounded_sum(coordinate, -error);
        return residual < Scalar{0}
                   ? std::nextafter(rounded, -std::numeric_limits<Scalar>::infinity())
                   : rounded;
    };
    const auto outward_upper = [&rounded_sum](const Scalar coordinate,
                                              const Scalar error) noexcept {
        const auto [rounded, residual] = rounded_sum(coordinate, error);
        return residual > Scalar{0}
                   ? std::nextafter(rounded, std::numeric_limits<Scalar>::infinity())
                   : rounded;
    };
    const auto minimum = Point3T<Scalar>{
        .x = outward_lower(position.x, absolute_position_error.x),
        .y = outward_lower(position.y, absolute_position_error.y),
        .z = outward_lower(position.z, absolute_position_error.z),
    };
    const auto maximum = Point3T<Scalar>{
        .x = outward_upper(position.x, absolute_position_error.x),
        .y = outward_upper(position.y, absolute_position_error.y),
        .z = outward_upper(position.z, absolute_position_error.z),
    };
    if (!light_detail::finite(minimum) || !light_detail::finite(maximum)) {
        return std::unexpected(invalid_punctual_light(
            "Punctual light uncertainty bounds are not representable in the requested "
            "precision."));
    }
    const auto bounds = Bounds3T<Scalar>::from_minimum_maximum(minimum, maximum);
    if (!bounds) {
        return std::unexpected(bounds.error());
    }
    return *bounds;
}

template <SpectrumScalar Scalar> struct FiniteLightGeometry final {
    Vector3T<Scalar> direction_to_light;
    Scalar distance;
    Scalar distance_scale;
    Scalar scaled_squared_distance;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<FiniteLightGeometry<Scalar>>
finite_light_geometry(const Point3T<Scalar> reference_position,
                      const Point3T<Scalar> light_position) {
    const auto displacement = light_position - reference_position;
    if (!std::isfinite(displacement.x) || !std::isfinite(displacement.y) ||
        !std::isfinite(displacement.z)) {
        return std::unexpected(invalid_punctual_light(
            "Finite punctual light separation is not representable in the requested precision."));
    }

    const auto distance_scale =
        std::max({std::abs(displacement.x), std::abs(displacement.y), std::abs(displacement.z)});
    if (!(distance_scale > Scalar{0}) || !std::isfinite(distance_scale)) {
        return std::unexpected(invalid_punctual_light(
            "A finite punctual light cannot be sampled at its own position."));
    }

    const auto scaled = displacement / distance_scale;
    const auto scaled_squared_distance =
        std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z));
    const auto scaled_distance = std::sqrt(scaled_squared_distance);
    const auto distance = distance_scale * scaled_distance;
    if (!std::isfinite(scaled_squared_distance) || !(scaled_squared_distance > Scalar{0}) ||
        !std::isfinite(scaled_distance) || !(scaled_distance > Scalar{0}) ||
        !std::isfinite(distance) || !(distance > Scalar{0})) {
        return std::unexpected(invalid_punctual_light(
            "Finite punctual light distance is not representable in the requested precision."));
    }

    const auto direction_to_light = scaled / scaled_distance;
    if (!light_detail::unit_direction(direction_to_light)) {
        return std::unexpected(invalid_punctual_light(
            "Finite punctual light direction is not representable as a unit vector."));
    }
    return FiniteLightGeometry<Scalar>{
        .direction_to_light = direction_to_light,
        .distance = distance,
        .distance_scale = distance_scale,
        .scaled_squared_distance = scaled_squared_distance,
    };
}

template <SpectrumScalar Scalar, std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] core::Result<Scalar>
scale_radiometric_value(const Scalar value,
                        const std::array<Scalar, NumeratorCount>& numerator_factors,
                        const std::array<Scalar, DenominatorCount>& denominator_factors,
                        const long long binary_exponent = 0) {
    if (!std::isfinite(value) || value < Scalar{0}) {
        return std::unexpected(invalid_punctual_light(
            "Punctual light radiometric values must be finite and non-negative."));
    }

    for (const auto factor : numerator_factors) {
        if (!std::isfinite(factor) || !(factor > Scalar{0})) {
            return std::unexpected(invalid_punctual_light(
                "A punctual light radiometric numerator is not representable."));
        }
    }
    for (const auto factor : denominator_factors) {
        if (!std::isfinite(factor) || !(factor > Scalar{0})) {
            return std::unexpected(invalid_punctual_light(
                "A punctual light radiometric denominator is not representable."));
        }
    }
    if (value == Scalar{0}) {
        return Scalar{0};
    }

    auto exponent = 0;
    auto fraction = std::frexp(value, &exponent);
    auto accumulated_exponent = static_cast<long long>(exponent) + binary_exponent;
    for (const auto factor : numerator_factors) {
        auto factor_exponent = 0;
        fraction *= std::frexp(factor, &factor_exponent);
        accumulated_exponent += factor_exponent;
    }
    for (const auto factor : denominator_factors) {
        auto factor_exponent = 0;
        fraction /= std::frexp(factor, &factor_exponent);
        accumulated_exponent -= factor_exponent;
    }

    auto normalization_exponent = 0;
    fraction = std::frexp(fraction, &normalization_exponent);
    accumulated_exponent += normalization_exponent;
    if (accumulated_exponent < std::numeric_limits<int>::min() ||
        accumulated_exponent > std::numeric_limits<int>::max()) {
        return std::unexpected(
            invalid_punctual_light("A punctual light radiometric exponent is not representable."));
    }

    const auto result = std::scalbn(fraction, static_cast<int>(accumulated_exponent));
    if (!std::isfinite(result) || !(result > Scalar{0})) {
        return std::unexpected(invalid_punctual_light(
            "A positive punctual light result is not representable in the requested precision."));
    }
    return result;
}

template <SpectrumScalar Scalar, std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] core::Result<LightSpectrumT<Scalar>>
scale_radiometric_spectrum(const LightSpectrumT<Scalar>& spectrum,
                           const std::array<Scalar, NumeratorCount>& numerator_factors,
                           const std::array<Scalar, DenominatorCount>& denominator_factors,
                           const long long binary_exponent = 0) {
    auto result = LightSpectrumT<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto scaled = scale_radiometric_value(spectrum[lane], numerator_factors,
                                                    denominator_factors, binary_exponent);
        if (!scaled) {
            return std::unexpected(scaled.error());
        }
        result[lane] = *scaled;
    }
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] Scalar exact_special_cosine(const Scalar angle) noexcept {
    if (angle == Scalar{0}) {
        return Scalar{1};
    }
    if (angle == std::numbers::pi_v<Scalar> / Scalar{2}) {
        return Scalar{0};
    }
    if (angle == std::numbers::pi_v<Scalar>) {
        return Scalar{-1};
    }
    return std::cos(angle);
}

template <SpectrumScalar Scalar> struct BoundingSpherePowerScale final {
    std::array<Scalar, 2> factors;
    long long binary_exponent;
    bool is_point;
};

template <SpectrumScalar Scalar> struct BinaryHalfExtent final {
    Scalar fraction;
    int exponent;
    bool is_zero;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BinaryHalfExtent<Scalar>> binary_half_extent(const Scalar lower,
                                                                        const Scalar upper) {
    if (lower == upper) {
        return BinaryHalfExtent<Scalar>{.fraction = Scalar{0}, .exponent = 0, .is_zero = true};
    }

    const auto full_extent = upper - lower;
    auto exponent = 0;
    if (std::isfinite(full_extent)) {
        if (!(full_extent > Scalar{0})) {
            return std::unexpected(invalid_punctual_light(
                "Scene bounds extent is not representable in the requested precision."));
        }
        const auto fraction = std::frexp(full_extent, &exponent);
        return BinaryHalfExtent<Scalar>{
            .fraction = fraction,
            .exponent = exponent - 1,
            .is_zero = false,
        };
    }

    const auto half_extent = upper / Scalar{2} - lower / Scalar{2};
    if (!std::isfinite(half_extent) || !(half_extent > Scalar{0})) {
        return std::unexpected(invalid_punctual_light(
            "Scene bounds extent is not representable in the requested precision."));
    }
    const auto fraction = std::frexp(half_extent, &exponent);
    return BinaryHalfExtent<Scalar>{
        .fraction = fraction,
        .exponent = exponent,
        .is_zero = false,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<BoundingSpherePowerScale<Scalar>>
bounding_sphere_power_scale(const Bounds3T<Scalar>& scene_bounds) {
    const auto bounds_status = validate_finite_scene_bounds(scene_bounds);
    if (!bounds_status) {
        return std::unexpected(bounds_status.error());
    }

    const auto& minimum = scene_bounds.minimum();
    const auto& maximum = scene_bounds.maximum();
    const auto x = binary_half_extent(minimum.x, maximum.x);
    const auto y = binary_half_extent(minimum.y, maximum.y);
    const auto z = binary_half_extent(minimum.z, maximum.z);
    if (!x) {
        return std::unexpected(x.error());
    }
    if (!y) {
        return std::unexpected(y.error());
    }
    if (!z) {
        return std::unexpected(z.error());
    }

    if (x->is_zero && y->is_zero && z->is_zero) {
        return BoundingSpherePowerScale<Scalar>{
            .factors = {Scalar{1}, Scalar{1}},
            .binary_exponent = 0,
            .is_point = true,
        };
    }

    auto maximum_exponent = std::numeric_limits<int>::min();
    for (const auto extent : std::array{*x, *y, *z}) {
        if (!extent.is_zero) {
            maximum_exponent = std::max(maximum_exponent, extent.exponent);
        }
    }
    const auto scaled_extent = [maximum_exponent](const BinaryHalfExtent<Scalar> extent) noexcept {
        return extent.is_zero ? Scalar{0}
                              : std::scalbn(extent.fraction, extent.exponent - maximum_exponent);
    };
    const auto scaled_x = scaled_extent(*x);
    const auto scaled_y = scaled_extent(*y);
    const auto scaled_z = scaled_extent(*z);
    const auto scaled_squared_radius =
        std::fma(scaled_x, scaled_x, std::fma(scaled_y, scaled_y, scaled_z * scaled_z));
    if (!std::isfinite(scaled_squared_radius) || !(scaled_squared_radius > Scalar{0})) {
        return std::unexpected(invalid_punctual_light(
            "Scene bounding-sphere radius is not representable in the requested precision."));
    }
    return BoundingSpherePowerScale<Scalar>{
        .factors = {std::numbers::pi_v<Scalar>, scaled_squared_radius},
        .binary_exponent = static_cast<long long>(maximum_exponent) * 2,
        .is_point = false,
    };
}

} // namespace punctual_light_detail

// An ideal isotropic point emitter. The stored packet is spectral radiant
// intensity (W sr^-1 nm^-1); sample_li applies the exact inverse-square law.
template <SpectrumScalar Scalar> class PointLightT final {
  public:
    [[nodiscard]] static core::Result<PointLightT>
    create(const Point3T<Scalar> position, const Vector3T<Scalar> absolute_position_error,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& spectral_radiant_intensity) {
        const auto emitter_bounds =
            punctual_light_detail::finite_emitter_bounds(position, absolute_position_error);
        if (!emitter_bounds) {
            return std::unexpected(emitter_bounds.error());
        }
        const auto intensity = punctual_light_detail::PacketSpectrumT<Scalar>::create(
            wavelengths, spectral_radiant_intensity);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return PointLightT{position, absolute_position_error, *emitter_bounds, *intensity};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status =
            punctual_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        const auto geometry =
            punctual_light_detail::finite_light_geometry(context.position(), position_);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }
        const auto radiance = punctual_light_detail::scale_radiometric_spectrum(
            *intensity, std::array<Scalar, 0>{},
            std::array{geometry->distance_scale, geometry->distance_scale,
                       geometry->scaled_squared_distance});
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        if (intensity_.is_black()) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }

        const auto endpoint =
            LightSampleEndpointT<Scalar>::create_point(position_, absolute_position_error_);
        if (!endpoint) {
            return std::unexpected(endpoint.error());
        }
        const auto sampled = IncidentLightSampleT<Scalar>::create_finite(
            context, *endpoint, *radiance, punctual_light_detail::delta_probability<Scalar>());
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sampled};
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        if (!light_detail::unit_direction(direction_to_light)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Punctual light PDF evaluation requires a finite unit direction."));
        }
        return DirectionalLightPdfT<Scalar>::create(Scalar{0});
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return punctual_light_detail::black_spectrum<Scalar>();
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status =
            punctual_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return punctual_light_detail::scale_radiometric_spectrum(
            *intensity, std::array{Scalar{4}, std::numbers::pi_v<Scalar>}, std::array<Scalar, 0>{});
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }

  private:
    constexpr PointLightT(const Point3T<Scalar> position,
                          const Vector3T<Scalar> absolute_position_error,
                          const Bounds3T<Scalar> bounds,
                          const punctual_light_detail::PacketSpectrumT<Scalar> intensity) noexcept
        : position_{position}, absolute_position_error_{absolute_position_error}, bounds_{bounds},
          intensity_{intensity} {}

    Point3T<Scalar> position_;
    Vector3T<Scalar> absolute_position_error_;
    Bounds3T<Scalar> bounds_;
    punctual_light_detail::PacketSpectrumT<Scalar> intensity_;
};

// An ideal directional source. The constructor direction is the propagation
// direction from the source into the scene. Stored lanes are spectral
// irradiance; power uses pi R^2 times that irradiance for the scene bounding
// sphere and is therefore an explicit finite-scene proxy.
template <SpectrumScalar Scalar> class DirectionalLightT final {
  public:
    [[nodiscard]] static core::Result<DirectionalLightT>
    create(const Vector3T<Scalar> propagation_direction,
           const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& spectral_irradiance) {
        if (!light_detail::unit_direction(propagation_direction)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "A directional light requires a finite unit propagation direction."));
        }
        const auto irradiance = punctual_light_detail::PacketSpectrumT<Scalar>::create(
            wavelengths, spectral_irradiance);
        if (!irradiance) {
            return std::unexpected(irradiance.error());
        }
        return DirectionalLightT{propagation_direction, *irradiance};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>&, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status =
            punctual_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto irradiance = irradiance_.evaluate(wavelengths);
        if (!irradiance) {
            return std::unexpected(irradiance.error());
        }
        if (irradiance_.is_black()) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }
        const auto sampled = IncidentLightSampleT<Scalar>::create_infinite(
            -propagation_direction_, *irradiance,
            punctual_light_detail::delta_probability<Scalar>());
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sampled};
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto irradiance = irradiance_.evaluate(wavelengths);
        if (!irradiance) {
            return std::unexpected(irradiance.error());
        }
        if (!light_detail::unit_direction(direction_to_light)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Punctual light PDF evaluation requires a finite unit direction."));
        }
        return DirectionalLightPdfT<Scalar>::create(Scalar{0});
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto irradiance = irradiance_.evaluate(wavelengths);
        if (!irradiance) {
            return std::unexpected(irradiance.error());
        }
        return punctual_light_detail::black_spectrum<Scalar>();
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto power_scale = punctual_light_detail::bounding_sphere_power_scale(scene_bounds);
        if (!power_scale) {
            return std::unexpected(power_scale.error());
        }
        const auto irradiance = irradiance_.evaluate(wavelengths);
        if (!irradiance) {
            return std::unexpected(irradiance.error());
        }
        if (power_scale->is_point) {
            return punctual_light_detail::black_spectrum<Scalar>();
        }
        return punctual_light_detail::scale_radiometric_spectrum(*irradiance, power_scale->factors,
                                                                 std::array<Scalar, 0>{},
                                                                 power_scale->binary_exponent);
    }

    [[nodiscard]] static constexpr Bounds3T<Scalar> bounds() noexcept {
        return Bounds3T<Scalar>::unbounded();
    }

  private:
    constexpr DirectionalLightT(
        const Vector3T<Scalar> propagation_direction,
        const punctual_light_detail::PacketSpectrumT<Scalar> irradiance) noexcept
        : propagation_direction_{propagation_direction}, irradiance_{irradiance} {}

    Vector3T<Scalar> propagation_direction_;
    punctual_light_detail::PacketSpectrumT<Scalar> irradiance_;
};

// An ideal point emitter with an axially symmetric smooth cone. The emission
// direction points from the light into the scene. Inner and outer values are
// half angles in radians; equal values select an explicit hard cone.
template <SpectrumScalar Scalar> class SpotLightT final {
  public:
    [[nodiscard]] static core::Result<SpotLightT>
    create(const Point3T<Scalar> position, const Vector3T<Scalar> absolute_position_error,
           const Vector3T<Scalar> emission_direction, const Scalar inner_half_angle_radians,
           const Scalar outer_half_angle_radians, const SampledWavelengthsT<Scalar>& wavelengths,
           const LightSpectrumT<Scalar>& on_axis_spectral_radiant_intensity) {
        const auto emitter_bounds =
            punctual_light_detail::finite_emitter_bounds(position, absolute_position_error);
        if (!emitter_bounds) {
            return std::unexpected(emitter_bounds.error());
        }
        if (!light_detail::unit_direction(emission_direction)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "A spot light requires a finite unit emission direction."));
        }
        if (!std::isfinite(inner_half_angle_radians) || !std::isfinite(outer_half_angle_radians) ||
            inner_half_angle_radians < Scalar{0} ||
            inner_half_angle_radians > outer_half_angle_radians ||
            !(outer_half_angle_radians > Scalar{0}) ||
            outer_half_angle_radians >= std::numbers::pi_v<Scalar>) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Spot half angles require 0 <= inner <= outer < pi and a positive outer cone."));
        }

        const auto cosine_inner =
            punctual_light_detail::exact_special_cosine(inner_half_angle_radians);
        const auto cosine_outer =
            punctual_light_detail::exact_special_cosine(outer_half_angle_radians);
        if (!std::isfinite(cosine_inner) || !std::isfinite(cosine_outer) ||
            cosine_outer == Scalar{1} || cosine_outer == Scalar{-1} ||
            (inner_half_angle_radians > Scalar{0} && cosine_inner == Scalar{1}) ||
            cosine_inner < cosine_outer ||
            (inner_half_angle_radians < outer_half_angle_radians && cosine_inner == cosine_outer)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Spot cone cosines are not distinguishable in the requested precision."));
        }

        const auto power_term =
            (Scalar{1} - cosine_inner) + (cosine_inner - cosine_outer) / Scalar{2};
        if (!std::isfinite(power_term) || !(power_term > Scalar{0})) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Spot cone power is not representable in the requested precision."));
        }

        const auto intensity = punctual_light_detail::PacketSpectrumT<Scalar>::create(
            wavelengths, on_axis_spectral_radiant_intensity);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return SpotLightT{position,
                          absolute_position_error,
                          emission_direction,
                          cosine_inner,
                          cosine_outer,
                          power_term,
                          inner_half_angle_radians == outer_half_angle_radians,
                          *emitter_bounds,
                          *intensity};
    }

    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto sample_status =
            punctual_light_detail::validate_canonical_sample(canonical_sample);
        if (!sample_status) {
            return std::unexpected(sample_status.error());
        }
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        const auto geometry =
            punctual_light_detail::finite_light_geometry(context.position(), position_);
        if (!geometry) {
            return std::unexpected(geometry.error());
        }

        const auto direction_from_light = -geometry->direction_to_light;
        const auto cone_cosine = std::fma(emission_direction_.x, direction_from_light.x,
                                          std::fma(emission_direction_.y, direction_from_light.y,
                                                   emission_direction_.z * direction_from_light.z));
        if (!std::isfinite(cone_cosine)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Spot light cone evaluation is not representable."));
        }
        if (cone_cosine <= cosine_outer_) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }

        auto falloff_factors = std::array{Scalar{1}, Scalar{1}, Scalar{1}};
        if (!hard_cone_ && cone_cosine < cosine_inner_) {
            const auto transition = (cone_cosine - cosine_outer_) / (cosine_inner_ - cosine_outer_);
            const auto smooth_factor = Scalar{3} - Scalar{2} * transition;
            if (!std::isfinite(transition) || !(transition > Scalar{0}) ||
                !(transition < Scalar{1}) || !std::isfinite(smooth_factor) ||
                !(smooth_factor > Scalar{0})) {
                return std::unexpected(punctual_light_detail::invalid_punctual_light(
                    "Spot light smooth falloff is not representable."));
            }
            falloff_factors = {transition, transition, smooth_factor};
        }

        const auto radiance = punctual_light_detail::scale_radiometric_spectrum(
            *intensity, falloff_factors,
            std::array{geometry->distance_scale, geometry->distance_scale,
                       geometry->scaled_squared_distance});
        if (!radiance) {
            return std::unexpected(radiance.error());
        }
        if (intensity_.is_black()) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }

        const auto endpoint =
            LightSampleEndpointT<Scalar>::create_point(position_, absolute_position_error_);
        if (!endpoint) {
            return std::unexpected(endpoint.error());
        }
        const auto sampled = IncidentLightSampleT<Scalar>::create_finite(
            context, *endpoint, *radiance, punctual_light_detail::delta_probability<Scalar>());
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sampled};
    }

    [[nodiscard]] core::Result<DirectionalLightPdfT<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar> direction_to_light,
           const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        if (!light_detail::unit_direction(direction_to_light)) {
            return std::unexpected(punctual_light_detail::invalid_punctual_light(
                "Punctual light PDF evaluation requires a finite unit direction."));
        }
        return DirectionalLightPdfT<Scalar>::create(Scalar{0});
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    le(const RayT<Scalar>&, const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return punctual_light_detail::black_spectrum<Scalar>();
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    power(const Bounds3T<Scalar>& scene_bounds,
          const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto bounds_status =
            punctual_light_detail::validate_finite_scene_bounds(scene_bounds);
        if (!bounds_status) {
            return std::unexpected(bounds_status.error());
        }
        const auto intensity = intensity_.evaluate(wavelengths);
        if (!intensity) {
            return std::unexpected(intensity.error());
        }
        return punctual_light_detail::scale_radiometric_spectrum(
            *intensity,
            std::array{Scalar{2}, std::numbers::pi_v<Scalar>, effective_solid_angle_term_},
            std::array<Scalar, 0>{});
    }

    [[nodiscard]] constexpr Bounds3T<Scalar> bounds() const noexcept {
        return bounds_;
    }

  private:
    constexpr SpotLightT(const Point3T<Scalar> position,
                         const Vector3T<Scalar> absolute_position_error,
                         const Vector3T<Scalar> emission_direction, const Scalar cosine_inner,
                         const Scalar cosine_outer, const Scalar effective_solid_angle_term,
                         const bool hard_cone, const Bounds3T<Scalar> bounds,
                         const punctual_light_detail::PacketSpectrumT<Scalar> intensity) noexcept
        : position_{position}, absolute_position_error_{absolute_position_error},
          emission_direction_{emission_direction}, cosine_inner_{cosine_inner},
          cosine_outer_{cosine_outer}, effective_solid_angle_term_{effective_solid_angle_term},
          hard_cone_{hard_cone}, bounds_{bounds}, intensity_{intensity} {}

    Point3T<Scalar> position_;
    Vector3T<Scalar> absolute_position_error_;
    Vector3T<Scalar> emission_direction_;
    Scalar cosine_inner_;
    Scalar cosine_outer_;
    Scalar effective_solid_angle_term_;
    bool hard_cone_;
    Bounds3T<Scalar> bounds_;
    punctual_light_detail::PacketSpectrumT<Scalar> intensity_;
};

using PointLight = PointLightT<TransportScalar>;
using ReferencePointLight = PointLightT<ReferenceScalar>;
using DirectionalLight = DirectionalLightT<TransportScalar>;
using ReferenceDirectionalLight = DirectionalLightT<ReferenceScalar>;
using SpotLight = SpotLightT<TransportScalar>;
using ReferenceSpotLight = SpotLightT<ReferenceScalar>;

static_assert(LightModelFor<PointLight, TransportScalar>);
static_assert(LightModelFor<ReferencePointLight, ReferenceScalar>);
static_assert(LightModelFor<DirectionalLight, TransportScalar>);
static_assert(LightModelFor<ReferenceDirectionalLight, ReferenceScalar>);
static_assert(LightModelFor<SpotLight, TransportScalar>);
static_assert(LightModelFor<ReferenceSpotLight, ReferenceScalar>);
static_assert(std::is_standard_layout_v<PointLight>);
static_assert(std::is_trivially_copyable_v<PointLight>);
static_assert(std::is_standard_layout_v<ReferencePointLight>);
static_assert(std::is_trivially_copyable_v<ReferencePointLight>);
static_assert(std::is_standard_layout_v<DirectionalLight>);
static_assert(std::is_trivially_copyable_v<DirectionalLight>);
static_assert(std::is_standard_layout_v<ReferenceDirectionalLight>);
static_assert(std::is_trivially_copyable_v<ReferenceDirectionalLight>);
static_assert(std::is_standard_layout_v<SpotLight>);
static_assert(std::is_trivially_copyable_v<SpotLight>);
static_assert(std::is_standard_layout_v<ReferenceSpotLight>);
static_assert(std::is_trivially_copyable_v<ReferenceSpotLight>);

} // namespace blackframe::renderer
