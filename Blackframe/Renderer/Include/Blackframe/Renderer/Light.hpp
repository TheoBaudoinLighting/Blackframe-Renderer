#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Bounds.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

template <SpectrumScalar Scalar>
using LightProbabilityDensityT =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, ProbabilityDensity,
                       ReferenceProbabilityDensity>;

template <SpectrumScalar Scalar>
using LightSpectrumT = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

namespace light_detail {

[[nodiscard]] inline core::Error invalid_light(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar> [[nodiscard]] bool finite(const Point3T<Scalar> point) noexcept {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_non_negative(const LightSpectrumT<Scalar>& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool finite_non_negative(const Vector3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           value.x >= Scalar{0} && value.y >= Scalar{0} && value.z >= Scalar{0};
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool unit_direction(const Vector3T<Scalar> direction) noexcept {
    const auto squared_length = std::fma(
        direction.x, direction.x, std::fma(direction.y, direction.y, direction.z * direction.z));
    constexpr auto tolerance = Scalar{128} * std::numeric_limits<Scalar>::epsilon();
    return std::isfinite(direction.x) && std::isfinite(direction.y) && std::isfinite(direction.z) &&
           std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <SpectrumScalar Scalar>
[[nodiscard]] bool unit_normal(const Normal3T<Scalar> normal) noexcept {
    const auto squared_length =
        std::fma(normal.x, normal.x, std::fma(normal.y, normal.y, normal.z * normal.z));
    constexpr auto tolerance = Scalar{128} * std::numeric_limits<Scalar>::epsilon();
    return std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z) &&
           std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <SpectrumScalar Scalar> struct PdfJacobianFactors final {
    Scalar distance_scale;
    Scalar scaled_distance_cubed;
    Scalar scaled_normal_alignment;
};

// For d = pLight - pReference, s = max(abs(d)), and u = d / s:
//   r^2 / abs(dot(Ng, -normalize(d)))
//     = s^2 * pow(dot(u, u), 3/2) / abs(dot(Ng, -u)).
// Keeping these factors separate avoids materializing either r^2 or 1 / r^2.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<PdfJacobianFactors<Scalar>>
pdf_jacobian_factors(const Point3T<Scalar> reference_position,
                     const Point3T<Scalar> surface_position,
                     const Normal3T<Scalar> surface_geometric_normal) {
    if (!finite(reference_position) || !finite(surface_position)) {
        return std::unexpected(
            invalid_light("Light PDF conversion requires finite reference and surface positions."));
    }
    if (!unit_normal(surface_geometric_normal)) {
        return std::unexpected(
            invalid_light("Light PDF conversion requires a finite unit surface geometric normal."));
    }

    const auto displacement = surface_position - reference_position;
    if (!std::isfinite(displacement.x) || !std::isfinite(displacement.y) ||
        !std::isfinite(displacement.z)) {
        return std::unexpected(invalid_light(
            "Light PDF conversion separation is not representable in the requested precision."));
    }
    const auto distance_scale =
        std::max({std::abs(displacement.x), std::abs(displacement.y), std::abs(displacement.z)});
    if (!(distance_scale > Scalar{0}) || !std::isfinite(distance_scale)) {
        return std::unexpected(
            invalid_light("Light PDF conversion requires two distinct positions."));
    }

    const auto scaled = displacement / distance_scale;
    const auto scaled_squared_length =
        std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z));
    const auto scaled_length = std::sqrt(scaled_squared_length);
    const auto scaled_distance_cubed = scaled_squared_length * scaled_length;
    if (!std::isfinite(scaled_squared_length) || !(scaled_squared_length > Scalar{0}) ||
        !std::isfinite(scaled_distance_cubed) || !(scaled_distance_cubed > Scalar{0})) {
        return std::unexpected(
            invalid_light("Light PDF conversion distance is not representable."));
    }

    const auto signed_alignment = std::fma(
        surface_geometric_normal.x, -scaled.x,
        std::fma(surface_geometric_normal.y, -scaled.y, surface_geometric_normal.z * -scaled.z));
    const auto absolute_alignment = std::abs(signed_alignment);
    const auto absolute_term_sum =
        std::fma(std::abs(surface_geometric_normal.x), std::abs(scaled.x),
                 std::fma(std::abs(surface_geometric_normal.y), std::abs(scaled.y),
                          std::abs(surface_geometric_normal.z) * std::abs(scaled.z)));
    constexpr auto rounding_factor = Scalar{4} * std::numeric_limits<Scalar>::epsilon();
    constexpr auto underflow_allowance = Scalar{4} * std::numeric_limits<Scalar>::denorm_min();
    const auto alignment_uncertainty =
        std::fma(rounding_factor, absolute_term_sum, underflow_allowance);
    if (!std::isfinite(absolute_alignment) || !std::isfinite(absolute_term_sum) ||
        !std::isfinite(alignment_uncertainty) || absolute_alignment <= alignment_uncertainty) {
        return std::unexpected(
            invalid_light("Light PDF conversion Jacobian is singular or numerically ambiguous."));
    }

    return PdfJacobianFactors<Scalar>{
        .distance_scale = distance_scale,
        .scaled_distance_cubed = scaled_distance_cubed,
        .scaled_normal_alignment = absolute_alignment,
    };
}

template <SpectrumScalar Scalar> struct BinaryFactor final {
    Scalar fraction;
    int exponent;
};

template <SpectrumScalar Scalar>
[[nodiscard]] BinaryFactor<Scalar> binary_factor(const Scalar value) noexcept {
    auto exponent = 0;
    const auto fraction = std::frexp(value, &exponent);
    return BinaryFactor<Scalar>{.fraction = fraction, .exponent = exponent};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar> converted_pdf_value(const Scalar source_value,
                                                       const PdfJacobianFactors<Scalar>& factors,
                                                       const bool area_to_solid_angle) {
    if (source_value == Scalar{0}) {
        return Scalar{0};
    }

    const auto source = binary_factor(source_value);
    const auto scale = binary_factor(factors.distance_scale);
    const auto distance = binary_factor(factors.scaled_distance_cubed);
    const auto alignment = binary_factor(factors.scaled_normal_alignment);

    auto combined_fraction = Scalar{};
    auto combined_exponent = 0;
    if (area_to_solid_angle) {
        combined_fraction = source.fraction * scale.fraction * scale.fraction * distance.fraction /
                            alignment.fraction;
        combined_exponent =
            source.exponent + 2 * scale.exponent + distance.exponent - alignment.exponent;
    } else {
        combined_fraction = source.fraction * alignment.fraction /
                            (scale.fraction * scale.fraction * distance.fraction);
        combined_exponent =
            source.exponent + alignment.exponent - 2 * scale.exponent - distance.exponent;
    }

    auto normalization_exponent = 0;
    const auto normalized_fraction = std::frexp(combined_fraction, &normalization_exponent);
    const auto converted =
        std::scalbn(normalized_fraction, combined_exponent + normalization_exponent);
    if (!std::isfinite(converted) || !(converted > Scalar{0})) {
        return std::unexpected(
            invalid_light("Converted light PDF is not representable in the requested precision."));
    }
    return converted;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status validate_source_pdf(const LightProbabilityDensityT<Scalar> probability,
                                               const ProbabilityMeasure expected_measure) {
    if (probability.measure != expected_measure) {
        return std::unexpected(invalid_light(
            "Light PDF conversion source measure does not match the requested conversion."));
    }
    if (!std::isfinite(probability.value) || probability.value < Scalar{0}) {
        return std::unexpected(
            invalid_light("Light PDF conversion requires a finite non-negative density."));
    }
    return {};
}

} // namespace light_detail

// The result of pdf_li is always a density with respect to solid angle at the
// query point. Delta lights return exactly zero because their sampling
// probability is discrete and therefore has no solid-angle density.
template <SpectrumScalar Scalar> class DirectionalLightPdfT final {
  public:
    [[nodiscard]] static core::Result<DirectionalLightPdfT> create(const Scalar value) {
        if (!std::isfinite(value) || value < Scalar{0}) {
            return std::unexpected(light_detail::invalid_light(
                "A directional light PDF requires a finite non-negative solid-angle density."));
        }
        return DirectionalLightPdfT{value == Scalar{0} ? Scalar{0} : value};
    }

    [[nodiscard]] constexpr Scalar value() const noexcept {
        return value_;
    }

    [[nodiscard]] static constexpr ProbabilityMeasure measure() noexcept {
        return ProbabilityMeasure::solid_angle;
    }

    [[nodiscard]] constexpr LightProbabilityDensityT<Scalar> probability_density() const noexcept {
        return LightProbabilityDensityT<Scalar>{
            .value = value_,
            .measure = ProbabilityMeasure::solid_angle,
        };
    }

    [[nodiscard]] constexpr bool operator==(const DirectionalLightPdfT&) const noexcept = default;

  private:
    constexpr explicit DirectionalLightPdfT(const Scalar value) noexcept : value_{value} {}

    Scalar value_;
};

using DirectionalLightPdf = DirectionalLightPdfT<TransportScalar>;
using ReferenceDirectionalLightPdf = DirectionalLightPdfT<ReferenceScalar>;

enum class LightEndpointKind : std::uint8_t {
    finite_point,
    finite_surface,
    infinite,
};

// A sampled light endpoint retains the data needed to build a future robust
// visibility segment. Point emitters carry a position and absolute error;
// surface emitters additionally carry their geometric normal; directional and
// environment lights use the explicit infinite kind.
template <SpectrumScalar Scalar> class LightSampleEndpointT final {
  public:
    [[nodiscard]] static core::Result<LightSampleEndpointT>
    create_point(const Point3T<Scalar> position, const Vector3T<Scalar> absolute_position_error) {
        if (!light_detail::finite(position)) {
            return std::unexpected(light_detail::invalid_light(
                "A finite light endpoint requires a finite world-space position."));
        }
        if (!light_detail::finite_non_negative(absolute_position_error)) {
            return std::unexpected(light_detail::invalid_light(
                "A finite light endpoint requires finite non-negative position error."));
        }
        return LightSampleEndpointT{LightEndpointKind::finite_point, position,
                                    absolute_position_error, Normal3T<Scalar>{}};
    }

    [[nodiscard]] static core::Result<LightSampleEndpointT>
    create_surface(const Point3T<Scalar> position, const Vector3T<Scalar> absolute_position_error,
                   const Normal3T<Scalar> geometric_normal) {
        if (!light_detail::finite(position)) {
            return std::unexpected(light_detail::invalid_light(
                "A finite light endpoint requires a finite world-space position."));
        }
        if (!light_detail::finite_non_negative(absolute_position_error)) {
            return std::unexpected(light_detail::invalid_light(
                "A finite light endpoint requires finite non-negative position error."));
        }
        if (!light_detail::unit_normal(geometric_normal)) {
            return std::unexpected(light_detail::invalid_light(
                "A surface light endpoint requires a finite unit geometric normal."));
        }
        return LightSampleEndpointT{LightEndpointKind::finite_surface, position,
                                    absolute_position_error, geometric_normal};
    }

    [[nodiscard]] static constexpr LightSampleEndpointT infinite() noexcept {
        return LightSampleEndpointT{LightEndpointKind::infinite, Point3T<Scalar>{},
                                    Vector3T<Scalar>{}, Normal3T<Scalar>{}};
    }

    [[nodiscard]] constexpr LightEndpointKind kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] constexpr bool is_finite() const noexcept {
        return kind_ != LightEndpointKind::infinite;
    }

    [[nodiscard]] constexpr bool is_surface() const noexcept {
        return kind_ == LightEndpointKind::finite_surface;
    }

    [[nodiscard]] constexpr std::optional<Point3T<Scalar>> position() const noexcept {
        if (!is_finite()) {
            return std::nullopt;
        }
        return position_;
    }

    [[nodiscard]] constexpr std::optional<Vector3T<Scalar>>
    absolute_position_error() const noexcept {
        if (!is_finite()) {
            return std::nullopt;
        }
        return absolute_position_error_;
    }

    [[nodiscard]] constexpr std::optional<Normal3T<Scalar>> geometric_normal() const noexcept {
        if (!is_surface()) {
            return std::nullopt;
        }
        return geometric_normal_;
    }

    [[nodiscard]] constexpr bool operator==(const LightSampleEndpointT&) const noexcept = default;

  private:
    constexpr LightSampleEndpointT(const LightEndpointKind kind, const Point3T<Scalar> position,
                                   const Vector3T<Scalar> absolute_position_error,
                                   const Normal3T<Scalar> geometric_normal) noexcept
        : kind_{kind}, position_{position}, absolute_position_error_{absolute_position_error},
          geometric_normal_{geometric_normal} {}

    LightEndpointKind kind_;
    Point3T<Scalar> position_;
    Vector3T<Scalar> absolute_position_error_;
    Normal3T<Scalar> geometric_normal_;
};

using LightSampleEndpoint = LightSampleEndpointT<TransportScalar>;
using ReferenceLightSampleEndpoint = LightSampleEndpointT<ReferenceScalar>;

// A light query is evaluated at one world-space point and one explicit frame
// time. It intentionally does not contain a surface normal, so the same
// contract can later serve both surface and volume vertices.
template <SpectrumScalar Scalar> class LightSampleContextT final {
  public:
    [[nodiscard]] static core::Result<LightSampleContextT> create(const Point3T<Scalar> position,
                                                                  const Scalar time) {
        if (!light_detail::finite(position)) {
            return std::unexpected(light_detail::invalid_light(
                "A light sample context requires a finite world-space position."));
        }
        if (!std::isfinite(time)) {
            return std::unexpected(
                light_detail::invalid_light("A light sample context requires a finite time."));
        }
        return LightSampleContextT{position, time};
    }

    [[nodiscard]] constexpr const Point3T<Scalar>& position() const noexcept {
        return position_;
    }

    [[nodiscard]] constexpr Scalar time() const noexcept {
        return time_;
    }

    [[nodiscard]] constexpr bool operator==(const LightSampleContextT&) const noexcept = default;

  private:
    constexpr LightSampleContextT(const Point3T<Scalar> position, const Scalar time) noexcept
        : position_{position}, time_{time} {}

    Point3T<Scalar> position_;
    Scalar time_;
};

using LightSampleContext = LightSampleContextT<TransportScalar>;
using ReferenceLightSampleContext = LightSampleContextT<ReferenceScalar>;

// Incident radiance and its conditional sampling density. direction_to_light
// is a world-space unit vector from the query point toward the light.
// distance is finite for local lights and +infinity for directional or
// environment lights. The probability excludes the future LightSampler's
// discrete light-selection probability.
template <SpectrumScalar Scalar> class IncidentLightSampleT final {
  public:
    using spectrum_type = LightSpectrumT<Scalar>;
    using probability_density_type = LightProbabilityDensityT<Scalar>;

    [[nodiscard]] static core::Result<IncidentLightSampleT> create_finite(
        const LightSampleContextT<Scalar>& context, const LightSampleEndpointT<Scalar> endpoint,
        const spectrum_type incident_radiance, const probability_density_type probability) {
        if (!endpoint.is_finite()) {
            return std::unexpected(light_detail::invalid_light(
                "A finite incident light sample requires a finite endpoint."));
        }
        const auto endpoint_position = endpoint.position();
        if (!endpoint_position) {
            return std::unexpected(light_detail::invalid_light(
                "A finite incident light sample endpoint is internally inconsistent."));
        }

        const auto displacement = *endpoint_position - context.position();
        if (!std::isfinite(displacement.x) || !std::isfinite(displacement.y) ||
            !std::isfinite(displacement.z)) {
            return std::unexpected(light_detail::invalid_light(
                "Finite incident light sample separation is not representable in the requested "
                "precision."));
        }
        const auto distance_scale = std::max(
            {std::abs(displacement.x), std::abs(displacement.y), std::abs(displacement.z)});
        if (!(distance_scale > Scalar{0}) || !std::isfinite(distance_scale)) {
            return std::unexpected(light_detail::invalid_light(
                "A finite incident light sample requires distinct context and endpoint "
                "positions."));
        }

        const auto scaled = displacement / distance_scale;
        const auto scaled_squared_length =
            std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z));
        const auto scaled_length = std::sqrt(scaled_squared_length);
        const auto distance = distance_scale * scaled_length;
        if (!std::isfinite(scaled_length) || !(scaled_length > Scalar{0}) ||
            !std::isfinite(distance) || !(distance > Scalar{0})) {
            return std::unexpected(light_detail::invalid_light(
                "Finite incident light sample distance is not representable in the requested "
                "precision."));
        }

        const auto direction_to_light = scaled / scaled_length;
        return create_validated(endpoint, direction_to_light, distance, incident_radiance,
                                probability);
    }

    [[nodiscard]] static core::Result<IncidentLightSampleT>
    create_infinite(const Vector3T<Scalar> direction_to_light,
                    const spectrum_type incident_radiance,
                    const probability_density_type probability) {
        return create_validated(LightSampleEndpointT<Scalar>::infinite(), direction_to_light,
                                std::numeric_limits<Scalar>::infinity(), incident_radiance,
                                probability);
    }

    [[nodiscard]] constexpr const LightSampleEndpointT<Scalar>& endpoint() const noexcept {
        return endpoint_;
    }

    [[nodiscard]] constexpr const Vector3T<Scalar>& direction_to_light() const noexcept {
        return direction_to_light_;
    }

    [[nodiscard]] constexpr Scalar distance() const noexcept {
        return distance_;
    }

    [[nodiscard]] constexpr const spectrum_type& incident_radiance() const noexcept {
        return incident_radiance_;
    }

    [[nodiscard]] constexpr const probability_density_type& probability() const noexcept {
        return probability_;
    }

    [[nodiscard]] constexpr bool operator==(const IncidentLightSampleT& other) const noexcept {
        return endpoint_ == other.endpoint_ && direction_to_light_ == other.direction_to_light_ &&
               distance_ == other.distance_ && incident_radiance_ == other.incident_radiance_ &&
               probability_.value == other.probability_.value &&
               probability_.measure == other.probability_.measure;
    }

  private:
    [[nodiscard]] static core::Result<IncidentLightSampleT>
    create_validated(const LightSampleEndpointT<Scalar> endpoint,
                     const Vector3T<Scalar> direction_to_light, const Scalar distance,
                     const spectrum_type incident_radiance,
                     const probability_density_type probability) {
        if (!light_detail::unit_direction(direction_to_light)) {
            return std::unexpected(light_detail::invalid_light(
                "An incident light sample requires a finite unit direction."));
        }
        if (!light_detail::finite_non_negative(incident_radiance)) {
            return std::unexpected(light_detail::invalid_light(
                "Incident light radiance requires finite non-negative spectral lanes."));
        }
        const auto solid_angle_density = probability.measure == ProbabilityMeasure::solid_angle &&
                                         std::isfinite(probability.value) &&
                                         probability.value > Scalar{0};
        const auto discrete_probability = probability.measure == ProbabilityMeasure::discrete &&
                                          std::isfinite(probability.value) &&
                                          probability.value > Scalar{0} &&
                                          probability.value <= Scalar{1};
        if (!solid_angle_density && !discrete_probability) {
            return std::unexpected(light_detail::invalid_light(
                "An incident light sample requires a finite positive solid-angle density or a "
                "discrete probability in (0, 1]."));
        }
        if (endpoint.kind() == LightEndpointKind::finite_point && !discrete_probability) {
            return std::unexpected(light_detail::invalid_light(
                "A finite point light sample requires a discrete probability."));
        }
        if (endpoint.kind() == LightEndpointKind::finite_surface && !solid_angle_density) {
            return std::unexpected(light_detail::invalid_light(
                "A finite surface light sample requires a solid-angle density."));
        }
        return IncidentLightSampleT{endpoint, direction_to_light, distance, incident_radiance,
                                    probability};
    }

    constexpr IncidentLightSampleT(const LightSampleEndpointT<Scalar> endpoint,
                                   const Vector3T<Scalar> direction_to_light, const Scalar distance,
                                   const spectrum_type incident_radiance,
                                   const probability_density_type probability) noexcept
        : endpoint_{endpoint}, direction_to_light_{direction_to_light}, distance_{distance},
          incident_radiance_{incident_radiance}, probability_{probability} {}

    LightSampleEndpointT<Scalar> endpoint_;
    Vector3T<Scalar> direction_to_light_;
    Scalar distance_;
    spectrum_type incident_radiance_;
    probability_density_type probability_;
};

using IncidentLightSample = IncidentLightSampleT<TransportScalar>;
using ReferenceIncidentLightSample = IncidentLightSampleT<ReferenceScalar>;

// Individual light implementations satisfy this static contract. It is not a
// sixth runtime transport interface: the reserved LightSampler remains the
// component that will select heterogeneous lights.
//
// sample_li consumes a canonical sample in [0, 1)^2 and returns Li with a
// conditional solid-angle density for its explicit endpoint, or a discrete
// probability for a delta light. A successful empty result is reserved for
// valid samples outside the light's support or with zero contribution; an
// unavailable implementation must return an explicit error. pdf_li consumes a
// world-space unit direction and returns the model's conditional solid-angle
// density in that direction. Finite surface lights resolve their closest
// geometric surface along the direction; any model may return zero outside its
// directional support. A delta light reports exactly zero in solid-angle
// measure. Both operations receive the path's
// validated four-wavelength packet. le
// evaluates only radiance carried by an escaped ray; hit-surface emission
// remains owned by OneSidedSurfaceEmission. Returned radiance is evaluated at
// the packet wavelengths and is not divided by their wavelength PDFs. power
// returns four-lane spectral radiant flux for explicit finite scene bounds.
// bounds returns conservative world-space support, using unbounded bounds for
// directional and environment lights. Implementations must report invalid
// inputs and unavailable operations through core::Error; black, zero density,
// empty sampling, and unbounded support are valid only when they describe the
// physical light rather than substituting for a missing capability.
template <typename Model, typename Scalar>
concept LightModelFor =
    SpectrumScalar<Scalar> &&
    requires(const Model& light, const LightSampleContextT<Scalar>& context,
             const Point2T<Scalar> canonical_sample, const Vector3T<Scalar> direction_to_light,
             const RayT<Scalar>& escaped_ray, const SampledWavelengthsT<Scalar>& wavelengths,
             const Bounds3T<Scalar>& scene_bounds) {
        {
            light.sample_li(context, canonical_sample, wavelengths)
        } -> std::same_as<core::Result<std::optional<IncidentLightSampleT<Scalar>>>>;
        {
            light.pdf_li(context, direction_to_light, wavelengths)
        } -> std::same_as<core::Result<DirectionalLightPdfT<Scalar>>>;
        {
            light.le(escaped_ray, wavelengths)
        } -> std::same_as<core::Result<LightSpectrumT<Scalar>>>;
        {
            light.power(scene_bounds, wavelengths)
        } -> std::same_as<core::Result<LightSpectrumT<Scalar>>>;
        { light.bounds() } -> std::same_as<Bounds3T<Scalar>>;
    };

// Converts an area density on a sampled surface to the corresponding
// incident-direction density. The geometric Jacobian is
// p_omega = p_area * r^2 / abs(dot(Ng, -omega)).
// The absolute cosine is only the two-sided change of measure: a one-sided
// emitter must reject its non-emitting side before this conversion.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LightProbabilityDensityT<Scalar>>
convert_area_pdf_to_solid_angle(const LightProbabilityDensityT<Scalar> area_probability,
                                const Point3T<Scalar> reference_position,
                                const Point3T<Scalar> surface_position,
                                const Normal3T<Scalar> surface_geometric_normal) {
    const auto source_status =
        light_detail::validate_source_pdf<Scalar>(area_probability, ProbabilityMeasure::area);
    if (!source_status) {
        return std::unexpected(source_status.error());
    }
    const auto factors = light_detail::pdf_jacobian_factors(reference_position, surface_position,
                                                            surface_geometric_normal);
    if (!factors) {
        return std::unexpected(factors.error());
    }
    const auto converted =
        light_detail::converted_pdf_value(area_probability.value, *factors, true);
    if (!converted) {
        return std::unexpected(converted.error());
    }
    return LightProbabilityDensityT<Scalar>{
        .value = *converted,
        .measure = ProbabilityMeasure::solid_angle,
    };
}

// Converts an incident-direction density back to its area density. A tangent
// configuration is rejected in both directions because its Jacobian is not
// invertible, even though the formal product in this direction would be zero.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<LightProbabilityDensityT<Scalar>>
convert_solid_angle_pdf_to_area(const LightProbabilityDensityT<Scalar> solid_angle_probability,
                                const Point3T<Scalar> reference_position,
                                const Point3T<Scalar> surface_position,
                                const Normal3T<Scalar> surface_geometric_normal) {
    const auto source_status = light_detail::validate_source_pdf<Scalar>(
        solid_angle_probability, ProbabilityMeasure::solid_angle);
    if (!source_status) {
        return std::unexpected(source_status.error());
    }
    const auto factors = light_detail::pdf_jacobian_factors(reference_position, surface_position,
                                                            surface_geometric_normal);
    if (!factors) {
        return std::unexpected(factors.error());
    }
    const auto converted =
        light_detail::converted_pdf_value(solid_angle_probability.value, *factors, false);
    if (!converted) {
        return std::unexpected(converted.error());
    }
    return LightProbabilityDensityT<Scalar>{
        .value = *converted,
        .measure = ProbabilityMeasure::area,
    };
}

static_assert(std::is_standard_layout_v<LightSampleContext>);
static_assert(std::is_trivially_copyable_v<LightSampleContext>);
static_assert(std::is_standard_layout_v<ReferenceLightSampleContext>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSampleContext>);
static_assert(std::is_standard_layout_v<DirectionalLightPdf>);
static_assert(std::is_trivially_copyable_v<DirectionalLightPdf>);
static_assert(std::is_standard_layout_v<ReferenceDirectionalLightPdf>);
static_assert(std::is_trivially_copyable_v<ReferenceDirectionalLightPdf>);
static_assert(std::is_standard_layout_v<LightSampleEndpoint>);
static_assert(std::is_trivially_copyable_v<LightSampleEndpoint>);
static_assert(std::is_standard_layout_v<ReferenceLightSampleEndpoint>);
static_assert(std::is_trivially_copyable_v<ReferenceLightSampleEndpoint>);
static_assert(std::is_standard_layout_v<IncidentLightSample>);
static_assert(std::is_trivially_copyable_v<IncidentLightSample>);
static_assert(std::is_standard_layout_v<ReferenceIncidentLightSample>);
static_assert(std::is_trivially_copyable_v<ReferenceIncidentLightSample>);

} // namespace blackframe::renderer
