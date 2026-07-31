#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/RayOriginOffset.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace blackframe::renderer {
namespace shadow_ray_detail {

[[nodiscard]] inline core::Error invalid_shadow_ray(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <SpectrumScalar Scalar> [[nodiscard]] bool finite(const Point3T<Scalar> point) noexcept {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Point3T<Scalar>>
offset_surface_point(const Point3T<Scalar> point, const Vector3T<Scalar> absolute_position_error,
                     const Normal3T<Scalar> geometric_normal,
                     const Vector3T<Scalar> outgoing_direction) {
    auto offset =
        offset_ray_origin(point, absolute_position_error, geometric_normal, outgoing_direction);
    if (!offset) {
        return std::unexpected(offset.error());
    }
    if (*offset != point) {
        return *offset;
    }

    // A valid zero error bound still needs an endpoint-exclusive surface
    // segment. Move exactly one representable value along the oriented
    // geometric normal instead of introducing a scene-scale epsilon.
    auto oriented_normal = geometric_normal;
    if (dot(geometric_normal, outgoing_direction) < Scalar{0}) {
        oriented_normal = -oriented_normal;
    }
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    auto shifted = point;
    if (oriented_normal.x != Scalar{0}) {
        shifted.x = std::nextafter(point.x, oriented_normal.x > Scalar{0} ? infinity : -infinity);
    }
    if (oriented_normal.y != Scalar{0}) {
        shifted.y = std::nextafter(point.y, oriented_normal.y > Scalar{0} ? infinity : -infinity);
    }
    if (oriented_normal.z != Scalar{0}) {
        shifted.z = std::nextafter(point.z, oriented_normal.z > Scalar{0} ? infinity : -infinity);
    }
    if (!finite(shifted) || shifted == point) {
        return std::unexpected(invalid_shadow_ray(
            "A shadow-ray surface offset is not representable in the requested precision."));
    }
    return shifted;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Point3T<Scalar>>
contract_point_endpoint(const Point3T<Scalar> point, const Vector3T<Scalar> absolute_position_error,
                        const Vector3T<Scalar> direction_to_light) {
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    auto contraction = Scalar{0};
    const auto accumulate_support = [&](const Scalar direction, const Scalar error) {
        const auto candidate = std::fma(std::abs(direction), error, contraction);
        if (!std::isfinite(candidate) || candidate < contraction) {
            return false;
        }
        if (direction != Scalar{0} && error != Scalar{0}) {
            contraction = std::nextafter(candidate, infinity);
            return std::isfinite(contraction);
        }
        contraction = candidate;
        return true;
    };
    if (!accumulate_support(direction_to_light.x, absolute_position_error.x) ||
        !accumulate_support(direction_to_light.y, absolute_position_error.y) ||
        !accumulate_support(direction_to_light.z, absolute_position_error.z)) {
        return std::unexpected(
            invalid_shadow_ray("A shadow-ray point-endpoint contraction is not representable."));
    }
    if (contraction == Scalar{0}) {
        return point;
    }

    auto contracted = Point3T<Scalar>{
        .x = std::fma(-direction_to_light.x, contraction, point.x),
        .y = std::fma(-direction_to_light.y, contraction, point.y),
        .z = std::fma(-direction_to_light.z, contraction, point.z),
    };
    if (direction_to_light.x != Scalar{0}) {
        contracted.x =
            std::nextafter(contracted.x, direction_to_light.x > Scalar{0} ? -infinity : infinity);
    }
    if (direction_to_light.y != Scalar{0}) {
        contracted.y =
            std::nextafter(contracted.y, direction_to_light.y > Scalar{0} ? -infinity : infinity);
    }
    if (direction_to_light.z != Scalar{0}) {
        contracted.z =
            std::nextafter(contracted.z, direction_to_light.z > Scalar{0} ? -infinity : infinity);
    }
    if (!finite(contracted)) {
        return std::unexpected(
            invalid_shadow_ray("A shadow-ray point endpoint exceeds the finite coordinate range."));
    }
    return contracted;
}

template <SpectrumScalar Scalar> struct UnitSegment final {
    Vector3T<Scalar> direction;
    Scalar length;
};

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<UnitSegment<Scalar>>
unit_segment(const Point3T<Scalar> origin, const Point3T<Scalar> endpoint,
             const Vector3T<Scalar> expected_direction) {
    const auto displacement = endpoint - origin;
    if (!std::isfinite(displacement.x) || !std::isfinite(displacement.y) ||
        !std::isfinite(displacement.z)) {
        return std::unexpected(invalid_shadow_ray(
            "A finite shadow-ray segment is not representable in the requested precision."));
    }
    const auto scale =
        std::max({std::abs(displacement.x), std::abs(displacement.y), std::abs(displacement.z)});
    if (!(scale > Scalar{0}) || !std::isfinite(scale)) {
        return std::unexpected(
            invalid_shadow_ray("A finite shadow-ray segment must have positive length."));
    }

    const auto scaled = displacement / scale;
    const auto scaled_squared_length =
        std::fma(scaled.x, scaled.x, std::fma(scaled.y, scaled.y, scaled.z * scaled.z));
    const auto scaled_length = std::sqrt(scaled_squared_length);
    const auto length = scale * scaled_length;
    if (!(scaled_length > Scalar{0}) || !std::isfinite(scaled_length) || !(length > Scalar{0}) ||
        !std::isfinite(length)) {
        return std::unexpected(invalid_shadow_ray(
            "A finite shadow-ray length is not representable in the requested precision."));
    }
    const auto direction = scaled / scaled_length;
    const auto alignment = dot(direction, expected_direction);
    if (!std::isfinite(alignment) || !(alignment > Scalar{0})) {
        return std::unexpected(invalid_shadow_ray(
            "Shadow-ray endpoint contraction reversed the sampled light direction."));
    }
    return UnitSegment<Scalar>{.direction = direction, .length = length};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
validate_finite_sample_source(const SurfaceInteractionT<Scalar>& source,
                              const IncidentLightSampleT<Scalar>& sample) {
    const auto context = LightSampleContextT<Scalar>::create(source.position(), source.time());
    if (!context) {
        return std::unexpected(context.error());
    }
    const auto rebuilt = IncidentLightSampleT<Scalar>::create_finite(
        *context, sample.endpoint(), sample.incident_radiance(), sample.probability());
    if (!rebuilt) {
        return std::unexpected(invalid_shadow_ray(
            "A finite incident-light sample is inconsistent with the shadow-ray source."));
    }
    if (rebuilt->direction_to_light() != sample.direction_to_light() ||
        rebuilt->distance() != sample.distance()) {
        return std::unexpected(invalid_shadow_ray(
            "A finite incident-light sample was evaluated at a different source point."));
    }
    return {};
}

} // namespace shadow_ray_detail

// Constructs an opaque visibility ray from a validated surface interaction to
// one sampled light endpoint. Finite endpoints are excluded from the closed
// ray interval by robustly contracting both endpoint errors and moving tMax by
// one representable value. The function neither evaluates occlusion nor
// assumes vacuum transmittance.
template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>>
make_shadow_ray(const SurfaceInteractionT<Scalar>& source,
                const Vector3T<Scalar> source_absolute_position_error,
                const IncidentLightSampleT<Scalar>& light_sample, const RayMask visibility_mask,
                const MediumId current_medium) {
    if (source.time() < Scalar{0} || source.time() > Scalar{1}) {
        return std::unexpected(shadow_ray_detail::invalid_shadow_ray(
            "A shadow-ray source time must lie in the normalized [0, 1] range."));
    }

    const auto origin = shadow_ray_detail::offset_surface_point(
        source.position(), source_absolute_position_error, source.geometric_normal(),
        light_sample.direction_to_light());
    if (!origin) {
        return std::unexpected(origin.error());
    }

    if (!light_sample.endpoint().is_finite()) {
        return RayT<Scalar>::create(*origin, light_sample.direction_to_light(), Scalar{0},
                                    std::numeric_limits<Scalar>::infinity(), source.time(),
                                    visibility_mask, current_medium);
    }

    const auto source_validation =
        shadow_ray_detail::validate_finite_sample_source(source, light_sample);
    if (!source_validation) {
        return std::unexpected(source_validation.error());
    }
    const auto endpoint_position = light_sample.endpoint().position();
    const auto endpoint_error = light_sample.endpoint().absolute_position_error();
    if (!endpoint_position || !endpoint_error) {
        return std::unexpected(shadow_ray_detail::invalid_shadow_ray(
            "A finite light endpoint is missing its visibility data."));
    }

    auto endpoint = core::Result<Point3T<Scalar>>{*endpoint_position};
    if (light_sample.endpoint().is_surface()) {
        const auto endpoint_normal = light_sample.endpoint().geometric_normal();
        if (!endpoint_normal) {
            return std::unexpected(shadow_ray_detail::invalid_shadow_ray(
                "A surface light endpoint is missing its geometric normal."));
        }
        endpoint = shadow_ray_detail::offset_surface_point(
            *endpoint_position, *endpoint_error, *endpoint_normal, *origin - *endpoint_position);
    } else {
        const auto nominal_segment = shadow_ray_detail::unit_segment(
            *origin, *endpoint_position, light_sample.direction_to_light());
        if (!nominal_segment) {
            return std::unexpected(nominal_segment.error());
        }
        endpoint = shadow_ray_detail::contract_point_endpoint(*endpoint_position, *endpoint_error,
                                                              nominal_segment->direction);
    }
    if (!endpoint) {
        return std::unexpected(endpoint.error());
    }

    const auto segment =
        shadow_ray_detail::unit_segment(*origin, *endpoint, light_sample.direction_to_light());
    if (!segment) {
        return std::unexpected(segment.error());
    }
    const auto t_max = std::nextafter(segment->length, Scalar{0});
    if (!(t_max > Scalar{0}) || !std::isfinite(t_max)) {
        return std::unexpected(shadow_ray_detail::invalid_shadow_ray(
            "A finite shadow-ray interval is not representable in the requested precision."));
    }
    return RayT<Scalar>::create(*origin, segment->direction, Scalar{0}, t_max, source.time(),
                                visibility_mask, current_medium);
}

} // namespace blackframe::renderer
