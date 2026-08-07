#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/Plane.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <Blackframe/Renderer/TextureCoordinateDifferentials.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <GeometryScalar Scalar> class RayDifferentialT;

namespace ray_differential_detail {

[[nodiscard]] inline core::Error invalid_ray_differential(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Point3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Vector3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Normal3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] bool non_zero(const Vector3T<Scalar> value) noexcept {
    return value.x != Scalar{0} || value.y != Scalar{0} || value.z != Scalar{0};
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_normal(const Normal3T<Scalar> value) noexcept {
    if (!finite(value)) {
        return false;
    }
    const auto squared_length =
        std::fma(value.x, value.x, std::fma(value.y, value.y, value.z * value.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_direction(const Vector3T<Scalar> value) noexcept {
    if (!finite(value)) {
        return false;
    }
    const auto squared_length =
        std::fma(value.x, value.x, std::fma(value.y, value.y, value.z * value.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    return std::isfinite(squared_length) && std::abs(squared_length - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> robust_unit_direction(const Vector3T<Scalar> direction,
                                                                   const char* const message) {
    if (!finite(direction)) {
        return std::unexpected(invalid_ray_differential(message));
    }
    const auto maximum_component =
        std::max({std::abs(direction.x), std::abs(direction.y), std::abs(direction.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(invalid_ray_differential(message));
    }
    const auto scaled = direction / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    const auto result = scaled / magnitude;
    if (!std::isfinite(magnitude) || !(magnitude > Scalar{0}) || !finite(result)) {
        return std::unexpected(invalid_ray_differential(message));
    }
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Point3T<Scalar>>
intersect_tangent_plane(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
                        const Point3T<Scalar> surface_position,
                        const Normal3T<Scalar> geometric_normal) {
    const auto relevant_component = [](const Scalar component,
                                       const Scalar normal_component) noexcept {
        return normal_component == Scalar{0} ? Scalar{0} : component;
    };
    const auto surface_coordinates = Vector3T<Scalar>{
        .x = relevant_component(surface_position.x, geometric_normal.x),
        .y = relevant_component(surface_position.y, geometric_normal.y),
        .z = relevant_component(surface_position.z, geometric_normal.z),
    };
    const auto origin_coordinates = Vector3T<Scalar>{
        .x = relevant_component(origin.x, geometric_normal.x),
        .y = relevant_component(origin.y, geometric_normal.y),
        .z = relevant_component(origin.z, geometric_normal.z),
    };
    const auto direction_coordinates = Vector3T<Scalar>{
        .x = relevant_component(direction.x, geometric_normal.x),
        .y = relevant_component(direction.y, geometric_normal.y),
        .z = relevant_component(direction.z, geometric_normal.z),
    };
    const auto spatial_maximum =
        std::max({std::abs(surface_coordinates.x), std::abs(surface_coordinates.y),
                  std::abs(surface_coordinates.z), std::abs(origin_coordinates.x),
                  std::abs(origin_coordinates.y), std::abs(origin_coordinates.z)});
    const auto direction_maximum =
        std::max({std::abs(direction_coordinates.x), std::abs(direction_coordinates.y),
                  std::abs(direction_coordinates.z)});
    auto spatial_exponent = 0;
    auto direction_exponent = 0;
    static_cast<void>(std::frexp(spatial_maximum, &spatial_exponent));
    static_cast<void>(std::frexp(direction_maximum, &direction_exponent));
    const auto scaled_surface = Vector3T<Scalar>{
        .x = std::ldexp(surface_coordinates.x, -spatial_exponent),
        .y = std::ldexp(surface_coordinates.y, -spatial_exponent),
        .z = std::ldexp(surface_coordinates.z, -spatial_exponent),
    };
    const auto scaled_origin = Vector3T<Scalar>{
        .x = std::ldexp(origin_coordinates.x, -spatial_exponent),
        .y = std::ldexp(origin_coordinates.y, -spatial_exponent),
        .z = std::ldexp(origin_coordinates.z, -spatial_exponent),
    };
    const auto scaled_direction = Vector3T<Scalar>{
        .x = std::ldexp(direction_coordinates.x, -direction_exponent),
        .y = std::ldexp(direction_coordinates.y, -direction_exponent),
        .z = std::ldexp(direction_coordinates.z, -direction_exponent),
    };
    if (!plane_detail::scaling_preserves(surface_coordinates.x, scaled_surface.x) ||
        !plane_detail::scaling_preserves(surface_coordinates.y, scaled_surface.y) ||
        !plane_detail::scaling_preserves(surface_coordinates.z, scaled_surface.z) ||
        !plane_detail::scaling_preserves(origin_coordinates.x, scaled_origin.x) ||
        !plane_detail::scaling_preserves(origin_coordinates.y, scaled_origin.y) ||
        !plane_detail::scaling_preserves(origin_coordinates.z, scaled_origin.z) ||
        !plane_detail::scaling_preserves(direction_coordinates.x, scaled_direction.x) ||
        !plane_detail::scaling_preserves(direction_coordinates.y, scaled_direction.y) ||
        !plane_detail::scaling_preserves(direction_coordinates.z, scaled_direction.z)) {
        return std::unexpected(invalid_ray_differential(
            "A ray differential tangent-plane scaling is not representable."));
    }

    const auto numerator = plane_detail::exact_plane_numerator_expansion(
        scaled_surface, scaled_origin, geometric_normal);
    const auto denominator = plane_detail::exact_dot_expansion(scaled_direction, geometric_normal);
    if (!numerator || !denominator) {
        return std::unexpected(invalid_ray_differential(
            "A ray differential tangent-plane equation is not exactly representable."));
    }
    if (plane_detail::expansion_sign(*denominator) == 0) {
        return std::unexpected(
            invalid_ray_differential("A ray differential is parallel to the tangent plane."));
    }
    const auto scaled_parameter =
        plane_detail::correctly_rounded_quotient(*numerator, *denominator);
    if (!scaled_parameter) {
        return std::unexpected(invalid_ray_differential(
            "A ray differential tangent-plane parameter is not representable."));
    }
    const auto parameter = std::ldexp(*scaled_parameter, spatial_exponent - direction_exponent);
    if (!std::isfinite(parameter) || (*scaled_parameter != Scalar{0} && parameter == Scalar{0})) {
        return std::unexpected(invalid_ray_differential(
            "A ray differential tangent-plane intersection is not representable."));
    }
    const auto point = Point3T<Scalar>{
        .x = std::fma(direction.x, parameter, origin.x),
        .y = std::fma(direction.y, parameter, origin.y),
        .z = std::fma(direction.z, parameter, origin.z),
    };
    if (!finite(point)) {
        return std::unexpected(invalid_ray_differential(
            "A ray differential tangent-plane intersection is not representable."));
    }
    return point;
}

template <GeometryScalar Scalar>
[[nodiscard]] std::array<Scalar, 2> projected_components(const Vector3T<Scalar> value,
                                                         const std::uint32_t dropped_axis) {
    switch (dropped_axis) {
    case 0U:
        return {value.y, value.z};
    case 1U:
        return {value.x, value.z};
    default:
        return {value.x, value.y};
    }
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> reflected_direction(const Vector3T<Scalar> incident,
                                                                 const Normal3T<Scalar> normal) {
    if (!unit_direction(incident) || !unit_normal(normal)) {
        return std::unexpected(invalid_ray_differential(
            "Specular reflection requires a finite direction and a finite unit normal."));
    }
    return robust_unit_direction(
        incident - Scalar{2} * dot(incident, normal) *
                       Vector3T<Scalar>{.x = normal.x, .y = normal.y, .z = normal.z},
        "A reflected ray differential direction is not representable.");
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<std::optional<Vector3T<Scalar>>>
refracted_direction(const Vector3T<Scalar> incident, const Normal3T<Scalar> shading_normal,
                    const bool entering, const Scalar exterior_eta, const Scalar interior_eta) {
    if (!unit_direction(incident) || !unit_normal(shading_normal) || !std::isfinite(exterior_eta) ||
        !(exterior_eta > Scalar{0}) || !std::isfinite(interior_eta) ||
        !(interior_eta > Scalar{0})) {
        return std::unexpected(invalid_ray_differential(
            "Specular transmission requires finite directions, unit normals, and positive "
            "indices."));
    }

    const auto incident_normal = entering ? shading_normal : -shading_normal;
    const auto normal_vector = Vector3T<Scalar>{
        .x = incident_normal.x,
        .y = incident_normal.y,
        .z = incident_normal.z,
    };
    const auto incident_cosine = -dot(incident, incident_normal);
    if (!std::isfinite(incident_cosine)) {
        return std::unexpected(invalid_ray_differential(
            "A transmitted ray differential incident cosine is not representable."));
    }
    if (!(incident_cosine > Scalar{0})) {
        return std::optional<Vector3T<Scalar>>{};
    }

    const auto incident_eta = entering ? exterior_eta : interior_eta;
    const auto transmitted_eta = entering ? interior_eta : exterior_eta;
    if (incident_eta == transmitted_eta) {
        return std::optional<Vector3T<Scalar>>{incident};
    }

    const auto incident_tangent = incident + incident_cosine * normal_vector;
    const auto raw_incident_sine =
        std::hypot(incident_tangent.x, incident_tangent.y, incident_tangent.z);
    if (!std::isfinite(raw_incident_sine)) {
        return std::unexpected(
            invalid_ray_differential("A transmitted ray differential sine is not representable."));
    }
    auto incident_sine = raw_incident_sine;
    auto tangent_normalization = Scalar{1};
    if (raw_incident_sine > Scalar{1}) {
        if (incident_cosine > Scalar{1}) {
            return std::unexpected(invalid_ray_differential(
                "A transmitted ray differential incident direction is inconsistent."));
        }
        incident_sine = std::sqrt((Scalar{1} - incident_cosine) * (Scalar{1} + incident_cosine));
        tangent_normalization = incident_sine / raw_incident_sine;
        if (!std::isfinite(incident_sine) || !std::isfinite(tangent_normalization) ||
            !(tangent_normalization > Scalar{0})) {
            return std::unexpected(invalid_ray_differential(
                "A transmitted ray differential sine is not representable."));
        }
    }

    if (incident_sine > Scalar{0}) {
        const auto critical_sine = transmitted_eta / incident_eta;
        if (critical_sine <= Scalar{1} && incident_sine >= critical_sine) {
            return std::optional<Vector3T<Scalar>>{};
        }
    }

    const auto relative_eta = incident_eta / transmitted_eta;
    if (!std::isfinite(relative_eta) || !(relative_eta > Scalar{0})) {
        return std::unexpected(invalid_ray_differential(
            "A transmitted ray differential relative index is not representable."));
    }

    const auto tangent_scale = relative_eta * tangent_normalization;
    const auto tangent = tangent_scale * incident_tangent;
    if (!finite(tangent) || (incident_tangent.x != Scalar{0} && tangent.x == Scalar{0}) ||
        (incident_tangent.y != Scalar{0} && tangent.y == Scalar{0}) ||
        (incident_tangent.z != Scalar{0} && tangent.z == Scalar{0})) {
        return std::unexpected(invalid_ray_differential(
            "A transmitted ray differential tangent is not representable."));
    }
    const auto transmitted_sine = std::hypot(tangent.x, tangent.y, tangent.z);
    if (!std::isfinite(transmitted_sine)) {
        return std::unexpected(
            invalid_ray_differential("A transmitted ray differential sine is not representable."));
    }
    if (transmitted_sine >= Scalar{1}) {
        return std::unexpected(invalid_ray_differential(
            "A subcritical transmitted ray differential is not representable."));
    }
    const auto transmitted_cosine =
        std::sqrt((Scalar{1} - transmitted_sine) * (Scalar{1} + transmitted_sine));
    if (!std::isfinite(transmitted_cosine) || !(transmitted_cosine > Scalar{0})) {
        return std::unexpected(invalid_ray_differential(
            "A transmitted ray differential cosine is not representable."));
    }
    auto transmitted =
        robust_unit_direction(tangent - transmitted_cosine * normal_vector,
                              "A transmitted ray differential direction is not representable.");
    if (!transmitted) {
        return std::unexpected(transmitted.error());
    }
    return std::optional<Vector3T<Scalar>>{*transmitted};
}

template <GeometryScalar Scalar>
[[nodiscard]] bool same_unit_direction(const Vector3T<Scalar> left,
                                       const Vector3T<Scalar> right) noexcept {
    if (!unit_direction(left) || !unit_direction(right)) {
        return false;
    }
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{256};
    return std::abs(left.x - right.x) <= tolerance && std::abs(left.y - right.y) <= tolerance &&
           std::abs(left.z - right.z) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Status validate_next_specular_ray(const RayDifferentialT<Scalar>& incident,
                                                      const RayT<Scalar>& next_ray,
                                                      const Vector3T<Scalar> expected_direction) {
    if (next_ray.time() != incident.ray().time()) {
        return std::unexpected(invalid_ray_differential(
            "A propagated ray differential must preserve the path time sample."));
    }
    if (!same_unit_direction(next_ray.direction(), expected_direction)) {
        return std::unexpected(invalid_ray_differential(
            "A propagated ray differential requires the exact central specular direction."));
    }
    return {};
}

} // namespace ray_differential_detail

// RayT remains the only traversal and host/device ABI ray. This companion carries two analytic
// one-pixel neighbors for the scalar reference path; auxiliary directions are validated and kept
// unit length at camera and specular events, but are never submitted as visibility queries.
template <GeometryScalar Scalar> class RayDifferentialT final {
  public:
    [[nodiscard]] static core::Result<RayDifferentialT>
    create(const RayT<Scalar> ray, const Point3T<Scalar> rx_origin,
           const Vector3T<Scalar> rx_direction, const Point3T<Scalar> ry_origin,
           const Vector3T<Scalar> ry_direction) {
        if (!ray_differential_detail::finite(rx_origin) ||
            !ray_differential_detail::finite(ry_origin) ||
            !ray_differential_detail::finite(rx_direction) ||
            !ray_differential_detail::finite(ry_direction) ||
            !ray_differential_detail::non_zero(rx_direction) ||
            !ray_differential_detail::non_zero(ry_direction)) {
            return std::unexpected(ray_differential_detail::invalid_ray_differential(
                "A ray differential requires finite origins and finite non-zero directions."));
        }
        return RayDifferentialT{ray, rx_origin, rx_direction, ry_origin, ry_direction};
    }

    [[nodiscard]] constexpr const RayT<Scalar>& ray() const noexcept {
        return ray_;
    }
    [[nodiscard]] constexpr const Point3T<Scalar>& rx_origin() const noexcept {
        return rx_origin_;
    }
    [[nodiscard]] constexpr const Vector3T<Scalar>& rx_direction() const noexcept {
        return rx_direction_;
    }
    [[nodiscard]] constexpr const Point3T<Scalar>& ry_origin() const noexcept {
        return ry_origin_;
    }
    [[nodiscard]] constexpr const Vector3T<Scalar>& ry_direction() const noexcept {
        return ry_direction_;
    }

  private:
    constexpr RayDifferentialT(const RayT<Scalar> ray, const Point3T<Scalar> rx_origin,
                               const Vector3T<Scalar> rx_direction, const Point3T<Scalar> ry_origin,
                               const Vector3T<Scalar> ry_direction) noexcept
        : ray_{ray}, rx_origin_{rx_origin}, rx_direction_{rx_direction}, ry_origin_{ry_origin},
          ry_direction_{ry_direction} {}

    RayT<Scalar> ray_;
    Point3T<Scalar> rx_origin_;
    Vector3T<Scalar> rx_direction_;
    Point3T<Scalar> ry_origin_;
    Vector3T<Scalar> ry_direction_;
};

using RayDifferential = RayDifferentialT<TransportScalar>;
using ReferenceRayDifferential = RayDifferentialT<ReferenceScalar>;

template <GeometryScalar Scalar> struct SurfacePointDifferentialsT final {
    Vector3T<Scalar> dpdx;
    Vector3T<Scalar> dpdy;

    [[nodiscard]] constexpr bool
    operator==(const SurfacePointDifferentialsT&) const noexcept = default;
};

using SurfacePointDifferentials = SurfacePointDifferentialsT<TransportScalar>;
using ReferenceSurfacePointDifferentials = SurfacePointDifferentialsT<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<SurfacePointDifferentialsT<Scalar>>
surface_point_differentials(const RayDifferentialT<Scalar>& differential,
                            const SurfaceInteractionT<Scalar>& surface) {
    if (differential.ray().time() != surface.time()) {
        return std::unexpected(ray_differential_detail::invalid_ray_differential(
            "Ray and surface differentials must describe the same time sample."));
    }
    const auto px = ray_differential_detail::intersect_tangent_plane(
        differential.rx_origin(), differential.rx_direction(), surface.position(),
        surface.geometric_normal());
    if (!px) {
        return std::unexpected(px.error());
    }
    const auto py = ray_differential_detail::intersect_tangent_plane(
        differential.ry_origin(), differential.ry_direction(), surface.position(),
        surface.geometric_normal());
    if (!py) {
        return std::unexpected(py.error());
    }
    return SurfacePointDifferentialsT<Scalar>{
        .dpdx = *px - surface.position(),
        .dpdy = *py - surface.position(),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<TextureCoordinateDifferentialsT<Scalar>>
texture_coordinate_differentials(const SurfaceInteractionT<Scalar>& surface,
                                 const SurfacePointDifferentialsT<Scalar> differentials) {
    const auto normal = surface.geometric_normal();
    const auto absolute = std::array{std::abs(normal.x), std::abs(normal.y), std::abs(normal.z)};
    const auto dropped_axis = absolute[0] > absolute[1] ? (absolute[0] > absolute[2] ? 0U : 2U)
                                                        : (absolute[1] > absolute[2] ? 1U : 2U);
    const auto dpdu = ray_differential_detail::projected_components(surface.dpdu(), dropped_axis);
    const auto dpdv = ray_differential_detail::projected_components(surface.dpdv(), dropped_axis);
    const auto dpdx =
        ray_differential_detail::projected_components(differentials.dpdx, dropped_axis);
    const auto dpdy =
        ray_differential_detail::projected_components(differentials.dpdy, dropped_axis);
    const auto determinant = std::fma(dpdu[0], dpdv[1], -dpdu[1] * dpdv[0]);
    if (!std::isfinite(determinant) || determinant == Scalar{0}) {
        return std::unexpected(ray_differential_detail::invalid_ray_differential(
            "Texture-coordinate differentials require a non-singular surface UV Jacobian."));
    }
    const auto reciprocal = Scalar{1} / determinant;
    const auto solve = [&](const std::array<Scalar, 2> value) {
        return std::array{
            std::fma(value[0], dpdv[1], -value[1] * dpdv[0]) * reciprocal,
            std::fma(dpdu[0], value[1], -dpdu[1] * value[0]) * reciprocal,
        };
    };
    const auto x = solve(dpdx);
    const auto y = solve(dpdy);
    if (!std::isfinite(reciprocal) || !std::isfinite(x[0]) || !std::isfinite(x[1]) ||
        !std::isfinite(y[0]) || !std::isfinite(y[1])) {
        return std::unexpected(ray_differential_detail::invalid_ray_differential(
            "Texture-coordinate differentials are not representable."));
    }
    return TextureCoordinateDifferentialsT<Scalar>{
        .dudx = x[0],
        .dvdx = x[1],
        .dudy = y[0],
        .dvdy = y[1],
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayDifferentialT<Scalar>> propagate_specular_reflection(
    const RayDifferentialT<Scalar>& incident, const RayT<Scalar> reflected_ray,
    const Normal3T<Scalar> central_shading_normal, const Point3T<Scalar> rx_position,
    const Normal3T<Scalar> rx_shading_normal, const Point3T<Scalar> ry_position,
    const Normal3T<Scalar> ry_shading_normal) {
    const auto central_direction = ray_differential_detail::reflected_direction(
        incident.ray().direction(), central_shading_normal);
    if (!central_direction) {
        return std::unexpected(central_direction.error());
    }
    const auto central_status = ray_differential_detail::validate_next_specular_ray(
        incident, reflected_ray, *central_direction);
    if (!central_status) {
        return std::unexpected(central_status.error());
    }
    const auto rx_direction =
        ray_differential_detail::reflected_direction(incident.rx_direction(), rx_shading_normal);
    const auto ry_direction =
        ray_differential_detail::reflected_direction(incident.ry_direction(), ry_shading_normal);
    if (!rx_direction) {
        return std::unexpected(rx_direction.error());
    }
    if (!ry_direction) {
        return std::unexpected(ry_direction.error());
    }
    return RayDifferentialT<Scalar>::create(reflected_ray, rx_position, *rx_direction, ry_position,
                                            *ry_direction);
}

// An empty successful result is an explicitly detected side change, critical ray, or TIR in one
// of the auxiliary rays. It is never converted into reflection or a zero footprint.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<std::optional<RayDifferentialT<Scalar>>> propagate_specular_transmission(
    const RayDifferentialT<Scalar>& incident, const RayT<Scalar> transmitted_ray,
    const Normal3T<Scalar> central_shading_normal, const Point3T<Scalar> rx_position,
    const Normal3T<Scalar> rx_shading_normal, const Point3T<Scalar> ry_position,
    const Normal3T<Scalar> ry_shading_normal, const bool entering, const Scalar exterior_eta,
    const Scalar interior_eta) {
    const auto central_direction = ray_differential_detail::refracted_direction(
        incident.ray().direction(), central_shading_normal, entering, exterior_eta, interior_eta);
    if (!central_direction) {
        return std::unexpected(central_direction.error());
    }
    if (!*central_direction) {
        return std::unexpected(ray_differential_detail::invalid_ray_differential(
            "A transmitted central ray cannot represent total internal reflection."));
    }
    const auto central_status = ray_differential_detail::validate_next_specular_ray(
        incident, transmitted_ray, **central_direction);
    if (!central_status) {
        return std::unexpected(central_status.error());
    }
    const auto rx_direction = ray_differential_detail::refracted_direction(
        incident.rx_direction(), rx_shading_normal, entering, exterior_eta, interior_eta);
    if (!rx_direction) {
        return std::unexpected(rx_direction.error());
    }
    const auto ry_direction = ray_differential_detail::refracted_direction(
        incident.ry_direction(), ry_shading_normal, entering, exterior_eta, interior_eta);
    if (!ry_direction) {
        return std::unexpected(ry_direction.error());
    }
    if (!*rx_direction || !*ry_direction) {
        return std::optional<RayDifferentialT<Scalar>>{};
    }
    const auto propagated = RayDifferentialT<Scalar>::create(
        transmitted_ray, rx_position, **rx_direction, ry_position, **ry_direction);
    if (!propagated) {
        return std::unexpected(propagated.error());
    }
    return std::optional<RayDifferentialT<Scalar>>{*propagated};
}

static_assert(std::is_standard_layout_v<RayDifferential>);
static_assert(std::is_trivially_copyable_v<RayDifferential>);
static_assert(std::is_standard_layout_v<ReferenceRayDifferential>);
static_assert(std::is_trivially_copyable_v<ReferenceRayDifferential>);
static_assert(std::is_standard_layout_v<SurfacePointDifferentials>);
static_assert(std::is_trivially_copyable_v<SurfacePointDifferentials>);
static_assert(std::is_standard_layout_v<ReferenceSurfacePointDifferentials>);
static_assert(std::is_trivially_copyable_v<ReferenceSurfacePointDifferentials>);

} // namespace blackframe::renderer
