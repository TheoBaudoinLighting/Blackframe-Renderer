#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/ClosureSet.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {

namespace ray_cone_detail {

[[nodiscard]] inline core::Error invalid_ray_cone(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

template <GeometryScalar Scalar> [[nodiscard]] bool finite(const Vector3T<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> unit_direction(const Vector3T<Scalar> value) {
    if (!finite(value)) {
        return std::unexpected(invalid_ray_cone("A ray-cone direction must be finite."));
    }
    const auto maximum = std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    if (!(maximum > Scalar{0})) {
        return std::unexpected(invalid_ray_cone("A ray-cone direction must be non-zero."));
    }
    const auto scaled = value / maximum;
    const auto magnitude = std::hypot(scaled.x, scaled.y, scaled.z);
    const auto normalized = scaled / magnitude;
    if (!std::isfinite(magnitude) || !(magnitude > Scalar{0}) || !finite(normalized)) {
        return std::unexpected(
            invalid_ray_cone("A ray-cone direction cannot be normalized representably."));
    }
    return normalized;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> vector_length(const Vector3T<Scalar> value,
                                                 const char* const message) {
    if (!finite(value)) {
        return std::unexpected(invalid_ray_cone(message));
    }
    const auto length = std::hypot(value.x, value.y, value.z);
    if (!std::isfinite(length)) {
        return std::unexpected(invalid_ray_cone(message));
    }
    return length;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> maximum_singular_value(const Vector3T<Scalar> first,
                                                          const Vector3T<Scalar> second,
                                                          const char* const message) {
    if (!finite(first) || !finite(second)) {
        return std::unexpected(invalid_ray_cone(message));
    }
    const auto scale = std::max({std::abs(first.x), std::abs(first.y), std::abs(first.z),
                                 std::abs(second.x), std::abs(second.y), std::abs(second.z)});
    if (scale == Scalar{0}) {
        return Scalar{0};
    }
    const auto x = first / scale;
    const auto y = second / scale;
    const auto a = dot(x, x);
    const auto b = dot(x, y);
    const auto c = dot(y, y);
    const auto largest_eigenvalue = Scalar{0.5} * (a + c + std::hypot(a - c, Scalar{2} * b));
    const auto normalized = std::sqrt(std::max(Scalar{0}, largest_eigenvalue));
    const auto result = scale * normalized;
    if (!std::isfinite(result) || !(result >= Scalar{0}) ||
        (normalized != Scalar{0} && result == Scalar{0})) {
        return std::unexpected(invalid_ray_cone(message));
    }
    return result;
}

template <GeometryScalar Scalar> struct DifferentialCrossSection final {
    Vector3T<Scalar> origin_offset;
    Vector3T<Scalar> slope;
};

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<DifferentialCrossSection<Scalar>> differential_cross_section(
    const Point3T<Scalar> central_origin, const Vector3T<Scalar> central_direction,
    const Point3T<Scalar> auxiliary_origin, const Vector3T<Scalar> auxiliary_direction) {
    const auto central = unit_direction(central_direction);
    const auto auxiliary = unit_direction(auxiliary_direction);
    if (!central) {
        return std::unexpected(central.error());
    }
    if (!auxiliary) {
        return std::unexpected(auxiliary.error());
    }
    const auto denominator = dot(*auxiliary, *central);
    if (!std::isfinite(denominator) || !(denominator > Scalar{0})) {
        return std::unexpected(invalid_ray_cone(
            "A ray cone cannot represent an auxiliary direction at or beyond ninety degrees."));
    }
    const auto origin_delta = auxiliary_origin - central_origin;
    const auto origin_projection = dot(origin_delta, *central);
    const auto parameter_at_origin = -origin_projection / denominator;
    const auto parameter_at_unit_distance = (Scalar{1} - origin_projection) / denominator;
    const auto origin_offset = origin_delta + parameter_at_origin * *auxiliary;
    const auto unit_offset = origin_delta + parameter_at_unit_distance * *auxiliary - *central;
    const auto slope = unit_offset - origin_offset;
    if (!std::isfinite(parameter_at_origin) || !std::isfinite(parameter_at_unit_distance) ||
        !finite(origin_offset) || !finite(slope)) {
        return std::unexpected(
            invalid_ray_cone("A ray-differential cross-section is not representable."));
    }
    return DifferentialCrossSection<Scalar>{.origin_offset = origin_offset, .slope = slope};
}

template <GeometryScalar Scalar>
[[nodiscard]] bool unit_local_direction(const Vector3T<Scalar> value) noexcept {
    if (!finite(value)) {
        return false;
    }
    const auto squared = std::fma(value.x, value.x, std::fma(value.y, value.y, value.z * value.z));
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{256};
    return std::isfinite(squared) && std::abs(squared - Scalar{1}) <= tolerance;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> checked_product(const Scalar left, const Scalar right,
                                                   const char* const message) {
    const auto product = left * right;
    if (!std::isfinite(product) ||
        (left != Scalar{0} && right != Scalar{0} && product == Scalar{0})) {
        return std::unexpected(invalid_ray_cone(message));
    }
    return product;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> checked_ratio(const Scalar numerator, const Scalar denominator,
                                                 const char* const message) {
    if (!std::isfinite(numerator) || !(numerator > Scalar{0}) || !std::isfinite(denominator) ||
        !(denominator > Scalar{0})) {
        return std::unexpected(invalid_ray_cone(message));
    }
    const auto ratio = numerator / denominator;
    if (!std::isfinite(ratio) || !(ratio > Scalar{0})) {
        return std::unexpected(invalid_ray_cone(message));
    }
    return ratio;
}

} // namespace ray_cone_detail

// A circular ray cone is represented at the current ray origin by a world-space radius and a
// non-negative radial slope. Its radius after a world-space distance d is width + spread * d.
// The type deliberately has no default constructor: every wavefront path must name its footprint.
template <GeometryScalar Scalar> class RayConeT final {
  public:
    [[nodiscard]] static core::Result<RayConeT> create(const Scalar width, const Scalar spread) {
        if (!std::isfinite(width) || !(width >= Scalar{0}) || !std::isfinite(spread) ||
            !(spread >= Scalar{0})) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A ray cone requires finite non-negative width and spread."));
        }
        return RayConeT{width, spread};
    }

    [[nodiscard]] constexpr Scalar width() const noexcept {
        return width_;
    }
    [[nodiscard]] constexpr Scalar spread() const noexcept {
        return spread_;
    }

    [[nodiscard]] constexpr bool operator==(const RayConeT&) const noexcept = default;

  private:
    constexpr RayConeT(const Scalar width, const Scalar spread) noexcept
        : width_{width}, spread_{spread} {}

    Scalar width_;
    Scalar spread_;
};

using RayCone = RayConeT<TransportScalar>;
using ReferenceRayCone = RayConeT<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayConeT<Scalar>>
ray_cone_from_differential(const RayDifferentialT<Scalar>& differential) {
    const auto x = ray_cone_detail::differential_cross_section(
        differential.ray().origin(), differential.ray().direction(), differential.rx_origin(),
        differential.rx_direction());
    const auto y = ray_cone_detail::differential_cross_section(
        differential.ray().origin(), differential.ray().direction(), differential.ry_origin(),
        differential.ry_direction());
    if (!x) {
        return std::unexpected(x.error());
    }
    if (!y) {
        return std::unexpected(y.error());
    }
    const auto width = ray_cone_detail::maximum_singular_value(
        x->origin_offset, y->origin_offset, "A ray-cone origin radius is not representable.");
    const auto spread = ray_cone_detail::maximum_singular_value(
        x->slope, y->slope, "A ray-cone angular spread is not representable.");
    if (!width) {
        return std::unexpected(width.error());
    }
    if (!spread) {
        return std::unexpected(spread.error());
    }
    return RayConeT<Scalar>::create(*width, *spread);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayConeT<Scalar>> advance_ray_cone(const RayConeT<Scalar> cone,
                                                              const Scalar distance) {
    if (!std::isfinite(distance) || !(distance >= Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray cone requires a finite non-negative travel distance."));
    }
    const auto growth = ray_cone_detail::checked_product(
        cone.spread(), distance, "A ray-cone distance growth is not representable.");
    if (!growth) {
        return std::unexpected(growth.error());
    }
    const auto width = std::fma(cone.spread(), distance, cone.width());
    if (!std::isfinite(width) || !(width >= cone.width()) ||
        (*growth != Scalar{0} && width == cone.width())) {
        return std::unexpected(
            ray_cone_detail::invalid_ray_cone("A propagated ray-cone width is not representable."));
    }
    return RayConeT<Scalar>::create(width, cone.spread());
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayConeT<Scalar>>
advance_ray_cone(const RayConeT<Scalar> cone, const RayT<Scalar>& ray, const Scalar parameter) {
    if (!std::isfinite(parameter) || !(parameter >= Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray cone requires a finite non-negative ray parameter."));
    }
    const auto direction_length = ray_cone_detail::vector_length(
        ray.direction(), "A ray-cone travel direction length is not representable.");
    if (!direction_length) {
        return std::unexpected(direction_length.error());
    }
    const auto distance = ray_cone_detail::checked_product(
        parameter, *direction_length, "A ray-cone travel distance is not representable.");
    if (!distance) {
        return std::unexpected(distance.error());
    }
    return advance_ray_cone(cone, *distance);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
ray_differential_surface_footprint_radius(const SurfacePointDifferentialsT<Scalar> differentials) {
    return ray_cone_detail::maximum_singular_value(
        differentials.dpdx, differentials.dpdy,
        "A ray-differential surface footprint radius is not representable.");
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
ray_cone_surface_footprint_radius(const RayConeT<Scalar> cone, const Vector3T<Scalar> ray_direction,
                                  const Normal3T<Scalar> geometric_normal) {
    const auto direction = ray_cone_detail::unit_direction(ray_direction);
    if (!direction) {
        return std::unexpected(direction.error());
    }
    const auto normal_length =
        std::hypot(geometric_normal.x, geometric_normal.y, geometric_normal.z);
    if (!std::isfinite(normal_length) || !(normal_length > Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray-cone surface footprint requires a finite non-zero geometric normal."));
    }
    const auto cosine = std::abs(dot(*direction, geometric_normal)) / normal_length;
    if (!std::isfinite(cosine) || !(cosine > Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray cone tangent to a surface has no finite circular footprint bound."));
    }
    const auto radius = cone.width() / cosine;
    if (!std::isfinite(radius) || !(radius >= Scalar{0}) ||
        (cone.width() != Scalar{0} && radius == Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray-cone surface footprint is not representable."));
    }
    return radius;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
ray_cone_surface_footprint_radius(const RayConeT<Scalar> cone, const RayT<Scalar>& ray,
                                  const Scalar parameter, const Normal3T<Scalar> geometric_normal) {
    const auto advanced = advance_ray_cone(cone, ray, parameter);
    if (!advanced) {
        return std::unexpected(advanced.error());
    }
    return ray_cone_surface_footprint_radius(*advanced, ray.direction(), geometric_normal);
}

// With a locally constant closure frame, propagation is exact for planar ideal reflection. Ideal
// transmission bounds both singular derivatives of Snell's law and reprojects the incident
// cross-section into the outgoing ray plane. Since continuous lobes have unbounded directional
// support, they use a documented finite footprint policy rather than pretending to bound that
// support: unit slope for diffuse and max(alphaX, alphaY) for GGX.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayConeT<Scalar>>
propagate_ray_cone_scattering(const RayConeT<Scalar> cone, const ClosureT<Scalar>& closure,
                              const ScatteringLobe event, const Vector3T<Scalar> outgoing_local,
                              const Vector3T<Scalar> incoming_local) {
    if (!is_valid_surface_scattering_event(event) ||
        !ray_cone_detail::unit_local_direction(outgoing_local) ||
        !ray_cone_detail::unit_local_direction(incoming_local) || outgoing_local.z == Scalar{0} ||
        incoming_local.z == Scalar{0}) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "Ray-cone scattering requires a valid surface event and unit local directions."));
    }

    const auto reflection = has_scattering_lobe(event, ScatteringLobe::reflection);
    const auto transmission = has_scattering_lobe(event, ScatteringLobe::transmission);
    const auto same_hemisphere = (outgoing_local.z > Scalar{0}) == (incoming_local.z > Scalar{0});
    if ((reflection && !same_hemisphere) || (transmission && same_hemisphere)) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray-cone event direction does not match its local hemispheres."));
    }
    auto spread_floor = Scalar{0};
    auto transmission_scale = Scalar{1};
    auto width_scale = Scalar{1};
    switch (closure.kind) {
    case ClosureKind::lambertian_reflection:
    case ClosureKind::rough_diffuse_reflection:
        if (!reflection || !has_scattering_lobe(event, ScatteringLobe::diffuse)) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A diffuse ray-cone event does not match its selected closure."));
        }
        spread_floor = Scalar{1};
        break;
    case ClosureKind::rough_conductor_reflection:
        if (!reflection || !has_scattering_lobe(event, ScatteringLobe::glossy)) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A conductor ray-cone event does not match its selected closure."));
        }
        spread_floor = std::max(closure.parameters[8U], closure.parameters[9U]);
        break;
    case ClosureKind::rough_dielectric:
        if (!has_scattering_lobe(event, ScatteringLobe::glossy)) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A rough-dielectric ray-cone event does not match its selected closure."));
        }
        spread_floor = std::max(closure.parameters[2U], closure.parameters[3U]);
        break;
    case ClosureKind::specular_reflection:
        if (!reflection || !has_scattering_lobe(event, ScatteringLobe::specular)) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A reflected ray-cone event does not match its selected closure."));
        }
        break;
    case ClosureKind::specular_transmission:
        if (!transmission || !has_scattering_lobe(event, ScatteringLobe::specular)) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A transmitted ray-cone event does not match its selected closure."));
        }
        break;
    case ClosureKind::none:
        return std::unexpected(
            ray_cone_detail::invalid_ray_cone("An inactive closure cannot propagate a ray cone."));
    default:
        return std::unexpected(
            ray_cone_detail::invalid_ray_cone("An unknown closure cannot propagate a ray cone."));
    }
    if (!std::isfinite(spread_floor) || !(spread_floor >= Scalar{0})) {
        return std::unexpected(ray_cone_detail::invalid_ray_cone(
            "A ray-cone closure requires a finite non-negative spread policy."));
    }

    if (transmission) {
        if (closure.kind != ClosureKind::rough_dielectric &&
            closure.kind != ClosureKind::specular_transmission) {
            return std::unexpected(ray_cone_detail::invalid_ray_cone(
                "A transmitted ray-cone event requires a dielectric closure."));
        }
        const auto exterior_eta = closure.parameters[0U];
        const auto interior_eta = closure.parameters[1U];
        const auto incident_eta = outgoing_local.z > Scalar{0} ? exterior_eta : interior_eta;
        const auto transmitted_eta = outgoing_local.z > Scalar{0} ? interior_eta : exterior_eta;
        const auto eta_ratio = ray_cone_detail::checked_ratio(
            incident_eta, transmitted_eta, "A ray-cone Snell eta ratio is not representable.");
        const auto cosine_ratio =
            ray_cone_detail::checked_ratio(std::abs(outgoing_local.z), std::abs(incoming_local.z),
                                           "A ray-cone Snell cosine ratio is not representable.");
        const auto inverse_cosine_ratio = ray_cone_detail::checked_ratio(
            std::abs(incoming_local.z), std::abs(outgoing_local.z),
            "A ray-cone transmission projection is not representable.");
        if (!eta_ratio) {
            return std::unexpected(eta_ratio.error());
        }
        if (!cosine_ratio) {
            return std::unexpected(cosine_ratio.error());
        }
        if (!inverse_cosine_ratio) {
            return std::unexpected(inverse_cosine_ratio.error());
        }
        const auto meridional_scale = ray_cone_detail::checked_product(
            *eta_ratio, *cosine_ratio,
            "A ray-cone meridional Snell derivative is not representable.");
        if (!meridional_scale) {
            return std::unexpected(meridional_scale.error());
        }
        // Refraction has eta_i / eta_t as its azimuthal singular value and
        // eta_i cos(theta_i) / (eta_t cos(theta_t)) in the meridional plane.
        transmission_scale = std::max(*eta_ratio, *meridional_scale);
        width_scale = std::max(Scalar{1}, *inverse_cosine_ratio);
    }

    const auto transported_width = ray_cone_detail::checked_product(
        cone.width(), width_scale, "A scattered ray-cone width is not representable.");
    const auto transported = ray_cone_detail::checked_product(
        cone.spread(), transmission_scale,
        "A scattered ray-cone angular spread is not representable.");
    if (!transported_width) {
        return std::unexpected(transported_width.error());
    }
    if (!transported) {
        return std::unexpected(transported.error());
    }
    return RayConeT<Scalar>::create(*transported_width, std::max(*transported, spread_floor));
}

static_assert(sizeof(RayCone) == 2U * sizeof(TransportScalar));
static_assert(sizeof(ReferenceRayCone) == 2U * sizeof(ReferenceScalar));
static_assert(std::is_standard_layout_v<RayCone>);
static_assert(std::is_trivially_copyable_v<RayCone>);
static_assert(std::is_standard_layout_v<ReferenceRayCone>);
static_assert(std::is_trivially_copyable_v<ReferenceRayCone>);
static_assert(!std::is_default_constructible_v<RayCone>);
static_assert(!std::is_default_constructible_v<ReferenceRayCone>);

} // namespace blackframe::renderer
