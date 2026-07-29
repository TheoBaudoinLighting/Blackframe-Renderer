#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace blackframe::renderer {
namespace ray_origin_detail {

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
[[nodiscard]] core::Result<Normal3T<Scalar>> unit_normal(const Normal3T<Scalar> normal) {
    const auto maximum_component =
        std::max({std::abs(normal.x), std::abs(normal.y), std::abs(normal.z)});
    if (!finite(normal) || maximum_component == Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A ray origin offset requires a finite non-zero geometric normal.",
        });
    }

    const auto scaled = normal / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    return scaled / magnitude;
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar outward_component(const Scalar value, const Scalar offset) noexcept {
    if (offset > Scalar{0}) {
        return std::nextafter(value, std::numeric_limits<Scalar>::infinity());
    }
    if (offset < Scalar{0}) {
        return std::nextafter(value, -std::numeric_limits<Scalar>::infinity());
    }
    return value;
}

} // namespace ray_origin_detail

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Point3T<Scalar>>
offset_ray_origin(const Point3T<Scalar> point, const Vector3T<Scalar> absolute_position_error,
                  const Normal3T<Scalar> geometric_normal,
                  const Vector3T<Scalar> outgoing_direction) {
    if (!ray_origin_detail::finite(point) || !ray_origin_detail::finite(absolute_position_error) ||
        absolute_position_error.x < Scalar{0} || absolute_position_error.y < Scalar{0} ||
        absolute_position_error.z < Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A ray origin offset requires a finite point and finite non-negative error.",
        });
    }
    if (!ray_origin_detail::finite(outgoing_direction) ||
        (outgoing_direction.x == Scalar{0} && outgoing_direction.y == Scalar{0} &&
         outgoing_direction.z == Scalar{0})) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A ray origin offset requires a finite non-zero outgoing direction.",
        });
    }

    const auto normal = ray_origin_detail::unit_normal(geometric_normal);
    if (!normal.has_value()) {
        return std::unexpected(normal.error());
    }

    const auto offset_distance = std::abs(normal->x) * absolute_position_error.x +
                                 std::abs(normal->y) * absolute_position_error.y +
                                 std::abs(normal->z) * absolute_position_error.z;
    if (!std::isfinite(offset_distance)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The projected ray origin error is not finite.",
        });
    }

    const auto direction_scale =
        std::max({std::abs(outgoing_direction.x), std::abs(outgoing_direction.y),
                  std::abs(outgoing_direction.z)});
    const auto scaled_direction = outgoing_direction / direction_scale;
    auto offset = Normal3T<Scalar>{
        .x = normal->x * offset_distance,
        .y = normal->y * offset_distance,
        .z = normal->z * offset_distance,
    };
    if (dot(*normal, scaled_direction) < Scalar{0}) {
        offset = -offset;
    }

    const auto shifted = Point3T<Scalar>{
        .x = ray_origin_detail::outward_component(point.x + offset.x, offset.x),
        .y = ray_origin_detail::outward_component(point.y + offset.y, offset.y),
        .z = ray_origin_detail::outward_component(point.z + offset.z, offset.z),
    };
    if (!ray_origin_detail::finite(shifted)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The ray origin offset exceeds the finite coordinate range.",
        });
    }
    return shifted;
}

} // namespace blackframe::renderer
