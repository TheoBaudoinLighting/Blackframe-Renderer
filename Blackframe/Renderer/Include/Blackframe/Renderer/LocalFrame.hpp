#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace blackframe::renderer {
namespace local_frame_detail {

template <GeometryScalar Scalar>
[[nodiscard]] Vector3T<Scalar> as_vector(const Normal3T<Scalar> normal) noexcept {
    return {.x = normal.x, .y = normal.y, .z = normal.z};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>> robust_unit_normal(const Normal3T<Scalar> value) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A frame normal must be finite and non-zero.",
        });
    }
    const auto maximum_component =
        std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A frame normal must be finite and non-zero.",
        });
    }

    const auto scaled = Normal3T<Scalar>{
        .x = value.x / maximum_component,
        .y = value.y / maximum_component,
        .z = value.z / maximum_component,
    };
    const auto magnitude =
        std::sqrt(scaled.x * scaled.x + scaled.y * scaled.y + scaled.z * scaled.z);
    return Normal3T<Scalar>{
        .x = scaled.x / magnitude,
        .y = scaled.y / magnitude,
        .z = scaled.z / magnitude,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> robust_unit_vector(const Vector3T<Scalar> value,
                                                                const char* const error_message) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = error_message,
        });
    }
    const auto maximum_component =
        std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    if (maximum_component == Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = error_message,
        });
    }

    const auto scaled = value / maximum_component;
    const auto magnitude = std::sqrt(length_squared(scaled));
    return scaled / magnitude;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool orthonormal(const Vector3T<Scalar> tangent, const Vector3T<Scalar> bitangent,
                               const Normal3T<Scalar> normal) noexcept {
    constexpr auto tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{128};
    const auto normal_vector = as_vector(normal);
    return std::abs(length_squared(tangent) - Scalar{1}) <= tolerance &&
           std::abs(length_squared(bitangent) - Scalar{1}) <= tolerance &&
           std::abs(length_squared(normal) - Scalar{1}) <= tolerance &&
           std::abs(dot(tangent, bitangent)) <= tolerance &&
           std::abs(dot(tangent, normal)) <= tolerance &&
           std::abs(dot(bitangent, normal)) <= tolerance &&
           std::abs(dot(cross(tangent, bitangent), normal_vector) - Scalar{1}) <= tolerance;
}

} // namespace local_frame_detail

template <GeometryScalar Scalar> class OrthonormalFrameT final {
  public:
    [[nodiscard]] static core::Result<OrthonormalFrameT>
    from_normal(const Normal3T<Scalar> normal) {
        auto unit_normal = local_frame_detail::robust_unit_normal(normal);
        if (!unit_normal) {
            return std::unexpected(std::move(unit_normal.error()));
        }

        // Duff et al.'s branch-free basis avoids the singularity at both normal poles.
        const auto sign = std::copysign(Scalar{1}, unit_normal->z);
        const auto coefficient = Scalar{-1} / (sign + unit_normal->z);
        const auto product = unit_normal->x * unit_normal->y * coefficient;
        const auto tangent_seed = Vector3T<Scalar>{
            .x = Scalar{1} + sign * unit_normal->x * unit_normal->x * coefficient,
            .y = sign * product,
            .z = -sign * unit_normal->x,
        };
        return from_unit_normal_and_tangent(*unit_normal, tangent_seed);
    }

    [[nodiscard]] static core::Result<OrthonormalFrameT>
    from_normal_and_tangent(const Normal3T<Scalar> normal, const Vector3T<Scalar> tangent) {
        auto unit_normal = local_frame_detail::robust_unit_normal(normal);
        if (!unit_normal) {
            return std::unexpected(std::move(unit_normal.error()));
        }
        return from_unit_normal_and_tangent(*unit_normal, tangent);
    }

    [[nodiscard]] const Vector3T<Scalar>& tangent() const noexcept {
        return tangent_;
    }
    [[nodiscard]] const Vector3T<Scalar>& bitangent() const noexcept {
        return bitangent_;
    }
    [[nodiscard]] const Normal3T<Scalar>& normal() const noexcept {
        return normal_;
    }

    [[nodiscard]] Vector3T<Scalar> to_local(const Vector3T<Scalar> world) const noexcept {
        return {
            .x = dot(world, tangent_),
            .y = dot(world, bitangent_),
            .z = dot(world, normal_),
        };
    }

    [[nodiscard]] Vector3T<Scalar> to_world(const Vector3T<Scalar> local) const noexcept {
        const auto normal_vector = local_frame_detail::as_vector(normal_);
        return tangent_ * local.x + bitangent_ * local.y + normal_vector * local.z;
    }

    [[nodiscard]] Normal3T<Scalar> to_local(const Normal3T<Scalar> world) const noexcept {
        return {
            .x = dot(world, tangent_),
            .y = dot(world, bitangent_),
            .z = dot(world, normal_),
        };
    }

    [[nodiscard]] Normal3T<Scalar> to_world(const Normal3T<Scalar> local) const noexcept {
        return {
            .x = tangent_.x * local.x + bitangent_.x * local.y + normal_.x * local.z,
            .y = tangent_.y * local.x + bitangent_.y * local.y + normal_.y * local.z,
            .z = tangent_.z * local.x + bitangent_.z * local.y + normal_.z * local.z,
        };
    }

  private:
    constexpr OrthonormalFrameT(Vector3T<Scalar> tangent, Vector3T<Scalar> bitangent,
                                Normal3T<Scalar> normal) noexcept
        : tangent_{tangent}, bitangent_{bitangent}, normal_{normal} {}

    [[nodiscard]] static core::Result<OrthonormalFrameT>
    from_unit_normal_and_tangent(const Normal3T<Scalar> unit_normal,
                                 const Vector3T<Scalar> tangent) {
        const auto normal_vector = local_frame_detail::as_vector(unit_normal);
        auto tangent_direction = local_frame_detail::robust_unit_vector(
            tangent, "A frame tangent must be finite and non-parallel to its normal.");
        if (!tangent_direction) {
            return std::unexpected(std::move(tangent_direction.error()));
        }

        const auto projected_tangent =
            *tangent_direction - normal_vector * dot(*tangent_direction, unit_normal);
        auto unit_tangent = local_frame_detail::robust_unit_vector(
            projected_tangent, "A frame tangent must be finite and non-parallel to its normal.");
        if (!unit_tangent) {
            return std::unexpected(std::move(unit_tangent.error()));
        }

        auto unit_bitangent = local_frame_detail::robust_unit_vector(
            cross(normal_vector, *unit_tangent),
            "A frame tangent cannot form a stable basis with its normal.");
        if (!unit_bitangent) {
            return std::unexpected(std::move(unit_bitangent.error()));
        }

        unit_tangent = local_frame_detail::robust_unit_vector(
            cross(*unit_bitangent, normal_vector),
            "A frame tangent cannot form a stable basis with its normal.");
        if (!unit_tangent) {
            return std::unexpected(std::move(unit_tangent.error()));
        }

        if (!local_frame_detail::orthonormal(*unit_tangent, *unit_bitangent, unit_normal)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "The supplied directions cannot form a stable orthonormal frame.",
            });
        }
        return OrthonormalFrameT{*unit_tangent, *unit_bitangent, unit_normal};
    }

    Vector3T<Scalar> tangent_;
    Vector3T<Scalar> bitangent_;
    Normal3T<Scalar> normal_;
};

using OrthonormalFrame = OrthonormalFrameT<TransportScalar>;
using ReferenceOrthonormalFrame = OrthonormalFrameT<ReferenceScalar>;

} // namespace blackframe::renderer
