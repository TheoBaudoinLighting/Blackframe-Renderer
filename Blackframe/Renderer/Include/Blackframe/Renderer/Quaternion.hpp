#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/MatrixTypes.hpp>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <GeometryScalar Scalar> struct QuaternionT final {
    Scalar x{};
    Scalar y{};
    Scalar z{};
    Scalar w{Scalar{1}};

    [[nodiscard]] constexpr bool operator==(const QuaternionT&) const noexcept = default;
};

using Quaternion = QuaternionT<TransportScalar>;
using ReferenceQuaternion = QuaternionT<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] constexpr QuaternionT<Scalar> operator*(const QuaternionT<Scalar> left,
                                                      const QuaternionT<Scalar> right) noexcept {
    return {
        .x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        .y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        .z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        .w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr QuaternionT<Scalar> conjugated(const QuaternionT<Scalar> value) noexcept {
    return {
        .x = -value.x,
        .y = -value.y,
        .z = -value.z,
        .w = value.w,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar length_squared(const QuaternionT<Scalar> value) noexcept {
    return value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
}

template <GeometryScalar Scalar>
[[nodiscard]] bool finite(const QuaternionT<Scalar> value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           std::isfinite(value.w);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<QuaternionT<Scalar>> normalized(const QuaternionT<Scalar> value) {
    const auto squared_length = length_squared(value);
    if (!finite(value) || !std::isfinite(squared_length) || squared_length <= Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A quaternion cannot be normalized from a zero or non-finite magnitude.",
        });
    }
    const auto magnitude = std::sqrt(squared_length);
    return QuaternionT<Scalar>{
        .x = value.x / magnitude,
        .y = value.y / magnitude,
        .z = value.z / magnitude,
        .w = value.w / magnitude,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<QuaternionT<Scalar>> inverse(const QuaternionT<Scalar> value) {
    const auto squared_length = length_squared(value);
    if (!finite(value) || !std::isfinite(squared_length) || squared_length <= Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A zero or non-finite quaternion has no inverse.",
        });
    }
    const auto conjugate = conjugated(value);
    return QuaternionT<Scalar>{
        .x = conjugate.x / squared_length,
        .y = conjugate.y / squared_length,
        .z = conjugate.z / squared_length,
        .w = conjugate.w / squared_length,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<QuaternionT<Scalar>>
quaternion_from_axis_angle(const Vector3T<Scalar> axis, const Scalar angle_radians) {
    if (!std::isfinite(angle_radians)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A quaternion rotation angle must be finite.",
        });
    }
    auto unit_axis = normalized(axis);
    if (!unit_axis) {
        return std::unexpected(std::move(unit_axis.error()));
    }

    const auto half_angle = angle_radians / Scalar{2};
    const auto sine = std::sin(half_angle);
    return QuaternionT<Scalar>{
        .x = unit_axis->x * sine,
        .y = unit_axis->y * sine,
        .z = unit_axis->z * sine,
        .w = std::cos(half_angle),
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Matrix4T<Scalar>> rotation_matrix(const QuaternionT<Scalar> rotation) {
    const auto squared_length = length_squared(rotation);
    constexpr auto unit_tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar{32};
    if (!finite(rotation) || !std::isfinite(squared_length) ||
        std::abs(squared_length - Scalar{1}) > unit_tolerance) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "A rotation quaternion must be finite and normalized.",
        });
    }

    const auto xx = rotation.x * rotation.x;
    const auto yy = rotation.y * rotation.y;
    const auto zz = rotation.z * rotation.z;
    const auto xy = rotation.x * rotation.y;
    const auto xz = rotation.x * rotation.z;
    const auto yz = rotation.y * rotation.z;
    const auto wx = rotation.w * rotation.x;
    const auto wy = rotation.w * rotation.y;
    const auto wz = rotation.w * rotation.z;

    auto result = identity_matrix<Scalar>();
    result(0, 0) = Scalar{1} - Scalar{2} * (yy + zz);
    result(0, 1) = Scalar{2} * (xy - wz);
    result(0, 2) = Scalar{2} * (xz + wy);
    result(1, 0) = Scalar{2} * (xy + wz);
    result(1, 1) = Scalar{1} - Scalar{2} * (xx + zz);
    result(1, 2) = Scalar{2} * (yz - wx);
    result(2, 0) = Scalar{2} * (xz - wy);
    result(2, 1) = Scalar{2} * (yz + wx);
    result(2, 2) = Scalar{1} - Scalar{2} * (xx + yy);
    return result;
}

static_assert(std::is_standard_layout_v<Quaternion>);
static_assert(std::is_trivially_copyable_v<Quaternion>);
static_assert(sizeof(Quaternion) == 4 * sizeof(TransportScalar));
static_assert(sizeof(ReferenceQuaternion) == 4 * sizeof(ReferenceScalar));

} // namespace blackframe::renderer
