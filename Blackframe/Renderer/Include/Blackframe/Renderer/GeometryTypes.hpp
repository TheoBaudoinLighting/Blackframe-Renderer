#pragma once

#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <concepts>
#include <type_traits>

namespace blackframe::renderer {

template <typename Scalar>
concept GeometryScalar =
    std::same_as<Scalar, TransportScalar> || std::same_as<Scalar, ReferenceScalar>;

template <GeometryScalar Scalar> struct Point2T final {
    Scalar x{};
    Scalar y{};

    [[nodiscard]] constexpr bool operator==(const Point2T&) const noexcept = default;
};

template <GeometryScalar Scalar> struct Vector3T final {
    Scalar x{};
    Scalar y{};
    Scalar z{};

    [[nodiscard]] constexpr bool operator==(const Vector3T&) const noexcept = default;
};

template <GeometryScalar Scalar> struct Point3T final {
    Scalar x{};
    Scalar y{};
    Scalar z{};

    [[nodiscard]] constexpr bool operator==(const Point3T&) const noexcept = default;
};

template <GeometryScalar Scalar> struct Normal3T final {
    Scalar x{};
    Scalar y{};
    Scalar z{};

    [[nodiscard]] constexpr bool operator==(const Normal3T&) const noexcept = default;
};

using Point2 = Point2T<TransportScalar>;
using ReferencePoint2 = Point2T<ReferenceScalar>;

using Vector3 = Vector3T<TransportScalar>;
using Point3 = Point3T<TransportScalar>;
using Normal3 = Normal3T<TransportScalar>;

using ReferenceVector3 = Vector3T<ReferenceScalar>;
using ReferencePoint3 = Point3T<ReferenceScalar>;
using ReferenceNormal3 = Normal3T<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator+(const Vector3T<Scalar> left,
                                                   const Vector3T<Scalar> right) noexcept {
    return {
        .x = left.x + right.x,
        .y = left.y + right.y,
        .z = left.z + right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator-(const Vector3T<Scalar> left,
                                                   const Vector3T<Scalar> right) noexcept {
    return {
        .x = left.x - right.x,
        .y = left.y - right.y,
        .z = left.z - right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator-(const Vector3T<Scalar> value) noexcept {
    return {
        .x = -value.x,
        .y = -value.y,
        .z = -value.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator*(const Vector3T<Scalar> value,
                                                   const Scalar scale) noexcept {
    return {
        .x = value.x * scale,
        .y = value.y * scale,
        .z = value.z * scale,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator*(const Scalar scale,
                                                   const Vector3T<Scalar> value) noexcept {
    return value * scale;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator/(const Vector3T<Scalar> value,
                                                   const Scalar scale) noexcept {
    return {
        .x = value.x / scale,
        .y = value.y / scale,
        .z = value.z / scale,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Point3T<Scalar> operator+(const Point3T<Scalar> point,
                                                  const Vector3T<Scalar> offset) noexcept {
    return {
        .x = point.x + offset.x,
        .y = point.y + offset.y,
        .z = point.z + offset.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Point3T<Scalar> operator+(const Vector3T<Scalar> offset,
                                                  const Point3T<Scalar> point) noexcept {
    return point + offset;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Point3T<Scalar> operator-(const Point3T<Scalar> point,
                                                  const Vector3T<Scalar> offset) noexcept {
    return {
        .x = point.x - offset.x,
        .y = point.y - offset.y,
        .z = point.z - offset.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> operator-(const Point3T<Scalar> left,
                                                   const Point3T<Scalar> right) noexcept {
    return {
        .x = left.x - right.x,
        .y = left.y - right.y,
        .z = left.z - right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator+(const Normal3T<Scalar> left,
                                                   const Normal3T<Scalar> right) noexcept {
    return {
        .x = left.x + right.x,
        .y = left.y + right.y,
        .z = left.z + right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator-(const Normal3T<Scalar> left,
                                                   const Normal3T<Scalar> right) noexcept {
    return {
        .x = left.x - right.x,
        .y = left.y - right.y,
        .z = left.z - right.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator-(const Normal3T<Scalar> value) noexcept {
    return {
        .x = -value.x,
        .y = -value.y,
        .z = -value.z,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator*(const Normal3T<Scalar> value,
                                                   const Scalar scale) noexcept {
    return {
        .x = value.x * scale,
        .y = value.y * scale,
        .z = value.z * scale,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator*(const Scalar scale,
                                                   const Normal3T<Scalar> value) noexcept {
    return value * scale;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Normal3T<Scalar> operator/(const Normal3T<Scalar> value,
                                                   const Scalar scale) noexcept {
    return {
        .x = value.x / scale,
        .y = value.y / scale,
        .z = value.z / scale,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar dot(const Vector3T<Scalar> left,
                                   const Vector3T<Scalar> right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar dot(const Normal3T<Scalar> left,
                                   const Vector3T<Scalar> right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar dot(const Vector3T<Scalar> left,
                                   const Normal3T<Scalar> right) noexcept {
    return dot(right, left);
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar dot(const Normal3T<Scalar> left,
                                   const Normal3T<Scalar> right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Vector3T<Scalar> cross(const Vector3T<Scalar> left,
                                               const Vector3T<Scalar> right) noexcept {
    return {
        .x = left.y * right.z - left.z * right.y,
        .y = left.z * right.x - left.x * right.z,
        .z = left.x * right.y - left.y * right.x,
    };
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar length_squared(const Vector3T<Scalar> value) noexcept {
    return dot(value, value);
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar length_squared(const Normal3T<Scalar> value) noexcept {
    return dot(value, value);
}

static_assert(!std::is_same_v<Vector3, Point3>);
static_assert(!std::is_same_v<Vector3, Normal3>);
static_assert(!std::is_same_v<Point3, Normal3>);
static_assert(std::is_standard_layout_v<Vector3>);
static_assert(std::is_standard_layout_v<Point2>);
static_assert(std::is_standard_layout_v<Point3>);
static_assert(std::is_standard_layout_v<Normal3>);
static_assert(std::is_trivially_copyable_v<Vector3>);
static_assert(std::is_trivially_copyable_v<Point2>);
static_assert(std::is_trivially_copyable_v<Point3>);
static_assert(std::is_trivially_copyable_v<Normal3>);
static_assert(sizeof(Vector3) == 3 * sizeof(TransportScalar));
static_assert(sizeof(Point2) == 2 * sizeof(TransportScalar));
static_assert(sizeof(Point3) == 3 * sizeof(TransportScalar));
static_assert(sizeof(Normal3) == 3 * sizeof(TransportScalar));

} // namespace blackframe::renderer
