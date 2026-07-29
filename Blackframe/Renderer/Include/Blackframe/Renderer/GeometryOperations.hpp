#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <cmath>

namespace blackframe::renderer {
namespace geometry_detail {

template <typename Value, GeometryScalar Scalar>
[[nodiscard]] core::Result<Value> normalized_value(const Value value, const Scalar squared_length,
                                                   const char* const type_name) {
    if (!std::isfinite(squared_length) || squared_length <= Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = std::string{type_name} +
                       " cannot be normalized from a zero or non-finite magnitude.",
        });
    }

    const auto magnitude = std::sqrt(squared_length);
    return value / magnitude;
}

} // namespace geometry_detail

template <GeometryScalar Scalar>
[[nodiscard]] Scalar length(const Vector3T<Scalar> value) noexcept {
    return std::sqrt(length_squared(value));
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar length(const Normal3T<Scalar> value) noexcept {
    return std::sqrt(length_squared(value));
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Vector3T<Scalar>> normalized(const Vector3T<Scalar> value) {
    return geometry_detail::normalized_value(value, length_squared(value), "Vector");
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Normal3T<Scalar>> normalized(const Normal3T<Scalar> value) {
    return geometry_detail::normalized_value(value, length_squared(value), "Normal");
}

} // namespace blackframe::renderer
