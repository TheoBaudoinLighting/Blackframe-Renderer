#pragma once

#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <array>
#include <cstddef>
#include <type_traits>

namespace blackframe::renderer {

// Matrices use row-major storage and multiply column vectors. In A * B, B is applied first.
template <GeometryScalar Scalar> struct Matrix4T final {
    std::array<Scalar, 16> elements{};

    [[nodiscard]] constexpr Scalar& operator()(const std::size_t row,
                                               const std::size_t column) noexcept {
        return elements[row * 4 + column];
    }

    [[nodiscard]] constexpr Scalar operator()(const std::size_t row,
                                              const std::size_t column) const noexcept {
        return elements[row * 4 + column];
    }

    [[nodiscard]] constexpr bool operator==(const Matrix4T&) const noexcept = default;
};

using Matrix4 = Matrix4T<TransportScalar>;
using ReferenceMatrix4 = Matrix4T<ReferenceScalar>;

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Matrix4T<Scalar> identity_matrix() noexcept {
    auto result = Matrix4T<Scalar>{};
    result(0, 0) = Scalar{1};
    result(1, 1) = Scalar{1};
    result(2, 2) = Scalar{1};
    result(3, 3) = Scalar{1};
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Matrix4T<Scalar> operator*(const Matrix4T<Scalar>& left,
                                                   const Matrix4T<Scalar>& right) noexcept {
    auto result = Matrix4T<Scalar>{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t component = 0; component < 4; ++component) {
                result(row, column) += left(row, component) * right(component, column);
            }
        }
    }
    return result;
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Matrix4T<Scalar> transposed(const Matrix4T<Scalar>& matrix) noexcept {
    auto result = Matrix4T<Scalar>{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = matrix(column, row);
        }
    }
    return result;
}

static_assert(std::is_standard_layout_v<Matrix4>);
static_assert(std::is_trivially_copyable_v<Matrix4>);
static_assert(sizeof(Matrix4) == 16 * sizeof(TransportScalar));
static_assert(sizeof(ReferenceMatrix4) == 16 * sizeof(ReferenceScalar));

} // namespace blackframe::renderer
