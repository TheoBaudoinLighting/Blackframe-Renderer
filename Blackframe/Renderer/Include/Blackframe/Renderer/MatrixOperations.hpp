#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/MatrixTypes.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace blackframe::renderer {
namespace matrix_detail {

template <GeometryScalar Scalar>
[[nodiscard]] bool finite(const Matrix4T<Scalar>& matrix) noexcept {
    return std::ranges::all_of(matrix.elements,
                               [](const Scalar value) { return std::isfinite(value); });
}

template <typename Value>
[[nodiscard]] core::Result<Value> inversion_error(const char* const message) {
    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    });
}

} // namespace matrix_detail

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Matrix4T<Scalar>> inverse(const Matrix4T<Scalar>& matrix) {
    if (!matrix_detail::finite(matrix)) {
        return matrix_detail::inversion_error<Matrix4T<Scalar>>(
            "A matrix must be finite before inversion.");
    }

    auto augmented = std::array<Scalar, 32>{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            augmented[row * 8 + column] = matrix(row, column);
            augmented[row * 8 + column + 4] = row == column ? Scalar{1} : Scalar{0};
        }
    }

    for (std::size_t pivot_column = 0; pivot_column < 4; ++pivot_column) {
        auto pivot_row = pivot_column;
        auto pivot_magnitude = std::abs(augmented[pivot_row * 8 + pivot_column]);
        for (std::size_t candidate = pivot_column + 1; candidate < 4; ++candidate) {
            const auto candidate_magnitude = std::abs(augmented[candidate * 8 + pivot_column]);
            if (candidate_magnitude > pivot_magnitude) {
                pivot_magnitude = candidate_magnitude;
                pivot_row = candidate;
            }
        }

        if (pivot_magnitude == Scalar{0}) {
            return matrix_detail::inversion_error<Matrix4T<Scalar>>(
                "A singular matrix has no inverse.");
        }

        if (pivot_row != pivot_column) {
            for (std::size_t column = 0; column < 8; ++column) {
                std::swap(augmented[pivot_column * 8 + column], augmented[pivot_row * 8 + column]);
            }
        }

        const auto pivot = augmented[pivot_column * 8 + pivot_column];
        for (std::size_t column = 0; column < 8; ++column) {
            augmented[pivot_column * 8 + column] /= pivot;
        }

        for (std::size_t row = 0; row < 4; ++row) {
            if (row == pivot_column) {
                continue;
            }
            const auto factor = augmented[row * 8 + pivot_column];
            for (std::size_t column = 0; column < 8; ++column) {
                augmented[row * 8 + column] -= factor * augmented[pivot_column * 8 + column];
            }
        }
    }

    auto result = Matrix4T<Scalar>{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = augmented[row * 8 + column + 4];
        }
    }
    if (!matrix_detail::finite(result)) {
        return matrix_detail::inversion_error<Matrix4T<Scalar>>(
            "Matrix inversion produced a non-finite result.");
    }
    return result;
}

} // namespace blackframe::renderer
