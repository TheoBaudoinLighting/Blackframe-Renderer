#pragma once

#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/Quaternion.hpp>
#include <cmath>
#include <numbers>
#include <utility>

namespace blackframe::renderer {

template <GeometryScalar Scalar> class AffineTransformT final {
  public:
    [[nodiscard]] static core::Result<AffineTransformT>
    from_matrix(const Matrix4T<Scalar>& matrix) {
        if (matrix(3, 0) != Scalar{0} || matrix(3, 1) != Scalar{0} || matrix(3, 2) != Scalar{0} ||
            matrix(3, 3) != Scalar{1}) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "An affine transform must have the homogeneous row [0, 0, 0, 1].",
            });
        }
        auto inverse_matrix = blackframe::renderer::inverse(matrix);
        if (!inverse_matrix) {
            return std::unexpected(std::move(inverse_matrix.error()));
        }
        return AffineTransformT{matrix, *inverse_matrix};
    }

    [[nodiscard]] static core::Result<AffineTransformT> translation(const Vector3T<Scalar> offset) {
        auto matrix = identity_matrix<Scalar>();
        matrix(0, 3) = offset.x;
        matrix(1, 3) = offset.y;
        matrix(2, 3) = offset.z;
        return from_matrix(matrix);
    }

    [[nodiscard]] static core::Result<AffineTransformT> scale(const Vector3T<Scalar> factors) {
        auto matrix = identity_matrix<Scalar>();
        matrix(0, 0) = factors.x;
        matrix(1, 1) = factors.y;
        matrix(2, 2) = factors.z;
        return from_matrix(matrix);
    }

    [[nodiscard]] static core::Result<AffineTransformT>
    rotation(const QuaternionT<Scalar> quaternion) {
        auto matrix = rotation_matrix(quaternion);
        if (!matrix) {
            return std::unexpected(std::move(matrix.error()));
        }
        return from_matrix(*matrix);
    }

    [[nodiscard]] const Matrix4T<Scalar>& matrix() const noexcept {
        return matrix_;
    }
    [[nodiscard]] const Matrix4T<Scalar>& inverse_matrix() const noexcept {
        return inverse_matrix_;
    }

    [[nodiscard]] Point3T<Scalar> apply(const Point3T<Scalar> point) const noexcept {
        return transform_point(matrix_, point);
    }

    [[nodiscard]] Vector3T<Scalar> apply(const Vector3T<Scalar> vector) const noexcept {
        return transform_vector(matrix_, vector);
    }

    [[nodiscard]] Normal3T<Scalar> apply(const Normal3T<Scalar> normal) const noexcept {
        return transform_normal(inverse_matrix_, normal);
    }

    [[nodiscard]] Point3T<Scalar> apply_inverse(const Point3T<Scalar> point) const noexcept {
        return transform_point(inverse_matrix_, point);
    }

    [[nodiscard]] Vector3T<Scalar> apply_inverse(const Vector3T<Scalar> vector) const noexcept {
        return transform_vector(inverse_matrix_, vector);
    }

    [[nodiscard]] Normal3T<Scalar> apply_inverse(const Normal3T<Scalar> normal) const noexcept {
        return transform_normal(matrix_, normal);
    }

  private:
    constexpr AffineTransformT(Matrix4T<Scalar> matrix, Matrix4T<Scalar> inverse_matrix) noexcept
        : matrix_{matrix}, inverse_matrix_{inverse_matrix} {}

    [[nodiscard]] static Point3T<Scalar> transform_point(const Matrix4T<Scalar>& matrix,
                                                         const Point3T<Scalar> point) noexcept {
        return {
            .x = matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2) * point.z +
                 matrix(0, 3),
            .y = matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2) * point.z +
                 matrix(1, 3),
            .z = matrix(2, 0) * point.x + matrix(2, 1) * point.y + matrix(2, 2) * point.z +
                 matrix(2, 3),
        };
    }

    [[nodiscard]] static Vector3T<Scalar> transform_vector(const Matrix4T<Scalar>& matrix,
                                                           const Vector3T<Scalar> vector) noexcept {
        return {
            .x = matrix(0, 0) * vector.x + matrix(0, 1) * vector.y + matrix(0, 2) * vector.z,
            .y = matrix(1, 0) * vector.x + matrix(1, 1) * vector.y + matrix(1, 2) * vector.z,
            .z = matrix(2, 0) * vector.x + matrix(2, 1) * vector.y + matrix(2, 2) * vector.z,
        };
    }

    [[nodiscard]] static Normal3T<Scalar> transform_normal(const Matrix4T<Scalar>& inverse_matrix,
                                                           const Normal3T<Scalar> normal) noexcept {
        return {
            .x = inverse_matrix(0, 0) * normal.x + inverse_matrix(1, 0) * normal.y +
                 inverse_matrix(2, 0) * normal.z,
            .y = inverse_matrix(0, 1) * normal.x + inverse_matrix(1, 1) * normal.y +
                 inverse_matrix(2, 1) * normal.z,
            .z = inverse_matrix(0, 2) * normal.x + inverse_matrix(1, 2) * normal.y +
                 inverse_matrix(2, 2) * normal.z,
        };
    }

    Matrix4T<Scalar> matrix_;
    Matrix4T<Scalar> inverse_matrix_;
};

using AffineTransform = AffineTransformT<TransportScalar>;
using ReferenceAffineTransform = AffineTransformT<ReferenceScalar>;

template <GeometryScalar Scalar> class ProjectiveTransformT final {
  public:
    [[nodiscard]] static core::Result<ProjectiveTransformT>
    from_matrix(const Matrix4T<Scalar>& matrix) {
        auto inverse_matrix = blackframe::renderer::inverse(matrix);
        if (!inverse_matrix) {
            return std::unexpected(std::move(inverse_matrix.error()));
        }
        return ProjectiveTransformT{matrix, *inverse_matrix};
    }

    [[nodiscard]] static core::Result<ProjectiveTransformT>
    perspective(const Scalar vertical_field_of_view_radians, const Scalar aspect_ratio,
                const Scalar near_distance, const Scalar far_distance) {
        if (!std::isfinite(vertical_field_of_view_radians) || !std::isfinite(aspect_ratio) ||
            !std::isfinite(near_distance) || !std::isfinite(far_distance) ||
            vertical_field_of_view_radians <= Scalar{0} ||
            vertical_field_of_view_radians >= std::numbers::pi_v<Scalar> ||
            aspect_ratio <= Scalar{0} || near_distance <= Scalar{0} ||
            far_distance <= near_distance) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Perspective parameters must define a finite, ordered view volume.",
            });
        }

        const auto focal_scale = Scalar{1} / std::tan(vertical_field_of_view_radians / Scalar{2});
        auto matrix = Matrix4T<Scalar>{};
        matrix(0, 0) = focal_scale / aspect_ratio;
        matrix(1, 1) = focal_scale;
        matrix(2, 2) = (far_distance + near_distance) / (near_distance - far_distance);
        matrix(2, 3) = (Scalar{2} * far_distance * near_distance) / (near_distance - far_distance);
        matrix(3, 2) = Scalar{-1};
        return from_matrix(matrix);
    }

    [[nodiscard]] const Matrix4T<Scalar>& matrix() const noexcept {
        return matrix_;
    }
    [[nodiscard]] const Matrix4T<Scalar>& inverse_matrix() const noexcept {
        return inverse_matrix_;
    }

    [[nodiscard]] core::Result<Point3T<Scalar>> apply(const Point3T<Scalar> point) const {
        return transform_point(matrix_, point);
    }

    [[nodiscard]] core::Result<Point3T<Scalar>> apply_inverse(const Point3T<Scalar> point) const {
        return transform_point(inverse_matrix_, point);
    }

  private:
    constexpr ProjectiveTransformT(Matrix4T<Scalar> matrix,
                                   Matrix4T<Scalar> inverse_matrix) noexcept
        : matrix_{matrix}, inverse_matrix_{inverse_matrix} {}

    [[nodiscard]] static core::Result<Point3T<Scalar>>
    transform_point(const Matrix4T<Scalar>& matrix, const Point3T<Scalar> point) {
        const auto x =
            matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2) * point.z + matrix(0, 3);
        const auto y =
            matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2) * point.z + matrix(1, 3);
        const auto z =
            matrix(2, 0) * point.x + matrix(2, 1) * point.y + matrix(2, 2) * point.z + matrix(2, 3);
        const auto w =
            matrix(3, 0) * point.x + matrix(3, 1) * point.y + matrix(3, 2) * point.z + matrix(3, 3);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w) ||
            w == Scalar{0}) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Projective transformation produced an invalid homogeneous point.",
            });
        }

        const auto result = Point3T<Scalar>{.x = x / w, .y = y / w, .z = z / w};
        if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::invalid_argument,
                .message = "Projective division produced a non-finite point.",
            });
        }
        return result;
    }

    Matrix4T<Scalar> matrix_;
    Matrix4T<Scalar> inverse_matrix_;
};

using ProjectiveTransform = ProjectiveTransformT<TransportScalar>;
using ReferenceProjectiveTransform = ProjectiveTransformT<ReferenceScalar>;

} // namespace blackframe::renderer
