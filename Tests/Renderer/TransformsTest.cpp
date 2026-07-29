#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/Quaternion.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <typename Transform, typename Value>
concept Applicable =
    requires(const Transform& transform, const Value value) { transform.apply(value); };

template <GeometryScalar Scalar>
void expect_vector_near(const Vector3T<Scalar> actual, const Vector3T<Scalar> expected,
                        const Scalar tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

template <GeometryScalar Scalar>
void expect_point_near(const Point3T<Scalar> actual, const Point3T<Scalar> expected,
                       const Scalar tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

template <GeometryScalar Scalar>
void expect_normal_near(const Normal3T<Scalar> actual, const Normal3T<Scalar> expected,
                        const Scalar tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

TEST(MatrixTypesTest, UsesExplicitPrecisionAndCompositionOrder) {
    static_assert(!std::same_as<Matrix4, ReferenceMatrix4>);
    static_assert(!std::convertible_to<Matrix4, ReferenceMatrix4>);
    static_assert(!std::same_as<Matrix4, Quaternion>);

    EXPECT_EQ(sizeof(Matrix4), 64U);
    EXPECT_EQ(sizeof(ReferenceMatrix4), 128U);

    constexpr auto identity = identity_matrix<TransportScalar>();
    constexpr auto matrix = Matrix4{.elements = {
                                        1.0F,
                                        2.0F,
                                        3.0F,
                                        4.0F,
                                        5.0F,
                                        6.0F,
                                        7.0F,
                                        8.0F,
                                        9.0F,
                                        10.0F,
                                        11.0F,
                                        12.0F,
                                        0.0F,
                                        0.0F,
                                        0.0F,
                                        1.0F,
                                    }};
    static_assert(identity * matrix == matrix);
    static_assert(matrix * identity == matrix);
    static_assert(transposed(transposed(matrix)) == matrix);

    EXPECT_EQ(identity * matrix, matrix);
    EXPECT_EQ(matrix * identity, matrix);
}

TEST(MatrixOperationsTest, InvertsAComposedMatrix) {
    auto matrix = identity_matrix<ReferenceScalar>();
    matrix(0, 0) = 2.0;
    matrix(1, 1) = -3.0;
    matrix(2, 2) = 0.5;
    matrix(0, 3) = 7.0;
    matrix(1, 3) = -4.0;
    matrix(2, 3) = 2.0;

    const auto inverted = inverse(matrix);
    ASSERT_TRUE(inverted.has_value());
    const auto product = matrix * *inverted;
    const auto identity = identity_matrix<ReferenceScalar>();
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            EXPECT_NEAR(product(row, column), identity(row, column), 1.0e-12);
        }
    }
}

TEST(MatrixOperationsTest, RejectsSingularMatricesWithoutIdentityFallback) {
    auto singular = identity_matrix<TransportScalar>();
    singular(2, 2) = 0.0F;

    const auto inverted = inverse(singular);

    ASSERT_FALSE(inverted.has_value());
    EXPECT_EQ(inverted.error().code, core::StatusCode::invalid_argument);
}

TEST(QuaternionTest, ComposesRotationAndProvidesAnInverse) {
    const auto rotation = quaternion_from_axis_angle(Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
                                                     std::numbers::pi_v<TransportScalar> / 2.0F);
    ASSERT_TRUE(rotation.has_value());

    const auto inverted = inverse(*rotation);
    ASSERT_TRUE(inverted.has_value());
    const auto identity = *rotation * *inverted;
    EXPECT_NEAR(identity.x, 0.0F, 1.0e-6F);
    EXPECT_NEAR(identity.y, 0.0F, 1.0e-6F);
    EXPECT_NEAR(identity.z, 0.0F, 1.0e-6F);
    EXPECT_NEAR(identity.w, 1.0F, 1.0e-6F);

    const auto matrix = rotation_matrix(*rotation);
    ASSERT_TRUE(matrix.has_value());
    EXPECT_NEAR((*matrix)(0, 0), 0.0F, 1.0e-6F);
    EXPECT_NEAR((*matrix)(0, 1), -1.0F, 1.0e-6F);
    EXPECT_NEAR((*matrix)(1, 0), 1.0F, 1.0e-6F);
}

TEST(QuaternionTest, RejectsInvalidRotationInputs) {
    const auto zero_axis = quaternion_from_axis_angle(Vector3{}, 1.0F);
    ASSERT_FALSE(zero_axis.has_value());

    const auto zero_inverse = inverse(Quaternion{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 0.0F});
    ASSERT_FALSE(zero_inverse.has_value());

    const auto non_unit_matrix =
        rotation_matrix(Quaternion{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 2.0F});
    ASSERT_FALSE(non_unit_matrix.has_value());
    EXPECT_EQ(non_unit_matrix.error().code, core::StatusCode::invalid_argument);
}

TEST(AffineTransformTest, PreservesPointVectorAndNormalSemantics) {
    static_assert(Applicable<AffineTransform, Point3>);
    static_assert(Applicable<AffineTransform, Vector3>);
    static_assert(Applicable<AffineTransform, Normal3>);
    static_assert(!Applicable<ProjectiveTransform, Vector3>);
    static_assert(!Applicable<ProjectiveTransform, Normal3>);

    const auto translation =
        AffineTransform::translation(Vector3{.x = 3.0F, .y = -2.0F, .z = 5.0F});
    ASSERT_TRUE(translation.has_value());

    EXPECT_EQ(translation->apply(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}),
              (Point3{.x = 4.0F, .y = 0.0F, .z = 8.0F}));
    EXPECT_EQ(translation->apply(Vector3{.x = 1.0F, .y = 2.0F, .z = 3.0F}),
              (Vector3{.x = 1.0F, .y = 2.0F, .z = 3.0F}));
    EXPECT_EQ(translation->apply(Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F}),
              (Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
}

TEST(AffineTransformTest, RoundTripsThroughStoredInverseWithinTolerance) {
    const auto rotation =
        quaternion_from_axis_angle(ReferenceVector3{.x = 1.0, .y = 2.0, .z = -1.0}, 0.7);
    ASSERT_TRUE(rotation.has_value());
    const auto rotation_part = rotation_matrix(*rotation);
    ASSERT_TRUE(rotation_part.has_value());

    auto translation_part = identity_matrix<ReferenceScalar>();
    translation_part(0, 3) = 4.0;
    translation_part(1, 3) = -3.0;
    translation_part(2, 3) = 2.0;

    auto scale_part = identity_matrix<ReferenceScalar>();
    scale_part(0, 0) = 2.0;
    scale_part(1, 1) = 0.5;
    scale_part(2, 2) = -1.5;

    const auto transform =
        ReferenceAffineTransform::from_matrix(translation_part * *rotation_part * scale_part);
    ASSERT_TRUE(transform.has_value());

    constexpr auto point = ReferencePoint3{.x = 0.25, .y = -1.0, .z = 3.0};
    constexpr auto vector = ReferenceVector3{.x = -2.0, .y = 0.5, .z = 1.0};
    constexpr auto normal = ReferenceNormal3{.x = 0.0, .y = 1.0, .z = 0.0};
    constexpr auto tolerance = 1.0e-12;

    expect_point_near(transform->apply_inverse(transform->apply(point)), point, tolerance);
    expect_vector_near(transform->apply_inverse(transform->apply(vector)), vector, tolerance);
    expect_normal_near(transform->apply_inverse(transform->apply(normal)), normal, tolerance);
}

TEST(AffineTransformTest, RejectsSingularAndProjectiveMatrices) {
    auto singular = identity_matrix<TransportScalar>();
    singular(1, 1) = 0.0F;
    const auto singular_transform = AffineTransform::from_matrix(singular);
    ASSERT_FALSE(singular_transform.has_value());

    auto projective = identity_matrix<TransportScalar>();
    projective(3, 2) = -1.0F;
    const auto affine_transform = AffineTransform::from_matrix(projective);
    ASSERT_FALSE(affine_transform.has_value());
    EXPECT_EQ(affine_transform.error().code, core::StatusCode::invalid_argument);
}

TEST(ProjectiveTransformTest, RoundTripsPerspectiveProjectionWithinTolerance) {
    const auto projection = ReferenceProjectiveTransform::perspective(
        std::numbers::pi_v<ReferenceScalar> / 3.0, 16.0 / 9.0, 0.1, 100.0);
    ASSERT_TRUE(projection.has_value());

    constexpr auto point = ReferencePoint3{.x = 0.5, .y = -0.25, .z = -3.0};
    const auto projected = projection->apply(point);
    ASSERT_TRUE(projected.has_value());
    const auto restored = projection->apply_inverse(*projected);
    ASSERT_TRUE(restored.has_value());
    expect_point_near(*restored, point, 1.0e-11);
}

TEST(ProjectiveTransformTest, RejectsInvalidVolumesAndHomogeneousDivision) {
    const auto invalid_volume = ProjectiveTransform::perspective(0.0F, 1.0F, 0.1F, 100.0F);
    ASSERT_FALSE(invalid_volume.has_value());

    const auto projection = ProjectiveTransform::perspective(
        std::numbers::pi_v<TransportScalar> / 3.0F, 1.0F, 0.1F, 100.0F);
    ASSERT_TRUE(projection.has_value());
    const auto camera_plane = projection->apply(Point3{});
    ASSERT_FALSE(camera_plane.has_value());
    EXPECT_EQ(camera_plane.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
