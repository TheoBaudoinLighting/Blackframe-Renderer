#include <Blackframe/Renderer/LocalFrame.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <typename Frame, typename Value>
concept ConvertsToLocal =
    requires(const Frame& frame, const Value value) { frame.to_local(value); };

template <GeometryScalar Scalar>
void expect_orthonormal(const OrthonormalFrameT<Scalar>& frame, const Scalar tolerance) {
    const auto normal_vector = Vector3T<Scalar>{
        .x = frame.normal().x,
        .y = frame.normal().y,
        .z = frame.normal().z,
    };
    EXPECT_NEAR(length_squared(frame.tangent()), Scalar{1}, tolerance);
    EXPECT_NEAR(length_squared(frame.bitangent()), Scalar{1}, tolerance);
    EXPECT_NEAR(length_squared(frame.normal()), Scalar{1}, tolerance);
    EXPECT_NEAR(dot(frame.tangent(), frame.bitangent()), Scalar{0}, tolerance);
    EXPECT_NEAR(dot(frame.tangent(), frame.normal()), Scalar{0}, tolerance);
    EXPECT_NEAR(dot(frame.bitangent(), frame.normal()), Scalar{0}, tolerance);
    EXPECT_NEAR(dot(cross(frame.tangent(), frame.bitangent()), normal_vector), Scalar{1},
                tolerance);
}

template <GeometryScalar Scalar>
void expect_vector_near(const Vector3T<Scalar> actual, const Vector3T<Scalar> expected,
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

TEST(LocalFrameTest, KeepsFrameSemanticsAndPrecisionExplicit) {
    static_assert(!std::same_as<OrthonormalFrame, ReferenceOrthonormalFrame>);
    static_assert(ConvertsToLocal<OrthonormalFrame, Vector3>);
    static_assert(ConvertsToLocal<OrthonormalFrame, Normal3>);
    static_assert(!ConvertsToLocal<OrthonormalFrame, Point3>);
    static_assert(!ConvertsToLocal<OrthonormalFrame, ReferenceVector3>);
}

TEST(LocalFrameTest, RemainsOrthonormalAtAxisAndDegeneratePoleCases) {
    constexpr auto smallest = std::numeric_limits<TransportScalar>::denorm_min();
    constexpr auto largest = std::numeric_limits<TransportScalar>::max();
    constexpr auto cases = std::array{
        Normal3{.x = 1.0F, .y = 0.0F, .z = 0.0F},
        Normal3{.x = -1.0F, .y = 0.0F, .z = 0.0F},
        Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F},
        Normal3{.x = 0.0F, .y = -1.0F, .z = 0.0F},
        Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
        Normal3{.x = 0.0F, .y = 0.0F, .z = -1.0F},
        Normal3{.x = 1.0e-20F, .y = -2.0e-20F, .z = -1.0F},
        Normal3{.x = smallest, .y = 0.0F, .z = 0.0F},
        Normal3{.x = largest / 2.0F, .y = -largest / 4.0F, .z = largest / 8.0F},
    };

    for (const auto normal : cases) {
        const auto frame = OrthonormalFrame::from_normal(normal);
        ASSERT_TRUE(frame.has_value());
        expect_orthonormal(*frame, 2.0e-5F);
    }
}

TEST(LocalFrameTest, HandlesTheReferenceSouthPoleWithoutPrecisionLoss) {
    const auto frame = ReferenceOrthonormalFrame::from_normal(
        ReferenceNormal3{.x = 1.0e-14, .y = -2.0e-14, .z = -1.0});

    ASSERT_TRUE(frame.has_value());
    expect_orthonormal(*frame, 5.0e-14);
}

TEST(LocalFrameTest, OrthogonalizesAnExplicitTangent) {
    const auto frame = OrthonormalFrame::from_normal_and_tangent(
        Normal3{.x = 0.0F, .y = 0.0F, .z = 2.0F}, Vector3{.x = 2.0F, .y = 1.0F, .z = 8.0F});

    ASSERT_TRUE(frame.has_value());
    expect_orthonormal(*frame, 2.0e-5F);
    EXPECT_GT(dot(frame->tangent(), Vector3{.x = 2.0F, .y = 1.0F, .z = 0.0F}), 0.0F);
}

TEST(LocalFrameTest, RejectsDegenerateInputsWithoutChoosingAnotherAxis) {
    const auto zero_normal = OrthonormalFrame::from_normal(Normal3{});
    ASSERT_FALSE(zero_normal.has_value());
    EXPECT_EQ(zero_normal.error().code, core::StatusCode::invalid_argument);

    const auto non_finite_normal = OrthonormalFrame::from_normal(Normal3{
        .x = std::numeric_limits<TransportScalar>::infinity(),
        .y = 0.0F,
        .z = 0.0F,
    });
    ASSERT_FALSE(non_finite_normal.has_value());

    const auto parallel_tangent = OrthonormalFrame::from_normal_and_tangent(
        Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, Vector3{.x = 0.0F, .y = 0.0F, .z = 4.0F});
    ASSERT_FALSE(parallel_tangent.has_value());
    EXPECT_EQ(parallel_tangent.error().code, core::StatusCode::invalid_argument);
}

TEST(LocalFrameTest, MapsItsBasisToCanonicalLocalAxes) {
    const auto frame = OrthonormalFrame::from_normal(Normal3{.x = 1.0F, .y = -2.0F, .z = 3.0F});
    ASSERT_TRUE(frame.has_value());

    expect_vector_near(frame->to_local(frame->tangent()), Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F},
                       2.0e-6F);
    expect_vector_near(frame->to_local(frame->bitangent()),
                       Vector3{.x = 0.0F, .y = 1.0F, .z = 0.0F}, 2.0e-6F);
    expect_normal_near(frame->to_local(frame->normal()), Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
                       2.0e-6F);
}

TEST(LocalFrameTest, RoundTripsWorldAndLocalDirectionsWithinTolerance) {
    const auto frame = ReferenceOrthonormalFrame::from_normal_and_tangent(
        ReferenceNormal3{.x = -2.0, .y = 3.0, .z = 5.0},
        ReferenceVector3{.x = 4.0, .y = 1.0, .z = -2.0});
    ASSERT_TRUE(frame.has_value());

    constexpr auto vector = ReferenceVector3{.x = 0.25, .y = -4.0, .z = 7.0};
    constexpr auto normal = ReferenceNormal3{.x = -1.5, .y = 0.75, .z = 2.0};
    constexpr auto tolerance = 2.0e-13;

    expect_vector_near(frame->to_world(frame->to_local(vector)), vector, tolerance);
    expect_normal_near(frame->to_world(frame->to_local(normal)), normal, tolerance);
}

} // namespace
} // namespace blackframe::renderer
