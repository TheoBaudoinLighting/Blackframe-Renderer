#include <Blackframe/Renderer/LocalFrame.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

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

template <GeometryScalar Scalar> [[nodiscard]] constexpr Scalar rotation_tolerance() noexcept {
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        return Scalar{8.0e-6F};
    }
    return Scalar{8.0e-14};
}

template <GeometryScalar Scalar>
void expect_frame_near(const OrthonormalFrameT<Scalar>& actual,
                       const OrthonormalFrameT<Scalar>& expected, const Scalar tolerance) {
    expect_vector_near(actual.tangent(), expected.tangent(), tolerance);
    expect_vector_near(actual.bitangent(), expected.bitangent(), tolerance);
    expect_normal_near(actual.normal(), expected.normal(), tolerance);
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

template <GeometryScalar Scalar> void expect_positive_quarter_turn() {
    const auto frame = OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}});
    ASSERT_TRUE(frame.has_value()) << frame.error().message;

    const auto rotated = frame->rotated_about_normal(std::numbers::pi_v<Scalar> / Scalar{2});
    ASSERT_TRUE(rotated.has_value()) << rotated.error().message;
    const auto tolerance = rotation_tolerance<Scalar>();
    expect_orthonormal(*rotated, tolerance);
    expect_vector_near(rotated->tangent(), Vector3T<Scalar>{.y = Scalar{1}}, tolerance);
    expect_vector_near(rotated->bitangent(), Vector3T<Scalar>{.x = Scalar{-1}}, tolerance);
    expect_normal_near(rotated->normal(), frame->normal(), tolerance);
}

TEST(LocalFrameTest, PositiveQuarterTurnRotatesTangentTowardBitangent) {
    expect_positive_quarter_turn<TransportScalar>();
    expect_positive_quarter_turn<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_rotated_world_local_round_trip() {
    const auto frame = OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.x = Scalar{-2}, .y = Scalar{3}, .z = Scalar{5}},
        Vector3T<Scalar>{.x = Scalar{4}, .y = Scalar{1}, .z = Scalar{-2}});
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    constexpr auto angle = Scalar{0.37};
    const auto rotated = frame->rotated_about_normal(angle);
    ASSERT_TRUE(rotated.has_value()) << rotated.error().message;

    const auto tolerance = rotation_tolerance<Scalar>();
    expect_orthonormal(*rotated, tolerance);
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    expect_vector_near(rotated->to_local(frame->tangent()),
                       Vector3T<Scalar>{.x = cosine, .y = -sine}, tolerance);
    expect_vector_near(rotated->to_local(frame->bitangent()),
                       Vector3T<Scalar>{.x = sine, .y = cosine}, tolerance);

    constexpr auto vector = Vector3T<Scalar>{.x = Scalar{0.25}, .y = Scalar{-4}, .z = Scalar{7}};
    constexpr auto normal = Normal3T<Scalar>{.x = Scalar{-1.5}, .y = Scalar{0.75}, .z = Scalar{2}};
    expect_vector_near(rotated->to_world(rotated->to_local(vector)), vector, tolerance);
    expect_normal_near(rotated->to_world(rotated->to_local(normal)), normal, tolerance);
}

TEST(LocalFrameTest, RotatedFrameRoundTripsWorldAndLocalValues) {
    expect_rotated_world_local_round_trip<TransportScalar>();
    expect_rotated_world_local_round_trip<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_periodic_composition() {
    const auto frame = OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.x = Scalar{1}, .y = Scalar{-2}, .z = Scalar{3}},
        Vector3T<Scalar>{.x = Scalar{2}, .y = Scalar{1}});
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    constexpr auto first_angle = Scalar{0.81};
    constexpr auto second_angle = Scalar{-0.44};
    constexpr auto combined_angle = first_angle + second_angle;

    const auto direct = frame->rotated_about_normal(combined_angle);
    const auto periodic =
        frame->rotated_about_normal(combined_angle + Scalar{2} * std::numbers::pi_v<Scalar>);
    const auto first = frame->rotated_about_normal(first_angle);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    ASSERT_TRUE(periodic.has_value()) << periodic.error().message;
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto composed = first->rotated_about_normal(second_angle);
    ASSERT_TRUE(composed.has_value()) << composed.error().message;

    const auto tolerance = rotation_tolerance<Scalar>();
    expect_frame_near(*periodic, *direct, tolerance);
    expect_frame_near(*composed, *direct, tolerance);

    constexpr auto local = Vector3T<Scalar>{.x = Scalar{0.3}, .y = Scalar{-0.7}, .z = Scalar{1.2}};
    expect_vector_near(periodic->to_world(local), direct->to_world(local), tolerance);
    expect_vector_near(composed->to_world(local), direct->to_world(local), tolerance);
}

TEST(LocalFrameTest, TangentRotationIsPeriodicAndCompositionallyCoherent) {
    expect_periodic_composition<TransportScalar>();
    expect_periodic_composition<ReferenceScalar>();
}

template <GeometryScalar Scalar> void expect_invalid_rotations() {
    const auto frame = OrthonormalFrameT<Scalar>::from_normal(Normal3T<Scalar>{.z = Scalar{1}});
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    for (const auto angle : std::array{
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
             -std::numeric_limits<Scalar>::infinity(),
         }) {
        const auto rotated = frame->rotated_about_normal(angle);
        ASSERT_FALSE(rotated.has_value());
        EXPECT_EQ(rotated.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(rotated.error().message.empty());
    }
}

TEST(LocalFrameTest, RejectsNonFiniteTangentRotationsWithoutFallback) {
    expect_invalid_rotations<TransportScalar>();
    expect_invalid_rotations<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
