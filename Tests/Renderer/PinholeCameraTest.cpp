#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<OrthonormalFrameT<Scalar>> identity_camera_frame() {
    return OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{1}},
        Vector3T<Scalar>{.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{0}});
}

TEST(PinholeCameraTest, GeneratesTheCentralPrimaryRayExactly) {
    static_assert(!std::is_same_v<PinholeCamera, ReferencePinholeCamera>);

    const auto frame = identity_camera_frame<TransportScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto origin = Point3{.x = 1.0F, .y = -2.0F, .z = 3.0F};
    const auto medium = MediumId{.value = 9};
    const auto camera = PinholeCamera::create(origin, *frame, RenderExtent{.width = 3, .height = 3},
                                              1.0F, 0.25F, 100.0F, 0x00FF00FFU, medium);
    ASSERT_TRUE(camera.has_value());

    const auto ray = camera->generate_primary_ray(Point2{.x = 1.5F, .y = 1.5F}, 0.75F);
    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(ray->origin(), origin);
    EXPECT_EQ(ray->direction(), (Vector3{.x = 0.0F, .y = 0.0F, .z = -1.0F}));
    EXPECT_FLOAT_EQ(ray->t_min(), 0.25F);
    EXPECT_FLOAT_EQ(ray->t_max(), 100.0F);
    EXPECT_FLOAT_EQ(ray->time(), 0.75F);
    EXPECT_EQ(ray->mask(), 0x00FF00FFU);
    EXPECT_EQ(ray->current_medium(), medium);
}

TEST(PinholeCameraTest, MapsUpperLeftRasterSamplesToNegativeXAndPositiveY) {
    const auto frame = identity_camera_frame<ReferenceScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto camera = ReferencePinholeCamera::create(
        ReferencePoint3{}, *frame, RenderExtent{.width = 2, .height = 2},
        std::numbers::pi_v<ReferenceScalar> / 2.0, 0.0,
        std::numeric_limits<ReferenceScalar>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    const auto upper_left = camera->generate_primary_ray(ReferencePoint2{.x = 0.5, .y = 0.5}, 0.0);
    const auto lower_right = camera->generate_primary_ray(ReferencePoint2{.x = 1.5, .y = 1.5}, 0.0);
    ASSERT_TRUE(upper_left.has_value());
    ASSERT_TRUE(lower_right.has_value());
    EXPECT_LT(upper_left->direction().x, 0.0);
    EXPECT_GT(upper_left->direction().y, 0.0);
    EXPECT_LT(upper_left->direction().z, 0.0);
    EXPECT_DOUBLE_EQ(lower_right->direction().x, -upper_left->direction().x);
    EXPECT_DOUBLE_EQ(lower_right->direction().y, -upper_left->direction().y);
    EXPECT_DOUBLE_EQ(lower_right->direction().z, upper_left->direction().z);
}

TEST(PinholeCameraTest, MapsTheCameraBasisToWorldSpace) {
    const auto frame = OrthonormalFrame::from_normal_and_tangent(
        Normal3{.x = 1.0F, .y = 0.0F, .z = 0.0F}, Vector3{.x = 0.0F, .y = 1.0F, .z = 0.0F});
    ASSERT_TRUE(frame.has_value());
    const auto camera = PinholeCamera::create(Point3{.x = 4.0F, .y = 5.0F, .z = 6.0F}, *frame,
                                              RenderExtent{.width = 1, .height = 1}, 1.0F, 0.0F,
                                              1.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    const auto ray = camera->generate_primary_ray(Point2{.x = 0.5F, .y = 0.5F}, 0.0F);
    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(ray->origin(), (Point3{.x = 4.0F, .y = 5.0F, .z = 6.0F}));
    EXPECT_EQ(ray->direction(), (Vector3{.x = -1.0F, .y = 0.0F, .z = 0.0F}));
}

TEST(PinholeCameraTest, GeneratesPositiveOnePixelDifferentialsWithStableMetadata) {
    const auto frame = identity_camera_frame<ReferenceScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto origin = ReferencePoint3{.x = 1.0, .y = -2.0, .z = 3.0};
    const auto medium = MediumId{.value = 17U};
    const auto camera = ReferencePinholeCamera::create(
        origin, *frame, RenderExtent{.width = 4U, .height = 4U},
        std::numbers::pi_v<ReferenceScalar> / 2.0, 0.125, 12.0, 0x13579BDFU, medium);
    ASSERT_TRUE(camera.has_value());

    const auto differential =
        camera->generate_primary_ray_differential(ReferencePoint2{.x = 2.0, .y = 2.0}, 0.75);
    ASSERT_TRUE(differential.has_value()) << differential.error().message;
    EXPECT_EQ(differential->ray().origin(), origin);
    EXPECT_EQ(differential->ray().direction(), (ReferenceVector3{.z = -1.0}));
    EXPECT_DOUBLE_EQ(differential->ray().t_min(), 0.125);
    EXPECT_DOUBLE_EQ(differential->ray().t_max(), 12.0);
    EXPECT_DOUBLE_EQ(differential->ray().time(), 0.75);
    EXPECT_EQ(differential->ray().mask(), 0x13579BDFU);
    EXPECT_EQ(differential->ray().current_medium(), medium);
    EXPECT_EQ(differential->rx_origin(), origin);
    EXPECT_EQ(differential->ry_origin(), origin);

    const auto inverse_sqrt_five = 1.0 / std::sqrt(5.0);
    EXPECT_NEAR(differential->rx_direction().x, inverse_sqrt_five, 2.0e-15);
    EXPECT_DOUBLE_EQ(differential->rx_direction().y, 0.0);
    EXPECT_NEAR(differential->rx_direction().z, -2.0 * inverse_sqrt_five, 2.0e-15);
    EXPECT_DOUBLE_EQ(differential->ry_direction().x, 0.0);
    EXPECT_NEAR(differential->ry_direction().y, -inverse_sqrt_five, 2.0e-15);
    EXPECT_NEAR(differential->ry_direction().z, -2.0 * inverse_sqrt_five, 2.0e-15);
}

TEST(PinholeCameraTest, ExtrapolatesForwardDifferentialsBeyondTheFilmEdges) {
    const auto frame = identity_camera_frame<TransportScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto camera = PinholeCamera::create(
        Point3{}, *frame, RenderExtent{.width = 4U, .height = 4U},
        std::numbers::pi_v<TransportScalar> / 2.0F, 0.0F,
        std::numeric_limits<TransportScalar>::infinity(), AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    const auto differential =
        camera->generate_primary_ray_differential(Point2{.x = 3.5F, .y = 3.5F}, 0.0F);
    ASSERT_TRUE(differential.has_value()) << differential.error().message;
    EXPECT_GT(differential->rx_direction().x, differential->ray().direction().x);
    EXPECT_LT(differential->ry_direction().y, differential->ray().direction().y);
    EXPECT_NE(differential->rx_direction(), differential->ray().direction());
    EXPECT_NE(differential->ry_direction(), differential->ray().direction());
}

TEST(PinholeCameraTest, RejectsOnePixelOffsetsLostToProjectionPrecision) {
    const auto frame = identity_camera_frame<TransportScalar>();
    ASSERT_TRUE(frame.has_value());
    constexpr auto tiny_field_of_view = 2.0F * std::numeric_limits<TransportScalar>::denorm_min();
    const auto camera =
        PinholeCamera::create(Point3{}, *frame, RenderExtent{.width = 4U, .height = 4U},
                              tiny_field_of_view, 0.0F, 1.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());

    const auto unresolved =
        camera->generate_primary_ray_differential(Point2{.x = 2.0F, .y = 2.0F}, 0.0F);

    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(unresolved.error().message.empty());
}

TEST(PinholeCameraTest, RejectsInvalidConfigurationWithoutFallback) {
    const auto frame = identity_camera_frame<TransportScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto create = [&](const Point3 origin, const RenderExtent extent,
                            const TransportScalar field_of_view, const TransportScalar t_min,
                            const TransportScalar t_max) {
        return PinholeCamera::create(origin, *frame, extent, field_of_view, t_min, t_max,
                                     AllRayVisibility, VacuumMedium);
    };
    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();

    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 0, .height = 1}, 1.0F, 0.0F, 1.0F).has_value());
    EXPECT_FALSE(
        create(Point3{.x = infinity}, RenderExtent{.width = 1, .height = 1}, 1.0F, 0.0F, 1.0F)
            .has_value());
    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 1, .height = 1}, 0.0F, 0.0F, 1.0F).has_value());
    EXPECT_FALSE(create(Point3{}, RenderExtent{.width = 1, .height = 1},
                        std::numbers::pi_v<TransportScalar>, 0.0F, 1.0F)
                     .has_value());
    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 1, .height = 1}, nan, 0.0F, 1.0F).has_value());
    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 1, .height = 1}, 1.0F, -1.0F, 1.0F).has_value());
    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 1, .height = 1}, 1.0F, 2.0F, 1.0F).has_value());
    EXPECT_FALSE(
        create(Point3{}, RenderExtent{.width = 1, .height = 1}, 1.0F, 0.0F, nan).has_value());
}

TEST(PinholeCameraTest, RejectsInvalidRasterSamplesAndTimeWithoutFallback) {
    const auto frame = identity_camera_frame<TransportScalar>();
    ASSERT_TRUE(frame.has_value());
    const auto camera =
        PinholeCamera::create(Point3{}, *frame, RenderExtent{.width = 2, .height = 2}, 1.0F, 0.0F,
                              1.0F, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(camera.has_value());
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();
    const auto infinity = std::numeric_limits<TransportScalar>::infinity();

    EXPECT_FALSE(camera->generate_primary_ray(Point2{.x = -0.5F, .y = 0.5F}, 0.0F).has_value());
    EXPECT_FALSE(camera->generate_primary_ray(Point2{.x = 2.0F, .y = 0.5F}, 0.0F).has_value());
    EXPECT_FALSE(camera->generate_primary_ray(Point2{.x = 0.5F, .y = 2.0F}, 0.0F).has_value());
    EXPECT_FALSE(camera->generate_primary_ray(Point2{.x = nan, .y = 0.5F}, 0.0F).has_value());
    EXPECT_FALSE(camera->generate_primary_ray(Point2{.x = 0.5F, .y = 0.5F}, infinity).has_value());
    EXPECT_FALSE(
        camera->generate_primary_ray_differential(Point2{.x = 2.0F, .y = 0.5F}, 0.0F).has_value());
    EXPECT_FALSE(camera->generate_primary_ray_differential(Point2{.x = 0.5F, .y = 0.5F}, infinity)
                     .has_value());
}

} // namespace
} // namespace blackframe::renderer
