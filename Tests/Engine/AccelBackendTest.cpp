#include "AccelBackendContract.hpp"

#include <Blackframe/Engine/AccelBackend.hpp>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <type_traits>

namespace blackframe::engine {
namespace {

static_assert(std::is_same_v<decltype(&create_analytic_accel_backend), AccelBackendFactory>);

TEST(AnalyticAccelBackendTest, ImplementsTheSharedClosestHitAndOcclusionContract) {
    const auto geometries = test::make_geometry_set();
    auto backend = create_analytic_accel_backend(geometries);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_backend_contract(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, RejectsMissingMeshesAndAmbiguousSurfaceIdentities) {
    const auto missing_mesh = std::array{AccelGeometry{}};
    const auto missing = create_analytic_accel_backend(missing_mesh);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(missing.error().message.empty());

    const auto mesh = test::make_triangle_mesh(0.0F);
    const auto duplicates = std::array{
        AccelGeometry{
            .mesh = mesh,
            .instance = renderer::InstanceId{.value = 9U},
            .geometry = renderer::GeometryId{.value = 10U},
        },
        AccelGeometry{
            .mesh = mesh,
            .instance = renderer::InstanceId{.value = 9U},
            .geometry = renderer::GeometryId{.value = 10U},
        },
    };
    const auto duplicate = create_analytic_accel_backend(duplicates);
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(duplicate.error().message.empty());
}

TEST(AnalyticAccelBackendTest, RepresentsAnEmptyWorldWithoutSyntheticGeometry) {
    auto backend = create_analytic_accel_backend(std::span<const AccelGeometry>{});
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_empty_backend(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, RetainsTheImmutableMeshesItWasBuiltFrom) {
    auto mesh = test::make_triangle_mesh(0.0F);
    const auto weak_mesh = std::weak_ptr<const TriangleMesh>{mesh};
    auto geometries = std::array{AccelGeometry{
        .mesh = mesh,
        .instance = renderer::InstanceId{.value = 1U},
        .geometry = renderer::GeometryId{.value = 2U},
        .material = renderer::MaterialId{.value = 3U},
    }};
    auto backend = create_analytic_accel_backend(geometries);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    geometries[0].mesh.reset();
    mesh.reset();
    EXPECT_FALSE(weak_mesh.expired());

    const auto ray = test::make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                    renderer::Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    const auto hit = (*backend)->closest_hit(ray);
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    EXPECT_TRUE(hit->has_value());

    backend->reset();
    EXPECT_TRUE(weak_mesh.expired());
}

} // namespace
} // namespace blackframe::engine
