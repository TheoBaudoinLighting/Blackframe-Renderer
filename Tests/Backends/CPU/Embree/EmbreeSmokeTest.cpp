#include "../../../Engine/AccelBackendContract.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <vector>

namespace blackframe::engine {
namespace {

static_assert(std::is_same_v<decltype(&create_analytic_accel_backend), AccelBackendFactory>);
static_assert(std::is_same_v<decltype(&create_embree_accel_backend), AccelBackendFactory>);

TEST(AccelBackendSelectionTest, SelectsBothImplementationsThroughTheSameSceneInterface) {
    auto scene = test::make_basic_scene();
    const auto weak_scene = std::weak_ptr<const FrameScene>{scene};
    auto retained_meshes = std::vector<std::weak_ptr<const TriangleMesh>>{};
    retained_meshes.reserve(scene->geometries().size());
    for (const auto& geometry : scene->geometries()) {
        retained_meshes.emplace_back(geometry.mesh);
    }
    const auto capability = renderer::require_backend_capability("cpu_embree");
    ASSERT_TRUE(capability.has_value()) << capability.error().message;

    const AccelBackendFactory analytic_factory = &create_analytic_accel_backend;
    const AccelBackendFactory embree_factory = &create_embree_accel_backend;
    auto analytic = analytic_factory(scene);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = embree_factory(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    scene.reset();
    EXPECT_FALSE(weak_scene.expired());
    for (const auto& mesh : retained_meshes) {
        EXPECT_FALSE(mesh.expired());
    }

    test::expect_backend_contract(**analytic, AccelBackendKind::analytic_reference);
    test::expect_backend_contract(**embree, AccelBackendKind::embree);

    analytic->reset();
    EXPECT_FALSE(weak_scene.expired());
    for (const auto& mesh : retained_meshes) {
        EXPECT_FALSE(mesh.expired());
    }
    embree->reset();
    EXPECT_TRUE(weak_scene.expired());
    for (const auto& mesh : retained_meshes) {
        EXPECT_TRUE(mesh.expired());
    }
}

TEST(AccelBackendParityTest, MatchesAnalyticClosestHitsForHierarchicalInstances) {
    const auto scene = test::make_instanced_scene();
    auto analytic = create_analytic_accel_backend(scene);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    test::expect_instanced_closest_hits(**analytic, AccelBackendKind::analytic_reference);
    test::expect_instanced_closest_hits(**embree, AccelBackendKind::embree);
    test::expect_closest_hit_parity(**analytic, **embree);
}

TEST(EmbreeAccelBackendTest, RejectsMissingScenesAndUnrepresentableInstances) {
    const auto missing = create_embree_accel_backend({});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(missing.error().message.empty());

    const auto unrepresentable =
        create_embree_accel_backend(test::make_unrepresentable_instance_scene());
    ASSERT_FALSE(unrepresentable.has_value());
    EXPECT_EQ(unrepresentable.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(unrepresentable.error().message.empty());
}

TEST(AccelBackendSelectionTest, BothFactoriesRepresentAnEmptyWorldExplicitly) {
    const auto scene = test::make_empty_scene();
    auto analytic = create_analytic_accel_backend(scene);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    test::expect_empty_backend(**analytic, AccelBackendKind::analytic_reference);
    test::expect_empty_backend(**embree, AccelBackendKind::embree);
}

} // namespace
} // namespace blackframe::engine
