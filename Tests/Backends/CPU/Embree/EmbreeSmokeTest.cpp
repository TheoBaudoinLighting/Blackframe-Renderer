#include "../../../Engine/AccelBackendContract.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/CapabilityRegistry.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace blackframe::engine {
namespace {

static_assert(std::is_same_v<decltype(&create_analytic_accel_backend), AccelBackendFactory>);
static_assert(std::is_same_v<decltype(&create_embree_accel_backend), AccelBackendFactory>);

[[nodiscard]] FrameSceneHandle
make_embree_spatial_domain_scene(const std::shared_ptr<const TriangleMesh>& mesh,
                                 const renderer::Matrix4& transform) {
    auto scene = FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 901U}}},
        .geometries = {SceneGeometry{.id = {.value = 902U}, .mesh = mesh}},
        .materials = {SceneMaterial{.id = {.value = 903U}}},
        .instances =
            {
                SceneInstance{
                    .id = {.value = 904U},
                    .parent = std::nullopt,
                    .object = {.value = 901U},
                    .geometry = {.value = 902U},
                    .material = {.value = 903U},
                    .local_to_parent = transform,
                },
            },
    });
    if (!scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

[[nodiscard]] renderer::Matrix4 outside_embree_spatial_domain() {
    auto transform = test::scale_matrix(renderer::Vector3{
        .x = 1.0e12F,
        .y = 1.0e12F,
        .z = 1.0F,
    });
    transform(0, 3) = 2.0e18F;
    return transform;
}

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

TEST(AccelBackendParityTest, MatchesAnalyticAnyHitShadowsAndVisibilityMasks) {
    const auto scene = test::make_instanced_scene();
    auto analytic = create_analytic_accel_backend(scene);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = create_embree_accel_backend(scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    test::expect_instanced_occlusion(**analytic, AccelBackendKind::analytic_reference);
    test::expect_instanced_occlusion(**embree, AccelBackendKind::embree);
    test::expect_occlusion_parity(**analytic, **embree);
}

TEST(EmbreeAccelBackendTest, RefitsAnimatedTransformsWithoutAnImplicitRebuild) {
    const auto scenes = test::make_accel_lifecycle_scenes();
    auto backend = create_embree_accel_backend(scenes.initial);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_refit_and_rebuild_lifecycle(**backend, scenes, AccelBackendKind::embree);
}

TEST(AccelBackendParityTest, MatchesAnalyticQueriesAfterAnAnimatedTransformRefit) {
    const auto scenes = test::make_accel_lifecycle_scenes();
    auto analytic = create_analytic_accel_backend(scenes.initial);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto embree = create_embree_accel_backend(scenes.initial);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    test::expect_lifecycle_refit_parity(**analytic, **embree, scenes);
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

TEST(EmbreeAccelBackendTest, RejectsSpatialDomainOverflowWithoutPublishingSceneUpdates) {
    const auto mesh = test::make_triangle_mesh(0.0F);
    const auto committed_scene =
        make_embree_spatial_domain_scene(mesh, test::identity_transform_matrix());
    const auto oversized_scene =
        make_embree_spatial_domain_scene(mesh, outside_embree_spatial_domain());

    const auto rejected_create = create_embree_accel_backend(oversized_scene);
    ASSERT_FALSE(rejected_create.has_value());
    EXPECT_EQ(rejected_create.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(rejected_create.error().message.find("1.844e18"), std::string::npos);

    auto backend = create_embree_accel_backend(committed_scene);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    EXPECT_EQ((*backend)->build_statistics(), (AccelBuildStatistics{.commits = 1U}));

    const auto ray = test::make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                    renderer::Vector3{.z = 1.0F});
    const auto original_hit = (*backend)->closest_hit(ray);
    ASSERT_TRUE(original_hit.has_value()) << original_hit.error().message;
    ASSERT_TRUE(original_hit->has_value());

    const auto rejected_refit = (*backend)->refit(oversized_scene);
    ASSERT_FALSE(rejected_refit.has_value());
    EXPECT_EQ(rejected_refit.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(rejected_refit.error().message.find("1.844e18"), std::string::npos);
    EXPECT_EQ((*backend)->build_statistics(), (AccelBuildStatistics{.commits = 1U}));
    const auto hit_after_refit = (*backend)->closest_hit(ray);
    ASSERT_TRUE(hit_after_refit.has_value()) << hit_after_refit.error().message;
    EXPECT_EQ(*hit_after_refit, *original_hit);

    const auto rejected_rebuild = (*backend)->rebuild(oversized_scene);
    ASSERT_FALSE(rejected_rebuild.has_value());
    EXPECT_EQ(rejected_rebuild.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(rejected_rebuild.error().message.find("1.844e18"), std::string::npos);
    EXPECT_EQ((*backend)->build_statistics(), (AccelBuildStatistics{.commits = 1U}));
    const auto hit_after_rebuild = (*backend)->closest_hit(ray);
    ASSERT_TRUE(hit_after_rebuild.has_value()) << hit_after_rebuild.error().message;
    EXPECT_EQ(*hit_after_rebuild, *original_hit);
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
