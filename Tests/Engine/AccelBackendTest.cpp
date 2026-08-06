#include "AccelBackendContract.hpp"

#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

static_assert(std::is_same_v<decltype(&create_analytic_accel_backend), AccelBackendFactory>);

TEST(AnalyticAccelBackendTest, ImplementsTheSharedClosestHitAndOcclusionContract) {
    auto backend = create_analytic_accel_backend(test::make_basic_scene());
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_backend_contract(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, ResolvesHierarchicalInstancesInWorldSpace) {
    auto backend = create_analytic_accel_backend(test::make_instanced_scene());
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_instanced_closest_hits(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, EvaluatesAnyHitShadowsAndVisibilityMasks) {
    auto backend = create_analytic_accel_backend(test::make_instanced_scene());
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_instanced_occlusion(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, RefitsAnimatedTransformsWithoutAnImplicitRebuild) {
    const auto scenes = test::make_accel_lifecycle_scenes();
    auto backend = create_analytic_accel_backend(scenes.initial);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_refit_and_rebuild_lifecycle(**backend, scenes,
                                             AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, RejectsPunctualLightChangesDuringTransformOnlyRefit) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    ASSERT_TRUE(wavelengths.has_value()) << wavelengths.error().message;
    const auto make_scene = [&](const float intensity) {
        auto description = FrameSceneDescription{};
        description.punctual_lights = {
            ScenePointLight{
                .position = {.z = 2.0F},
                .absolute_position_error = {},
                .spectral_radiant_intensity =
                    renderer::TransportSpectrum{
                        .values = {intensity, intensity, intensity, intensity},
                    },
            },
        };
        description.spectral_environment = SceneSpectralEnvironment{
            .wavelengths = *wavelengths,
            .radiance = {},
        };
        return FrameScene::create(std::move(description));
    };

    const auto initial = make_scene(1.0F);
    const auto changed = make_scene(2.0F);
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    ASSERT_TRUE(changed.has_value()) << changed.error().message;
    auto backend = create_analytic_accel_backend(*initial);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    const auto rejected = (*backend)->refit(*changed);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::incompatible);
    EXPECT_EQ((*backend)->frame_scene(), *initial);
    EXPECT_EQ((*backend)->build_statistics(),
              (AccelBuildStatistics{.commits = 1U, .rebuilds = 0U, .refits = 0U}));

    const auto rebuilt = (*backend)->rebuild(*changed);
    ASSERT_TRUE(rebuilt.has_value()) << rebuilt.error().message;
    EXPECT_EQ((*backend)->frame_scene(), *changed);
    EXPECT_EQ((*backend)->build_statistics(),
              (AccelBuildStatistics{.commits = 2U, .rebuilds = 1U, .refits = 0U}));
}

TEST(AnalyticAccelBackendTest, RejectsConstantTextureChangesDuringTransformOnlyRefit) {
    const auto make_scene = [](const float value) {
        auto description = FrameSceneDescription{};
        description.constant_textures = {
            SceneConstantTexture{
                .id = {.value = 7U},
                .texture = renderer::ConstantFloatTexture::create(value).value(),
            },
        };
        return FrameScene::create(std::move(description));
    };

    const auto initial = make_scene(1.0F);
    const auto changed = make_scene(2.0F);
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    ASSERT_TRUE(changed.has_value()) << changed.error().message;
    auto backend = create_analytic_accel_backend(*initial);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    const auto rejected = (*backend)->refit(*changed);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::incompatible);
    EXPECT_EQ((*backend)->frame_scene(), *initial);
    EXPECT_EQ((*backend)->build_statistics(),
              (AccelBuildStatistics{.commits = 1U, .rebuilds = 0U, .refits = 0U}));

    const auto rebuilt = (*backend)->rebuild(*changed);
    ASSERT_TRUE(rebuilt.has_value()) << rebuilt.error().message;
    EXPECT_EQ((*backend)->frame_scene(), *changed);
    EXPECT_EQ((*backend)->build_statistics(),
              (AccelBuildStatistics{.commits = 2U, .rebuilds = 1U, .refits = 0U}));
}

TEST(AnalyticAccelBackendTest, DistinguishesSignedZeroTextureChangesDuringRefit) {
    const auto make_scene = [](const float value) {
        auto description = FrameSceneDescription{};
        description.constant_textures = {
            SceneConstantTexture{
                .id = {.value = 3U},
                .texture = renderer::ConstantFloatTexture::create(value).value(),
            },
        };
        return FrameScene::create(std::move(description));
    };

    const auto initial = make_scene(0.0F);
    const auto changed = make_scene(-0.0F);
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    ASSERT_TRUE(changed.has_value()) << changed.error().message;
    auto backend = create_analytic_accel_backend(*initial);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    const auto rejected = (*backend)->refit(*changed);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::incompatible);
    EXPECT_EQ((*backend)->frame_scene(), *initial);
}

TEST(AnalyticAccelBackendTest, RejectsMissingScenesAndUnrepresentableInstances) {
    const auto missing = create_analytic_accel_backend({});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(missing.error().message.empty());

    const auto unrepresentable =
        create_analytic_accel_backend(test::make_unrepresentable_instance_scene());
    ASSERT_FALSE(unrepresentable.has_value());
    EXPECT_EQ(unrepresentable.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(unrepresentable.error().message.empty());
}

TEST(AnalyticAccelBackendTest, RepresentsAnEmptyWorldWithoutSyntheticGeometry) {
    auto backend = create_analytic_accel_backend(test::make_empty_scene());
    ASSERT_TRUE(backend.has_value()) << backend.error().message;

    test::expect_empty_backend(**backend, AccelBackendKind::analytic_reference);
}

TEST(AnalyticAccelBackendTest, RetainsTheImmutableFrameSceneAndItsMeshes) {
    auto scene = test::make_basic_scene();
    const auto weak_scene = std::weak_ptr<const FrameScene>{scene};
    auto weak_meshes = std::vector<std::weak_ptr<const TriangleMesh>>{};
    weak_meshes.reserve(scene->geometries().size());
    for (const auto& geometry : scene->geometries()) {
        weak_meshes.emplace_back(geometry.mesh);
    }

    auto backend = create_analytic_accel_backend(scene);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    scene.reset();
    EXPECT_FALSE(weak_scene.expired());
    for (const auto& mesh : weak_meshes) {
        EXPECT_FALSE(mesh.expired());
    }

    const auto ray = test::make_ray(renderer::Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F},
                                    renderer::Vector3{.z = 1.0F});
    const auto hit = (*backend)->closest_hit(ray);
    ASSERT_TRUE(hit.has_value()) << hit.error().message;
    EXPECT_TRUE(hit->has_value());

    backend->reset();
    EXPECT_TRUE(weak_scene.expired());
    for (const auto& mesh : weak_meshes) {
        EXPECT_TRUE(mesh.expired());
    }
}

} // namespace
} // namespace blackframe::engine
