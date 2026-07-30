#include <Blackframe/Engine/FrameScene.hpp>
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] FrameSceneDescription make_scene_description() {
    return FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 91}},
                SceneObject{.id = {.value = 7}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 900}},
                SceneGeometry{.id = {.value = 4}},
            },
        .materials =
            {
                SceneMaterial{.id = {.value = 55}},
                SceneMaterial{.id = {.value = 2}},
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 800},
                    .object = {.value = 91},
                    .geometry = {.value = 900},
                    .material = {.value = 55},
                },
                SceneInstance{
                    .id = {.value = 1},
                    .object = {.value = 7},
                    .geometry = {.value = 4},
                    .material = {.value = 2},
                },
            },
    };
}

template <typename Result>
void expect_error_code(const Result& result, const core::StatusCode expected_code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, expected_code) << result.error().message;
    EXPECT_FALSE(result.error().message.empty());
}

TEST(FrameSceneTest, KeepsEveryIdentifierDomainStrongAndSnapshotAccessReadOnly) {
    static_assert(!std::same_as<renderer::ObjectId, renderer::InstanceId>);
    static_assert(!std::same_as<renderer::ObjectId, renderer::GeometryId>);
    static_assert(!std::same_as<renderer::ObjectId, renderer::MaterialId>);
    static_assert(!std::same_as<renderer::InstanceId, renderer::GeometryId>);
    static_assert(!std::same_as<renderer::InstanceId, renderer::MaterialId>);
    static_assert(!std::same_as<renderer::GeometryId, renderer::MaterialId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::ObjectId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::InstanceId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::GeometryId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::MaterialId>);
    static_assert(std::same_as<FrameSceneHandle::element_type, const FrameScene>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().objects()),
                               std::span<const SceneObject>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().geometries()),
                               std::span<const SceneGeometry>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().materials()),
                               std::span<const SceneMaterial>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().instances()),
                               std::span<const SceneInstance>>);
    static_assert(!std::default_initializable<FrameScene>);
    static_assert(!std::copyable<FrameScene>);
    static_assert(std::is_nothrow_destructible_v<FrameScene>);
}

TEST(FrameSceneTest, OwnsCanonicalStorageAndResolvesEveryStableIdentifier) {
    auto source = make_scene_description();
    const auto scene_result = FrameScene::create(source);
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    const auto scene = *scene_result;

    source.objects.clear();
    source.geometries.clear();
    source.materials.clear();
    source.instances.clear();

    ASSERT_EQ(scene->objects().size(), 2U);
    EXPECT_EQ(scene->objects()[0].id, (renderer::ObjectId{.value = 7}));
    EXPECT_EQ(scene->objects()[1].id, (renderer::ObjectId{.value = 91}));
    ASSERT_EQ(scene->geometries().size(), 2U);
    EXPECT_EQ(scene->geometries()[0].id, (renderer::GeometryId{.value = 4}));
    EXPECT_EQ(scene->geometries()[1].id, (renderer::GeometryId{.value = 900}));
    ASSERT_EQ(scene->materials().size(), 2U);
    EXPECT_EQ(scene->materials()[0].id, (renderer::MaterialId{.value = 2}));
    EXPECT_EQ(scene->materials()[1].id, (renderer::MaterialId{.value = 55}));
    ASSERT_EQ(scene->instances().size(), 2U);
    EXPECT_EQ(scene->instances()[0].id, (renderer::InstanceId{.value = 1}));
    EXPECT_EQ(scene->instances()[1].id, (renderer::InstanceId{.value = 800}));

    const auto object = scene->object(renderer::ObjectId{.value = 91});
    const auto geometry = scene->geometry(renderer::GeometryId{.value = 900});
    const auto material = scene->material(renderer::MaterialId{.value = 55});
    const auto instance = scene->instance(renderer::InstanceId{.value = 800});
    ASSERT_TRUE(object.has_value()) << object.error().message;
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_TRUE(material.has_value()) << material.error().message;
    ASSERT_TRUE(instance.has_value()) << instance.error().message;
    EXPECT_EQ(object->get(), (SceneObject{.id = {.value = 91}}));
    EXPECT_EQ(geometry->get(), (SceneGeometry{.id = {.value = 900}}));
    EXPECT_EQ(material->get(), (SceneMaterial{.id = {.value = 55}}));
    EXPECT_EQ(instance->get(), (SceneInstance{
                                   .id = {.value = 800},
                                   .object = {.value = 91},
                                   .geometry = {.value = 900},
                                   .material = {.value = 55},
                               }));
}

TEST(FrameSceneTest, IdentifierResolutionDoesNotDependOnInsertionOrder) {
    auto forward = make_scene_description();
    auto reverse = forward;
    std::ranges::reverse(reverse.objects);
    std::ranges::reverse(reverse.geometries);
    std::ranges::reverse(reverse.materials);
    std::ranges::reverse(reverse.instances);

    const auto forward_result = FrameScene::create(std::move(forward));
    const auto reverse_result = FrameScene::create(std::move(reverse));
    ASSERT_TRUE(forward_result.has_value()) << forward_result.error().message;
    ASSERT_TRUE(reverse_result.has_value()) << reverse_result.error().message;

    EXPECT_TRUE(std::ranges::equal((*forward_result)->objects(), (*reverse_result)->objects()));
    EXPECT_TRUE(
        std::ranges::equal((*forward_result)->geometries(), (*reverse_result)->geometries()));
    EXPECT_TRUE(std::ranges::equal((*forward_result)->materials(), (*reverse_result)->materials()));
    EXPECT_TRUE(std::ranges::equal((*forward_result)->instances(), (*reverse_result)->instances()));
}

TEST(FrameSceneTest, AcceptsEveryUint32ValueWithoutCrossDomainSentinels) {
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    auto description = FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 0}},
                SceneObject{.id = {.value = maximum}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 0}},
                SceneGeometry{.id = {.value = maximum}},
            },
        .materials =
            {
                SceneMaterial{.id = {.value = 0}},
                SceneMaterial{.id = {.value = maximum}},
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 0},
                    .object = {.value = 0},
                    .geometry = {.value = 0},
                    .material = {.value = 0},
                },
                SceneInstance{
                    .id = {.value = maximum},
                    .object = {.value = maximum},
                    .geometry = {.value = maximum},
                    .material = {.value = maximum},
                },
            },
    };

    const auto result = FrameScene::create(std::move(description));
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE((*result)->object(renderer::ObjectId{.value = 0}).has_value());
    EXPECT_TRUE((*result)->geometry(renderer::GeometryId{.value = maximum}).has_value());
    EXPECT_TRUE((*result)->material(renderer::MaterialId{.value = 0}).has_value());
    EXPECT_TRUE((*result)->instance(renderer::InstanceId{.value = maximum}).has_value());
}

TEST(FrameSceneTest, RejectsDuplicateIdentifiersInEveryDomain) {
    {
        auto description = make_scene_description();
        description.objects.push_back(description.objects.front());
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.geometries.push_back(description.geometries.front());
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.materials.push_back(description.materials.front());
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.instances.push_back(description.instances.front());
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
}

TEST(FrameSceneTest, RejectsEveryDanglingInstanceReferenceWithoutRepair) {
    {
        auto description = make_scene_description();
        description.instances.front().object = renderer::ObjectId{.value = 1234};
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.instances.front().geometry = renderer::GeometryId{.value = 1234};
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.instances.front().material = renderer::MaterialId{.value = 1234};
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
}

TEST(FrameSceneTest, ReportsUnknownLookupsInsteadOfReturningAnotherRecord) {
    const auto scene_result = FrameScene::create(make_scene_description());
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    const auto& scene = **scene_result;

    expect_error_code(scene.object(renderer::ObjectId{.value = 8}), core::StatusCode::not_found);
    expect_error_code(scene.geometry(renderer::GeometryId{.value = 5}),
                      core::StatusCode::not_found);
    expect_error_code(scene.material(renderer::MaterialId{.value = 3}),
                      core::StatusCode::not_found);
    expect_error_code(scene.instance(renderer::InstanceId{.value = 2}),
                      core::StatusCode::not_found);
}

TEST(FrameSceneTest, ReleasesOwnedGraphAtTheEndOfTheLastSnapshotLifetime) {
    std::weak_ptr<const FrameScene> observer;
    {
        const auto scene_result = FrameScene::create(make_scene_description());
        ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
        observer = *scene_result;
        EXPECT_FALSE(observer.expired());
    }
    EXPECT_TRUE(observer.expired());
}

} // namespace
} // namespace blackframe::engine
