#include <Blackframe/Engine/FrameScene.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

[[nodiscard]] renderer::Matrix4 identity_transform_matrix() {
    return renderer::identity_matrix<renderer::TransportScalar>();
}

[[nodiscard]] renderer::Matrix4 translation_matrix(const renderer::Vector3 offset) {
    auto matrix = identity_transform_matrix();
    matrix(0, 3) = offset.x;
    matrix(1, 3) = offset.y;
    matrix(2, 3) = offset.z;
    return matrix;
}

[[nodiscard]] renderer::Matrix4 scale_matrix(const renderer::Vector3 factors) {
    auto matrix = identity_transform_matrix();
    matrix(0, 0) = factors.x;
    matrix(1, 1) = factors.y;
    matrix(2, 2) = factors.z;
    return matrix;
}

[[nodiscard]] renderer::Matrix4 positive_quarter_turn_z_matrix() {
    auto matrix = identity_transform_matrix();
    matrix(0, 0) = 0.0F;
    matrix(0, 1) = -1.0F;
    matrix(1, 0) = 1.0F;
    matrix(1, 1) = 0.0F;
    return matrix;
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_scene_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.y = 1.0F},
        },
        std::vector(3, renderer::Normal3{.z = 1.0F}),
        std::vector{
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        std::vector{TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    if (!mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] renderer::SampledWavelengths make_scene_wavelengths(const float sample = 0.25F) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(sample);
    if (!wavelengths) {
        throw std::runtime_error{wavelengths.error().message};
    }
    return *wavelengths;
}

[[nodiscard]] renderer::TransportSpectrum scene_light_spectrum(const float scale = 1.0F) {
    return renderer::TransportSpectrum{
        .values = {scale, 2.0F * scale, 3.0F * scale, 4.0F * scale},
    };
}

[[nodiscard]] ScenePointLight make_scene_point_light() {
    return ScenePointLight{
        .position = {.x = 1.0F, .y = 2.0F, .z = 3.0F},
        .absolute_position_error = {.x = 0.01F, .y = 0.02F, .z = 0.03F},
        .spectral_radiant_intensity = scene_light_spectrum(1.0F),
    };
}

[[nodiscard]] SceneDirectionalLight make_scene_directional_light() {
    return SceneDirectionalLight{
        .propagation_direction = {.z = -1.0F},
        .spectral_irradiance = scene_light_spectrum(2.0F),
    };
}

[[nodiscard]] SceneSpotLight make_scene_spot_light() {
    return SceneSpotLight{
        .position = {.x = -2.0F, .y = 1.0F, .z = 4.0F},
        .absolute_position_error = {.x = 0.03F, .y = 0.02F, .z = 0.01F},
        .emission_direction = {.z = -1.0F},
        .inner_half_angle_radians = 0.25F,
        .outer_half_angle_radians = 0.5F,
        .on_axis_spectral_radiant_intensity = scene_light_spectrum(3.0F),
    };
}

[[nodiscard]] SceneSpectralEnvironment
make_scene_environment(const renderer::SampledWavelengths wavelengths) {
    return SceneSpectralEnvironment{
        .wavelengths = wavelengths,
        .radiance = renderer::TransportSpectrum{},
    };
}

[[nodiscard]] SceneSpectralMaterial
make_scene_spectral_material(const renderer::SampledWavelengths wavelengths,
                             const renderer::TransportSpectrum emitted_radiance = {}) {
    auto reflectance = renderer::TransportSpectrum{};
    reflectance.values.fill(0.5F);
    return SceneSpectralMaterial{
        .wavelengths = wavelengths,
        .reflectance = reflectance,
        .emitted_radiance = emitted_radiance,
    };
}

[[nodiscard]] FrameSceneDescription make_scene_description() {
    const auto mesh = make_scene_mesh();
    return FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 91}},
                SceneObject{.id = {.value = 7}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 900}, .mesh = mesh},
                SceneGeometry{.id = {.value = 4}, .mesh = mesh},
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
                    .parent = std::nullopt,
                    .object = {.value = 91},
                    .geometry = {.value = 900},
                    .material = {.value = 55},
                    .local_to_parent = identity_transform_matrix(),
                },
                SceneInstance{
                    .id = {.value = 1},
                    .parent = std::nullopt,
                    .object = {.value = 7},
                    .geometry = {.value = 4},
                    .material = {.value = 2},
                    .local_to_parent = identity_transform_matrix(),
                },
            },
    };
}

[[nodiscard]] FrameSceneDescription make_three_level_scene_description() {
    const auto mesh = make_scene_mesh();
    return FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 42}}},
        .geometries = {SceneGeometry{.id = {.value = 84}, .mesh = mesh}},
        .materials = {SceneMaterial{.id = {.value = 126}}},
        // Deliberately neither insertion- nor identifier-topological.
        .instances =
            {
                SceneInstance{
                    .id = {.value = 20},
                    .parent = renderer::InstanceId{.value = 10},
                    .object = {.value = 42},
                    .geometry = {.value = 84},
                    .material = {.value = 126},
                    .local_to_parent =
                        scale_matrix(renderer::Vector3{.x = 2.0F, .y = 3.0F, .z = 1.0F}),
                },
                SceneInstance{
                    .id = {.value = 30},
                    .parent = std::nullopt,
                    .object = {.value = 42},
                    .geometry = {.value = 84},
                    .material = {.value = 126},
                    .local_to_parent = translation_matrix(renderer::Vector3{.x = 10.0F}),
                },
                SceneInstance{
                    .id = {.value = 10},
                    .parent = renderer::InstanceId{.value = 30},
                    .object = {.value = 42},
                    .geometry = {.value = 84},
                    .material = {.value = 126},
                    .local_to_parent = positive_quarter_turn_z_matrix(),
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

void expect_point_near(const renderer::Point3 actual, const renderer::Point3 expected,
                       const renderer::TransportScalar tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
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
    static_assert(std::same_as<decltype(SceneGeometry::mesh), std::shared_ptr<const TriangleMesh>>);
    static_assert(!std::is_copy_assignable_v<TriangleMesh>);
    static_assert(!std::is_move_assignable_v<TriangleMesh>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().materials()),
                               std::span<const SceneMaterial>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().instances()),
                               std::span<const SceneInstance>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().punctual_lights()),
                               std::span<const ScenePunctualLight>>);
    static_assert(std::same_as<decltype(std::declval<const FrameScene&>().mesh_area_lights()),
                               std::span<const renderer::MeshAreaLight>>);
    static_assert(
        std::same_as<decltype(std::declval<const FrameScene&>().mesh_area_light_instance_ids()),
                     std::span<const renderer::InstanceId>>);
    static_assert(std::variant_size_v<ScenePunctualLight> == 3U);
    static_assert(std::same_as<std::variant_alternative_t<0, ScenePunctualLight>, ScenePointLight>);
    static_assert(
        std::same_as<std::variant_alternative_t<1, ScenePunctualLight>, SceneDirectionalLight>);
    static_assert(std::same_as<std::variant_alternative_t<2, ScenePunctualLight>, SceneSpotLight>);
    static_assert(
        std::same_as<decltype(std::declval<const FrameScene&>().local_transform(
                         renderer::InstanceId{})),
                     core::Result<std::reference_wrapper<const renderer::AffineTransform>>>);
    static_assert(
        std::same_as<decltype(std::declval<const FrameScene&>().world_transform(
                         renderer::InstanceId{})),
                     core::Result<std::reference_wrapper<const renderer::AffineTransform>>>);
    static_assert(!std::default_initializable<FrameScene>);
    static_assert(!std::copyable<FrameScene>);
    static_assert(std::is_nothrow_destructible_v<FrameScene>);
}

TEST(FrameSceneTest, RetainsValidatedPunctualLightsInStableSlotOrder) {
    const auto wavelengths = make_scene_wavelengths();
    const auto point = make_scene_point_light();
    const auto directional = make_scene_directional_light();
    const auto spot = make_scene_spot_light();
    auto description = FrameSceneDescription{};
    description.punctual_lights = {ScenePunctualLight{spot}, ScenePunctualLight{point},
                                   ScenePunctualLight{directional}};
    description.spectral_environment = make_scene_environment(wavelengths);

    const auto scene_result = FrameScene::create(description);
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    description.punctual_lights.clear();

    const auto lights = (*scene_result)->punctual_lights();
    ASSERT_EQ(lights.size(), 3U);
    ASSERT_TRUE(std::holds_alternative<SceneSpotLight>(lights[0]));
    EXPECT_EQ(std::get<SceneSpotLight>(lights[0]), spot);
    ASSERT_TRUE(std::holds_alternative<ScenePointLight>(lights[1]));
    EXPECT_EQ(std::get<ScenePointLight>(lights[1]), point);
    ASSERT_TRUE(std::holds_alternative<SceneDirectionalLight>(lights[2]));
    EXPECT_EQ(std::get<SceneDirectionalLight>(lights[2]), directional);
}

TEST(FrameSceneTest, RejectsPunctualLightsWithoutASpectralEnvironment) {
    auto description = FrameSceneDescription{};
    description.punctual_lights = {ScenePunctualLight{make_scene_point_light()}};

    expect_error_code(FrameScene::create(std::move(description)),
                      core::StatusCode::invalid_argument);
}

TEST(FrameSceneTest, ValidatesEveryPunctualRecordWithoutRepairingIt) {
    const auto wavelengths = make_scene_wavelengths();
    const auto environment = make_scene_environment(wavelengths);

    {
        auto invalid = make_scene_point_light();
        invalid.absolute_position_error.x = -1.0F;
        auto description = FrameSceneDescription{};
        description.punctual_lights = {ScenePunctualLight{invalid}};
        description.spectral_environment = environment;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto invalid = make_scene_directional_light();
        invalid.propagation_direction.z = 2.0F;
        auto description = FrameSceneDescription{};
        description.punctual_lights = {ScenePunctualLight{invalid}};
        description.spectral_environment = environment;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto invalid = make_scene_spot_light();
        invalid.inner_half_angle_radians = 0.75F;
        auto description = FrameSceneDescription{};
        description.punctual_lights = {ScenePunctualLight{invalid}};
        description.spectral_environment = environment;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto invalid = make_scene_point_light();
        invalid.spectral_radiant_intensity[2] =
            std::numeric_limits<renderer::TransportScalar>::quiet_NaN();
        auto description = FrameSceneDescription{};
        description.punctual_lights = {ScenePunctualLight{invalid}};
        description.spectral_environment = environment;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
}

TEST(FrameSceneTest, DerivesMeshAreaLightsFromCanonicalInstancesInStableOrder) {
    const auto wavelengths = make_scene_wavelengths();
    const auto emitted = scene_light_spectrum(0.5F);
    auto description = make_three_level_scene_description();
    description.spectral_environment = make_scene_environment(wavelengths);
    description.materials.front().spectral = make_scene_spectral_material(wavelengths, emitted);

    const auto scene_result = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    const auto& scene = **scene_result;
    const auto lights = scene.mesh_area_lights();
    const auto instance_ids = scene.mesh_area_light_instance_ids();
    ASSERT_EQ(lights.size(), 3U);
    ASSERT_EQ(instance_ids.size(), lights.size());
    EXPECT_EQ(instance_ids[0], (renderer::InstanceId{.value = 10}));
    EXPECT_EQ(instance_ids[1], (renderer::InstanceId{.value = 20}));
    EXPECT_EQ(instance_ids[2], (renderer::InstanceId{.value = 30}));

    for (auto light_index = std::size_t{}; light_index < lights.size(); ++light_index) {
        const auto instance = scene.instance(instance_ids[light_index]);
        const auto world = scene.world_transform(instance_ids[light_index]);
        ASSERT_TRUE(instance.has_value()) << instance.error().message;
        ASSERT_TRUE(world.has_value()) << world.error().message;
        const auto geometry = scene.geometry(instance->get().geometry);
        ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

        const auto source_positions = geometry->get().mesh->positions();
        ASSERT_EQ(lights[light_index].positions().size(), source_positions.size());
        for (auto vertex = std::size_t{}; vertex < source_positions.size(); ++vertex) {
            EXPECT_EQ(lights[light_index].positions()[vertex],
                      world->get().apply(source_positions[vertex]));
        }
        ASSERT_EQ(lights[light_index].triangles().size(), 1U);
        EXPECT_EQ(lights[light_index].triangles()[0], (renderer::AreaLightTriangleIndices{
                                                          .vertex0 = 0U,
                                                          .vertex1 = 1U,
                                                          .vertex2 = 2U,
                                                      }));
        EXPECT_EQ(lights[light_index].sidedness(), renderer::AreaLightSidedness::one_sided);

        const auto& positions = lights[light_index].positions();
        const auto center = renderer::Point3{
            .x = (positions[0].x + positions[1].x + positions[2].x) / 3.0F,
            .y = (positions[0].y + positions[1].y + positions[2].y) / 3.0F,
            .z = (positions[0].z + positions[1].z + positions[2].z) / 3.0F + 4.0F,
        };
        const auto context = renderer::LightSampleContext::create(center, 0.5F);
        ASSERT_TRUE(context.has_value()) << context.error().message;
        const auto sampled = lights[light_index].sample_li(
            *context, renderer::Point2{.x = 0.25F, .y = 0.5F}, wavelengths);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        ASSERT_TRUE(sampled->has_value());
        EXPECT_EQ((**sampled).incident_radiance(), emitted);
        const auto endpoint_error = (**sampled).endpoint().absolute_position_error();
        ASSERT_TRUE(endpoint_error.has_value());
        EXPECT_GT(endpoint_error->x, 0.0F);
        EXPECT_GT(endpoint_error->y, 0.0F);
        EXPECT_GT(endpoint_error->z, 0.0F);
    }
}

TEST(FrameSceneTest, OmitsBlackMaterialsWithoutInventingAreaLights) {
    const auto wavelengths = make_scene_wavelengths();
    auto description = make_scene_description();
    description.spectral_environment = make_scene_environment(wavelengths);
    for (auto& material : description.materials) {
        material.spectral = make_scene_spectral_material(wavelengths);
    }
    const auto emissive_material =
        std::ranges::find_if(description.materials, [](const auto& value) {
            return value.id == renderer::MaterialId{.value = 2};
        });
    ASSERT_NE(emissive_material, description.materials.end());
    emissive_material->spectral->emitted_radiance = scene_light_spectrum();

    const auto scene_result = FrameScene::create(std::move(description));
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    const auto lights = (*scene_result)->mesh_area_lights();
    const auto instance_ids = (*scene_result)->mesh_area_light_instance_ids();
    ASSERT_EQ(lights.size(), 1U);
    ASSERT_EQ(instance_ids.size(), 1U);
    EXPECT_EQ(instance_ids.front(), (renderer::InstanceId{.value = 1}));
}

TEST(FrameSceneTest, RejectsUnrepresentableEmissiveWorldMeshesInsteadOfSkippingThem) {
    const auto wavelengths = make_scene_wavelengths();
    auto description = make_scene_description();
    description.spectral_environment = make_scene_environment(wavelengths);
    for (auto& material : description.materials) {
        material.spectral = make_scene_spectral_material(wavelengths);
    }
    const auto instance = std::ranges::find_if(description.instances, [](const auto& value) {
        return value.id == renderer::InstanceId{.value = 1};
    });
    ASSERT_NE(instance, description.instances.end());
    instance->local_to_parent = identity_transform_matrix();
    instance->local_to_parent(0, 0) = std::numeric_limits<float>::max() / 2.0F;
    instance->local_to_parent(0, 3) = std::numeric_limits<float>::max();

    const auto black_scene = FrameScene::create(description);
    ASSERT_TRUE(black_scene.has_value()) << black_scene.error().message;
    EXPECT_TRUE((*black_scene)->mesh_area_lights().empty());

    const auto emissive_material =
        std::ranges::find_if(description.materials, [](const auto& value) {
            return value.id == renderer::MaterialId{.value = 2};
        });
    ASSERT_NE(emissive_material, description.materials.end());
    emissive_material->spectral->emitted_radiance = scene_light_spectrum();
    expect_error_code(FrameScene::create(std::move(description)),
                      core::StatusCode::invalid_argument);
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
    const auto local_transform = scene->local_transform(renderer::InstanceId{.value = 800});
    const auto world_transform = scene->world_transform(renderer::InstanceId{.value = 800});
    ASSERT_TRUE(object.has_value()) << object.error().message;
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_TRUE(material.has_value()) << material.error().message;
    ASSERT_TRUE(instance.has_value()) << instance.error().message;
    ASSERT_TRUE(local_transform.has_value()) << local_transform.error().message;
    ASSERT_TRUE(world_transform.has_value()) << world_transform.error().message;
    EXPECT_EQ(object->get(), (SceneObject{.id = {.value = 91}}));
    EXPECT_EQ(geometry->get().id, (renderer::GeometryId{.value = 900}));
    EXPECT_NE(geometry->get().mesh, nullptr);
    EXPECT_EQ(material->get(), (SceneMaterial{.id = {.value = 55}}));
    EXPECT_EQ(instance->get(), (SceneInstance{
                                   .id = {.value = 800},
                                   .parent = std::nullopt,
                                   .object = {.value = 91},
                                   .geometry = {.value = 900},
                                   .material = {.value = 55},
                                   .local_to_parent = identity_transform_matrix(),
                               }));
    EXPECT_EQ(local_transform->get().matrix(), identity_transform_matrix());
    EXPECT_EQ(world_transform->get().matrix(), identity_transform_matrix());
}

TEST(FrameSceneTest, IdentifierResolutionDoesNotDependOnInsertionOrder) {
    auto forward = make_three_level_scene_description();
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

    constexpr auto instance_ids = std::array{
        renderer::InstanceId{.value = 10},
        renderer::InstanceId{.value = 20},
        renderer::InstanceId{.value = 30},
    };
    for (const auto id : instance_ids) {
        const auto forward_local = (*forward_result)->local_transform(id);
        const auto reverse_local = (*reverse_result)->local_transform(id);
        const auto forward_world = (*forward_result)->world_transform(id);
        const auto reverse_world = (*reverse_result)->world_transform(id);
        ASSERT_TRUE(forward_local.has_value()) << forward_local.error().message;
        ASSERT_TRUE(reverse_local.has_value()) << reverse_local.error().message;
        ASSERT_TRUE(forward_world.has_value()) << forward_world.error().message;
        ASSERT_TRUE(reverse_world.has_value()) << reverse_world.error().message;
        EXPECT_EQ(forward_local->get().matrix(), reverse_local->get().matrix());
        EXPECT_EQ(forward_local->get().inverse_matrix(), reverse_local->get().inverse_matrix());
        EXPECT_EQ(forward_world->get().matrix(), reverse_world->get().matrix());
        EXPECT_EQ(forward_world->get().inverse_matrix(), reverse_world->get().inverse_matrix());
    }
}

TEST(FrameSceneTest, ComposesThreeNestedInstanceLevelsIntoExpectedWorldTransforms) {
    const auto scene_result = FrameScene::create(make_three_level_scene_description());
    ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
    const auto& scene = **scene_result;

    const auto root = scene.world_transform(renderer::InstanceId{.value = 30});
    const auto middle = scene.world_transform(renderer::InstanceId{.value = 10});
    const auto leaf_local = scene.local_transform(renderer::InstanceId{.value = 20});
    const auto leaf_world = scene.world_transform(renderer::InstanceId{.value = 20});
    ASSERT_TRUE(root.has_value()) << root.error().message;
    ASSERT_TRUE(middle.has_value()) << middle.error().message;
    ASSERT_TRUE(leaf_local.has_value()) << leaf_local.error().message;
    ASSERT_TRUE(leaf_world.has_value()) << leaf_world.error().message;

    constexpr auto tolerance = 1.0e-6F;
    expect_point_near(root->get().apply(renderer::Point3{}), renderer::Point3{.x = 10.0F},
                      tolerance);
    expect_point_near(middle->get().apply(renderer::Point3{.x = 1.0F}),
                      renderer::Point3{.x = 10.0F, .y = 1.0F}, tolerance);

    constexpr auto leaf_point = renderer::Point3{.x = 1.0F, .y = 2.0F};
    expect_point_near(leaf_local->get().apply(leaf_point), renderer::Point3{.x = 2.0F, .y = 6.0F},
                      tolerance);
    const auto world_point = leaf_world->get().apply(leaf_point);
    expect_point_near(world_point, renderer::Point3{.x = 4.0F, .y = 2.0F}, tolerance);
    expect_point_near(leaf_world->get().apply_inverse(world_point), leaf_point, tolerance);

    auto expected_world = identity_transform_matrix();
    expected_world(0, 0) = 0.0F;
    expected_world(0, 1) = -3.0F;
    expected_world(0, 3) = 10.0F;
    expected_world(1, 0) = 2.0F;
    expected_world(1, 1) = 0.0F;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            EXPECT_NEAR(leaf_world->get().matrix()(row, column), expected_world(row, column),
                        tolerance);
        }
    }

    const auto leaf = scene.instance(renderer::InstanceId{.value = 20});
    ASSERT_TRUE(leaf.has_value()) << leaf.error().message;
    ASSERT_TRUE(leaf->get().parent.has_value());
    EXPECT_EQ(*leaf->get().parent, (renderer::InstanceId{.value = 10}));
}

TEST(FrameSceneTest, AcceptsEveryUint32ValueWithoutCrossDomainSentinels) {
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    const auto mesh = make_scene_mesh();
    auto description = FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 0}},
                SceneObject{.id = {.value = maximum}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 0}, .mesh = mesh},
                SceneGeometry{.id = {.value = maximum}, .mesh = mesh},
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
                    .parent = std::nullopt,
                    .object = {.value = 0},
                    .geometry = {.value = 0},
                    .material = {.value = 0},
                    .local_to_parent = identity_transform_matrix(),
                },
                SceneInstance{
                    .id = {.value = maximum},
                    .parent = renderer::InstanceId{.value = 0},
                    .object = {.value = maximum},
                    .geometry = {.value = maximum},
                    .material = {.value = maximum},
                    .local_to_parent = identity_transform_matrix(),
                },
            },
    };

    const auto result = FrameScene::create(std::move(description));
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE((*result)->object(renderer::ObjectId{.value = 0}).has_value());
    EXPECT_TRUE((*result)->geometry(renderer::GeometryId{.value = maximum}).has_value());
    EXPECT_TRUE((*result)->material(renderer::MaterialId{.value = 0}).has_value());
    const auto maximum_instance = (*result)->instance(renderer::InstanceId{.value = maximum});
    ASSERT_TRUE(maximum_instance.has_value()) << maximum_instance.error().message;
    ASSERT_TRUE(maximum_instance->get().parent.has_value());
    EXPECT_EQ(*maximum_instance->get().parent, (renderer::InstanceId{.value = 0}));
}

TEST(FrameSceneTest, RejectsSelfParentingAndMultiInstanceCycles) {
    {
        auto description = make_three_level_scene_description();
        const auto root = std::ranges::find_if(description.instances, [](const auto& instance) {
            return instance.id == renderer::InstanceId{.value = 30};
        });
        ASSERT_NE(root, description.instances.end());
        root->parent = root->id;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_three_level_scene_description();
        const auto root = std::ranges::find_if(description.instances, [](const auto& instance) {
            return instance.id == renderer::InstanceId{.value = 30};
        });
        ASSERT_NE(root, description.instances.end());
        root->parent = renderer::InstanceId{.value = 20};
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
}

TEST(FrameSceneTest, RejectsInvalidLocalTransformsWithoutIdentityFallback) {
    {
        auto description = make_scene_description();
        description.instances.front().local_to_parent = identity_transform_matrix();
        description.instances.front().local_to_parent(0, 0) = 0.0F;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.instances.front().local_to_parent = identity_transform_matrix();
        description.instances.front().local_to_parent(3, 2) = -1.0F;
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
    {
        auto description = make_scene_description();
        description.instances.front().local_to_parent = identity_transform_matrix();
        description.instances.front().local_to_parent(0, 3) =
            std::numeric_limits<renderer::TransportScalar>::infinity();
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
}

TEST(FrameSceneTest, RejectsNonFiniteWorldCompositionWithoutKeepingOnlyTheLocalTransform) {
    auto description = make_three_level_scene_description();
    const auto root = std::ranges::find_if(description.instances, [](const auto& instance) {
        return instance.id == renderer::InstanceId{.value = 30};
    });
    const auto middle = std::ranges::find_if(description.instances, [](const auto& instance) {
        return instance.id == renderer::InstanceId{.value = 10};
    });
    ASSERT_NE(root, description.instances.end());
    ASSERT_NE(middle, description.instances.end());

    root->local_to_parent = scale_matrix(renderer::Vector3{
        .x = std::numeric_limits<renderer::TransportScalar>::max() / 2.0F,
        .y = 1.0F,
        .z = 1.0F,
    });
    middle->local_to_parent = scale_matrix(renderer::Vector3{.x = 4.0F, .y = 1.0F, .z = 1.0F});

    expect_error_code(FrameScene::create(std::move(description)),
                      core::StatusCode::invalid_argument);
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
        description.geometries.front().mesh.reset();
        expect_error_code(FrameScene::create(std::move(description)),
                          core::StatusCode::invalid_argument);
    }
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
    {
        auto description = make_scene_description();
        description.instances.front().parent = renderer::InstanceId{.value = 1234};
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
    expect_error_code(scene.local_transform(renderer::InstanceId{.value = 2}),
                      core::StatusCode::not_found);
    expect_error_code(scene.world_transform(renderer::InstanceId{.value = 2}),
                      core::StatusCode::not_found);
}

TEST(FrameSceneTest, ReleasesOwnedGraphAtTheEndOfTheLastSnapshotLifetime) {
    std::weak_ptr<const FrameScene> observer;
    std::weak_ptr<const TriangleMesh> mesh_observer;
    {
        auto description = make_scene_description();
        mesh_observer = description.geometries.front().mesh;
        const auto scene_result = FrameScene::create(std::move(description));
        ASSERT_TRUE(scene_result.has_value()) << scene_result.error().message;
        observer = *scene_result;
        EXPECT_FALSE(observer.expired());
        EXPECT_FALSE(mesh_observer.expired());
    }
    EXPECT_TRUE(observer.expired());
    EXPECT_TRUE(mesh_observer.expired());
}

} // namespace
} // namespace blackframe::engine
