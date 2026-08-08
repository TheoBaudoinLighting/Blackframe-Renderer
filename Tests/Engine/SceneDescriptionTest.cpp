#include <Blackframe/Engine/SceneDescription.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto LowIdentifier = std::uint32_t{7U};
inline constexpr auto SharedIdentifier = std::uint32_t{41U};
inline constexpr auto MissingIdentifier = std::uint32_t{9'999U};

inline constexpr auto ActiveFilmId = renderer::FilmId{.value = SharedIdentifier};
inline constexpr auto SecondaryFilmId = renderer::FilmId{.value = LowIdentifier};
inline constexpr auto ActiveCameraId = renderer::CameraId{.value = SharedIdentifier};
inline constexpr auto SecondaryCameraId = renderer::CameraId{.value = LowIdentifier};
inline constexpr auto ActiveOptionsId = renderer::RenderOptionsId{.value = SharedIdentifier};
inline constexpr auto SecondaryOptionsId = renderer::RenderOptionsId{.value = LowIdentifier};
inline constexpr auto ActiveTextureId = renderer::TextureId{.value = SharedIdentifier};
inline constexpr auto SecondaryTextureId = renderer::TextureId{.value = LowIdentifier};
inline constexpr auto ActiveObjectId = renderer::ObjectId{.value = SharedIdentifier};
inline constexpr auto SecondaryObjectId = renderer::ObjectId{.value = LowIdentifier};
inline constexpr auto ActiveGeometryId = renderer::GeometryId{.value = SharedIdentifier};
inline constexpr auto SecondaryGeometryId = renderer::GeometryId{.value = LowIdentifier};
inline constexpr auto ActiveMaterialId = renderer::MaterialId{.value = SharedIdentifier};
inline constexpr auto SecondaryMaterialId = renderer::MaterialId{.value = LowIdentifier};
inline constexpr auto ChildInstanceId = renderer::InstanceId{.value = SharedIdentifier};
inline constexpr auto RootInstanceId = renderer::InstanceId{.value = LowIdentifier};
inline constexpr auto PointLightId = renderer::LightId{.value = SharedIdentifier};
inline constexpr auto EnvironmentLightId = renderer::LightId{.value = LowIdentifier};

[[nodiscard]] renderer::OrthonormalFrame make_camera_frame() {
    const auto frame = renderer::OrthonormalFrame::from_normal(renderer::Normal3{.z = 1.0F});
    if (!frame) {
        throw std::runtime_error{frame.error().message};
    }
    return *frame;
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_triangle_mesh() {
    auto mesh = TriangleMesh::create(
        std::vector{
            renderer::Point3{},
            renderer::Point3{.x = 1.0F},
            renderer::Point3{.y = 1.0F},
        },
        std::vector(3U, renderer::Normal3{.z = 1.0F}),
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

[[nodiscard]] renderer::SampledWavelengths make_wavelengths() {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        throw std::runtime_error{wavelengths.error().message};
    }
    return *wavelengths;
}

[[nodiscard]] SceneSpectralMaterial
make_spectral_material(const renderer::SampledWavelengths wavelengths) {
    auto reflectance = renderer::TransportSpectrum{};
    reflectance.values.fill(0.5F);
    const auto mixture = SceneClosureMixture::create_lambertian(reflectance);
    if (!mixture) {
        throw std::runtime_error{mixture.error().message};
    }
    return SceneSpectralMaterial{
        .wavelengths = wavelengths,
        .closure_mixture = *mixture,
        .emitted_radiance = {},
    };
}

[[nodiscard]] SceneConstantTexture make_float_texture(const renderer::TextureId id,
                                                      const float value) {
    const auto texture = renderer::ConstantFloatTexture::create(value);
    if (!texture) {
        throw std::runtime_error{texture.error().message};
    }
    return SceneConstantTexture{.id = id, .texture = *texture};
}

[[nodiscard]] SceneConstantTexture make_color_texture(const renderer::TextureId id) {
    const auto texture = renderer::ConstantColorTexture::create(
        renderer::LinearRGB{.red = 0.25F, .green = 0.5F, .blue = 0.75F});
    if (!texture) {
        throw std::runtime_error{texture.error().message};
    }
    return SceneConstantTexture{.id = id, .texture = *texture};
}

[[nodiscard]] SceneDescriptionInput make_complete_input() {
    const auto frame = make_camera_frame();
    const auto mesh = make_triangle_mesh();
    const auto wavelengths = make_wavelengths();
    const auto spectral_material = make_spectral_material(wavelengths);

    auto environment_radiance = renderer::TransportSpectrum{};
    environment_radiance.values.fill(0.125F);
    const auto point_intensity = renderer::TransportSpectrum{.values = {1.0F, 2.0F, 3.0F, 4.0F}};

    return SceneDescriptionInput{
        .active_film = ActiveFilmId,
        .active_camera = ActiveCameraId,
        .active_render_options = ActiveOptionsId,
        // Every registry is deliberately authored in descending identifier order.
        .films =
            {
                SceneFilmDescription{
                    .id = ActiveFilmId,
                    .extent = {.width = 64U, .height = 32U},
                    .crop = {.minimum_x = 4U, .minimum_y = 2U, .maximum_x = 60U, .maximum_y = 30U},
                    .accumulation_precision = renderer::AccumulationPrecision::float64,
                },
                SceneFilmDescription{
                    .id = SecondaryFilmId,
                    .extent = {.width = 32U, .height = 16U},
                    .crop = {.maximum_x = 32U, .maximum_y = 16U},
                    .accumulation_precision = renderer::AccumulationPrecision::float32,
                },
            },
        .cameras =
            {
                SceneCameraDescription{
                    .id = ActiveCameraId,
                    .film = ActiveFilmId,
                    .model =
                        ScenePinholeCameraDescription{
                            .origin = {.x = 1.0F, .y = 2.0F, .z = 3.0F},
                            .orientation = frame,
                            .vertical_field_of_view_radians =
                                std::numbers::pi_v<renderer::TransportScalar> / 3.0F,
                            .t_min = 0.01F,
                            .t_max = 1'000.0F,
                            .visibility_mask = renderer::AllRayVisibility,
                            .current_medium = renderer::VacuumMedium,
                        },
                },
                SceneCameraDescription{
                    .id = SecondaryCameraId,
                    .film = SecondaryFilmId,
                    .model =
                        ScenePinholeCameraDescription{
                            .origin = {},
                            .orientation = frame,
                            .vertical_field_of_view_radians =
                                std::numbers::pi_v<renderer::TransportScalar> / 2.0F,
                            .t_min = 0.0F,
                            .t_max = std::numeric_limits<renderer::TransportScalar>::infinity(),
                        },
                },
            },
        .render_options =
            {
                SceneRenderOptionsDescription{
                    .id = ActiveOptionsId,
                    .film = ActiveFilmId,
                    .options =
                        SceneRenderOptions{
                            .samples_per_pixel = 8U,
                            .maximum_path_depth = 6U,
                            .tile_edge_length = 8U,
                            .seed = 0x1234'5678ULL,
                            .pixel_jitter = renderer::PixelJitterMode::center,
                            .mis_heuristic = renderer::MisHeuristic::balance,
                            .light_sampling_strategy =
                                renderer::LightSamplingStrategy::power_weighted,
                            .depth_limits = {.diffuse = 3U,
                                             .glossy = 2U,
                                             .specular = 4U,
                                             .transmission = 5U,
                                             .volume = 1U},
                            .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
                        },
                },
                SceneRenderOptionsDescription{
                    .id = SecondaryOptionsId,
                    .film = SecondaryFilmId,
                    .options = {},
                },
            },
        .constant_textures =
            {
                make_color_texture(ActiveTextureId),
                make_float_texture(SecondaryTextureId, -0.25F),
            },
        .objects =
            {
                SceneObject{.id = ActiveObjectId},
                SceneObject{.id = SecondaryObjectId},
            },
        .geometries =
            {
                SceneGeometry{.id = ActiveGeometryId, .mesh = mesh},
                SceneGeometry{.id = SecondaryGeometryId, .mesh = mesh},
            },
        .materials =
            {
                SceneMaterial{.id = ActiveMaterialId, .spectral = spectral_material},
                SceneMaterial{.id = SecondaryMaterialId, .spectral = spectral_material},
            },
        .instances =
            {
                SceneInstance{
                    .id = ChildInstanceId,
                    .parent = RootInstanceId,
                    .object = ActiveObjectId,
                    .geometry = ActiveGeometryId,
                    .material = ActiveMaterialId,
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
                SceneInstance{
                    .id = RootInstanceId,
                    .parent = std::nullopt,
                    .object = SecondaryObjectId,
                    .geometry = SecondaryGeometryId,
                    .material = SecondaryMaterialId,
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .lights =
            {
                SceneLightDescription{
                    .id = PointLightId,
                    .light =
                        ScenePointLight{
                            .position = {.x = 2.0F, .y = 3.0F, .z = 4.0F},
                            .absolute_position_error = {.x = 0.01F, .y = 0.01F, .z = 0.01F},
                            .spectral_radiant_intensity = point_intensity,
                        },
                },
                SceneLightDescription{
                    .id = EnvironmentLightId,
                    .light =
                        SceneSpectralEnvironment{
                            .wavelengths = wavelengths,
                            .radiance = environment_radiance,
                        },
                },
            },
    };
}

template <typename Result>
void expect_invalid_argument(const Result& result, const std::string_view diagnostic_fragment) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument) << result.error().message;
    EXPECT_NE(result.error().message.find(diagnostic_fragment), std::string::npos)
        << result.error().message;
}

template <typename Record, std::size_t Size>
void expect_identifier_order(const std::span<const Record> records,
                             const std::array<std::uint32_t, Size>& expected) {
    ASSERT_EQ(records.size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(records[index].id.value, expected[index]);
    }
}

TEST(SceneDescriptionTest, ConstructsACompleteTypedDescriptionWithoutBackendResources) {
    const auto result = SceneDescription::create(make_complete_input());

    ASSERT_TRUE(result.has_value()) << result.error().message;
    const auto& description = *result;
    EXPECT_EQ(description.active_film_id(), ActiveFilmId);
    EXPECT_EQ(description.active_camera_id(), ActiveCameraId);
    EXPECT_EQ(description.active_render_options_id(), ActiveOptionsId);
    EXPECT_EQ(description.active_film().id, ActiveFilmId);
    EXPECT_EQ(description.active_film().extent.width, 64U);
    EXPECT_EQ(description.active_film().crop.minimum_x, 4U);
    EXPECT_EQ(description.active_camera().id, ActiveCameraId);
    EXPECT_EQ(description.active_camera().film, ActiveFilmId);
    EXPECT_EQ(description.active_render_options().id, ActiveOptionsId);
    EXPECT_EQ(description.active_render_options().film, ActiveFilmId);
    EXPECT_EQ(description.active_render_options().options.samples_per_pixel, 8U);
    EXPECT_EQ(description.active_render_options().options.seed, 0x1234'5678ULL);

    const auto* camera_model =
        std::get_if<ScenePinholeCameraDescription>(&description.active_camera().model);
    ASSERT_NE(camera_model, nullptr);
    EXPECT_EQ(camera_model->origin, (renderer::Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}));
    EXPECT_EQ(camera_model->current_medium, renderer::VacuumMedium);

    EXPECT_EQ(description.films().size(), 2U);
    EXPECT_EQ(description.cameras().size(), 2U);
    EXPECT_EQ(description.render_options().size(), 2U);
    EXPECT_EQ(description.constant_textures().size(), 2U);
    EXPECT_TRUE(description.host_image_textures().empty());
    EXPECT_EQ(description.objects().size(), 2U);
    EXPECT_EQ(description.geometries().size(), 2U);
    EXPECT_EQ(description.materials().size(), 2U);
    EXPECT_EQ(description.instances().size(), 2U);
    EXPECT_EQ(description.lights().size(), 2U);

    const auto film = description.film(SecondaryFilmId);
    const auto camera = description.camera(SecondaryCameraId);
    const auto options = description.options(SecondaryOptionsId);
    const auto texture = description.constant_texture(ActiveTextureId);
    const auto object = description.object(ActiveObjectId);
    const auto geometry = description.geometry(ActiveGeometryId);
    const auto material = description.material(ActiveMaterialId);
    const auto instance = description.instance(ChildInstanceId);
    const auto point_light = description.light(PointLightId);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    ASSERT_TRUE(options.has_value()) << options.error().message;
    ASSERT_TRUE(texture.has_value()) << texture.error().message;
    ASSERT_TRUE(object.has_value()) << object.error().message;
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_TRUE(material.has_value()) << material.error().message;
    ASSERT_TRUE(instance.has_value()) << instance.error().message;
    ASSERT_TRUE(point_light.has_value()) << point_light.error().message;
    EXPECT_EQ(film->get().id, SecondaryFilmId);
    EXPECT_EQ(camera->get().film, SecondaryFilmId);
    EXPECT_EQ(options->get().film, SecondaryFilmId);
    EXPECT_EQ(texture->get().id, ActiveTextureId);
    EXPECT_EQ(object->get().id, ActiveObjectId);
    EXPECT_NE(geometry->get().mesh, nullptr);
    EXPECT_TRUE(material->get().spectral.has_value());
    ASSERT_TRUE(instance->get().parent.has_value());
    EXPECT_EQ(*instance->get().parent, RootInstanceId);
    EXPECT_TRUE(std::holds_alternative<ScenePointLight>(point_light->get().light));
    EXPECT_EQ(description.constant_textures().front().kind(),
              renderer::ConstantTextureKind::float_value);
    EXPECT_EQ(description.constant_textures().back().kind(), renderer::ConstantTextureKind::color);
}

TEST(SceneDescriptionTest, KeepsStrongStableIdsWhileCanonicalizingEveryRegistry) {
    static_assert(!std::same_as<renderer::FilmId, renderer::CameraId>);
    static_assert(!std::same_as<renderer::CameraId, renderer::RenderOptionsId>);
    static_assert(!std::same_as<renderer::ObjectId, renderer::GeometryId>);
    static_assert(!std::same_as<renderer::GeometryId, renderer::MaterialId>);
    static_assert(!std::same_as<renderer::MaterialId, renderer::TextureId>);
    static_assert(!std::same_as<renderer::TextureId, renderer::LightId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::FilmId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::CameraId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::RenderOptionsId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::ObjectId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::InstanceId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::GeometryId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::MaterialId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::TextureId>);
    static_assert(!std::convertible_to<std::uint32_t, renderer::LightId>);

    auto forward_input = make_complete_input();
    auto reverse_input = forward_input;
    std::ranges::reverse(reverse_input.films);
    std::ranges::reverse(reverse_input.cameras);
    std::ranges::reverse(reverse_input.render_options);
    std::ranges::reverse(reverse_input.constant_textures);
    std::ranges::reverse(reverse_input.objects);
    std::ranges::reverse(reverse_input.geometries);
    std::ranges::reverse(reverse_input.materials);
    std::ranges::reverse(reverse_input.instances);
    std::ranges::reverse(reverse_input.lights);

    const auto forward = SceneDescription::create(std::move(forward_input));
    const auto reverse = SceneDescription::create(std::move(reverse_input));
    ASSERT_TRUE(forward.has_value()) << forward.error().message;
    ASSERT_TRUE(reverse.has_value()) << reverse.error().message;

    constexpr auto expected = std::array{LowIdentifier, SharedIdentifier};
    expect_identifier_order(forward->films(), expected);
    expect_identifier_order(forward->cameras(), expected);
    expect_identifier_order(forward->render_options(), expected);
    expect_identifier_order(forward->constant_textures(), expected);
    expect_identifier_order(forward->objects(), expected);
    expect_identifier_order(forward->geometries(), expected);
    expect_identifier_order(forward->materials(), expected);
    expect_identifier_order(forward->instances(), expected);
    expect_identifier_order(forward->lights(), expected);
    expect_identifier_order(reverse->films(), expected);
    expect_identifier_order(reverse->cameras(), expected);
    expect_identifier_order(reverse->render_options(), expected);
    expect_identifier_order(reverse->constant_textures(), expected);
    expect_identifier_order(reverse->objects(), expected);
    expect_identifier_order(reverse->geometries(), expected);
    expect_identifier_order(reverse->materials(), expected);
    expect_identifier_order(reverse->instances(), expected);
    expect_identifier_order(reverse->lights(), expected);

    EXPECT_EQ(forward->active_film_id().value, SharedIdentifier);
    EXPECT_EQ(forward->active_camera_id().value, SharedIdentifier);
    EXPECT_EQ(forward->active_render_options_id().value, SharedIdentifier);
    EXPECT_TRUE(forward->object(ActiveObjectId).has_value());
    EXPECT_TRUE(forward->geometry(ActiveGeometryId).has_value());
    EXPECT_TRUE(forward->material(ActiveMaterialId).has_value());
    EXPECT_TRUE(forward->instance(ChildInstanceId).has_value());
    EXPECT_TRUE(forward->light(PointLightId).has_value());
    EXPECT_TRUE(reverse->object(ActiveObjectId).has_value());
    EXPECT_TRUE(reverse->geometry(ActiveGeometryId).has_value());
    EXPECT_TRUE(reverse->material(ActiveMaterialId).has_value());
    EXPECT_TRUE(reverse->instance(ChildInstanceId).has_value());
    EXPECT_TRUE(reverse->light(PointLightId).has_value());
}

TEST(SceneDescriptionTest, RejectsDuplicateIdentifiersInEveryRegistry) {
    {
        auto input = make_complete_input();
        input.films.push_back(input.films.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate film");
    }
    {
        auto input = make_complete_input();
        input.cameras.push_back(input.cameras.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate camera");
    }
    {
        auto input = make_complete_input();
        input.render_options.push_back(input.render_options.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "duplicate render-options");
    }
    {
        auto input = make_complete_input();
        input.constant_textures.push_back(input.constant_textures.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "duplicate constant-texture");
    }
    {
        auto input = make_complete_input();
        input.host_image_textures = {
            SceneHostImageTexture{.id = {.value = 80U}, .image = {}},
            SceneHostImageTexture{.id = {.value = 80U}, .image = {}},
        };
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "duplicate host-image-texture");
    }
    {
        auto input = make_complete_input();
        input.host_image_textures = {SceneHostImageTexture{.id = ActiveTextureId, .image = {}}};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "both a constant and a host image");
    }
    {
        auto input = make_complete_input();
        input.objects.push_back(input.objects.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate object");
    }
    {
        auto input = make_complete_input();
        input.geometries.push_back(input.geometries.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate geometry");
    }
    {
        auto input = make_complete_input();
        input.materials.push_back(input.materials.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate material");
    }
    {
        auto input = make_complete_input();
        input.instances.push_back(input.instances.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate instance");
    }
    {
        auto input = make_complete_input();
        input.lights.push_back(input.lights.front());
        expect_invalid_argument(SceneDescription::create(std::move(input)), "duplicate light");
    }
}

TEST(SceneDescriptionTest, RejectsEveryDanglingSelectionAndRecordReferenceWithoutRepair) {
    {
        auto input = make_complete_input();
        input.active_film = renderer::FilmId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)), "unknown active film");
    }
    {
        auto input = make_complete_input();
        input.active_camera = renderer::CameraId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "unknown active camera");
    }
    {
        auto input = make_complete_input();
        input.active_render_options = renderer::RenderOptionsId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "unknown active render-options");
    }
    {
        auto input = make_complete_input();
        input.cameras.front().film = renderer::FilmId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "camera references an unknown film");
    }
    {
        auto input = make_complete_input();
        input.render_options.front().film = renderer::FilmId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "options reference an unknown film");
    }
    {
        auto input = make_complete_input();
        input.cameras.front().film = SecondaryFilmId;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "active scene camera must reference the active film");
    }
    {
        auto input = make_complete_input();
        input.render_options.front().film = SecondaryFilmId;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "active scene render-options must reference the active film");
    }
    {
        auto input = make_complete_input();
        input.instances.front().object = renderer::ObjectId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "instance references an unknown object");
    }
    {
        auto input = make_complete_input();
        input.instances.front().geometry = renderer::GeometryId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "instance references an unknown geometry");
    }
    {
        auto input = make_complete_input();
        input.instances.front().material = renderer::MaterialId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "instance references an unknown material");
    }
    {
        auto input = make_complete_input();
        input.instances.front().parent = renderer::InstanceId{.value = MissingIdentifier};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "instance references an unknown parent");
    }
    {
        auto input = make_complete_input();
        input.materials.front().spectral->normal_map =
            SceneNormalMapBinding{.texture = ActiveTextureId};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "surface map references an unknown host-image texture");
    }
    {
        auto input = make_complete_input();
        input.materials.front().spectral->bump_map =
            SceneBumpMapBinding{.texture = SecondaryTextureId};
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "surface map references an unknown host-image texture");
    }
}

TEST(SceneDescriptionTest, RejectsNonCanonicalRecordsAndUnrepresentableWorldComposition) {
    {
        auto input = make_complete_input();
        auto& probabilities =
            input.materials.front().spectral->closure_mixture.component_probabilities;
        probabilities[1] = 0.25F;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "exactly zero inactive probability tail");
    }
    {
        auto input = make_complete_input();
        auto& child_transform = input.instances.front().local_to_parent;
        auto& root_transform = input.instances.back().local_to_parent;
        child_transform(0U, 0U) = 4.0F;
        root_transform(0U, 0U) = std::numeric_limits<renderer::TransportScalar>::max() / 2.0F;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "hierarchy produced an invalid world transform");
    }
}

TEST(SceneDescriptionTest, RejectsSelfAndIndirectInstanceCycles) {
    {
        auto input = make_complete_input();
        input.instances.front().parent = input.instances.front().id;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "hierarchy contains a cycle");
    }
    {
        auto input = make_complete_input();
        const auto middle_id = renderer::InstanceId{.value = 23U};
        input.instances.push_back(SceneInstance{
            .id = middle_id,
            .parent = RootInstanceId,
            .object = SecondaryObjectId,
            .geometry = SecondaryGeometryId,
            .material = SecondaryMaterialId,
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
        });
        input.instances.front().parent = middle_id;
        input.instances.back().parent = RootInstanceId;
        input.instances[1].parent = ChildInstanceId;
        expect_invalid_argument(SceneDescription::create(std::move(input)),
                                "hierarchy contains a cycle");
    }
}

TEST(SceneDescriptionTest, AcceptsABranchingInstanceDag) {
    auto input = make_complete_input();
    const auto sibling_id = renderer::InstanceId{.value = 23U};
    input.instances.push_back(SceneInstance{
        .id = sibling_id,
        .parent = RootInstanceId,
        .object = ActiveObjectId,
        .geometry = ActiveGeometryId,
        .material = ActiveMaterialId,
        .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
    });

    const auto result = SceneDescription::create(std::move(input));

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result->instances().size(), 3U);
    const auto child = result->instance(ChildInstanceId);
    const auto sibling = result->instance(sibling_id);
    ASSERT_TRUE(child.has_value()) << child.error().message;
    ASSERT_TRUE(sibling.has_value()) << sibling.error().message;
    ASSERT_TRUE(child->get().parent.has_value());
    ASSERT_TRUE(sibling->get().parent.has_value());
    EXPECT_EQ(*child->get().parent, RootInstanceId);
    EXPECT_EQ(*sibling->get().parent, RootInstanceId);
}

} // namespace
} // namespace blackframe::engine
