#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr auto HostTextureId = renderer::TextureId{.value = 70U};

[[nodiscard]] std::filesystem::path write_normal_map_fixture(const std::string_view name) {
    const auto path = std::filesystem::path{BLACKFRAME_ENGINE_SURFACE_MAP_TEST_OUTPUT_DIR} / name;
    constexpr auto width = std::uint32_t{4U};
    constexpr auto height = std::uint32_t{4U};
    auto pixels = std::vector<renderer::TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(width) * height * 3U);
    for (auto index = std::uint32_t{}; index < width * height; ++index) {
        pixels.insert(pixels.end(), {0.8F, 0.5F, 0.9F});
    }

    auto stream = std::ofstream{path, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "PF\n"
           << width << ' ' << height << '\n'
           << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size() * sizeof(pixels.front())));
    EXPECT_TRUE(stream.good());
    return path;
}

[[nodiscard]] renderer::HostImageMipChainHandle make_normal_map(const std::string_view name) {
    auto cache = renderer::HostImageCache::create().value();
    const auto image =
        cache.load(write_normal_map_fixture(name), renderer::TextureColorSpace::data);
    EXPECT_TRUE(image.has_value()) << image.error().message;
    if (!image) {
        return {};
    }
    const auto mip_chain = renderer::HostImageMipChain::generate(*image);
    EXPECT_TRUE(mip_chain.has_value()) << mip_chain.error().message;
    return mip_chain ? *mip_chain : renderer::HostImageMipChainHandle{};
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_surface_mesh() {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 2.0F, .y = 0.0F, .z = 0.0F},
            renderer::Point3{.x = 0.0F, .y = 2.0F, .z = 0.0F},
        },
        {
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = 1.0F},
        },
        {
            renderer::Point2{.x = 0.0F, .y = 0.0F},
            renderer::Point2{.x = 1.0F, .y = 0.0F},
            renderer::Point2{.x = 0.0F, .y = 1.0F},
        },
        {TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    return std::make_shared<const TriangleMesh>(std::move(mesh).value());
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> make_transport_plane(const float z,
                                                                       const bool upward) {
    constexpr auto extent = 10'000.0F;
    const auto left = renderer::Point3{.x = -extent, .y = -extent, .z = z};
    const auto right = renderer::Point3{.x = extent, .y = -extent, .z = z};
    const auto top = renderer::Point3{.x = 0.0F, .y = extent, .z = z};
    const auto normal = renderer::Normal3{.z = upward ? 1.0F : -1.0F};
    const auto positions = upward ? std::vector{left, right, top} : std::vector{left, top, right};
    const auto coordinates = upward ? std::vector{renderer::Point2{.x = 0.0F, .y = 0.0F},
                                                  renderer::Point2{.x = 1.0F, .y = 0.0F},
                                                  renderer::Point2{.x = 0.5F, .y = 1.0F}}
                                    : std::vector{renderer::Point2{.x = 0.0F, .y = 0.0F},
                                                  renderer::Point2{.x = 0.5F, .y = 1.0F},
                                                  renderer::Point2{.x = 1.0F, .y = 0.0F}};
    auto mesh = TriangleMesh::create(positions, {normal, normal, normal}, coordinates,
                                     {TriangleVertexIndices{.vertices = {0U, 1U, 2U}}});
    return std::make_shared<const TriangleMesh>(std::move(mesh).value());
}

[[nodiscard]] FrameSceneDescription
make_mapped_scene_description(const renderer::HostImageMipChainHandle& image) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F).value();
    auto closures = renderer::TransportSpectrum{};
    closures.values.fill(0.5F);
    return FrameSceneDescription{
        .host_image_textures = {SceneHostImageTexture{.id = HostTextureId, .image = image}},
        .objects = {SceneObject{.id = {.value = 41U}}},
        .geometries =
            {
                SceneGeometry{.id = {.value = 42U}, .mesh = make_surface_mesh()},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 43U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = wavelengths,
                            .closure_mixture =
                                SceneClosureMixture::create_lambertian(closures).value(),
                            .emitted_radiance = {},
                            .normal_map =
                                SceneNormalMapBinding{
                                    .texture = HostTextureId,
                                    .red_channel = 0U,
                                    .green_channel = 1U,
                                    .blue_channel = 2U,
                                    .y_convention =
                                        renderer::TangentSpaceNormalYConvention::positive_v,
                                    .u_wrap = renderer::TextureWrapMode::repeat,
                                    .v_wrap = renderer::TextureWrapMode::repeat,
                                    .ewa_limits = {},
                                },
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 44U},
                    .parent = std::nullopt,
                    .object = {.value = 41U},
                    .geometry = {.value = 42U},
                    .material = {.value = 43U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = wavelengths,
                .radiance = {},
            },
    };
}

[[nodiscard]] FrameSceneHandle
make_scalar_sidecar_scene(const renderer::HostImageMipChainHandle& image) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F).value();
    auto reflectance = renderer::TransportSpectrum{};
    reflectance.values.fill(0.5F);
    const auto lambert = SceneClosureMixture::create_lambertian(reflectance).value();
    return FrameScene::create(
               FrameSceneDescription{
                   .host_image_textures = {SceneHostImageTexture{.id = HostTextureId,
                                                                 .image = image}},
                   .objects =
                       {
                           SceneObject{.id = {.value = 101U}},
                           SceneObject{.id = {.value = 102U}},
                       },
                   .geometries =
                       {
                           SceneGeometry{.id = {.value = 111U},
                                         .mesh = make_transport_plane(0.0F, true)},
                           SceneGeometry{.id = {.value = 112U},
                                         .mesh = make_transport_plane(2.0F, false)},
                       },
                   .materials =
                       {
                           SceneMaterial{
                               .id = {.value = 121U},
                               .spectral =
                                   SceneSpectralMaterial{
                                       .wavelengths = wavelengths,
                                       .closure_mixture = lambert,
                                       .emitted_radiance = {},
                                   },
                           },
                           SceneMaterial{
                               .id = {.value = 122U},
                               .spectral =
                                   SceneSpectralMaterial{
                                       .wavelengths = wavelengths,
                                       .closure_mixture = lambert,
                                       .emitted_radiance = {},
                                       .normal_map =
                                           SceneNormalMapBinding{.texture = HostTextureId},
                                   },
                           },
                       },
                   .instances =
                       {
                           SceneInstance{
                               .id = {.value = 131U},
                               .parent = std::nullopt,
                               .object = {.value = 101U},
                               .geometry = {.value = 111U},
                               .material = {.value = 121U},
                               .local_to_parent =
                                   renderer::identity_matrix<renderer::TransportScalar>(),
                           },
                           SceneInstance{
                               .id = {.value = 132U},
                               .parent = std::nullopt,
                               .object = {.value = 102U},
                               .geometry = {.value = 112U},
                               .material = {.value = 122U},
                               .local_to_parent =
                                   renderer::identity_matrix<renderer::TransportScalar>(),
                           },
                       },
                   .punctual_lights =
                       {
                           ScenePointLight{
                               .position = {.z = -2.0F},
                               .absolute_position_error = {},
                               .spectral_radiant_intensity = reflectance,
                           },
                       },
                   .spectral_environment =
                       SceneSpectralEnvironment{
                           .wavelengths = wavelengths,
                           .radiance = {},
                       },
               })
        .value();
}

[[nodiscard]] renderer::Ray make_ray() {
    return renderer::Ray::create(renderer::Point3{.x = 0.5F, .y = 0.5F, .z = -2.0F},
                                 renderer::Vector3{.z = 1.0F}, 0.0F, 8.0F, 0.375F,
                                 renderer::AllRayVisibility, renderer::VacuumMedium)
        .value();
}

[[nodiscard]] renderer::RayDifferential make_differential(const renderer::Ray ray) {
    return renderer::RayDifferential::create(
               ray, renderer::Point3{.x = 0.52F, .y = 0.5F, .z = -2.0F}, ray.direction(),
               renderer::Point3{.x = 0.5F, .y = 0.52F, .z = -2.0F}, ray.direction())
        .value();
}

[[nodiscard]] AccelHit make_hit() {
    return AccelHit{
        .object = {.value = 41U},
        .triangle =
            {
                .parameter = 2.0F,
                .position = {.x = 0.5F, .y = 0.5F, .z = 0.0F},
                .geometric_normal = {.z = 1.0F},
                .barycentrics = {.vertex0 = 0.5F, .vertex1 = 0.25F, .vertex2 = 0.25F},
            },
        .identifiers =
            {
                .instance = {.value = 44U},
                .geometry = {.value = 42U},
                .primitive = {.value = 0U},
                .material = {.value = 43U},
            },
    };
}

TEST(SceneSurfaceTextureTest, ClosesOneSharedTextureIdentifierRegistryAndRejectsInvalidBindings) {
    const auto image = make_normal_map("scene-registry-normal-map.pfm");
    ASSERT_TRUE(image);
    auto description = make_mapped_scene_description(image);
    const auto scene = FrameScene::create(description);
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->host_image_textures().size(), 1U);
    EXPECT_EQ((*scene)->host_image_texture(HostTextureId)->get().image, image);

    auto shared_identifier = description;
    shared_identifier.constant_textures.push_back(SceneConstantTexture{
        .id = HostTextureId,
        .texture = renderer::ConstantFloatTexture::create(0.5F).value(),
    });
    const auto shared = FrameScene::create(std::move(shared_identifier));
    ASSERT_FALSE(shared.has_value());
    EXPECT_EQ(shared.error().code, core::StatusCode::invalid_argument);

    auto duplicate_channels = description;
    duplicate_channels.materials.front().spectral->normal_map->green_channel = 0U;
    const auto duplicate = FrameScene::create(std::move(duplicate_channels));
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, core::StatusCode::invalid_argument);

    auto invalid_convention = description;
    invalid_convention.materials.front().spectral->normal_map->y_convention =
        static_cast<renderer::TangentSpaceNormalYConvention>(99U);
    const auto convention = FrameScene::create(std::move(invalid_convention));
    ASSERT_FALSE(convention.has_value());
    EXPECT_EQ(convention.error().code, core::StatusCode::invalid_argument);

    auto invalid_bump = description;
    invalid_bump.materials.front().spectral->normal_map.reset();
    invalid_bump.materials.front().spectral->bump_map = SceneBumpMapBinding{
        .texture = HostTextureId,
        .channel = 0U,
        .scale = std::numeric_limits<float>::infinity(),
    };
    const auto bump = FrameScene::create(std::move(invalid_bump));
    ASSERT_FALSE(bump.has_value());
    EXPECT_EQ(bump.error().code, core::StatusCode::invalid_argument);
}

TEST(SceneSurfaceTextureTest, RejectsHostImageChangesDuringTransformOnlyRefit) {
    const auto initial_image = make_normal_map("scene-refit-initial-normal-map.pfm");
    const auto changed_image = make_normal_map("scene-refit-changed-normal-map.pfm");
    ASSERT_TRUE(initial_image);
    ASSERT_TRUE(changed_image);
    ASSERT_NE(initial_image, changed_image);
    auto initial_description = make_mapped_scene_description(initial_image);
    auto changed_description = initial_description;
    changed_description.host_image_textures.front().image = changed_image;
    const auto initial = FrameScene::create(std::move(initial_description));
    const auto changed = FrameScene::create(std::move(changed_description));
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    ASSERT_TRUE(changed.has_value()) << changed.error().message;
    const auto acceleration = create_analytic_accel_backend(*initial);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;

    const auto refit = (*acceleration)->refit(*changed);
    ASSERT_FALSE(refit.has_value());
    EXPECT_EQ(refit.error().code, core::StatusCode::incompatible);
    EXPECT_EQ((*acceleration)->frame_scene(), *initial);
}

TEST(SceneSurfaceTextureTest, RequiresFootprintAndPerturbsOnlyTheShadingNormal) {
    const auto scene = FrameScene::create(make_mapped_scene_description(
                                              make_normal_map("scene-footprint-normal-map.pfm")))
                           .value();
    const auto ray = make_ray();
    const auto hit = make_hit();

    const auto missing_footprint = resolve_scene_surface_hit(*scene, hit, ray);
    ASSERT_FALSE(missing_footprint.has_value());
    EXPECT_EQ(missing_footprint.error().code, core::StatusCode::unavailable);

    const auto differential = resolve_scene_surface_hit(*scene, hit, make_differential(ray));
    ASSERT_TRUE(differential.has_value()) << differential.error().message;
    EXPECT_EQ(differential->surface.interaction.geometric_normal(), (renderer::Normal3{.z = 1.0F}));
    EXPECT_NEAR(differential->surface.interaction.shading_normal().x, 0.6F, 2.0e-5F);
    EXPECT_NEAR(differential->surface.interaction.shading_normal().y, 0.0F, 2.0e-5F);
    EXPECT_NEAR(differential->surface.interaction.shading_normal().z, 0.8F, 2.0e-5F);
    EXPECT_EQ(differential->surface.closure_frame.normal(),
              differential->surface.interaction.shading_normal());
    EXPECT_NE(differential->differentials.rx_shading_normal, renderer::Normal3{.z = 1.0F});
    EXPECT_NE(differential->differentials.ry_shading_normal, renderer::Normal3{.z = 1.0F});

    const auto cone = renderer::RayCone::create(0.02F, 0.0F).value();
    const auto conical = resolve_scene_surface_hit(*scene, hit, ray, cone);
    ASSERT_TRUE(conical.has_value()) << conical.error().message;
    EXPECT_EQ(conical->interaction.geometric_normal(),
              differential->surface.interaction.geometric_normal());
    EXPECT_NEAR(conical->interaction.shading_normal().x,
                differential->surface.interaction.shading_normal().x, 2.0e-5F);
    EXPECT_NEAR(conical->interaction.shading_normal().z,
                differential->surface.interaction.shading_normal().z, 2.0e-5F);

    const auto zero_cone = renderer::RayCone::create(0.0F, 0.0F).value();
    const auto missing_cone_footprint = resolve_scene_surface_hit(*scene, hit, ray, zero_cone);
    ASSERT_FALSE(missing_cone_footprint.has_value());
    EXPECT_EQ(missing_cone_footprint.error().code, core::StatusCode::unavailable);
}

TEST(SceneSurfaceTextureTest, KeepsAConeSidecarAfterContinuousScalarScattering) {
    const auto scene = make_scalar_sidecar_scene(make_normal_map("scene-sidecar-normal-map.pfm"));
    const auto acceleration = create_analytic_accel_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto state = renderer::PathState::create_initial(
        scene->spectral_environment()->wavelengths, renderer::VacuumMedium);
    const auto ray =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<float>::infinity(), 0.25F,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(acceleration.has_value()) << acceleration.error().message;
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(state.has_value()) << state.error().message;
    ASSERT_TRUE(ray.has_value()) << ray.error().message;
    const auto differential = renderer::RayDifferential::create(
        *ray, renderer::Point3{.x = 0.02F, .z = 1.0F}, ray->direction(),
        renderer::Point3{.y = 0.02F, .z = 1.0F}, ray->direction());
    ASSERT_TRUE(differential.has_value()) << differential.error().message;

    const auto stream = renderer::IndependentSampler{0x243F6A8885A308D3ULL}.make_stream(0U, 0U, 0U);
    const auto result = trace_scene_mis_with_ray_differentials(
        *differential, *state, stream, **acceleration, *sampler, renderer::MisHeuristic::power,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_FALSE(result->terminal_differential.has_value());
    EXPECT_EQ(result->loss_reason, RayDifferentialLossReason::non_delta_scattering);
    EXPECT_EQ(result->path.termination, renderer::BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(result->path.state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 1U}));
}

} // namespace
} // namespace blackframe::engine
