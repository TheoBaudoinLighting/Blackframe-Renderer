#include "../../CPU/Embree/CornellWavefrontScene.hpp"
#if BLACKFRAME_CUDA_SURFACE_MAP_TESTS
#include "../../SurfaceMapSphereFixture.hpp"
#endif

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/Fresnel.hpp>
#if BLACKFRAME_CUDA_SURFACE_MAP_TESTS
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#endif
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#if BLACKFRAME_CUDA_SURFACE_MAP_TESTS
#include <filesystem>
#include <fstream>
#endif
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

constexpr auto EvaluationSeed = std::uint64_t{0x243F6A8885A308D3ULL};
constexpr auto PathTime = renderer::TransportScalar{0.5F};
constexpr auto MaximumLinearMse = renderer::ReferenceScalar{1.0e-10};
constexpr auto MaximumLinearRmse = renderer::ReferenceScalar{1.0e-5};
constexpr auto MaximumAbsoluteError = renderer::ReferenceScalar{1.0e-4};
constexpr auto MinimumDisplayPsnr = renderer::ReferenceScalar{80.0};
constexpr auto TerminalGeometryTolerance = renderer::TransportScalar{5.0e-4F};
constexpr auto OneSurfaceBounce = renderer::PathDepthLimits{
    .diffuse = 1U,
    .glossy = 1U,
    .specular = 1U,
    .transmission = 1U,
};

#if BLACKFRAME_CUDA_SURFACE_MAP_TESTS
[[nodiscard]] renderer::HostImageMipChainHandle
make_cuda_surface_map(const std::string_view name, const std::uint32_t width,
                      const std::uint32_t height,
                      const std::span<const renderer::TransportScalar> pixels, const bool rgb) {
    const auto path = std::filesystem::path{BLACKFRAME_CUDA_SURFACE_MAP_TEST_OUTPUT_DIR} / name;
    if (pixels.size() != static_cast<std::size_t>(width) * height * (rgb ? 3U : 1U)) {
        throw std::runtime_error{"CUDA transport surface-map fixture has an invalid pixel count."};
    }
    auto stream = std::ofstream{path, std::ios::binary | std::ios::trunc};
    if (!stream.is_open()) {
        throw std::runtime_error{"CUDA transport surface-map fixture could not be created."};
    }
    stream << (rgb ? "PF\n" : "Pf\n") << width << ' ' << height << '\n'
           << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size() * sizeof(pixels.front())));
    if (!stream.good()) {
        throw std::runtime_error{"CUDA transport surface-map fixture could not be written."};
    }
    stream.close();
    if (stream.fail()) {
        throw std::runtime_error{"CUDA transport surface-map fixture could not be closed."};
    }

    auto cache = renderer::HostImageCache::create();
    if (!cache) {
        throw std::runtime_error{cache.error().message};
    }
    const auto image = cache->load(path, renderer::TextureColorSpace::data);
    if (!image) {
        throw std::runtime_error{image.error().message};
    }
    const auto mip_chain = renderer::HostImageMipChain::generate(*image);
    if (!mip_chain) {
        throw std::runtime_error{mip_chain.error().message};
    }
    return *mip_chain;
}

[[nodiscard]] renderer::HostImageMipChainHandle make_cuda_normal_map() {
    const auto pixels = surface_map_sphere_fixture::normal_map_pixels();
    return make_cuda_surface_map("cuda-transport-sphere-normal-map.pfm",
                                 surface_map_sphere_fixture::NormalMapWidth,
                                 surface_map_sphere_fixture::NormalMapHeight, pixels, true);
}

[[nodiscard]] renderer::HostImageMipChainHandle make_cuda_bump_map() {
    const auto pixels = surface_map_sphere_fixture::bump_map_pixels();
    return make_cuda_surface_map("cuda-transport-sphere-bump-map.pfm",
                                 surface_map_sphere_fixture::BumpMapWidth,
                                 surface_map_sphere_fixture::BumpMapHeight, pixels, false);
}

[[nodiscard]] renderer::HostImageMipChainHandle make_cuda_flat_bump_map() {
    const auto pixels = std::vector<renderer::TransportScalar>(
        static_cast<std::size_t>(surface_map_sphere_fixture::BumpMapWidth) *
            surface_map_sphere_fixture::BumpMapHeight,
        0.5F);
    return make_cuda_surface_map("cuda-transport-sphere-flat-bump-map.pfm",
                                 surface_map_sphere_fixture::BumpMapWidth,
                                 surface_map_sphere_fixture::BumpMapHeight, pixels, false);
}
#endif

struct ParityConfiguration final {
    std::string_view scene_name;
    renderer::RenderExtent extent;
    std::uint32_t samples_per_pixel;
    std::uint64_t seed;
    renderer::MisHeuristic heuristic;
    renderer::PathDepthLimits depth_limits;
    renderer::RussianRoulettePolicy roulette_policy;
};

struct ParityResult final {
    renderer::LinearMetrics linear;
    renderer::ReferenceScalar display_psnr;
    renderer::ReferenceScalar maximum_path_radiance_absolute_error;
    renderer::ReferenceScalar scalar_positive_image_energy;
    renderer::ReferenceScalar cuda_positive_image_energy;
    std::vector<renderer::BsdfOnlyPathResult> scalar_paths;
    std::vector<CudaWavefrontPathResult> cuda_paths;
    std::vector<renderer::RayCone> cuda_terminal_cones;
    CudaWavefrontTransportReport cuda_report;
};

[[nodiscard]] core::Error parity_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto spectrum = renderer::TransportSpectrum{};
    spectrum.values.fill(value);
    return spectrum;
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

void require_closure_append(const renderer::ClosureAppendStatus status) {
    if (status != renderer::ClosureAppendStatus::appended) {
        throw std::runtime_error{"A CUDA material parity closure could not be appended."};
    }
}

[[nodiscard]] SceneClosureMixture require_scene_closure_mixture(
    renderer::ClosureSet closures,
    const std::span<const renderer::TransportScalar> component_probabilities,
    const renderer::TransportScalar tangent_rotation_radians = 0.0F) {
    auto result = SceneClosureMixture::create(std::move(closures), component_probabilities,
                                              SceneClosureFrameMode::surface_tangent,
                                              tangent_rotation_radians);
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] SceneClosureMixture rough_diffuse_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_rough_diffuse_reflection(constant_spectrum(0.72F), 0.65F));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture
rough_conductor_scene_closure(const renderer::TransportScalar alpha_x,
                              const renderer::TransportScalar alpha_y,
                              const renderer::TransportScalar tangent_rotation_radians = 0.0F) {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_rough_conductor_reflection(
        constant_spectrum(0.85F),
        renderer::TransportSpectrum{.values = {0.25F, 0.45F, 0.75F, 1.10F}},
        renderer::TransportSpectrum{.values = {3.2F, 2.7F, 2.2F, 1.8F}}, alpha_x, alpha_y));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities,
                                         tangent_rotation_radians);
}

[[nodiscard]] SceneClosureMixture
rough_dielectric_scene_closure(const renderer::TransportScalar alpha_x = 0.25F,
                               const renderer::TransportScalar alpha_y = 0.25F) {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_rough_dielectric(constant_spectrum(0.9F), 1.0F, 1.5F, alpha_x, alpha_y));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture mirror_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_specular_reflection(constant_spectrum(0.85F)));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture empty_scene_closure() {
    constexpr auto probabilities = std::array<renderer::TransportScalar, 0U>{};
    return require_scene_closure_mixture(renderer::ClosureSet{}, probabilities);
}

[[nodiscard]] SceneClosureMixture transmission_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_specular_transmission(constant_spectrum(0.9F), 1.0F, 1.5F));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture continuous_mixture_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_lambertian_reflection(constant_spectrum(0.25F)));
    require_closure_append(
        closures.append_rough_diffuse_reflection(constant_spectrum(0.75F), 0.7F));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.35F}, renderer::TransportScalar{0.65F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture
diffuse_glossy_mixture_scene_closure(const renderer::TransportScalar diffuse_weight = 0.25F) {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_lambertian_reflection(constant_spectrum(diffuse_weight)));
    require_closure_append(closures.append_rough_conductor_reflection(
        constant_spectrum(0.8F), constant_spectrum(0.5F), constant_spectrum(2.5F), 0.35F));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.25F}, renderer::TransportScalar{0.75F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture mixed_measure_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_lambertian_reflection(constant_spectrum(0.3F)));
    require_closure_append(closures.append_specular_reflection(constant_spectrum(0.8F)));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.4F}, renderer::TransportScalar{0.6F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
make_quad(const std::array<renderer::Point3, 4U>& positions, const renderer::Normal3 normal,
          const bool degenerate_texture_coordinates = false) {
    const auto texture_coordinates = degenerate_texture_coordinates
                                         ? std::vector<renderer::Point2>(positions.size())
                                         : std::vector<renderer::Point2>{
                                               renderer::Point2{},
                                               renderer::Point2{.x = 1.0F},
                                               renderer::Point2{.x = 1.0F, .y = 1.0F},
                                               renderer::Point2{.y = 1.0F},
                                           };
    auto mesh = TriangleMesh::create(
        std::vector<renderer::Point3>{positions.begin(), positions.end()},
        std::vector<renderer::Normal3>(positions.size(), normal), texture_coordinates,
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
        });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<FrameSceneHandle> make_punctual_light_scene(
    ScenePunctualLight punctual_light,
    const renderer::TransportSpectrum reflectance = constant_spectrum(0.65F),
    const renderer::TransportSpectrum environment_radiance = constant_spectrum(0.125F)) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }

    auto shared_receiver = make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F},
        },
        renderer::Normal3{.z = 1.0F});
    if (!shared_receiver) {
        return std::unexpected(shared_receiver.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*shared_receiver)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure(reflectance),
                            .emitted_radiance = {},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                std::move(punctual_light),
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = environment_radiance,
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_point_light_scene() {
    return make_punctual_light_scene(ScenePointLight{
        .position = renderer::Point3{.z = 1.5F},
        .absolute_position_error = {},
        .spectral_radiant_intensity = constant_spectrum(2.0F),
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_mixed_light_scene() {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    auto receiver = make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F},
        },
        renderer::Normal3{.z = 1.0F});
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    auto emitter = make_quad(
        {
            renderer::Point3{.x = -8.0F, .y = -8.0F, .z = 2.0F},
            renderer::Point3{.x = -8.0F, .y = 8.0F, .z = 2.0F},
            renderer::Point3{.x = 8.0F, .y = 8.0F, .z = 2.0F},
            renderer::Point3{.x = 8.0F, .y = -8.0F, .z = 2.0F},
        },
        renderer::Normal3{.z = -1.0F});
    if (!emitter) {
        return std::unexpected(emitter.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 1U}},
                SceneObject{.id = {.value = 2U}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*emitter)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure(
                                {.values = {0.35F, 0.50F, 0.65F, 0.80F}}),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure({}),
                            .emitted_radiance = {.values = {3.0F, 2.0F, 4.0F, 1.0F}},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                ScenePointLight{
                    .position = renderer::Point3{.z = 1.25F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = {.values = {1.0F, 2.0F, 3.0F, 4.0F}},
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {},
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_occluded_point_light_scene() {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    auto receiver = make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F},
        },
        renderer::Normal3{.z = 1.0F});
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    auto blocker = make_quad(
        {
            renderer::Point3{.x = -1.0e6F, .y = -1.0e6F, .z = 0.5F},
            renderer::Point3{.x = 1.0e6F, .y = -1.0e6F, .z = 0.5F},
            renderer::Point3{.x = 1.0e6F, .y = 1.0e6F, .z = 0.5F},
            renderer::Point3{.x = -1.0e6F, .y = 1.0e6F, .z = 0.5F},
        },
        renderer::Normal3{.z = 1.0F});
    if (!blocker) {
        return std::unexpected(blocker.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 1U}},
                SceneObject{.id = {.value = 2U}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*blocker)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture =
                                require_lambertian_scene_closure(constant_spectrum(0.5F)),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure({}),
                            .emitted_radiance = {},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                ScenePointLight{
                    .position = renderer::Point3{.z = 1.5F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = constant_spectrum(0x1p120F),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {.values = {0.25F, 0.50F, 0.75F, 1.0F}},
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_material_scene(SceneClosureMixture closure_mixture,
                    const bool degenerate_surface_tangent = false) {
    auto receiver = make_quad(
        {
            renderer::Point3{.x = -16.0F, .y = -16.0F},
            renderer::Point3{.x = 16.0F, .y = -16.0F},
            renderer::Point3{.x = 16.0F, .y = 16.0F},
            renderer::Point3{.x = -16.0F, .y = 16.0F},
        },
        renderer::Normal3{.z = 1.0F}, degenerate_surface_tangent);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)}},
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = std::move(closure_mixture),
                            .emitted_radiance = {},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                ScenePointLight{
                    .position = renderer::Point3{.z = 3.0F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = constant_spectrum(0.5F),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = constant_spectrum(0.35F),
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_delta_emitter_scene() {
    auto receiver = make_quad(
        {
            renderer::Point3{.x = -16.0F, .y = -16.0F},
            renderer::Point3{.x = 16.0F, .y = -16.0F},
            renderer::Point3{.x = 16.0F, .y = 16.0F},
            renderer::Point3{.x = -16.0F, .y = 16.0F},
        },
        renderer::Normal3{.z = 1.0F});
    auto emitter = make_quad(
        {
            renderer::Point3{.x = -64.0F, .y = -64.0F, .z = 2.0F},
            renderer::Point3{.x = -64.0F, .y = 64.0F, .z = 2.0F},
            renderer::Point3{.x = 64.0F, .y = 64.0F, .z = 2.0F},
            renderer::Point3{.x = 64.0F, .y = -64.0F, .z = 2.0F},
        },
        renderer::Normal3{.z = -1.0F});
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!emitter) {
        return std::unexpected(emitter.error());
    }
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}, SceneObject{.id = {.value = 2U}}},
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*emitter)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = mirror_scene_closure(),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = empty_scene_closure(),
                            .emitted_radiance = constant_spectrum(4.0F),
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {},
            },
    });
}

[[nodiscard]] core::Result<renderer::Ray>
material_primary_ray(const renderer::Vector3 outgoing_world) {
    return renderer::Ray::create(
        renderer::Point3{.x = outgoing_world.x, .y = outgoing_world.y, .z = outgoing_world.z},
        -outgoing_world, 0.0F, std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
        renderer::AllRayVisibility, renderer::VacuumMedium);
}

struct ComponentSampleBand final {
    renderer::TransportScalar lower{};
    renderer::TransportScalar upper{};
    std::size_t count{};
};

[[nodiscard]] core::Error material_test_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Result<std::vector<CudaWavefrontPathInput>>
make_material_inputs(const FrameSceneHandle& scene, const renderer::Ray& ray,
                     const std::span<const ComponentSampleBand> bands) {
    const auto state = renderer::PathState::create_initial(
        scene->spectral_environment()->wavelengths, renderer::VacuumMedium);
    if (!state) {
        return std::unexpected(state.error());
    }
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    if (!dimensions) {
        return std::unexpected(dimensions.error());
    }
    const auto primary_cone = renderer::RayCone::create(0.0F, 0.0F);
    if (!primary_cone) {
        return std::unexpected(primary_cone.error());
    }

    auto requested_count = std::size_t{};
    for (const auto& band : bands) {
        if (!std::isfinite(band.lower) || !std::isfinite(band.upper) || band.lower < 0.0F ||
            !(band.lower < band.upper) || band.upper > 1.0F || band.count == 0U ||
            requested_count > std::numeric_limits<std::uint32_t>::max() - band.count) {
            return std::unexpected(material_test_error(
                "CUDA material parity sample bands must be finite, canonical, and "
                "representable."));
        }
        requested_count += band.count;
    }
    auto inputs = std::vector<CudaWavefrontPathInput>{};
    inputs.reserve(requested_count);
    for (const auto& band : bands) {
        for (auto band_index = std::size_t{}; band_index < band.count; ++band_index) {
            const auto pixel_x = static_cast<std::uint32_t>(inputs.size());
            auto found = false;
            for (auto candidate = std::uint64_t{}; candidate < 100'000U; ++candidate) {
                const auto sample = renderer::SampleStreamIndex{
                    .pixel_x = pixel_x,
                    .pixel_y = 0U,
                    .sample_index = candidate,
                    .seed = EvaluationSeed,
                };
                const auto component =
                    renderer::SampleStream{sample}.sample_1d(dimensions->bsdf_component);
                if (component < band.lower || component >= band.upper) {
                    continue;
                }
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = ray,
                    .primary_cone = *primary_cone,
                    .initial_state = *state,
                    .sample = sample,
                });
                found = true;
                break;
            }
            if (!found) {
                return std::unexpected(material_test_error(
                    "A deterministic CUDA material parity sample band could not be populated."));
            }
        }
    }
    return inputs;
}

struct PrimaryRayWithCone final {
    renderer::Ray ray;
    renderer::RayCone cone;
};

[[nodiscard]] core::Result<PrimaryRayWithCone>
camera_primary_ray(const renderer::PinholeCamera& camera, const renderer::PixelSampleIndex index) {
    const auto ray =
        camera.generate_primary_ray(index, renderer::PixelJitterMode::uniform, PathTime);
    const auto cone =
        camera.generate_primary_ray_cone(index, renderer::PixelJitterMode::uniform, PathTime);
    if (!ray) {
        return std::unexpected(ray.error());
    }
    if (!cone) {
        return std::unexpected(cone.error());
    }
    return PrimaryRayWithCone{.ray = *ray, .cone = *cone};
}

[[nodiscard]] core::Result<PrimaryRayWithCone> point_primary_ray(core::Result<renderer::Ray> ray) {
    if (!ray) {
        return std::unexpected(ray.error());
    }
    const auto cone = renderer::RayCone::create(0.0F, 0.0F);
    if (!cone) {
        return std::unexpected(cone.error());
    }
    return PrimaryRayWithCone{.ray = *ray, .cone = *cone};
}

template <typename GenerateRay>
[[nodiscard]] core::Result<std::vector<CudaWavefrontPathInput>>
make_inputs(const renderer::RenderExtent extent, const std::uint32_t samples_per_pixel,
            const std::uint64_t seed, const renderer::SampledWavelengths& wavelengths,
            GenerateRay&& generate_ray) {
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }
    const auto sampler = renderer::IndependentSampler{seed};
    auto inputs = std::vector<CudaWavefrontPathInput>{};
    inputs.reserve(static_cast<std::size_t>(extent.width) * extent.height * samples_per_pixel);
    for (auto sample_index = std::uint64_t{}; sample_index < samples_per_pixel; ++sample_index) {
        for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
            for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
                const auto index = renderer::PixelSampleIndex{
                    .pixel_x = pixel_x,
                    .pixel_y = pixel_y,
                    .sample_index = sample_index,
                    .seed = seed,
                };
                const auto stream = sampler.make_stream(pixel_x, pixel_y, sample_index);
                const auto primary = generate_ray(index, stream);
                if (!primary) {
                    return std::unexpected(primary.error());
                }
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = primary->ray,
                    .primary_cone = primary->cone,
                    .initial_state = *initial_state,
                    .sample = stream.index(),
                });
            }
        }
    }
    return inputs;
}

[[nodiscard]] core::Result<renderer::Ray> make_planar_ray(const renderer::PixelSampleIndex& index,
                                                          const renderer::RenderExtent extent,
                                                          const bool toward_receiver) {
    const auto sample = renderer::generate_pixel_sample<renderer::TransportScalar>(
        index, renderer::PixelJitterMode::uniform);
    if (!sample) {
        return std::unexpected(sample.error());
    }
    const auto x = -0.75F + 1.5F * (static_cast<float>(sample->pixel_x) + sample->offset_x) /
                                static_cast<float>(extent.width);
    const auto y = -0.75F + 1.5F * (static_cast<float>(sample->pixel_y) + sample->offset_y) /
                                static_cast<float>(extent.height);
    return renderer::Ray::create(renderer::Point3{.x = x, .y = y, .z = 1.0F},
                                 renderer::Vector3{.z = toward_receiver ? -1.0F : 1.0F}, 0.0F,
                                 std::numeric_limits<renderer::TransportScalar>::infinity(),
                                 PathTime, renderer::AllRayVisibility, renderer::VacuumMedium);
}

[[nodiscard]] core::Result<renderer::LightSampler>
make_uniform_light_sampler(const FrameSceneHandle& scene) {
    if (!scene) {
        return std::unexpected(parity_error(core::StatusCode::invalid_argument,
                                            "CUDA transport parity requires a frame scene."));
    }
    if (scene->punctual_lights().size() >
        std::numeric_limits<std::size_t>::max() - scene->mesh_area_lights().size()) {
        return std::unexpected(parity_error(core::StatusCode::resource_exhausted,
                                            "CUDA parity light count is not representable."));
    }
    return renderer::LightSampler::create_uniform(scene->punctual_lights().size() +
                                                  scene->mesh_area_lights().size());
}

template <renderer::AccumulationPrecision Precision>
[[nodiscard]] core::Status accumulate_path(renderer::FilmT<Precision>& film,
                                           const CudaWavefrontPathInput& input,
                                           const renderer::PathState& state) {
    const auto xyz =
        renderer::cie_1931_spectrum_to_xyz(state.accumulated_radiance(), state.wavelengths());
    if (!xyz) {
        return std::unexpected(xyz.error());
    }
    const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
    if (!rgb) {
        return std::unexpected(rgb.error());
    }
    using Film = renderer::FilmT<Precision>;
    return film.add_sample(input.sample.pixel_x, input.sample.pixel_y,
                           typename Film::Color{
                               .red = static_cast<typename Film::Scalar>(rgb->red),
                               .green = static_cast<typename Film::Scalar>(rgb->green),
                               .blue = static_cast<typename Film::Scalar>(rgb->blue),
                           },
                           typename Film::Scalar{1});
}

template <renderer::AccumulationPrecision Precision>
[[nodiscard]] core::Result<renderer::ReferenceScalar>
positive_image_energy(const renderer::FilmT<Precision>& film) {
    auto energy = renderer::ReferenceScalar{};
    for (auto pixel_y = std::uint32_t{}; pixel_y < film.extent().height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{}; pixel_x < film.extent().width; ++pixel_x) {
            const auto pixel = film.resolved_pixel(pixel_x, pixel_y);
            if (!pixel) {
                return std::unexpected(pixel.error());
            }
            const auto red = static_cast<renderer::ReferenceScalar>(pixel->red);
            const auto green = static_cast<renderer::ReferenceScalar>(pixel->green);
            const auto blue = static_cast<renderer::ReferenceScalar>(pixel->blue);
            if (!std::isfinite(red) || !std::isfinite(green) || !std::isfinite(blue)) {
                return std::unexpected(parity_error(
                    core::StatusCode::invalid_argument,
                    "CUDA transport parity cannot measure a non-finite resolved pixel."));
            }
            energy += std::max(renderer::ReferenceScalar{}, red);
            energy += std::max(renderer::ReferenceScalar{}, green);
            energy += std::max(renderer::ReferenceScalar{}, blue);
        }
    }
    if (!std::isfinite(energy)) {
        return std::unexpected(
            parity_error(core::StatusCode::resource_exhausted,
                         "CUDA transport parity positive image energy is not representable."));
    }
    return energy;
}

[[nodiscard]] std::string metric_text(const renderer::ReferenceScalar value) {
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    auto stream = std::ostringstream{};
    stream << std::setprecision(std::numeric_limits<renderer::ReferenceScalar>::max_digits10)
           << value;
    return stream.str();
}

[[nodiscard]] core::Result<ParityResult>
run_parity(const FrameSceneHandle& scene, const std::span<const CudaWavefrontPathInput> inputs,
           const renderer::LightSampler& light_sampler, const ParityConfiguration& configuration,
           const std::span<const renderer::RayDifferential> scalar_differentials = {}) {
    if (!scene || inputs.empty()) {
        return std::unexpected(parity_error(
            core::StatusCode::invalid_argument,
            "CUDA transport parity requires an explicit scene and non-empty input batch."));
    }
    const auto expected_path_count = static_cast<std::size_t>(configuration.extent.width) *
                                     configuration.extent.height * configuration.samples_per_pixel;
    if (inputs.size() != expected_path_count) {
        return std::unexpected(parity_error(
            core::StatusCode::incompatible,
            "CUDA transport parity input count does not match its image configuration."));
    }
    if (!scalar_differentials.empty() && scalar_differentials.size() != inputs.size()) {
        return std::unexpected(parity_error(
            core::StatusCode::incompatible,
            "CUDA transport parity scalar differential count does not match its inputs."));
    }

    auto scalar_backend = create_analytic_accel_backend(scene);
    if (!scalar_backend) {
        return std::unexpected(scalar_backend.error());
    }
    if ((*scalar_backend)->kind() != AccelBackendKind::analytic_reference) {
        return std::unexpected(
            parity_error(core::StatusCode::incompatible,
                         "CUDA transport parity did not receive the analytic scalar oracle."));
    }
    auto scene_soa = CudaSceneSoA::upload(*scene);
    if (!scene_soa) {
        return std::unexpected(scene_soa.error());
    }
    auto scene_bvh = CudaSceneBvh::build(*scene_soa);
    if (!scene_bvh) {
        return std::unexpected(scene_bvh.error());
    }

    const auto cuda =
        trace_cuda_wavefront_transport(*scene_soa, *scene_bvh, inputs, std::cref(light_sampler),
                                       CudaWavefrontTransportOptions{
                                           .heuristic = configuration.heuristic,
                                           .depth_limits = configuration.depth_limits,
                                           .roulette_policy = configuration.roulette_policy,
                                       });
    if (!cuda) {
        return std::unexpected(cuda.error());
    }
    if (cuda->paths.size() != inputs.size() || cuda->terminal_cones.size() != inputs.size()) {
        return std::unexpected(
            parity_error(core::StatusCode::internal_error,
                         "CUDA transport parity received misaligned path and ray-cone outputs."));
    }

    auto scalar_film = renderer::ReferenceFilm::create(configuration.extent);
    auto cuda_film = renderer::Film::create(configuration.extent);
    if (!scalar_film) {
        return std::unexpected(scalar_film.error());
    }
    if (!cuda_film) {
        return std::unexpected(cuda_film.error());
    }

    auto scalar_paths = std::vector<renderer::BsdfOnlyPathResult>{};
    scalar_paths.reserve(inputs.size());
    auto maximum_path_error = renderer::ReferenceScalar{};
    for (auto path_index = std::size_t{}; path_index < inputs.size(); ++path_index) {
        const auto& input = inputs[path_index];
        const auto scalar = [&]() -> core::Result<renderer::BsdfOnlyPathResult> {
            if (scalar_differentials.empty()) {
                return trace_scene_mis(input.primary_ray, input.initial_state,
                                       renderer::SampleStream{input.sample}, **scalar_backend,
                                       light_sampler, configuration.heuristic,
                                       configuration.depth_limits, configuration.roulette_policy);
            }
            const auto tracked = trace_scene_mis_with_ray_differentials(
                scalar_differentials[path_index], input.initial_state,
                renderer::SampleStream{input.sample}, **scalar_backend, light_sampler,
                configuration.heuristic, configuration.depth_limits, configuration.roulette_policy);
            if (!tracked) {
                return std::unexpected(tracked.error());
            }
            return tracked->path;
        }();
        if (!scalar) {
            return std::unexpected(parity_error(scalar.error().code,
                                                "Scalar parity path " + std::to_string(path_index) +
                                                    " failed: " + scalar.error().message));
        }
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            const auto error = std::abs(
                static_cast<renderer::ReferenceScalar>(
                    cuda->paths[path_index].state.accumulated_radiance()[lane]) -
                static_cast<renderer::ReferenceScalar>(scalar->state.accumulated_radiance()[lane]));
            maximum_path_error = std::max(maximum_path_error, error);
        }
        if (auto status = accumulate_path(*scalar_film, input, scalar->state); !status) {
            return std::unexpected(status.error());
        }
        if (auto status = accumulate_path(*cuda_film, input, cuda->paths[path_index].state);
            !status) {
            return std::unexpected(status.error());
        }
        scalar_paths.push_back(*scalar);
    }

    const auto linear = renderer::compute_linear_metrics(*cuda_film, *scalar_film);
    const auto display = renderer::compute_display_psnr(*cuda_film, *scalar_film);
    if (!linear) {
        return std::unexpected(linear.error());
    }
    if (!display) {
        return std::unexpected(display.error());
    }
    const auto scalar_energy = positive_image_energy(*scalar_film);
    const auto cuda_energy = positive_image_energy(*cuda_film);
    if (!scalar_energy) {
        return std::unexpected(scalar_energy.error());
    }
    if (!cuda_energy) {
        return std::unexpected(cuda_energy.error());
    }

    auto cuda_paths = cuda->paths;
    auto cuda_terminal_cones = cuda->terminal_cones;
    const auto cuda_report = cuda->report;
    if (auto status = scene_bvh->close(); !status) {
        return std::unexpected(status.error());
    }
    if (auto status = scene_soa->close(); !status) {
        return std::unexpected(status.error());
    }
    return ParityResult{
        .linear = *linear,
        .display_psnr = display->psnr,
        .maximum_path_radiance_absolute_error = maximum_path_error,
        .scalar_positive_image_energy = *scalar_energy,
        .cuda_positive_image_energy = *cuda_energy,
        .scalar_paths = std::move(scalar_paths),
        .cuda_paths = std::move(cuda_paths),
        .cuda_terminal_cones = std::move(cuda_terminal_cones),
        .cuda_report = cuda_report,
    };
}

[[nodiscard]] std::optional<CudaWavefrontPathTermination>
cuda_termination(const renderer::BsdfOnlyPathTermination termination) {
    switch (termination) {
    case renderer::BsdfOnlyPathTermination::escaped_environment:
        return CudaWavefrontPathTermination::escaped_environment;
    case renderer::BsdfOnlyPathTermination::depth_limit:
        return CudaWavefrontPathTermination::depth_limit;
    case renderer::BsdfOnlyPathTermination::outside_bsdf_support:
        return CudaWavefrontPathTermination::outside_bsdf_support;
    case renderer::BsdfOnlyPathTermination::zero_throughput:
        return CudaWavefrontPathTermination::zero_throughput;
    case renderer::BsdfOnlyPathTermination::russian_roulette:
        return CudaWavefrontPathTermination::russian_roulette;
    }
    return std::nullopt;
}

void expect_finite_near(const float expected, const float actual, const float tolerance) {
    if (std::isfinite(expected)) {
        EXPECT_NEAR(actual, expected, tolerance);
    } else {
        EXPECT_EQ(actual, expected);
    }
}

void expect_path_parity(const renderer::BsdfOnlyPathResult& scalar,
                        const CudaWavefrontPathResult& cuda) {
    const auto expected_termination = cuda_termination(scalar.termination);
    ASSERT_TRUE(expected_termination.has_value());
    EXPECT_EQ(cuda.termination, *expected_termination);
    EXPECT_EQ(cuda.state.depth(), scalar.state.depth());
    EXPECT_EQ(cuda.state.depth_counters(), scalar.state.depth_counters());
    EXPECT_EQ(cuda.state.wavelengths(), scalar.state.wavelengths());
    EXPECT_EQ(cuda.state.delta_flags(), scalar.state.delta_flags());
    EXPECT_EQ(cuda.state.current_medium(), scalar.state.current_medium());
    EXPECT_EQ(cuda.blocked_depth_limits, scalar.blocked_depth_limits);
    EXPECT_NEAR(cuda.state.eta_scale(), scalar.state.eta_scale(),
                static_cast<float>(MaximumAbsoluteError));
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(cuda.state.beta()[lane], scalar.state.beta()[lane], MaximumAbsoluteError)
            << "spectral beta lane " << lane;
        EXPECT_NEAR(cuda.state.accumulated_radiance()[lane],
                    scalar.state.accumulated_radiance()[lane], MaximumAbsoluteError)
            << "spectral radiance lane " << lane;
    }

    expect_finite_near(scalar.terminal_ray.origin().x, cuda.terminal_ray.origin().x,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.origin().y, cuda.terminal_ray.origin().y,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.origin().z, cuda.terminal_ray.origin().z,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.direction().x, cuda.terminal_ray.direction().x,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.direction().y, cuda.terminal_ray.direction().y,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.direction().z, cuda.terminal_ray.direction().z,
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.t_min(), cuda.terminal_ray.t_min(),
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.t_max(), cuda.terminal_ray.t_max(),
                       TerminalGeometryTolerance);
    expect_finite_near(scalar.terminal_ray.time(), cuda.terminal_ray.time(),
                       TerminalGeometryTolerance);
    EXPECT_EQ(cuda.terminal_ray.mask(), scalar.terminal_ray.mask());
    EXPECT_EQ(cuda.terminal_ray.current_medium(), scalar.terminal_ray.current_medium());
}

void expect_parity(const ParityConfiguration& configuration, const ParityResult& result,
                   const bool require_positive_energy = true) {
    SCOPED_TRACE(configuration.scene_name);
    testing::Test::RecordProperty("scene", std::string{configuration.scene_name});
    testing::Test::RecordProperty("seed", std::to_string(configuration.seed));
    testing::Test::RecordProperty("width", static_cast<int>(configuration.extent.width));
    testing::Test::RecordProperty("height", static_cast<int>(configuration.extent.height));
    testing::Test::RecordProperty("samples_per_pixel",
                                  static_cast<int>(configuration.samples_per_pixel));
    testing::Test::RecordProperty("maximum_linear_mse", metric_text(MaximumLinearMse));
    testing::Test::RecordProperty("maximum_linear_rmse", metric_text(MaximumLinearRmse));
    testing::Test::RecordProperty("maximum_linear_absolute_error",
                                  metric_text(MaximumAbsoluteError));
    testing::Test::RecordProperty("maximum_path_radiance_absolute_error",
                                  metric_text(MaximumAbsoluteError));
    testing::Test::RecordProperty("minimum_display_psnr", metric_text(MinimumDisplayPsnr));
    testing::Test::RecordProperty("mse_linear", metric_text(result.linear.mse));
    testing::Test::RecordProperty("rmse_linear", metric_text(result.linear.rmse));
    testing::Test::RecordProperty("bias_mean", metric_text(result.linear.mean_bias));
    testing::Test::RecordProperty("max_abs", metric_text(result.linear.maximum_absolute_error));
    testing::Test::RecordProperty("path_radiance_max_abs",
                                  metric_text(result.maximum_path_radiance_absolute_error));
    testing::Test::RecordProperty("psnr_display", metric_text(result.display_psnr));
    testing::Test::RecordProperty("scalar_positive_image_energy",
                                  metric_text(result.scalar_positive_image_energy));
    testing::Test::RecordProperty("cuda_positive_image_energy",
                                  metric_text(result.cuda_positive_image_energy));

    EXPECT_LE(result.linear.mse, MaximumLinearMse);
    EXPECT_LE(result.linear.rmse, MaximumLinearRmse);
    EXPECT_LE(result.linear.maximum_absolute_error, MaximumAbsoluteError);
    EXPECT_LE(result.maximum_path_radiance_absolute_error, MaximumAbsoluteError);
    EXPECT_TRUE((std::isinf(result.display_psnr) && result.display_psnr > 0.0) ||
                (std::isfinite(result.display_psnr) && result.display_psnr >= MinimumDisplayPsnr))
        << "PSNR: " << result.display_psnr;
    if (require_positive_energy) {
        EXPECT_GT(result.scalar_positive_image_energy, renderer::ReferenceScalar{});
        EXPECT_GT(result.cuda_positive_image_energy, renderer::ReferenceScalar{});
    } else {
        EXPECT_GE(result.scalar_positive_image_energy, renderer::ReferenceScalar{});
        EXPECT_GE(result.cuda_positive_image_energy, renderer::ReferenceScalar{});
    }
    EXPECT_EQ(result.cuda_report.schema_version, CurrentCudaWavefrontTransportReportSchemaVersion);
    EXPECT_TRUE(result.cuda_report.has_light_sampler);
    EXPECT_GT(result.cuda_report.registered_light_count, 0U);
    EXPECT_EQ(result.cuda_report.heuristic, configuration.heuristic);
    EXPECT_EQ(result.cuda_report.light_sampling_strategy, renderer::LightSamplingStrategy::uniform);
    EXPECT_EQ(result.cuda_report.depth_limits, configuration.depth_limits);
    EXPECT_EQ(result.cuda_report.roulette_policy, configuration.roulette_policy);
    EXPECT_EQ(result.cuda_report.path_count, result.scalar_paths.size());
    EXPECT_EQ(result.cuda_report.terminated_paths, result.scalar_paths.size());
    EXPECT_EQ(result.cuda_report.queue_overflow_attempts, 0U);
    EXPECT_EQ(result.cuda_report.queue_rejected_lanes, 0U);
    ASSERT_EQ(result.cuda_paths.size(), result.scalar_paths.size());
    ASSERT_EQ(result.cuda_terminal_cones.size(), result.cuda_paths.size());
    for (auto path_index = std::size_t{}; path_index < result.scalar_paths.size(); ++path_index) {
        SCOPED_TRACE(path_index);
        expect_path_parity(result.scalar_paths[path_index], result.cuda_paths[path_index]);
        EXPECT_TRUE(std::isfinite(result.cuda_terminal_cones[path_index].width()));
        EXPECT_GE(result.cuda_terminal_cones[path_index].width(), 0.0F);
        EXPECT_TRUE(std::isfinite(result.cuda_terminal_cones[path_index].spread()));
        EXPECT_GE(result.cuda_terminal_cones[path_index].spread(), 0.0F);
    }
}

[[nodiscard]] ParityConfiguration
point_configuration(const std::string_view name, const renderer::RenderExtent extent,
                    const std::uint32_t samples_per_pixel,
                    const renderer::PathDepthLimits depth_limits,
                    const renderer::RussianRoulettePolicy roulette_policy) {
    return ParityConfiguration{
        .scene_name = name,
        .extent = extent,
        .samples_per_pixel = samples_per_pixel,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = depth_limits,
        .roulette_policy = roulette_policy,
    };
}

[[nodiscard]] ParityConfiguration
material_configuration(const std::string_view name, const std::size_t path_count,
                       const renderer::MisHeuristic heuristic = renderer::MisHeuristic::power) {
    if (path_count == 0U || path_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error{"CUDA material parity path count is not representable."};
    }
    return ParityConfiguration{
        .scene_name = name,
        .extent =
            renderer::RenderExtent{.width = static_cast<std::uint32_t>(path_count), .height = 1U},
        .samples_per_pixel = 1U,
        .seed = EvaluationSeed,
        .heuristic = heuristic,
        .depth_limits = OneSurfaceBounce,
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
}

[[nodiscard]] core::Result<ParityResult>
run_material_parity(const FrameSceneHandle& scene,
                    const std::span<const CudaWavefrontPathInput> inputs,
                    const std::string_view name,
                    const renderer::MisHeuristic heuristic = renderer::MisHeuristic::power) {
    const auto sampler = make_uniform_light_sampler(scene);
    if (!sampler) {
        return std::unexpected(sampler.error());
    }
    return run_parity(scene, inputs, *sampler,
                      material_configuration(name, inputs.size(), heuristic));
}

[[nodiscard]] core::Result<ParityResult> run_material_parity_with_limits(
    const FrameSceneHandle& scene, const std::span<const CudaWavefrontPathInput> inputs,
    const std::string_view name, const renderer::PathDepthLimits depth_limits,
    const renderer::MisHeuristic heuristic = renderer::MisHeuristic::power) {
    const auto sampler = make_uniform_light_sampler(scene);
    if (!sampler) {
        return std::unexpected(sampler.error());
    }
    auto configuration = material_configuration(name, inputs.size(), heuristic);
    configuration.depth_limits = depth_limits;
    return run_parity(scene, inputs, *sampler, configuration);
}

#if BLACKFRAME_CUDA_SURFACE_MAP_TESTS
struct SurfaceMapInputs final {
    std::vector<CudaWavefrontPathInput> cuda;
    std::vector<renderer::RayDifferential> scalar;
};

[[nodiscard]] core::Result<SurfaceMapInputs>
make_surface_map_inputs(const FrameSceneHandle& scene, const std::uint32_t path_count) {
    if (!scene || !scene->spectral_environment() || path_count == 0U) {
        return std::unexpected(parity_error(
            core::StatusCode::invalid_argument,
            "CUDA sphere surface-map parity requires an explicit scene and non-zero paths."));
    }
    const auto ray = surface_map_sphere_fixture::ray();
    const auto differential = surface_map_sphere_fixture::differential();
    const auto cone = surface_map_sphere_fixture::cone();
    const auto state = renderer::PathState::create_initial(
        scene->spectral_environment()->wavelengths, renderer::VacuumMedium);
    if (!ray) {
        return std::unexpected(ray.error());
    }
    if (!differential) {
        return std::unexpected(differential.error());
    }
    if (!cone) {
        return std::unexpected(cone.error());
    }
    if (!state) {
        return std::unexpected(state.error());
    }

    auto result = SurfaceMapInputs{};
    result.cuda.reserve(path_count);
    result.scalar.reserve(path_count);
    const auto sampler = renderer::IndependentSampler{EvaluationSeed};
    for (auto path_index = std::uint32_t{}; path_index < path_count; ++path_index) {
        const auto stream = sampler.make_stream(path_index, 0U, 0U);
        result.cuda.push_back(CudaWavefrontPathInput{
            .primary_ray = *ray,
            .primary_cone = *cone,
            .initial_state = *state,
            .sample = stream.index(),
        });
        result.scalar.push_back(*differential);
    }
    return result;
}

TEST(CudaTransportParityTest, MatchesNormalAndBumpMapsOnUvSphere) {
    ASSERT_TRUE(select_test_device());
    const auto normal_map = make_cuda_normal_map();
    const auto bump_map = make_cuda_bump_map();
    const auto flat_bump_map = make_cuda_flat_bump_map();
    ASSERT_TRUE(normal_map);
    ASSERT_TRUE(bump_map);
    ASSERT_TRUE(flat_bump_map);
    const auto mapped_scene = surface_map_sphere_fixture::make_scene(normal_map, bump_map);
    const auto negative_v_scene = surface_map_sphere_fixture::make_scene(
        normal_map, bump_map, renderer::TangentSpaceNormalYConvention::negative_v);
    const auto baseline_scene = surface_map_sphere_fixture::make_scene({}, {});
    const auto normal_only_scene = surface_map_sphere_fixture::make_scene(normal_map, {});
    const auto flat_bump_scene = surface_map_sphere_fixture::make_scene(normal_map, flat_bump_map);
    ASSERT_TRUE(mapped_scene) << mapped_scene.error().message;
    ASSERT_TRUE(negative_v_scene) << negative_v_scene.error().message;
    ASSERT_TRUE(baseline_scene) << baseline_scene.error().message;
    ASSERT_TRUE(normal_only_scene) << normal_only_scene.error().message;
    ASSERT_TRUE(flat_bump_scene) << flat_bump_scene.error().message;

    constexpr auto path_count = std::uint32_t{32U};
    const auto mapped_inputs = make_surface_map_inputs(*mapped_scene, path_count);
    const auto negative_v_inputs = make_surface_map_inputs(*negative_v_scene, path_count);
    const auto baseline_inputs = make_surface_map_inputs(*baseline_scene, path_count);
    const auto normal_only_inputs = make_surface_map_inputs(*normal_only_scene, path_count);
    const auto flat_bump_inputs = make_surface_map_inputs(*flat_bump_scene, path_count);
    const auto mapped_sampler = make_uniform_light_sampler(*mapped_scene);
    const auto negative_v_sampler = make_uniform_light_sampler(*negative_v_scene);
    const auto baseline_sampler = make_uniform_light_sampler(*baseline_scene);
    const auto normal_only_sampler = make_uniform_light_sampler(*normal_only_scene);
    const auto flat_bump_sampler = make_uniform_light_sampler(*flat_bump_scene);
    ASSERT_TRUE(mapped_inputs) << mapped_inputs.error().message;
    ASSERT_TRUE(negative_v_inputs) << negative_v_inputs.error().message;
    ASSERT_TRUE(baseline_inputs) << baseline_inputs.error().message;
    ASSERT_TRUE(normal_only_inputs) << normal_only_inputs.error().message;
    ASSERT_TRUE(flat_bump_inputs) << flat_bump_inputs.error().message;
    ASSERT_TRUE(mapped_sampler) << mapped_sampler.error().message;
    ASSERT_TRUE(negative_v_sampler) << negative_v_sampler.error().message;
    ASSERT_TRUE(baseline_sampler) << baseline_sampler.error().message;
    ASSERT_TRUE(normal_only_sampler) << normal_only_sampler.error().message;
    ASSERT_TRUE(flat_bump_sampler) << flat_bump_sampler.error().message;

    const auto configuration = material_configuration("SurfaceMapUvSphere", path_count);
    const auto mapped = run_parity(*mapped_scene, mapped_inputs->cuda, *mapped_sampler,
                                   configuration, mapped_inputs->scalar);
    ASSERT_TRUE(mapped) << mapped.error().message;
    expect_parity(configuration, *mapped);
    testing::Test::RecordProperty("surface_maps_mse_linear", metric_text(mapped->linear.mse));
    testing::Test::RecordProperty("surface_maps_rmse_linear", metric_text(mapped->linear.rmse));
    testing::Test::RecordProperty("surface_maps_bias_mean", metric_text(mapped->linear.mean_bias));
    testing::Test::RecordProperty("surface_maps_max_abs",
                                  metric_text(mapped->linear.maximum_absolute_error));
    testing::Test::RecordProperty("surface_maps_path_radiance_max_abs",
                                  metric_text(mapped->maximum_path_radiance_absolute_error));
    testing::Test::RecordProperty("surface_maps_psnr_display", metric_text(mapped->display_psnr));

    const auto negative_v_configuration =
        material_configuration("SurfaceMapUvSphereNegativeV", path_count);
    const auto negative_v =
        run_parity(*negative_v_scene, negative_v_inputs->cuda, *negative_v_sampler,
                   negative_v_configuration, negative_v_inputs->scalar);
    ASSERT_TRUE(negative_v) << negative_v.error().message;
    expect_parity(negative_v_configuration, *negative_v);

    const auto baseline =
        run_parity(*baseline_scene, baseline_inputs->cuda, *baseline_sampler,
                   material_configuration("SurfaceMapUvSphereBaseline", path_count));
    ASSERT_TRUE(baseline) << baseline.error().message;
    expect_parity(material_configuration("SurfaceMapUvSphereBaseline", path_count), *baseline);

    const auto normal_only_configuration =
        material_configuration("SurfaceMapUvSphereNormalOnly", path_count);
    const auto normal_only =
        run_parity(*normal_only_scene, normal_only_inputs->cuda, *normal_only_sampler,
                   normal_only_configuration, normal_only_inputs->scalar);
    ASSERT_TRUE(normal_only) << normal_only.error().message;
    expect_parity(normal_only_configuration, *normal_only);
    const auto flat_bump_configuration =
        material_configuration("SurfaceMapUvSphereFlatBump", path_count);
    const auto flat_bump = run_parity(*flat_bump_scene, flat_bump_inputs->cuda, *flat_bump_sampler,
                                      flat_bump_configuration, flat_bump_inputs->scalar);
    ASSERT_TRUE(flat_bump) << flat_bump.error().message;
    expect_parity(flat_bump_configuration, *flat_bump);

    auto maximum_terminal_direction_delta = renderer::ReferenceScalar{};
    auto maximum_radiance_delta = renderer::ReferenceScalar{};
    ASSERT_EQ(mapped->cuda_paths.size(), baseline->cuda_paths.size());
    for (auto path_index = std::size_t{}; path_index < mapped->cuda_paths.size(); ++path_index) {
        const auto& mapped_path = mapped->cuda_paths[path_index];
        const auto& baseline_path = baseline->cuda_paths[path_index];
        const auto direction_delta =
            std::abs(
                static_cast<renderer::ReferenceScalar>(mapped_path.terminal_ray.direction().x) -
                baseline_path.terminal_ray.direction().x) +
            std::abs(
                static_cast<renderer::ReferenceScalar>(mapped_path.terminal_ray.direction().y) -
                baseline_path.terminal_ray.direction().y) +
            std::abs(
                static_cast<renderer::ReferenceScalar>(mapped_path.terminal_ray.direction().z) -
                baseline_path.terminal_ray.direction().z);
        maximum_terminal_direction_delta =
            std::max(maximum_terminal_direction_delta, direction_delta);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            maximum_radiance_delta =
                std::max(maximum_radiance_delta,
                         std::abs(static_cast<renderer::ReferenceScalar>(
                                      mapped_path.state.accumulated_radiance()[lane]) -
                                  baseline_path.state.accumulated_radiance()[lane]));
        }
    }
    EXPECT_GT(maximum_terminal_direction_delta, 1.0e-3)
        << "Normal/bump maps must measurably perturb the CUDA shading frame.";
    EXPECT_GT(maximum_radiance_delta, 1.0e-5)
        << "Normal/bump maps must measurably change the CUDA sphere transport.";

    auto maximum_y_convention_direction_delta = renderer::ReferenceScalar{};
    auto maximum_y_convention_radiance_delta = renderer::ReferenceScalar{};
    ASSERT_EQ(mapped->cuda_paths.size(), negative_v->cuda_paths.size());
    for (auto path_index = std::size_t{}; path_index < mapped->cuda_paths.size(); ++path_index) {
        const auto& positive_path = mapped->cuda_paths[path_index];
        const auto& negative_path = negative_v->cuda_paths[path_index];
        const auto direction_delta =
            std::abs(
                static_cast<renderer::ReferenceScalar>(positive_path.terminal_ray.direction().x) -
                negative_path.terminal_ray.direction().x) +
            std::abs(
                static_cast<renderer::ReferenceScalar>(positive_path.terminal_ray.direction().y) -
                negative_path.terminal_ray.direction().y) +
            std::abs(
                static_cast<renderer::ReferenceScalar>(positive_path.terminal_ray.direction().z) -
                negative_path.terminal_ray.direction().z);
        maximum_y_convention_direction_delta =
            std::max(maximum_y_convention_direction_delta, direction_delta);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            maximum_y_convention_radiance_delta =
                std::max(maximum_y_convention_radiance_delta,
                         std::abs(static_cast<renderer::ReferenceScalar>(
                                      positive_path.state.accumulated_radiance()[lane]) -
                                  negative_path.state.accumulated_radiance()[lane]));
        }
    }
    EXPECT_GT(maximum_y_convention_direction_delta, 1.0e-3)
        << "positive-V and negative-V normal maps must produce distinct shading frames.";
    EXPECT_GT(maximum_y_convention_radiance_delta, 1.0e-5)
        << "The normal-map Y convention must measurably affect sphere transport.";
    testing::Test::RecordProperty("surface_maps_activation_direction_max_abs",
                                  metric_text(maximum_terminal_direction_delta));
    testing::Test::RecordProperty("surface_maps_activation_radiance_max_abs",
                                  metric_text(maximum_radiance_delta));
    testing::Test::RecordProperty("surface_maps_y_convention_direction_max_abs",
                                  metric_text(maximum_y_convention_direction_delta));
    testing::Test::RecordProperty("surface_maps_y_convention_radiance_max_abs",
                                  metric_text(maximum_y_convention_radiance_delta));

    ASSERT_EQ(normal_only->cuda_paths.size(), flat_bump->cuda_paths.size());
    for (auto path_index = std::size_t{}; path_index < normal_only->cuda_paths.size();
         ++path_index) {
        const auto& normal_only_path = normal_only->cuda_paths[path_index];
        const auto& flat_bump_path = flat_bump->cuda_paths[path_index];
        EXPECT_NEAR(flat_bump_path.terminal_ray.direction().x,
                    normal_only_path.terminal_ray.direction().x, 2.0e-5F);
        EXPECT_NEAR(flat_bump_path.terminal_ray.direction().y,
                    normal_only_path.terminal_ray.direction().y, 2.0e-5F);
        EXPECT_NEAR(flat_bump_path.terminal_ray.direction().z,
                    normal_only_path.terminal_ray.direction().z, 2.0e-5F);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            EXPECT_NEAR(flat_bump_path.state.accumulated_radiance()[lane],
                        normal_only_path.state.accumulated_radiance()[lane], 2.0e-5F);
        }
    }
}

TEST(CudaTransportParityTest, RejectsMalformedSurfaceMapColumnsOnDevice) {
    ASSERT_TRUE(select_test_device());
    const auto normal_map = make_cuda_normal_map();
    const auto bump_map = make_cuda_bump_map();
    ASSERT_TRUE(normal_map);
    ASSERT_TRUE(bump_map);
    const auto scene = surface_map_sphere_fixture::make_scene(normal_map, bump_map);
    ASSERT_TRUE(scene) << scene.error().message;
    const auto inputs = make_surface_map_inputs(*scene, 1U);
    const auto sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs) << inputs.error().message;
    ASSERT_TRUE(sampler) << sampler.error().message;

    const auto expect_rejected = [&](const std::uint64_t offset, const auto& value,
                                     const std::string_view failure_context) {
        auto scene_soa = CudaSceneSoA::upload(**scene);
        ASSERT_TRUE(scene_soa) << scene_soa.error().message;
        auto scene_bvh = CudaSceneBvh::build(*scene_soa);
        ASSERT_TRUE(scene_bvh) << scene_bvh.error().message;
        ASSERT_LE(offset, scene_soa->size_bytes());
        ASSERT_LE(sizeof(value), scene_soa->size_bytes() - static_cast<std::size_t>(offset));
        auto* destination = const_cast<std::uint8_t*>(scene_soa->device_data()) + offset;
        ASSERT_EQ(cudaMemcpy(destination, &value, sizeof(value), cudaMemcpyHostToDevice),
                  cudaSuccess);
        const auto traced = trace_cuda_wavefront_transport(
            *scene_soa, *scene_bvh, inputs->cuda, std::cref(*sampler),
            CudaWavefrontTransportOptions{
                .heuristic = renderer::MisHeuristic::power,
                .depth_limits = OneSurfaceBounce,
                .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
            });
        EXPECT_FALSE(traced) << failure_context;
        if (!traced) {
            EXPECT_NE(traced.error().code, core::StatusCode::success) << failure_context;
        }
        EXPECT_TRUE(scene_bvh->close()) << failure_context;
        EXPECT_TRUE(scene_soa->close()) << failure_context;
    };

    auto metadata_upload = CudaSceneSoA::upload(**scene);
    ASSERT_TRUE(metadata_upload) << metadata_upload.error().message;
    const auto pristine_header = metadata_upload->header();
    ASSERT_TRUE(metadata_upload->close());
    auto truncated_texels =
        pristine_header.columns[xpu::shared::scene_soa_column::image_texel_value];
    ASSERT_GT(truncated_texels.element_count, 0U);
    --truncated_texels.element_count;
    expect_rejected(offsetof(xpu::shared::SceneSoaHeader, columns) +
                        sizeof(xpu::shared::SceneSoaColumnDescriptor) *
                            xpu::shared::scene_soa_column::image_texel_value,
                    truncated_texels, "A truncated device texel column must fail before shading.");

    const auto invalid_channel = std::uint32_t{99U};
    expect_rejected(
        pristine_header.columns[xpu::shared::scene_soa_column::material_normal_map_red_channel]
            .offset_bytes,
        invalid_channel, "An out-of-range device normal-map channel must fail explicitly.");
    const auto invalid_y_convention = std::uint32_t{2U};
    expect_rejected(
        pristine_header.columns[xpu::shared::scene_soa_column::material_normal_map_y_convention]
            .offset_bytes,
        invalid_y_convention, "An unknown device normal-map Y convention must fail explicitly.");
    const auto color_storage =
        static_cast<std::uint32_t>(renderer::TextureColorSpace::scene_linear_srgb);
    expect_rejected(
        pristine_header.columns[xpu::shared::scene_soa_column::image_texture_storage_color_space]
            .offset_bytes,
        color_storage, "A color-tagged device normal map must fail instead of being sampled.");
    const auto empty_mip_range = std::uint64_t{};
    expect_rejected(pristine_header.columns[xpu::shared::scene_soa_column::image_texture_mip_count]
                        .offset_bytes,
                    empty_mip_range, "An empty device image mip range must fail explicitly.");
}
#endif

TEST(CudaTransportParityTest, MatchesPrimaryEnvironmentMiss) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = point_configuration(
        "EnvironmentMiss", renderer::RenderExtent{.width = 6U, .height = 4U}, 1U,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, false));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
}

TEST(CudaTransportParityTest, MatchesDirectionalLightNee) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_punctual_light_scene(SceneDirectionalLight{
        .propagation_direction = renderer::Vector3{.z = -1.0F},
        .spectral_irradiance = {.values = {1.0F, 2.0F, 3.0F, 4.0F}},
    });
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = point_configuration(
        "DirectionalLightNee", renderer::RenderExtent{.width = 6U, .height = 4U}, 1U,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

TEST(CudaTransportParityTest, MatchesWideSpotLightNee) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_punctual_light_scene(SceneSpotLight{
        .position = renderer::Point3{.z = 1.5F},
        .absolute_position_error = {},
        .emission_direction = renderer::Vector3{.z = -1.0F},
        .inner_half_angle_radians = 0.0F,
        .outer_half_angle_radians = std::numbers::pi_v<renderer::TransportScalar> / 2.0F,
        .on_axis_spectral_radiant_intensity = {.values = {4.0F, 3.0F, 2.0F, 1.0F}},
    });
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = point_configuration(
        "WideSpotLightNee", renderer::RenderExtent{.width = 6U, .height = 4U}, 1U,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

class CudaAreaEmitterParityTest : public testing::TestWithParam<renderer::MisHeuristic> {};

TEST_P(CudaAreaEmitterParityTest, MatchesBalanceAndPowerMis) {
    ASSERT_TRUE(select_test_device());
    const auto scene = cornell_wavefront_test::make_cornell_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = ParityConfiguration{
        .scene_name = GetParam() == renderer::MisHeuristic::balance ? "AreaEmitterBalance"
                                                                    : "AreaEmitterPower",
        .extent = renderer::RenderExtent{.width = 4U, .height = 4U},
        .samples_per_pixel = 1U,
        .seed = EvaluationSeed,
        .heuristic = GetParam(),
        .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    const auto camera = cornell_wavefront_test::make_camera(configuration.extent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&camera](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return camera_primary_ray(*camera, index);
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

INSTANTIATE_TEST_SUITE_P(BalanceAndPower, CudaAreaEmitterParityTest,
                         testing::Values(renderer::MisHeuristic::balance,
                                         renderer::MisHeuristic::power));

TEST(CudaTransportParityTest, MatchesMixedRegistryAndComplementaryEmitterHitMis) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_mixed_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->punctual_lights().size(), 1U);
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 1U);
    const auto configuration = ParityConfiguration{
        .scene_name = "MixedPunctualAreaMis",
        .extent = renderer::RenderExtent{.width = 8U, .height = 8U},
        .samples_per_pixel = 1U,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::balance,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    ASSERT_EQ(light_sampler->light_count(), 2U);

    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    auto selected_slots = std::array<std::size_t, 2U>{};
    for (const auto& input : *inputs) {
        const auto canonical =
            renderer::SampleStream{input.sample}.sample_1d(dimensions->light_selection);
        const auto selection = light_sampler->sample(canonical);
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ASSERT_LT(selection->light_index(), selected_slots.size());
        ++selected_slots[selection->light_index()];
    }
    EXPECT_GT(selected_slots[0], 0U);
    EXPECT_GT(selected_slots[1], 0U);

    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    const auto emitter_hits =
        std::ranges::count_if(parity->scalar_paths, [](const renderer::BsdfOnlyPathResult& path) {
            return path.termination == renderer::BsdfOnlyPathTermination::depth_limit;
        });
    EXPECT_GT(emitter_hits, 0);
    EXPECT_GT(parity->cuda_report.closure_samples, 0U);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

TEST(CudaTransportParityTest, MatchesRussianRouletteDecisions) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto roulette = renderer::RussianRoulettePolicy::create_enabled(1U, 0.5F, 0.5F);
    ASSERT_TRUE(roulette.has_value()) << roulette.error().message;
    const auto configuration =
        point_configuration("RussianRoulette", renderer::RenderExtent{.width = 6U, .height = 4U},
                            1U, renderer::PathDepthLimits{.diffuse = 2U}, *roulette);
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    const auto terminated =
        std::ranges::count_if(parity->scalar_paths, [](const renderer::BsdfOnlyPathResult& path) {
            return path.termination == renderer::BsdfOnlyPathTermination::russian_roulette;
        });
    EXPECT_GT(terminated, 0);
    EXPECT_LT(static_cast<std::size_t>(terminated), parity->scalar_paths.size());
}

TEST(CudaTransportParityTest, PreservesRepresentableSubnormalLambertProducts) {
    ASSERT_TRUE(select_test_device());
    constexpr auto initial_beta_value = renderer::TransportScalar{0x1p120F};
    const auto reflectance = std::numeric_limits<renderer::TransportScalar>::denorm_min();
    ASSERT_EQ(std::fpclassify(reflectance), FP_SUBNORMAL);
    const auto expected_throughput = initial_beta_value * reflectance;
    ASSERT_TRUE(std::isnormal(expected_throughput));

    const auto scene = make_punctual_light_scene(
        ScenePointLight{
            .position = renderer::Point3{.z = 1.5F},
            .absolute_position_error = {},
            .spectral_radiant_intensity = constant_spectrum(1.0F),
        },
        constant_spectrum(reflectance), constant_spectrum(1.0F));
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = ParityConfiguration{
        .scene_name = "SubnormalReflectance",
        .extent = renderer::RenderExtent{.width = 1U, .height = 1U},
        .samples_per_pixel = 1U,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::balance,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto high_beta_state = renderer::PathState::create(
        constant_spectrum(initial_beta_value), renderer::TransportSpectrum{},
        renderer::PathDepthCounters{}, renderer::TransportScalar{1},
        (*scene)->spectral_environment()->wavelengths, renderer::PathDeltaFlags::none,
        renderer::VacuumMedium);
    ASSERT_TRUE(high_beta_state.has_value()) << high_beta_state.error().message;
    inputs->front().initial_state = *high_beta_state;
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;

    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    ASSERT_EQ(parity->scalar_paths.size(), 1U);
    ASSERT_EQ(parity->cuda_paths.size(), 1U);
    EXPECT_EQ(parity->scalar_paths.front().termination,
              renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(parity->cuda_paths.front().termination,
              CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(parity->cuda_paths.front().blocked_depth_limits, renderer::ScatteringLobe::none);
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ(parity->scalar_paths.front().state.beta()[lane], expected_throughput);
        EXPECT_EQ(parity->cuda_paths.front().state.beta()[lane], expected_throughput);
        EXPECT_GT(parity->scalar_paths.front().state.accumulated_radiance()[lane], 0.0F);
        EXPECT_FLOAT_EQ(parity->cuda_paths.front().state.accumulated_radiance()[lane],
                        parity->scalar_paths.front().state.accumulated_radiance()[lane]);
    }
}

TEST(CudaTransportParityTest, MatchesPointLightNee) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = point_configuration(
        "PointLightNee", renderer::RenderExtent{.width = 8U, .height = 8U}, 4U,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&configuration](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, configuration.extent, true));
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

TEST(CudaTransportParityTest, SkipsOccludedUnrepresentablePointRadiometry) {
    ASSERT_TRUE(select_test_device());
    constexpr auto initial_beta_value = renderer::TransportScalar{0x1p120F};
    const auto scene = make_occluded_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = ParityConfiguration{
        .scene_name = "OccludedPointLight",
        .extent = renderer::RenderExtent{.width = 2U, .height = 1U},
        .samples_per_pixel = 1U,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    auto inputs =
        make_inputs(configuration.extent, configuration.samples_per_pixel, configuration.seed,
                    (*scene)->spectral_environment()->wavelengths,
                    [](const renderer::PixelSampleIndex&, const renderer::SampleStream&) {
                        return point_primary_ray(renderer::Ray::create(
                            renderer::Point3{.z = 0.25F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                            std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                            renderer::AllRayVisibility, renderer::VacuumMedium));
                    });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_EQ(inputs->size(), 2U);
    const auto miss_ray =
        renderer::Ray::create(renderer::Point3{.z = 0.25F}, renderer::Vector3{.x = 1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(miss_ray.has_value()) << miss_ray.error().message;
    inputs->front().primary_ray = *miss_ray;
    const auto high_beta_state = renderer::PathState::create(
        constant_spectrum(initial_beta_value), renderer::TransportSpectrum{},
        renderer::PathDepthCounters{}, renderer::TransportScalar{1},
        (*scene)->spectral_environment()->wavelengths, renderer::PathDeltaFlags::none,
        renderer::VacuumMedium);
    ASSERT_TRUE(high_beta_state.has_value()) << high_beta_state.error().message;
    inputs->back().initial_state = *high_beta_state;
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;

    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    ASSERT_EQ(parity->scalar_paths.size(), 2U);
    ASSERT_EQ(parity->cuda_paths.size(), 2U);
    EXPECT_EQ(parity->scalar_paths.front().termination,
              renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(parity->cuda_paths.front().termination,
              CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(parity->scalar_paths.back().termination,
              renderer::BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(parity->cuda_paths.back().termination, CudaWavefrontPathTermination::depth_limit);
    EXPECT_EQ(parity->cuda_paths.back().blocked_depth_limits, renderer::ScatteringLobe::diffuse);
    EXPECT_EQ(parity->scalar_paths.back().state.accumulated_radiance(),
              renderer::TransportSpectrum{});
    EXPECT_EQ(parity->cuda_paths.back().state.accumulated_radiance(),
              renderer::TransportSpectrum{});
    EXPECT_EQ(parity->cuda_report.light_samples, 1U);
    EXPECT_EQ(parity->cuda_report.shadow_queries, 1U);
}

TEST(CudaTransportParityTest, MatchesCornellDiffuse) {
    ASSERT_TRUE(select_test_device());
    const auto scene = cornell_wavefront_test::make_cornell_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto configuration = ParityConfiguration{
        .scene_name = "CornellDiffuse",
        .extent = renderer::RenderExtent{.width = 4U, .height = 4U},
        .samples_per_pixel = 2U,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 5U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    const auto camera = cornell_wavefront_test::make_camera(configuration.extent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_inputs(
        configuration.extent, configuration.samples_per_pixel, configuration.seed,
        (*scene)->spectral_environment()->wavelengths,
        [&camera](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return camera_primary_ray(*camera, index);
        });
    const auto light_sampler = make_uniform_light_sampler(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
    const auto parity = run_parity(*scene, *inputs, *light_sampler, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(configuration, *parity);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

TEST(CudaTransportAllLobesParityTest, MatchesLambertRoughDiffuseAndIsotropicConductor) {
    ASSERT_TRUE(select_test_device());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 16U}};

    const auto verify = [&](const std::string_view name, SceneClosureMixture closure,
                            const renderer::ScatteringLobe expected_family) {
        SCOPED_TRACE(name);
        const auto scene = make_material_scene(std::move(closure));
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = run_material_parity(*scene, *inputs, name);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        expect_parity(material_configuration(name, inputs->size()), *parity);

        auto accepted_events = std::size_t{};
        for (const auto& path : parity->scalar_paths) {
            if (path.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
                EXPECT_EQ(path.state.depth(), 0U);
                EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
                EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::none);
                continue;
            }
            ++accepted_events;
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth(), 1U);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
            EXPECT_GT(path.terminal_ray.direction().z, 0.0F);
            if (expected_family == renderer::ScatteringLobe::diffuse) {
                EXPECT_EQ(path.state.depth_counters(),
                          (renderer::PathDepthCounters{.diffuse = 1U}));
            } else {
                EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
            }
        }
        EXPECT_GT(accepted_events, 0U);
        EXPECT_GT(parity->cuda_report.closure_samples, 0U);
        EXPECT_GT(parity->cuda_report.light_samples, 0U);
        EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
    };

    verify("Lambert", require_lambertian_scene_closure(constant_spectrum(0.68F)),
           renderer::ScatteringLobe::diffuse);
    verify("RoughDiffuse", rough_diffuse_scene_closure(), renderer::ScatteringLobe::diffuse);
    verify("RoughConductorIsotropic", rough_conductor_scene_closure(0.35F, 0.35F),
           renderer::ScatteringLobe::glossy);
}

TEST(CudaTransportAllLobesParityTest, RotatesAnisotropicConductorFrameCoherently) {
    ASSERT_TRUE(select_test_device());
    const auto unrotated_scene = make_material_scene(rough_conductor_scene_closure(0.15F, 0.6F));
    const auto rotated_scene = make_material_scene(
        rough_conductor_scene_closure(0.15F, 0.6F, std::numbers::pi_v<float> / 2.0F));
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(unrotated_scene.has_value()) << unrotated_scene.error().message;
    ASSERT_TRUE(rotated_scene.has_value()) << rotated_scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 16U}};
    const auto inputs = make_material_inputs(*unrotated_scene, *primary, bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto unrotated =
        run_material_parity(*unrotated_scene, *inputs, "RoughConductorAnisotropic");
    const auto rotated =
        run_material_parity(*rotated_scene, *inputs, "RoughConductorAnisotropicRotated");
    ASSERT_TRUE(unrotated.has_value()) << unrotated.error().message;
    ASSERT_TRUE(rotated.has_value()) << rotated.error().message;
    expect_parity(material_configuration("RoughConductorAnisotropic", inputs->size()), *unrotated);
    expect_parity(material_configuration("RoughConductorAnisotropicRotated", inputs->size()),
                  *rotated);
    ASSERT_EQ(unrotated->scalar_paths.size(), rotated->scalar_paths.size());

    auto accepted_events = std::size_t{};
    auto observed_azimuthal_sample = false;
    for (auto index = std::size_t{}; index < unrotated->scalar_paths.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& base = unrotated->scalar_paths[index];
        const auto& quarter_turn = rotated->scalar_paths[index];
        EXPECT_EQ(quarter_turn.termination, base.termination);
        if (base.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
            EXPECT_EQ(base.state.depth_counters(), renderer::PathDepthCounters{});
            EXPECT_EQ(quarter_turn.state.depth_counters(), renderer::PathDepthCounters{});
            continue;
        }
        ++accepted_events;
        EXPECT_EQ(base.state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
        EXPECT_EQ(quarter_turn.state.depth_counters(), base.state.depth_counters());
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().x, -base.terminal_ray.direction().y,
                    TerminalGeometryTolerance);
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().y, base.terminal_ray.direction().x,
                    TerminalGeometryTolerance);
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().z, base.terminal_ray.direction().z,
                    TerminalGeometryTolerance);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            EXPECT_NEAR(quarter_turn.state.beta()[lane], base.state.beta()[lane],
                        MaximumAbsoluteError);
            EXPECT_NEAR(quarter_turn.state.accumulated_radiance()[lane],
                        base.state.accumulated_radiance()[lane], MaximumAbsoluteError);
        }
        observed_azimuthal_sample = observed_azimuthal_sample ||
                                    std::abs(base.terminal_ray.direction().x) > 1.0e-3F ||
                                    std::abs(base.terminal_ray.direction().y) > 1.0e-3F;
    }
    EXPECT_GT(accepted_events, 0U);
    EXPECT_TRUE(observed_azimuthal_sample);
}

TEST(CudaTransportAllLobesParityTest, MatchesRoughDielectricReflectionTransmissionAndTir) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_material_scene(rough_dielectric_scene_closure());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto branch_bands = std::array{
        ComponentSampleBand{.lower = 0.0F, .upper = 0.01F, .count = 8U},
        ComponentSampleBand{.lower = 0.5F, .upper = 0.95F, .count = 8U},
    };
    const auto inputs = make_material_inputs(*scene, *primary, branch_bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto parity = run_material_parity(*scene, *inputs, "RoughDielectricBranches");
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    expect_parity(material_configuration("RoughDielectricBranches", inputs->size()), *parity);

    auto reflected = std::size_t{};
    auto transmitted = std::size_t{};
    for (const auto& path : parity->scalar_paths) {
        if (path.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
            EXPECT_EQ(path.state.depth(), 0U);
            continue;
        }
        EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
        EXPECT_EQ(path.state.depth(), 1U);
        EXPECT_EQ(path.state.depth_counters().glossy, 1U);
        EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        if (path.terminal_ray.direction().z > 0.0F) {
            ++reflected;
            EXPECT_EQ(path.state.depth_counters().transmission, 0U);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
        } else {
            ++transmitted;
            EXPECT_EQ(path.state.depth_counters().transmission, 1U);
            EXPECT_NEAR(path.state.eta_scale(), 2.25F, MaximumAbsoluteError);
        }
    }
    EXPECT_GT(reflected, 0U);
    EXPECT_GT(transmitted, 0U);

    const auto tir_scene = make_material_scene(rough_dielectric_scene_closure(0.05F, 0.2F));
    constexpr auto inside_outgoing = renderer::Vector3{.x = 0.8F, .z = -0.6F};
    const auto tir_primary = material_primary_ray(inside_outgoing);
    ASSERT_TRUE(tir_scene.has_value()) << tir_scene.error().message;
    ASSERT_TRUE(tir_primary.has_value()) << tir_primary.error().message;
    constexpr auto tir_band =
        std::array{ComponentSampleBand{.lower = 0.95F, .upper = 1.0F, .count = 64U}};
    const auto tir_inputs = make_material_inputs(*tir_scene, *tir_primary, tir_band);
    ASSERT_TRUE(tir_inputs.has_value()) << tir_inputs.error().message;
    const auto tir_parity = run_material_parity(*tir_scene, *tir_inputs, "RoughDielectricTir");
    ASSERT_TRUE(tir_parity.has_value()) << tir_parity.error().message;
    expect_parity(material_configuration("RoughDielectricTir", tir_inputs->size()), *tir_parity);

    auto total_internal_reflections = std::size_t{};
    for (const auto& path : tir_parity->scalar_paths) {
        if (path.state.depth_counters().glossy != 1U ||
            path.state.depth_counters().transmission != 0U ||
            path.terminal_ray.direction().z >= 0.0F) {
            continue;
        }
        auto microfacet = inside_outgoing + path.terminal_ray.direction();
        const auto length = std::hypot(std::hypot(microfacet.x, microfacet.y), microfacet.z);
        ASSERT_TRUE(std::isfinite(length));
        ASSERT_GT(length, 0.0F);
        microfacet = microfacet / length;
        if (renderer::dot(inside_outgoing, microfacet) < 0.0F) {
            microfacet = microfacet * -1.0F;
        }
        const auto fresnel = renderer::dielectric_fresnel(
            std::abs(renderer::dot(inside_outgoing, microfacet)), 1.5F, 1.0F);
        ASSERT_TRUE(fresnel.has_value()) << fresnel.error().message;
        if (*fresnel == 1.0F) {
            ++total_internal_reflections;
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        }
    }
    EXPECT_GT(total_internal_reflections, 0U);
}

TEST(CudaTransportAllLobesParityTest, MatchesSpecularDeltaEntryExitAndTir) {
    ASSERT_TRUE(select_test_device());
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 4U}};
    const auto primary_cone = renderer::RayCone::create(0.125F, 0.25F);
    ASSERT_TRUE(primary_cone.has_value()) << primary_cone.error().message;

    const auto verify = [&](const std::string_view name, SceneClosureMixture closure,
                            const renderer::Vector3 outgoing, const bool expect_support,
                            const renderer::PathDepthCounters expected_depth,
                            const float expected_eta_scale, const renderer::Vector3 expected_ray) {
        SCOPED_TRACE(name);
        ASSERT_EQ(closure.closures.size(), 1U);
        const auto reference_closure = closure.closures.closures().front();
        const auto scene = make_material_scene(std::move(closure));
        const auto primary = material_primary_ray(outgoing);
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        ASSERT_TRUE(primary.has_value()) << primary.error().message;
        auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        for (auto& input : *inputs) {
            input.primary_cone = *primary_cone;
        }
        const auto parity = run_material_parity(*scene, *inputs, name);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        expect_parity(material_configuration(name, inputs->size()), *parity, expect_support);
        for (auto index = std::size_t{}; index < parity->scalar_paths.size(); ++index) {
            const auto& path = parity->scalar_paths[index];
            if (!expect_support) {
                EXPECT_EQ(path.termination,
                          renderer::BsdfOnlyPathTermination::outside_bsdf_support);
                EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
                EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::none);
                EXPECT_EQ(path.state.eta_scale(), 1.0F);
                EXPECT_EQ(path.state.accumulated_radiance(), renderer::TransportSpectrum{});
                EXPECT_EQ(path.terminal_ray.direction(), inputs->at(index).primary_ray.direction());
                EXPECT_EQ(parity->cuda_terminal_cones[index], inputs->at(index).primary_cone);
                continue;
            }
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(), expected_depth);
            EXPECT_EQ(path.state.delta_flags(),
                      renderer::PathDeltaFlags::previous_bounce_was_delta);
            EXPECT_NEAR(path.state.eta_scale(), expected_eta_scale, MaximumAbsoluteError);
            EXPECT_NEAR(path.terminal_ray.direction().x, expected_ray.x, TerminalGeometryTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().y, expected_ray.y, TerminalGeometryTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().z, expected_ray.z, TerminalGeometryTolerance);

            const auto hit_parameter = -inputs->at(index).primary_ray.origin().z /
                                       inputs->at(index).primary_ray.direction().z;
            const auto advanced = renderer::advance_ray_cone(
                inputs->at(index).primary_cone, inputs->at(index).primary_ray, hit_parameter);
            ASSERT_TRUE(advanced.has_value()) << advanced.error().message;
            const auto event =
                expected_depth.transmission == 0U
                    ? renderer::ScatteringLobe::specular | renderer::ScatteringLobe::reflection
                    : renderer::ScatteringLobe::specular | renderer::ScatteringLobe::transmission;
            const auto reference_cone = renderer::propagate_ray_cone_scattering(
                *advanced, reference_closure, event, outgoing, expected_ray);
            ASSERT_TRUE(reference_cone.has_value()) << reference_cone.error().message;
            EXPECT_NEAR(parity->cuda_terminal_cones[index].width(), reference_cone->width(),
                        1.0e-6F);
            EXPECT_NEAR(parity->cuda_terminal_cones[index].spread(), reference_cone->spread(),
                        1.0e-6F);
            EXPECT_GT(parity->cuda_terminal_cones[index].width(), 0.0F);
            EXPECT_GT(parity->cuda_terminal_cones[index].spread(), 0.0F);
        }
        EXPECT_EQ(parity->cuda_report.shadow_queries, 0U);
    };

    verify("SpecularMirror", mirror_scene_closure(), renderer::Vector3{.x = 0.6F, .z = 0.8F}, true,
           renderer::PathDepthCounters{.specular = 1U}, 1.0F,
           renderer::Vector3{.x = -0.6F, .z = 0.8F});
    verify("SpecularTransmissionEntry", transmission_scene_closure(),
           renderer::Vector3{.x = 0.6F, .z = 0.8F}, true,
           renderer::PathDepthCounters{.specular = 1U, .transmission = 1U}, 2.25F,
           renderer::Vector3{.x = -0.4F, .z = -std::sqrt(0.84F)});
    verify("SpecularTransmissionExit", transmission_scene_closure(),
           renderer::Vector3{.x = 0.6F, .z = -0.8F}, true,
           renderer::PathDepthCounters{.specular = 1U, .transmission = 1U}, 4.0F / 9.0F,
           renderer::Vector3{.x = -0.9F, .z = std::sqrt(0.19F)});
    verify("SpecularTransmissionTir", transmission_scene_closure(),
           renderer::Vector3{.x = 0.8F, .z = -0.6F}, false, renderer::PathDepthCounters{}, 1.0F,
           {});
}

TEST(CudaTransportAllLobesParityTest, MatchesContinuousAndMixedMeasureMixturesForBothMisRules) {
    ASSERT_TRUE(select_test_device());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;

    for (const auto heuristic : {renderer::MisHeuristic::balance, renderer::MisHeuristic::power}) {
        SCOPED_TRACE(heuristic == renderer::MisHeuristic::balance ? "balance" : "power");
        {
            const auto mixture = continuous_mixture_scene_closure();
            const auto first_probability = mixture.active_component_probabilities().front();
            const auto scene = make_material_scene(mixture);
            ASSERT_TRUE(scene.has_value()) << scene.error().message;
            const auto bands = std::array{
                ComponentSampleBand{.lower = 0.0F, .upper = first_probability * 0.9F, .count = 8U},
                ComponentSampleBand{.lower = first_probability + 0.05F, .upper = 1.0F, .count = 8U},
            };
            const auto inputs = make_material_inputs(*scene, *primary, bands);
            ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
            const auto parity =
                run_material_parity(*scene, *inputs, "ContinuousClosureMixture", heuristic);
            ASSERT_TRUE(parity.has_value()) << parity.error().message;
            expect_parity(
                material_configuration("ContinuousClosureMixture", inputs->size(), heuristic),
                *parity);
            auto first_selected = std::size_t{};
            auto second_selected = std::size_t{};
            for (auto index = std::size_t{}; index < parity->scalar_paths.size(); ++index) {
                const auto component = renderer::SampleStream{inputs->at(index).sample}.sample_1d(
                    dimensions->bsdf_component);
                component < first_probability ? ++first_selected : ++second_selected;
                const auto& path = parity->scalar_paths[index];
                EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
                EXPECT_EQ(path.state.depth_counters(),
                          (renderer::PathDepthCounters{.diffuse = 1U}));
                EXPECT_EQ(path.state.delta_flags(),
                          renderer::PathDeltaFlags::any_non_delta_bounces);
            }
            EXPECT_GT(first_selected, 0U);
            EXPECT_GT(second_selected, 0U);
        }
        {
            const auto mixture = mixed_measure_scene_closure();
            const auto first_probability = mixture.active_component_probabilities().front();
            const auto scene = make_material_scene(mixture);
            ASSERT_TRUE(scene.has_value()) << scene.error().message;
            const auto bands = std::array{
                ComponentSampleBand{.lower = 0.0F, .upper = first_probability * 0.9F, .count = 8U},
                ComponentSampleBand{.lower = first_probability + 0.05F, .upper = 1.0F, .count = 8U},
            };
            const auto inputs = make_material_inputs(*scene, *primary, bands);
            ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
            const auto parity =
                run_material_parity(*scene, *inputs, "MixedMeasureClosureMixture", heuristic);
            ASSERT_TRUE(parity.has_value()) << parity.error().message;
            expect_parity(
                material_configuration("MixedMeasureClosureMixture", inputs->size(), heuristic),
                *parity);
            auto diffuse_selected = std::size_t{};
            auto mirror_selected = std::size_t{};
            for (auto index = std::size_t{}; index < parity->scalar_paths.size(); ++index) {
                const auto component = renderer::SampleStream{inputs->at(index).sample}.sample_1d(
                    dimensions->bsdf_component);
                const auto& path = parity->scalar_paths[index];
                EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
                EXPECT_EQ(path.state.eta_scale(), 1.0F);
                if (component < first_probability) {
                    ++diffuse_selected;
                    EXPECT_EQ(path.state.depth_counters(),
                              (renderer::PathDepthCounters{.diffuse = 1U}));
                    EXPECT_EQ(path.state.delta_flags(),
                              renderer::PathDeltaFlags::any_non_delta_bounces);
                } else {
                    ++mirror_selected;
                    EXPECT_EQ(path.state.depth_counters(),
                              (renderer::PathDepthCounters{.specular = 1U}));
                    EXPECT_EQ(path.state.delta_flags(),
                              renderer::PathDeltaFlags::previous_bounce_was_delta);
                }
            }
            EXPECT_GT(diffuse_selected, 0U);
            EXPECT_GT(mirror_selected, 0U);
        }
    }
}

TEST(CudaTransportAllLobesParityTest,
     FiltersDisabledDiffuseBeforeGlossySelectionNeeAndPdfEvaluation) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_material_scene(diffuse_glossy_mixture_scene_closure());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands = std::array{
        ComponentSampleBand{.lower = 0.0F, .upper = 0.2F, .count = 8U},
        ComponentSampleBand{.lower = 0.8F, .upper = 1.0F, .count = 8U},
    };
    const auto inputs = make_material_inputs(*scene, *primary, bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    constexpr auto limits = renderer::PathDepthLimits{.diffuse = 0U, .glossy = 1U};
    const auto parity =
        run_material_parity_with_limits(*scene, *inputs, "FilteredGlossyMixture", limits);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    auto configuration = material_configuration("FilteredGlossyMixture", inputs->size());
    configuration.depth_limits = limits;
    expect_parity(configuration, *parity);

    auto continued = std::size_t{};
    for (const auto& path : parity->scalar_paths) {
        EXPECT_EQ(path.state.depth_counters().diffuse, 0U);
        EXPECT_EQ(path.state.depth_counters().specular, 0U);
        EXPECT_EQ(path.state.depth_counters().transmission, 0U);
        if (path.termination == renderer::BsdfOnlyPathTermination::escaped_environment) {
            ++continued;
            EXPECT_EQ(path.state.depth_counters().glossy, 1U);
        } else {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::outside_bsdf_support);
            EXPECT_EQ(path.state.depth_counters().glossy, 0U);
        }
    }
    EXPECT_GT(continued, 0U);
    EXPECT_GT(parity->cuda_report.light_samples, 0U);
    EXPECT_GT(parity->cuda_report.shadow_queries, 0U);
}

TEST(CudaTransportAllLobesParityTest,
     ConditionsRoughDielectricOnReflectionWhenTransmissionDepthIsDisabled) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_material_scene(rough_dielectric_scene_closure());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto transmission_biased_band =
        std::array{ComponentSampleBand{.lower = 0.75F, .upper = 1.0F, .count = 32U}};
    const auto inputs = make_material_inputs(*scene, *primary, transmission_biased_band);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    constexpr auto limits = renderer::PathDepthLimits{.glossy = 1U, .transmission = 0U};
    const auto parity =
        run_material_parity_with_limits(*scene, *inputs, "RoughDielectricReflectionOnly", limits);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    auto configuration = material_configuration("RoughDielectricReflectionOnly", inputs->size());
    configuration.depth_limits = limits;
    expect_parity(configuration, *parity);

    auto reflected = std::size_t{};
    for (const auto& path : parity->scalar_paths) {
        EXPECT_EQ(path.state.depth_counters().transmission, 0U);
        EXPECT_EQ(path.state.eta_scale(), 1.0F);
        if (path.termination == renderer::BsdfOnlyPathTermination::escaped_environment) {
            ++reflected;
            EXPECT_EQ(path.state.depth_counters().glossy, 1U);
            EXPECT_GT(path.terminal_ray.direction().z, 0.0F);
        } else {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::outside_bsdf_support);
            EXPECT_EQ(path.state.depth_counters().glossy, 0U);
        }
    }
    EXPECT_GT(reflected, 0U);
}

TEST(CudaTransportAllLobesParityTest, DistinguishesEmptySourceFromEveryClosureBlockedByDepth) {
    ASSERT_TRUE(select_test_device());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 4U}};
    constexpr auto limits = renderer::PathDepthLimits{};

    const auto verify = [&](const std::string_view name, SceneClosureMixture closure,
                            const renderer::BsdfOnlyPathTermination termination,
                            const renderer::ScatteringLobe blocked) {
        SCOPED_TRACE(name);
        const auto scene = make_material_scene(std::move(closure));
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = run_material_parity_with_limits(*scene, *inputs, name, limits);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        auto configuration = material_configuration(name, inputs->size());
        configuration.depth_limits = limits;
        expect_parity(configuration, *parity, false);
        for (const auto& path : parity->scalar_paths) {
            EXPECT_EQ(path.termination, termination);
            EXPECT_EQ(path.blocked_depth_limits, blocked);
            EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
            EXPECT_EQ(path.state.accumulated_radiance(), renderer::TransportSpectrum{});
        }
    };

    verify("EmptyClosureSource", empty_scene_closure(),
           renderer::BsdfOnlyPathTermination::outside_bsdf_support, renderer::ScatteringLobe::none);
    verify("AllClosuresDepthFiltered", require_lambertian_scene_closure(constant_spectrum(0.5F)),
           renderer::BsdfOnlyPathTermination::depth_limit, renderer::ScatteringLobe::diffuse);
}

TEST(CudaTransportAllLobesParityTest,
     RejectsDegenerateSurfaceTangentBeforeEmptyOrDepthLimitedTermination) {
    ASSERT_TRUE(select_test_device());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 1U}};
    constexpr auto limits = renderer::PathDepthLimits{};

    const auto verify = [&](const std::string_view name, SceneClosureMixture closure) {
        SCOPED_TRACE(name);
        const auto scene = make_material_scene(std::move(closure), true);
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        const auto sampler = make_uniform_light_sampler(*scene);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
        auto scene_soa = CudaSceneSoA::upload(**scene);
        ASSERT_TRUE(scene_soa.has_value()) << scene_soa.error().message;
        auto scene_bvh = CudaSceneBvh::build(*scene_soa);
        ASSERT_TRUE(scene_bvh.has_value()) << scene_bvh.error().message;

        const auto cuda = trace_cuda_wavefront_transport(
            *scene_soa, *scene_bvh, *inputs, std::cref(*sampler),
            CudaWavefrontTransportOptions{
                .heuristic = renderer::MisHeuristic::power,
                .depth_limits = limits,
                .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
            });
        ASSERT_FALSE(cuda.has_value());
        EXPECT_EQ(cuda.error().code, core::StatusCode::internal_error);
        EXPECT_NE(cuda.error().message.find("CUDA wavefront shade failed"), std::string::npos);

        const auto scalar_backend = create_analytic_accel_backend(*scene);
        ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
        const auto scalar = trace_scene_mis(
            inputs->front().primary_ray, inputs->front().initial_state,
            renderer::SampleStream{inputs->front().sample}, **scalar_backend, *sampler,
            renderer::MisHeuristic::power, limits, renderer::RussianRoulettePolicy::disabled());
        ASSERT_FALSE(scalar.has_value());

        EXPECT_TRUE(scene_bvh->close().has_value());
        EXPECT_TRUE(scene_soa->close().has_value());
    };

    verify("DegenerateTangentEmptyClosure", empty_scene_closure());
    verify("DegenerateTangentDepthFiltered", rough_diffuse_scene_closure());
}

TEST(CudaTransportAllLobesParityTest, PreservesExactSubnormalLambertProductAfterFilteringGlossy) {
    ASSERT_TRUE(select_test_device());
    constexpr auto high_beta = renderer::TransportScalar{0x1p120F};
    const auto reflectance = std::numeric_limits<renderer::TransportScalar>::denorm_min();
    const auto expected_beta = high_beta * reflectance;
    ASSERT_TRUE(std::isnormal(expected_beta));
    const auto scene = make_material_scene(diffuse_glossy_mixture_scene_closure(reflectance));
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto glossy_source_band =
        std::array{ComponentSampleBand{.lower = 0.8F, .upper = 1.0F, .count = 4U}};
    auto inputs = make_material_inputs(*scene, *primary, glossy_source_band);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto state = renderer::PathState::create(
        constant_spectrum(high_beta), renderer::TransportSpectrum{}, renderer::PathDepthCounters{},
        renderer::TransportScalar{1}, (*scene)->spectral_environment()->wavelengths,
        renderer::PathDeltaFlags::none, renderer::VacuumMedium);
    ASSERT_TRUE(state.has_value()) << state.error().message;
    for (auto& input : *inputs) {
        input.initial_state = *state;
    }
    constexpr auto limits = renderer::PathDepthLimits{.diffuse = 1U, .glossy = 0U};
    const auto parity =
        run_material_parity_with_limits(*scene, *inputs, "FilteredSubnormalLambert", limits);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    auto configuration = material_configuration("FilteredSubnormalLambert", inputs->size());
    configuration.depth_limits = limits;
    expect_parity(configuration, *parity);
    for (const auto& path : parity->scalar_paths) {
        EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
        EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 1U}));
        for (const auto lane : path.state.beta().values) {
            EXPECT_EQ(lane, expected_beta);
        }
    }
}

TEST(CudaTransportAllLobesParityTest, DeltaEmitterHitIsNotDoubleCountedByMis) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_delta_emitter_scene();
    const auto primary = material_primary_ray(renderer::Vector3{.x = 0.6F, .z = 0.8F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 4U}};
    const auto inputs = make_material_inputs(*scene, *primary, bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    auto heuristic_radiance = std::array<renderer::TransportSpectrum, 2U>{};
    auto heuristic_index = std::size_t{};
    for (const auto heuristic : {renderer::MisHeuristic::balance, renderer::MisHeuristic::power}) {
        const auto name = heuristic == renderer::MisHeuristic::balance ? "DeltaEmitterBalance"
                                                                       : "DeltaEmitterPower";
        const auto parity = run_material_parity(*scene, *inputs, name, heuristic);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        expect_parity(material_configuration(name, inputs->size(), heuristic), *parity);
        ASSERT_FALSE(parity->scalar_paths.empty());
        heuristic_radiance[heuristic_index++] =
            parity->scalar_paths.front().state.accumulated_radiance();
        EXPECT_EQ(parity->cuda_report.shadow_queries, 0U);
        for (const auto& path : parity->scalar_paths) {
            EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.specular = 1U}));
            EXPECT_EQ(path.state.delta_flags(),
                      renderer::PathDeltaFlags::previous_bounce_was_delta);
            for (const auto radiance : path.state.accumulated_radiance().values) {
                EXPECT_NEAR(radiance, 3.4F, MaximumAbsoluteError);
            }
        }
    }
    EXPECT_EQ(heuristic_index, heuristic_radiance.size());
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(heuristic_radiance[0][lane], heuristic_radiance[1][lane], MaximumAbsoluteError);
    }
}

TEST(CudaTransportParityTest, RejectsUnsupportedOrMismatchedLightSamplersWithoutFallback) {
    ASSERT_TRUE(select_test_device());
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto extent = renderer::RenderExtent{.width = 1U, .height = 1U};
    const auto inputs = make_inputs(
        extent, 1U, EvaluationSeed, (*scene)->spectral_environment()->wavelengths,
        [extent](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return point_primary_ray(make_planar_ray(index, extent, true));
        });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto scene_soa = CudaSceneSoA::upload(**scene);
    ASSERT_TRUE(scene_soa.has_value()) << scene_soa.error().message;
    auto scene_bvh = CudaSceneBvh::build(*scene_soa);
    ASSERT_TRUE(scene_bvh.has_value()) << scene_bvh.error().message;

    const auto absent =
        trace_cuda_wavefront_transport(*scene_soa, *scene_bvh, *inputs, std::nullopt,
                                       CudaWavefrontTransportOptions{
                                           .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
                                       });
    ASSERT_FALSE(absent.has_value());
    EXPECT_EQ(absent.error().code, core::StatusCode::incompatible);
    EXPECT_NE(absent.error().message.find("requires an explicit light sampler"), std::string::npos);

    const auto powers = std::array{constant_spectrum(1.0F)};
    const auto non_uniform = renderer::LightSampler::create_power_weighted(
        std::span<const renderer::LightSpectrumT<renderer::TransportScalar>>{powers});
    ASSERT_TRUE(non_uniform.has_value()) << non_uniform.error().message;
    const auto unsupported =
        trace_cuda_wavefront_transport(*scene_soa, *scene_bvh, *inputs, std::cref(*non_uniform),
                                       CudaWavefrontTransportOptions{
                                           .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
                                       });
    ASSERT_FALSE(unsupported.has_value());
    EXPECT_EQ(unsupported.error().code, core::StatusCode::unavailable);
    EXPECT_NE(unsupported.error().message.find("not ported"), std::string::npos);
    EXPECT_NE(unsupported.error().message.find("substituted"), std::string::npos);

    const auto valid_sampler = renderer::LightSampler::create_uniform(1U);
    ASSERT_TRUE(valid_sampler.has_value()) << valid_sampler.error().message;
    const auto invalid_heuristic =
        trace_cuda_wavefront_transport(*scene_soa, *scene_bvh, *inputs, std::cref(*valid_sampler),
                                       CudaWavefrontTransportOptions{
                                           .heuristic = static_cast<renderer::MisHeuristic>(0xFFU),
                                           .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
                                       });
    ASSERT_FALSE(invalid_heuristic.has_value());
    EXPECT_EQ(invalid_heuristic.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(invalid_heuristic.error().message.find("balance or power"), std::string::npos);

    const auto wrong_count = renderer::LightSampler::create_uniform(2U);
    ASSERT_TRUE(wrong_count.has_value()) << wrong_count.error().message;
    const auto mismatched =
        trace_cuda_wavefront_transport(*scene_soa, *scene_bvh, *inputs, std::cref(*wrong_count),
                                       CudaWavefrontTransportOptions{
                                           .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
                                       });
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, core::StatusCode::incompatible);
    EXPECT_NE(mismatched.error().message.find("does not match"), std::string::npos);

    EXPECT_TRUE(scene_bvh->close().has_value());
    EXPECT_TRUE(scene_soa->close().has_value());
}

} // namespace
} // namespace blackframe::engine
