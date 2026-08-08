#pragma once

#include "CPU/Embree/CornellWavefrontScene.hpp"

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/RayCone.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace blackframe::engine::surface_map_sphere_fixture {

inline constexpr auto NormalTextureId = renderer::TextureId{.value = 901U};
inline constexpr auto BumpTextureId = renderer::TextureId{.value = 902U};
inline constexpr auto ObjectId = renderer::ObjectId{.value = 911U};
inline constexpr auto GeometryId = renderer::GeometryId{.value = 912U};
inline constexpr auto MaterialId = renderer::MaterialId{.value = 913U};
inline constexpr auto InstanceId = renderer::InstanceId{.value = 914U};
inline constexpr auto NormalMapWidth = std::uint32_t{4U};
inline constexpr auto NormalMapHeight = std::uint32_t{4U};
inline constexpr auto BumpMapWidth = std::uint32_t{16U};
inline constexpr auto BumpMapHeight = std::uint32_t{16U};

[[nodiscard]] inline std::vector<renderer::TransportScalar> normal_map_pixels() {
    auto pixels = std::vector<renderer::TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(NormalMapWidth) * NormalMapHeight * 3U);
    for (auto index = std::uint32_t{}; index < NormalMapWidth * NormalMapHeight; ++index) {
        // Decodes to the unit tangent-space direction (0.3, 0.2, sqrt(0.87)); the non-zero +V
        // component makes the shared CPU/CUDA sphere exercise tangent handedness.
        pixels.insert(pixels.end(), {0.65F, 0.60F, 0.96636895F});
    }
    return pixels;
}

[[nodiscard]] inline std::vector<renderer::TransportScalar> bump_map_pixels() {
    auto pixels = std::vector<renderer::TransportScalar>{};
    pixels.reserve(static_cast<std::size_t>(BumpMapWidth) * BumpMapHeight);
    for (auto y = std::uint32_t{}; y < BumpMapHeight; ++y) {
        for (auto x = std::uint32_t{}; x < BumpMapWidth; ++x) {
            // A bounded affine U+V ramp keeps both filtered partial derivatives non-zero.
            const auto numerator = static_cast<float>(x) + 0.5F * static_cast<float>(y);
            const auto denominator = static_cast<float>(BumpMapWidth - 1U) +
                                     0.5F * static_cast<float>(BumpMapHeight - 1U);
            pixels.push_back(numerator / denominator);
        }
    }
    return pixels;
}

[[nodiscard]] inline core::Result<FrameSceneHandle>
make_scene(renderer::HostImageMipChainHandle normal_map, renderer::HostImageMipChainHandle bump_map,
           const renderer::TangentSpaceNormalYConvention y_convention =
               renderer::TangentSpaceNormalYConvention::positive_v) {
    const auto sphere = cornell_wavefront_test::detail::make_unit_sphere_mesh();
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!sphere) {
        return std::unexpected(sphere.error());
    }
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    auto reflectance = renderer::TransportSpectrum{};
    reflectance.values.fill(0.5F);
    const auto closure = SceneClosureMixture::create_lambertian(reflectance);
    if (!closure) {
        return std::unexpected(closure.error());
    }

    auto description = FrameSceneDescription{
        .objects = {SceneObject{.id = ObjectId}},
        .geometries = {SceneGeometry{.id = GeometryId, .mesh = *sphere}},
        .materials =
            {
                SceneMaterial{
                    .id = MaterialId,
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = *closure,
                            .emitted_radiance = {},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = InstanceId,
                    .parent = std::nullopt,
                    .object = ObjectId,
                    .geometry = GeometryId,
                    .material = MaterialId,
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                ScenePointLight{
                    .position = {.x = -2.0F, .y = 3.0F, .z = -3.0F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = reflectance,
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {},
            },
    };
    if (normal_map) {
        description.host_image_textures.push_back(
            SceneHostImageTexture{.id = NormalTextureId, .image = std::move(normal_map)});
        description.materials.front().spectral->normal_map = SceneNormalMapBinding{
            .texture = NormalTextureId,
            .red_channel = 0U,
            .green_channel = 1U,
            .blue_channel = 2U,
            .y_convention = y_convention,
            .u_wrap = renderer::TextureWrapMode::repeat,
            .v_wrap = renderer::TextureWrapMode::repeat,
            .ewa_limits = {},
        };
    }
    if (bump_map) {
        description.host_image_textures.push_back(
            SceneHostImageTexture{.id = BumpTextureId, .image = std::move(bump_map)});
        description.materials.front().spectral->bump_map = SceneBumpMapBinding{
            .texture = BumpTextureId,
            .channel = 0U,
            .scale = 0.2F,
            .u_wrap = renderer::TextureWrapMode::clamp,
            .v_wrap = renderer::TextureWrapMode::clamp,
            .ewa_limits = {},
        };
    }
    return FrameScene::create(std::move(description));
}

[[nodiscard]] inline core::Result<renderer::Ray> ray() {
    return renderer::Ray::create(renderer::Point3{.x = 0.2F, .z = -3.0F},
                                 renderer::Vector3{.z = 1.0F}, 0.0F,
                                 std::numeric_limits<renderer::TransportScalar>::infinity(), 0.25F,
                                 renderer::AllRayVisibility, renderer::VacuumMedium);
}

[[nodiscard]] inline core::Result<renderer::RayDifferential> differential() {
    const auto central = ray();
    if (!central) {
        return std::unexpected(central.error());
    }
    return renderer::RayDifferential::create(
        *central, renderer::Point3{.x = 0.21F, .z = -3.0F}, central->direction(),
        renderer::Point3{.x = 0.2F, .y = 0.01F, .z = -3.0F}, central->direction());
}

[[nodiscard]] inline core::Result<renderer::RayCone> cone() {
    return renderer::RayCone::create(0.01F, 0.0F);
}

} // namespace blackframe::engine::surface_map_sphere_fixture
