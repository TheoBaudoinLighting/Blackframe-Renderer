#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/MatrixTypes.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine::cornell_wavefront_test {
namespace detail {

inline constexpr auto SphereLongitudeSegments = std::uint32_t{64U};
inline constexpr auto SphereLatitudeSegments = std::uint32_t{32U};

[[nodiscard]] inline core::Error cornell_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] inline renderer::TransportSpectrum constant_spectrum(const float value) {
    auto result = renderer::TransportSpectrum{};
    result.values.fill(value);
    return result;
}

// These lane values map to neutral scene-linear RGB at the fixed wavelength
// packet. Neutrality is established in spectral transport, never as a display
// correction after rendering.
[[nodiscard]] inline renderer::TransportSpectrum neutral_emission(const float scale) {
    return renderer::TransportSpectrum{
        .values =
            {
                7.28694487F * scale,
                0.952785134F * scale,
                1.53811407F * scale,
                0.0F,
            },
    };
}

[[nodiscard]] inline renderer::TransportSpectrum red_wall_spectrum() {
    return renderer::TransportSpectrum{
        .values = {0.05950854F, 0.10644369F, 0.42375090F, 0.10F},
    };
}

[[nodiscard]] inline renderer::TransportSpectrum green_wall_spectrum() {
    return renderer::TransportSpectrum{
        .values = {0.11711656F, 0.38914749F, 0.26961990F, 0.10F},
    };
}

[[nodiscard]] inline core::Result<renderer::SampledWavelengths> cornell_wavelengths() {
    return renderer::sample_uniform_visible_wavelengths(0.1F);
}

[[nodiscard]] inline renderer::Matrix4 sphere_transform(const renderer::Point3 center,
                                                        const float radius) {
    auto matrix = renderer::identity_matrix<renderer::TransportScalar>();
    matrix(0, 0) = radius;
    matrix(1, 1) = radius;
    matrix(2, 2) = radius;
    matrix(0, 3) = center.x;
    matrix(1, 3) = center.y;
    matrix(2, 3) = center.z;
    return matrix;
}

[[nodiscard]] inline core::Result<std::shared_ptr<const TriangleMesh>>
make_quad(const std::array<renderer::Point3, 4U>& positions, const renderer::Normal3 normal) {
    auto mesh =
        TriangleMesh::create(std::vector<renderer::Point3>{positions.begin(), positions.end()},
                             std::vector<renderer::Normal3>(positions.size(), normal),
                             {
                                 renderer::Point2{},
                                 renderer::Point2{.x = 1.0F},
                                 renderer::Point2{.x = 1.0F, .y = 1.0F},
                                 renderer::Point2{.y = 1.0F},
                             },
                             {
                                 TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
                                 TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
                             });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] inline core::Result<std::shared_ptr<const TriangleMesh>> make_unit_sphere_mesh() {
    constexpr auto ring_stride = SphereLongitudeSegments + 1U;
    auto positions = std::vector<renderer::Point3>{};
    auto normals = std::vector<renderer::Normal3>{};
    auto texture_coordinates = std::vector<renderer::Point2>{};
    auto triangles = std::vector<TriangleVertexIndices>{};

    const auto vertex_count =
        std::size_t{2U} + static_cast<std::size_t>(SphereLatitudeSegments - 1U) * ring_stride;
    const auto triangle_count = static_cast<std::size_t>(SphereLongitudeSegments) *
                                static_cast<std::size_t>(2U * (SphereLatitudeSegments - 1U));
    positions.reserve(vertex_count);
    normals.reserve(vertex_count);
    texture_coordinates.reserve(vertex_count);
    triangles.reserve(triangle_count);

    positions.push_back(renderer::Point3{.y = 1.0F});
    normals.push_back(renderer::Normal3{.y = 1.0F});
    texture_coordinates.push_back(renderer::Point2{.x = 0.5F});

    for (auto latitude = std::uint32_t{1U}; latitude < SphereLatitudeSegments; ++latitude) {
        const auto v = static_cast<float>(latitude) / static_cast<float>(SphereLatitudeSegments);
        const auto theta = std::numbers::pi_v<float> * v;
        const auto sin_theta = std::sin(theta);
        const auto cos_theta = std::cos(theta);
        for (auto longitude = std::uint32_t{}; longitude <= SphereLongitudeSegments; ++longitude) {
            const auto u =
                static_cast<float>(longitude) / static_cast<float>(SphereLongitudeSegments);
            const auto phi = 2.0F * std::numbers::pi_v<float> * u;
            const auto x = sin_theta * std::cos(phi);
            const auto z = sin_theta * std::sin(phi);
            positions.push_back(renderer::Point3{.x = x, .y = cos_theta, .z = z});
            normals.push_back(renderer::Normal3{.x = x, .y = cos_theta, .z = z});
            texture_coordinates.push_back(renderer::Point2{.x = u, .y = v});
        }
    }

    const auto bottom_index = static_cast<std::uint32_t>(positions.size());
    positions.push_back(renderer::Point3{.y = -1.0F});
    normals.push_back(renderer::Normal3{.y = -1.0F});
    texture_coordinates.push_back(renderer::Point2{.x = 0.5F, .y = 1.0F});

    const auto first_ring = std::uint32_t{1U};
    for (auto longitude = std::uint32_t{}; longitude < SphereLongitudeSegments; ++longitude) {
        triangles.push_back(TriangleVertexIndices{
            .vertices = {0U, first_ring + longitude + 1U, first_ring + longitude},
        });
    }

    for (auto latitude = std::uint32_t{1U}; latitude + 1U < SphereLatitudeSegments; ++latitude) {
        const auto upper_ring =
            std::uint32_t{1U} + (latitude - 1U) * static_cast<std::uint32_t>(ring_stride);
        const auto lower_ring = upper_ring + static_cast<std::uint32_t>(ring_stride);
        for (auto longitude = std::uint32_t{}; longitude < SphereLongitudeSegments; ++longitude) {
            const auto upper = upper_ring + longitude;
            const auto lower = lower_ring + longitude;
            triangles.push_back(TriangleVertexIndices{
                .vertices = {upper, lower + 1U, lower},
            });
            triangles.push_back(TriangleVertexIndices{
                .vertices = {upper, upper + 1U, lower + 1U},
            });
        }
    }

    const auto last_ring = bottom_index - ring_stride;
    for (auto longitude = std::uint32_t{}; longitude < SphereLongitudeSegments; ++longitude) {
        triangles.push_back(TriangleVertexIndices{
            .vertices = {bottom_index, last_ring + longitude, last_ring + longitude + 1U},
        });
    }

    auto mesh = TriangleMesh::create(std::move(positions), std::move(normals),
                                     std::move(texture_coordinates), std::move(triangles));
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

} // namespace detail

[[nodiscard]] inline core::Result<FrameSceneHandle> make_cornell_scene() {
    auto floor = detail::make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = -1.0F, .y = -1.0F, .z = 2.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F, .z = 2.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F},
        },
        renderer::Normal3{.y = 1.0F});
    auto ceiling = detail::make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = 1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F, .z = 2.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F, .z = 2.0F},
        },
        renderer::Normal3{.y = -1.0F});
    auto back = detail::make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F},
        },
        renderer::Normal3{.z = 1.0F});
    auto left = detail::make_quad(
        {
            renderer::Point3{.x = -1.0F, .y = -1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F},
            renderer::Point3{.x = -1.0F, .y = 1.0F, .z = 2.0F},
            renderer::Point3{.x = -1.0F, .y = -1.0F, .z = 2.0F},
        },
        renderer::Normal3{.x = 1.0F});
    auto right = detail::make_quad(
        {
            renderer::Point3{.x = 1.0F, .y = -1.0F},
            renderer::Point3{.x = 1.0F, .y = -1.0F, .z = 2.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F, .z = 2.0F},
            renderer::Point3{.x = 1.0F, .y = 1.0F},
        },
        renderer::Normal3{.x = -1.0F});
    auto emitter = detail::make_quad(
        {
            renderer::Point3{.x = -0.32F, .y = 0.985F, .z = 0.62F},
            renderer::Point3{.x = 0.32F, .y = 0.985F, .z = 0.62F},
            renderer::Point3{.x = 0.32F, .y = 0.985F, .z = 1.18F},
            renderer::Point3{.x = -0.32F, .y = 0.985F, .z = 1.18F},
        },
        renderer::Normal3{.y = -1.0F});
    auto sphere = detail::make_unit_sphere_mesh();

    const auto meshes = std::array{&floor, &ceiling, &back, &left, &right, &emitter, &sphere};
    for (const auto* mesh : meshes) {
        if (!mesh->has_value()) {
            return std::unexpected(mesh->error());
        }
    }

    const auto wavelengths = detail::cornell_wavelengths();
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    return FrameScene::create(
        FrameSceneDescription{
            .objects =
                {
                    SceneObject{.id = {.value = 1U}},
                    SceneObject{.id = {.value = 2U}},
                    SceneObject{.id = {.value = 3U}},
                    SceneObject{.id = {.value = 4U}},
                    SceneObject{.id = {.value = 5U}},
                    SceneObject{.id = {.value = 6U}},
                    SceneObject{.id = {.value = 7U}},
                },
            .geometries =
                {
                    SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*floor)},
                    SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*ceiling)},
                    SceneGeometry{.id = {.value = 13U}, .mesh = std::move(*back)},
                    SceneGeometry{.id = {.value = 14U}, .mesh = std::move(*left)},
                    SceneGeometry{.id = {.value = 15U}, .mesh = std::move(*right)},
                    SceneGeometry{.id = {.value = 16U}, .mesh = std::move(*emitter)},
                    SceneGeometry{.id = {.value = 17U}, .mesh = std::move(*sphere)},
                },
            .materials =
                {
                    SceneMaterial{
                        .id = {.value = 21U},
                        .spectral =
                            SceneSpectralMaterial{
                                .wavelengths = *wavelengths,
                                .reflectance = detail::constant_spectrum(0.72F),
                                .emitted_radiance = {},
                            },
                    },
                    SceneMaterial{
                        .id = {.value = 22U},
                        .spectral =
                            SceneSpectralMaterial{
                                .wavelengths = *wavelengths,
                                .reflectance = detail::red_wall_spectrum(),
                                .emitted_radiance = {},
                            },
                    },
                    SceneMaterial{
                        .id = {.value = 23U},
                        .spectral =
                            SceneSpectralMaterial{
                                .wavelengths = *wavelengths,
                                .reflectance = detail::green_wall_spectrum(),
                                .emitted_radiance = {},
                            },
                    },
                    SceneMaterial{
                        .id = {.value = 24U},
                        .spectral =
                            SceneSpectralMaterial{
                                .wavelengths = *wavelengths,
                                .reflectance = {},
                                .emitted_radiance = detail::neutral_emission(6.0F),
                            },
                    },
                },
            .instances =
                {
                    SceneInstance{.id = {.value = 31U},
                                  .parent = std::nullopt,
                                  .object = {.value = 1U},
                                  .geometry = {.value = 11U},
                                  .material = {.value = 21U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 32U},
                                  .parent = std::nullopt,
                                  .object = {.value = 2U},
                                  .geometry = {.value = 12U},
                                  .material = {.value = 21U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 33U},
                                  .parent = std::nullopt,
                                  .object = {.value = 3U},
                                  .geometry = {.value = 13U},
                                  .material = {.value = 21U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 34U},
                                  .parent = std::nullopt,
                                  .object = {.value = 4U},
                                  .geometry = {.value = 14U},
                                  .material = {.value = 22U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 35U},
                                  .parent = std::nullopt,
                                  .object = {.value = 5U},
                                  .geometry = {.value = 15U},
                                  .material = {.value = 23U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 36U},
                                  .parent = std::nullopt,
                                  .object = {.value = 6U},
                                  .geometry = {.value = 16U},
                                  .material = {.value = 24U},
                                  .local_to_parent =
                                      renderer::identity_matrix<renderer::TransportScalar>()},
                    SceneInstance{.id = {.value = 37U},
                                  .parent = std::nullopt,
                                  .object = {.value = 7U},
                                  .geometry = {.value = 17U},
                                  .material = {.value = 21U},
                                  .local_to_parent =
                                      detail::sphere_transform(
                                          renderer::Point3{.x = -0.40F, .y = -0.57F, .z = 1.22F},
                                          0.43F)},
                    SceneInstance{.id = {.value = 38U},
                                  .parent = std::nullopt,
                                  .object = {.value = 7U},
                                  .geometry = {.value = 17U},
                                  .material = {.value = 21U},
                                  .local_to_parent =
                                      detail::sphere_transform(
                                          renderer::Point3{.x = 0.39F, .y = -0.50F, .z = 0.70F},
                                          0.50F)},
                },
            .spectral_environment =
                SceneSpectralEnvironment{
                    .wavelengths = *wavelengths,
                    .radiance = {},
                },
        });
}

[[nodiscard]] inline core::Result<renderer::OrthonormalFrame> make_camera_frame() {
    constexpr auto origin = renderer::Point3{.z = 4.0F};
    constexpr auto target = renderer::Point3{.y = -0.05F, .z = 0.92F};
    const auto forward_seed = target - origin;
    const auto forward_length = std::sqrt(renderer::length_squared(forward_seed));
    if (!std::isfinite(forward_length) || !(forward_length > 0.0F)) {
        return std::unexpected(detail::cornell_error("The Cornell camera direction is invalid."));
    }
    const auto forward = forward_seed / forward_length;
    return renderer::OrthonormalFrame::from_normal_and_tangent(
        renderer::Normal3{.x = -forward.x, .y = -forward.y, .z = -forward.z},
        renderer::Vector3{.x = 1.0F});
}

[[nodiscard]] inline core::Result<renderer::PinholeCamera>
make_camera(const renderer::RenderExtent extent) {
    const auto frame = make_camera_frame();
    if (!frame) {
        return std::unexpected(frame.error());
    }
    return renderer::PinholeCamera::create(
        renderer::Point3{.z = 4.0F}, *frame, extent, 0.70F, 0.0F,
        std::numeric_limits<renderer::TransportScalar>::infinity(), renderer::AllRayVisibility,
        renderer::VacuumMedium);
}

} // namespace blackframe::engine::cornell_wavefront_test
