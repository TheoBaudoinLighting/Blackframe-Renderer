#pragma once

#include "CornellDiffuseTestScene.hpp"

#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/FilmTile.hpp>
#include <Blackframe/Renderer/NumericConversion.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer::cornell_test {

inline constexpr auto CornellSceneIdentifier = "S02_CornellDiffuse";
inline constexpr auto CornellReferenceSeed = std::uint64_t{0x243F6A8885A308D3ULL};
inline constexpr auto CornellEvaluationSeed = std::uint64_t{0xA4093822299F31D0ULL};
inline constexpr auto CornellMaximumDiffuseDepth = std::uint32_t{3};
inline constexpr auto CornellMaximumWorkerCount = std::uint32_t{64};
inline constexpr auto CornellReferenceBackend = "scalar_ref";
inline constexpr auto CornellReferenceCapabilities =
    "bsdf_only,cie1931,film_float64_compensated,lambert,spectrum4";

struct CornellImageSpecification final {
    const char* variant;
    RenderExtent extent;
    std::uint32_t reference_samples_per_pixel;
    const char* scene_filename;
    const char* reference_filename;

    [[nodiscard]] constexpr bool operator==(const CornellImageSpecification& other) const noexcept {
        if (variant == nullptr || scene_filename == nullptr || reference_filename == nullptr ||
            other.variant == nullptr || other.scene_filename == nullptr ||
            other.reference_filename == nullptr) {
            return false;
        }
        return std::string_view{variant} == other.variant && extent.width == other.extent.width &&
               extent.height == other.extent.height &&
               reference_samples_per_pixel == other.reference_samples_per_pixel &&
               std::string_view{scene_filename} == other.scene_filename &&
               std::string_view{reference_filename} == other.reference_filename;
    }
};

inline constexpr auto Cornell64Specification = CornellImageSpecification{
    .variant = "64x64",
    .extent = {.width = 64, .height = 64},
    .reference_samples_per_pixel = 4096,
    .scene_filename = "S02_CornellDiffuse_64x64.json",
    .reference_filename = "S02_CornellDiffuse_64x64_scalar_ref_4096spp.exr",
};

inline constexpr auto Cornell256Specification = CornellImageSpecification{
    .variant = "256x256",
    .extent = {.width = 256, .height = 256},
    .reference_samples_per_pixel = 1024,
    .scene_filename = "S02_CornellDiffuse_256x256.json",
    .reference_filename = "S02_CornellDiffuse_256x256_scalar_ref_1024spp.exr",
};

[[nodiscard]] inline std::string
cornell_reference_scene_path(const CornellImageSpecification& specification) {
    return "scenes/" + std::string{specification.scene_filename};
}

[[nodiscard]] inline std::string
cornell_reference_options(const CornellImageSpecification& specification) {
    return "spp=" + std::to_string(specification.reference_samples_per_pixel) +
           ";jitter=uniform;diffuse=3;glossy=0;specular=0;transmission=0;volume=0;rr=disabled;"
           "wavelength_packet=fixed-v1";
}

template <SpectrumScalar Scalar>
using CornellFilmFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, Film, ReferenceFilm>;

template <SpectrumScalar Scalar>
using CornellFilmTileFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, FilmTile, ReferenceFilmTile>;

template <SpectrumScalar Scalar>
using CornellColorFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, LinearRGB, ReferenceLinearRGB>;

template <SpectrumScalar Scalar> struct CornellImageScene final {
    WavelengthsFor<Scalar> wavelengths;
    std::vector<SurfaceFor<Scalar>> surfaces;
    BsdfOnlyEnvironmentT<Scalar> environment;
    PinholeCameraT<Scalar> camera;
};

namespace image_renderer_detail {

[[nodiscard]] inline core::Error image_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] constexpr bool
known_specification(const CornellImageSpecification& specification) noexcept {
    return specification == Cornell64Specification || specification == Cornell256Specification;
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<CornellColorFor<Scalar>>
render_sample(const CornellImageScene<Scalar>& scene, const std::uint32_t pixel_x,
              const std::uint32_t pixel_y, const std::uint64_t sample_index,
              const std::uint64_t seed) {
    const auto index = PixelSampleIndex{
        .pixel_x = pixel_x,
        .pixel_y = pixel_y,
        .sample_index = sample_index,
        .seed = seed,
    };
    const auto ray = scene.camera.generate_primary_ray(index, PixelJitterMode::uniform, Scalar{0});
    if (!ray.has_value()) {
        return std::unexpected(ray.error());
    }
    const auto state = PathStateT<Scalar>::create_initial(scene.wavelengths, VacuumMedium);
    if (!state.has_value()) {
        return std::unexpected(state.error());
    }
    const auto stream =
        IndependentSamplerT<Scalar>{seed}.make_stream(pixel_x, pixel_y, sample_index);
    const auto traced =
        trace_bsdf_only(*ray, *state, stream, std::span<const SurfaceFor<Scalar>>{scene.surfaces},
                        scene.environment, PathDepthLimits{.diffuse = CornellMaximumDiffuseDepth},
                        RussianRoulettePolicyT<Scalar>::disabled());
    if (!traced.has_value()) {
        return std::unexpected(traced.error());
    }
    const auto xyz =
        cie_1931_spectrum_to_xyz(traced->state.accumulated_radiance(), scene.wavelengths);
    if (!xyz.has_value()) {
        return std::unexpected(xyz.error());
    }
    const auto rgb = xyz_to_linear_rgb(*xyz);
    if (!rgb.has_value()) {
        return std::unexpected(rgb.error());
    }
    return *rgb;
}

} // namespace image_renderer_detail

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<CornellImageScene<Scalar>>
make_cornell_image_scene(const CornellImageSpecification& specification) {
    if (!image_renderer_detail::known_specification(specification)) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::invalid_argument,
            "The Cornell image renderer received an unsupported closed fixture specification."));
    }

    const auto wavelengths = fixed_wavelengths<Scalar>();
    auto surfaces = make_cornell_surfaces(wavelengths);
    if (!surfaces.has_value()) {
        return std::unexpected(surfaces.error());
    }
    const auto black_environment = ConstantEnvironmentT<Scalar>::create(SpectrumFor<Scalar>{});
    if (!black_environment.has_value()) {
        return std::unexpected(black_environment.error());
    }
    const auto frame = OrthonormalFrameT<Scalar>::from_normal_and_tangent(
        Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}});
    if (!frame.has_value()) {
        return std::unexpected(frame.error());
    }
    const auto camera = PinholeCameraT<Scalar>::create(
        Point3T<Scalar>{.z = Scalar{3}}, *frame, specification.extent,
        std::numbers::pi_v<Scalar> / Scalar{2}, Scalar{0}, std::numeric_limits<Scalar>::infinity(),
        AllRayVisibility, VacuumMedium);
    if (!camera.has_value()) {
        return std::unexpected(camera.error());
    }

    return CornellImageScene<Scalar>{
        .wavelengths = wavelengths,
        .surfaces = std::move(*surfaces),
        .environment = BsdfOnlyEnvironmentT<Scalar>{*black_environment, wavelengths},
        .camera = *camera,
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<CornellFilmFor<Scalar>>
render_cornell_image(const CornellImageSpecification& specification,
                     const std::uint32_t samples_per_pixel, const std::uint64_t seed,
                     const std::uint32_t worker_count) {
    if (!image_renderer_detail::known_specification(specification)) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::invalid_argument,
            "The Cornell image renderer received an unsupported closed fixture specification."));
    }
    if (samples_per_pixel == 0) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::invalid_argument,
            "Cornell image rendering requires at least one sample per pixel."));
    }
    if (worker_count == 0 || worker_count > CornellMaximumWorkerCount) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::invalid_argument,
            "Cornell image rendering requires between one and 64 explicit workers."));
    }

    const auto scene = make_cornell_image_scene<Scalar>(specification);
    if (!scene.has_value()) {
        return std::unexpected(scene.error());
    }

    try {
        const auto effective_worker_count = std::min(worker_count, specification.extent.height);
        auto tiles = std::vector<CornellFilmTileFor<Scalar>>{};
        tiles.reserve(effective_worker_count);
        for (auto worker_index = std::uint32_t{0}; worker_index < effective_worker_count;
             ++worker_index) {
            const auto minimum_y =
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(specification.extent.height) *
                                           worker_index / effective_worker_count);
            const auto maximum_y =
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(specification.extent.height) *
                                           (worker_index + 1U) / effective_worker_count);
            auto tile = CornellFilmTileFor<Scalar>::create(
                specification.extent, FilmCrop{
                                          .minimum_y = minimum_y,
                                          .maximum_x = specification.extent.width,
                                          .maximum_y = maximum_y,
                                      });
            if (!tile.has_value()) {
                return std::unexpected(tile.error());
            }
            tiles.push_back(std::move(*tile));
        }

        auto worker_errors = std::vector<std::optional<core::Error>>(effective_worker_count);
        {
            auto workers = std::vector<std::jthread>{};
            workers.reserve(effective_worker_count);
            for (auto worker_index = std::uint32_t{0}; worker_index < effective_worker_count;
                 ++worker_index) {
                workers.emplace_back([&, worker_index] {
                    try {
                        auto& tile = tiles[worker_index];
                        const auto crop = tile.crop();
                        for (auto pixel_y = crop.minimum_y; pixel_y < crop.maximum_y; ++pixel_y) {
                            for (auto pixel_x = std::uint32_t{0};
                                 pixel_x < specification.extent.width; ++pixel_x) {
                                for (auto sample_index = std::uint64_t{0};
                                     sample_index < samples_per_pixel; ++sample_index) {
                                    const auto sample = image_renderer_detail::render_sample(
                                        *scene, pixel_x, pixel_y, sample_index, seed);
                                    if (!sample.has_value()) {
                                        worker_errors[worker_index] = sample.error();
                                        return;
                                    }
                                    const auto accumulation_status =
                                        tile.add_sample(pixel_x, pixel_y, *sample, Scalar{1});
                                    if (!accumulation_status.has_value()) {
                                        worker_errors[worker_index] = accumulation_status.error();
                                        return;
                                    }
                                }
                            }
                        }
                    } catch (const std::bad_alloc&) {
                        worker_errors[worker_index] = image_renderer_detail::image_error(
                            core::StatusCode::resource_exhausted,
                            "A Cornell image worker exhausted host memory.");
                    } catch (const std::exception& error) {
                        worker_errors[worker_index] = image_renderer_detail::image_error(
                            core::StatusCode::internal_error,
                            "A Cornell image worker failed unexpectedly: " +
                                std::string{error.what()});
                    } catch (...) {
                        worker_errors[worker_index] = image_renderer_detail::image_error(
                            core::StatusCode::internal_error,
                            "A Cornell image worker failed with an unknown exception.");
                    }
                });
            }
        }
        for (const auto& error : worker_errors) {
            if (error.has_value()) {
                return std::unexpected(*error);
            }
        }

        auto film = CornellFilmFor<Scalar>::create(specification.extent);
        if (!film.has_value()) {
            return std::unexpected(film.error());
        }
        const auto merge_status =
            film->merge_tiles(std::span<const CornellFilmTileFor<Scalar>>{tiles});
        if (!merge_status.has_value()) {
            return std::unexpected(merge_status.error());
        }
        return std::move(*film);
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            image_renderer_detail::image_error(core::StatusCode::resource_exhausted,
                                               "Cornell image rendering exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::resource_exhausted,
            "Cornell image rendering exceeded a host container limit."));
    } catch (const std::system_error& error) {
        return std::unexpected(image_renderer_detail::image_error(
            core::StatusCode::platform_error,
            "Cornell image worker creation failed: " + std::string{error.what()}));
    }
}

[[nodiscard]] inline core::Result<Film> quantize_cornell_reference(const ReferenceFilm& reference) {
    auto result = Film::create(reference.extent(), reference.crop());
    if (!result.has_value()) {
        return std::unexpected(result.error());
    }
    const auto crop = reference.crop();
    for (auto pixel_y = crop.minimum_y; pixel_y < crop.maximum_y; ++pixel_y) {
        for (auto pixel_x = crop.minimum_x; pixel_x < crop.maximum_x; ++pixel_x) {
            const auto resolved = reference.resolved_pixel(pixel_x, pixel_y);
            if (!resolved.has_value()) {
                return std::unexpected(resolved.error());
            }
            const auto red = to_transport_scalar(resolved->red);
            const auto green = to_transport_scalar(resolved->green);
            const auto blue = to_transport_scalar(resolved->blue);
            if (!red.has_value()) {
                return std::unexpected(red.error());
            }
            if (!green.has_value()) {
                return std::unexpected(green.error());
            }
            if (!blue.has_value()) {
                return std::unexpected(blue.error());
            }
            const auto status = result->add_sample(
                pixel_x, pixel_y, LinearRGB{.red = *red, .green = *green, .blue = *blue}, 1.0F);
            if (!status.has_value()) {
                return std::unexpected(status.error());
            }
        }
    }
    return std::move(*result);
}

} // namespace blackframe::renderer::cornell_test
