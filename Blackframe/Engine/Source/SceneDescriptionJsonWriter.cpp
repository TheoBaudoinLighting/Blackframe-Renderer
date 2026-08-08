#include "SceneJsonSyntax.hpp"

#include <Blackframe/Engine/SceneDescriptionJson.hpp>
#include <Blackframe/Renderer/HostImage.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace blackframe::engine {
namespace {

using scene_json_syntax::Writer;

template <typename> inline constexpr bool AlwaysFalse = false;

[[nodiscard]] core::Error writer_error(const core::StatusCode code,
                                       const std::string_view message) {
    return core::Error{.code = code, .message = std::string{message}};
}

[[nodiscard]] core::Result<std::string> path_utf8(const std::filesystem::path& path);

[[nodiscard]] core::Status add_to_budget(std::uint64_t& current, const std::uint64_t amount,
                                         const std::uint64_t maximum,
                                         const std::string_view diagnostic) {
    if (amount > maximum - current) {
        return std::unexpected(writer_error(core::StatusCode::resource_exhausted, diagnostic));
    }
    current += amount;
    return {};
}

template <typename Record>
[[nodiscard]] core::Status validate_registry(const std::span<const Record> records,
                                             const SceneDescriptionJsonLimits limits,
                                             std::uint64_t& decoded_bytes) {
    if (records.size() > limits.maximum_records_per_registry) {
        return std::unexpected(
            writer_error(core::StatusCode::resource_exhausted,
                         "A scene JSON registry exceeds the configured record-count limit."));
    }
    if (records.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(Record)) {
        return std::unexpected(
            writer_error(core::StatusCode::resource_exhausted,
                         "A scene JSON registry decoded byte count is not representable."));
    }
    return add_to_budget(decoded_bytes, static_cast<std::uint64_t>(records.size()) * sizeof(Record),
                         limits.maximum_decoded_bytes,
                         "A scene JSON description exceeds the configured decoded-byte limit.");
}

[[nodiscard]] core::Status preflight_description(const SceneDescription& description,
                                                 const SceneDescriptionJsonLimits limits) {
    if (auto status = scene_json_syntax::validate_limits(limits); !status) {
        return status;
    }

    auto decoded_bytes = std::uint64_t{};
    if (auto status = validate_registry(description.films(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.cameras(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.render_options(), limits, decoded_bytes);
        !status) {
        return status;
    }
    if (auto status = validate_registry(description.constant_textures(), limits, decoded_bytes);
        !status) {
        return status;
    }
    if (auto status = validate_registry(description.host_image_textures(), limits, decoded_bytes);
        !status) {
        return status;
    }
    if (auto status = validate_registry(description.objects(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.geometries(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.materials(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.instances(), limits, decoded_bytes); !status) {
        return status;
    }
    if (auto status = validate_registry(description.lights(), limits, decoded_bytes); !status) {
        return status;
    }

    auto mesh_vertices = std::uint64_t{};
    auto mesh_triangles = std::uint64_t{};
    for (const auto& geometry : description.geometries()) {
        if (!geometry.mesh) {
            return std::unexpected(writer_error(core::StatusCode::invalid_argument,
                                                "A scene JSON geometry has no mesh snapshot."));
        }
        if (auto status = add_to_budget(mesh_vertices, geometry.mesh->positions().size(),
                                        limits.maximum_mesh_vertices,
                                        "Scene JSON meshes exceed the configured vertex limit.");
            !status) {
            return status;
        }
        if (auto status = add_to_budget(mesh_triangles, geometry.mesh->triangles().size(),
                                        limits.maximum_mesh_triangles,
                                        "Scene JSON meshes exceed the configured triangle limit.");
            !status) {
            return status;
        }
        if (auto status = add_to_budget(
                decoded_bytes, geometry.mesh->memory_report().payload_bytes,
                limits.maximum_decoded_bytes,
                "A scene JSON description exceeds the configured decoded-byte limit.");
            !status) {
            return status;
        }
    }

    auto image_scalar_values = std::uint64_t{};
    for (const auto& texture : description.host_image_textures()) {
        if (!texture.image || !texture.image->source_image()) {
            return std::unexpected(writer_error(
                core::StatusCode::invalid_argument,
                "A scene JSON host-image texture has no complete immutable snapshot."));
        }
        const auto source = texture.image->source_image();
        const auto source_path = path_utf8(source->source_path());
        if (!source_path) {
            return std::unexpected(source_path.error());
        }
        if (auto status = add_to_budget(
                image_scalar_values, source->pixels().size(), limits.maximum_image_scalar_values,
                "Scene JSON images exceed the configured scalar-value limit.");
            !status) {
            return status;
        }
        if (auto status = add_to_budget(
                decoded_bytes, texture.image->total_pixel_bytes(), limits.maximum_decoded_bytes,
                "A scene JSON description exceeds the configured decoded-byte limit.");
            !status) {
            return status;
        }
        auto metadata_bytes_per_level = std::uint64_t{};
        if (auto status = add_to_budget(metadata_bytes_per_level, source_path->size(),
                                        std::numeric_limits<std::uint64_t>::max(),
                                        "Scene JSON image metadata size is not representable.");
            !status) {
            return status;
        }
        if (auto status = add_to_budget(metadata_bytes_per_level, source->format_name().size(),
                                        std::numeric_limits<std::uint64_t>::max(),
                                        "Scene JSON image metadata size is not representable.");
            !status) {
            return status;
        }
        for (const auto& channel_name : source->channel_names()) {
            if (auto status = add_to_budget(metadata_bytes_per_level,
                                            sizeof(std::string) + channel_name.size(),
                                            std::numeric_limits<std::uint64_t>::max(),
                                            "Scene JSON image metadata size is not representable.");
                !status) {
                return status;
            }
        }
        for (auto level = std::uint32_t{}; level < texture.image->level_count(); ++level) {
            if (auto status = add_to_budget(
                    decoded_bytes, metadata_bytes_per_level, limits.maximum_decoded_bytes,
                    "A scene JSON description exceeds the configured decoded-byte limit.");
                !status) {
                return status;
            }
        }
    }
    return {};
}

[[nodiscard]] core::Status write_key(Writer& writer, const std::string_view key) {
    return writer.key(key);
}

[[nodiscard]] core::Status write_u64_member(Writer& writer, const std::string_view key,
                                            const std::uint64_t value) {
    if (auto status = write_key(writer, key); !status) {
        return status;
    }
    return writer.write_u64(value);
}

[[nodiscard]] core::Status write_float_member(Writer& writer, const std::string_view key,
                                              const renderer::TransportScalar value) {
    if (auto status = write_key(writer, key); !status) {
        return status;
    }
    return writer.write_float(value);
}

[[nodiscard]] core::Status write_string_member(Writer& writer, const std::string_view key,
                                               const std::string_view value) {
    if (auto status = write_key(writer, key); !status) {
        return status;
    }
    return writer.write_string(value);
}

template <typename Range, typename ElementWriter>
[[nodiscard]] core::Status write_array(Writer& writer, const Range& values,
                                       ElementWriter&& write_element) {
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    for (const auto& value : values) {
        if (auto status = write_element(value); !status) {
            return status;
        }
    }
    return writer.end_array();
}

template <typename Value> [[nodiscard]] core::Status write_xyz(Writer& writer, const Value value) {
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    for (const auto component : {value.x, value.y, value.z}) {
        if (auto status = writer.write_float(component); !status) {
            return status;
        }
    }
    return writer.end_array();
}

[[nodiscard]] core::Status write_xy(Writer& writer, const renderer::Point2 value) {
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    if (auto status = writer.write_float(value.x); !status) {
        return status;
    }
    if (auto status = writer.write_float(value.y); !status) {
        return status;
    }
    return writer.end_array();
}

[[nodiscard]] core::Status write_spectrum(Writer& writer,
                                          const renderer::TransportSpectrum spectrum) {
    return write_array(writer, spectrum.values,
                       [&](const auto value) { return writer.write_float(value); });
}

[[nodiscard]] core::Status write_wavelengths(Writer& writer,
                                             const renderer::SampledWavelengths& wavelengths) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "nanometers"); !status) {
        return status;
    }
    if (auto status =
            write_array(writer, wavelengths.samples,
                        [&](const auto& sample) { return writer.write_float(sample.nanometers); });
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "pdf"); !status) {
        return status;
    }
    if (auto status = write_array(
            writer, wavelengths.samples,
            [&](const auto& sample) {
                if (sample.probability.measure != renderer::ProbabilityMeasure::wavelength) {
                    return core::Status{std::unexpected(writer_error(
                        core::StatusCode::invalid_argument,
                        "Scene JSON wavelengths require wavelength-measure PDF values."))};
                }
                return writer.write_float(sample.probability.value);
            });
        !status) {
        return status;
    }
    return writer.end_object();
}

template <typename Enum>
[[nodiscard]] core::Result<std::string_view> unknown_enum(const std::string_view diagnostic) {
    return std::unexpected(writer_error(core::StatusCode::invalid_argument, diagnostic));
}

[[nodiscard]] core::Result<std::string_view>
accumulation_name(const renderer::AccumulationPrecision precision) {
    switch (precision) {
    case renderer::AccumulationPrecision::float32:
        return "float32";
    case renderer::AccumulationPrecision::float64:
        return "float64";
    }
    return unknown_enum<renderer::AccumulationPrecision>(
        "A scene JSON film has an unknown accumulation precision.");
}

[[nodiscard]] core::Result<std::string_view>
pixel_jitter_name(const renderer::PixelJitterMode mode) {
    switch (mode) {
    case renderer::PixelJitterMode::center:
        return "center";
    case renderer::PixelJitterMode::uniform:
        return "uniform";
    }
    return unknown_enum<renderer::PixelJitterMode>(
        "Scene JSON render options have an unknown pixel-jitter mode.");
}

[[nodiscard]] core::Result<std::string_view>
mis_heuristic_name(const renderer::MisHeuristic heuristic) {
    switch (heuristic) {
    case renderer::MisHeuristic::balance:
        return "balance";
    case renderer::MisHeuristic::power:
        return "power";
    }
    return unknown_enum<renderer::MisHeuristic>(
        "Scene JSON render options have an unknown MIS heuristic.");
}

[[nodiscard]] core::Result<std::string_view>
light_sampling_name(const renderer::LightSamplingStrategy strategy) {
    switch (strategy) {
    case renderer::LightSamplingStrategy::uniform:
        return "uniform";
    case renderer::LightSamplingStrategy::power_weighted:
        return "power_weighted";
    case renderer::LightSamplingStrategy::spatial_tree:
        return "spatial_tree";
    }
    return unknown_enum<renderer::LightSamplingStrategy>(
        "Scene JSON render options have an unknown light-sampling strategy.");
}

[[nodiscard]] core::Result<std::string_view> frame_mode_name(const SceneClosureFrameMode mode) {
    switch (mode) {
    case SceneClosureFrameMode::shading_normal:
        return "shading_normal";
    case SceneClosureFrameMode::surface_tangent:
        return "surface_tangent";
    }
    return unknown_enum<SceneClosureFrameMode>(
        "A scene JSON closure mixture has an unknown frame mode.");
}

[[nodiscard]] core::Result<std::string_view> wrap_mode_name(const renderer::TextureWrapMode mode) {
    switch (mode) {
    case renderer::TextureWrapMode::repeat:
        return "repeat";
    case renderer::TextureWrapMode::clamp:
        return "clamp";
    case renderer::TextureWrapMode::mirror:
        return "mirror";
    case renderer::TextureWrapMode::black:
        return "black";
    }
    return unknown_enum<renderer::TextureWrapMode>(
        "A scene JSON surface map has an unknown wrap mode.");
}

[[nodiscard]] core::Result<std::string_view>
normal_y_name(const renderer::TangentSpaceNormalYConvention convention) {
    switch (convention) {
    case renderer::TangentSpaceNormalYConvention::positive_v:
        return "positive_v";
    case renderer::TangentSpaceNormalYConvention::negative_v:
        return "negative_v";
    }
    return unknown_enum<renderer::TangentSpaceNormalYConvention>(
        "A scene JSON normal map has an unknown Y convention.");
}

[[nodiscard]] core::Result<std::string_view>
color_space_name(const renderer::TextureColorSpace color_space) {
    switch (color_space) {
    case renderer::TextureColorSpace::data:
        return "data";
    case renderer::TextureColorSpace::srgb:
        return "srgb";
    case renderer::TextureColorSpace::scene_linear_srgb:
        return "scene_linear_srgb";
    }
    return unknown_enum<renderer::TextureColorSpace>(
        "A scene JSON host image has an unknown color space.");
}

[[nodiscard]] core::Result<std::string_view> closure_kind_name(const renderer::ClosureKind kind) {
    switch (kind) {
    case renderer::ClosureKind::lambertian_reflection:
        return "lambertian_reflection";
    case renderer::ClosureKind::rough_diffuse_reflection:
        return "rough_diffuse_reflection";
    case renderer::ClosureKind::rough_conductor_reflection:
        return "rough_conductor_reflection";
    case renderer::ClosureKind::rough_dielectric:
        return "rough_dielectric";
    case renderer::ClosureKind::specular_reflection:
        return "specular_reflection";
    case renderer::ClosureKind::specular_transmission:
        return "specular_transmission";
    case renderer::ClosureKind::none:
        break;
    }
    return unknown_enum<renderer::ClosureKind>(
        "A scene JSON material contains an unknown or inactive closure kind.");
}

[[nodiscard]] core::Status write_film(Writer& writer, const SceneFilmDescription& film) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", film.id.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "extent"); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    if (auto status = writer.write_u64(film.extent.width); !status) {
        return status;
    }
    if (auto status = writer.write_u64(film.extent.height); !status) {
        return status;
    }
    if (auto status = writer.end_array(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "crop"); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    for (const auto value :
         {film.crop.minimum_x, film.crop.minimum_y, film.crop.maximum_x, film.crop.maximum_y}) {
        if (auto status = writer.write_u64(value); !status) {
            return status;
        }
    }
    if (auto status = writer.end_array(); !status) {
        return status;
    }
    const auto accumulation = accumulation_name(film.accumulation_precision);
    if (!accumulation) {
        return std::unexpected(accumulation.error());
    }
    if (auto status = write_string_member(writer, "accumulation", *accumulation); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_pinhole_model(Writer& writer,
                                               const ScenePinholeCameraDescription& model) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "type", "pinhole"); !status) {
        return status;
    }
    if (auto status = write_key(writer, "origin"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, model.origin); !status) {
        return status;
    }
    if (auto status = write_key(writer, "normal"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, model.orientation.normal()); !status) {
        return status;
    }
    if (auto status = write_key(writer, "tangent"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, model.orientation.tangent()); !status) {
        return status;
    }
    if (auto status = write_key(writer, "bitangent"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, model.orientation.bitangent()); !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "vertical_fov_radians",
                                         model.vertical_field_of_view_radians);
        !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "t_min", model.t_min); !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "t_max", model.t_max); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "visibility_mask", model.visibility_mask); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "current_medium", model.current_medium.value);
        !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_camera(Writer& writer, const SceneCameraDescription& camera) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", camera.id.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "film", camera.film.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "model"); !status) {
        return status;
    }
    if (camera.model.valueless_by_exception()) {
        return std::unexpected(writer_error(core::StatusCode::invalid_argument,
                                            "A scene JSON camera model has no value."));
    }
    const auto model_status = std::visit(
        [&](const auto& model) -> core::Status {
            using Model = std::remove_cvref_t<decltype(model)>;
            if constexpr (std::is_same_v<Model, ScenePinholeCameraDescription>) {
                return write_pinhole_model(writer, model);
            }
        },
        camera.model);
    if (!model_status) {
        return model_status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_depth_limits(Writer& writer,
                                              const renderer::PathDepthLimits limits) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "diffuse", limits.diffuse); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "glossy", limits.glossy); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "specular", limits.specular); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "transmission", limits.transmission); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "volume", limits.volume); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_russian_roulette(Writer& writer,
                                                  const renderer::RussianRoulettePolicy& policy) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    switch (policy.mode()) {
    case renderer::RussianRouletteMode::disabled:
        if (auto status = write_string_member(writer, "mode", "disabled"); !status) {
            return status;
        }
        break;
    case renderer::RussianRouletteMode::enabled:
        if (auto status = write_string_member(writer, "mode", "enabled"); !status) {
            return status;
        }
        if (auto status =
                write_u64_member(writer, "first_eligible_depth", policy.first_eligible_depth());
            !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "minimum_survival_probability",
                                             policy.minimum_survival_probability());
            !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "maximum_survival_probability",
                                             policy.maximum_survival_probability());
            !status) {
            return status;
        }
        break;
    default:
        return std::unexpected(
            writer_error(core::StatusCode::invalid_argument,
                         "Scene JSON render options have an unknown Russian-roulette mode."));
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_render_options(Writer& writer,
                                                const SceneRenderOptionsDescription& description) {
    const auto jitter = pixel_jitter_name(description.options.pixel_jitter);
    const auto heuristic = mis_heuristic_name(description.options.mis_heuristic);
    const auto sampling = light_sampling_name(description.options.light_sampling_strategy);
    if (!jitter) {
        return std::unexpected(jitter.error());
    }
    if (!heuristic) {
        return std::unexpected(heuristic.error());
    }
    if (!sampling) {
        return std::unexpected(sampling.error());
    }

    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", description.id.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "film", description.film.value); !status) {
        return status;
    }
    const auto& options = description.options;
    if (auto status = write_u64_member(writer, "samples_per_pixel", options.samples_per_pixel);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "maximum_path_depth", options.maximum_path_depth);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "tile_edge_length", options.tile_edge_length);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "seed", options.seed); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "pixel_jitter", *jitter); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "mis_heuristic", *heuristic); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "light_sampling_strategy", *sampling); !status) {
        return status;
    }
    if (auto status = write_key(writer, "depth_limits"); !status) {
        return status;
    }
    if (auto status = write_depth_limits(writer, options.depth_limits); !status) {
        return status;
    }
    if (auto status = write_key(writer, "russian_roulette"); !status) {
        return status;
    }
    if (auto status = write_russian_roulette(writer, options.roulette_policy); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_constant_texture(Writer& writer,
                                                  const SceneConstantTexture& texture) {
    if (texture.texture.valueless_by_exception()) {
        return std::unexpected(writer_error(core::StatusCode::invalid_argument,
                                            "A scene JSON constant texture has no value."));
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", texture.id.value); !status) {
        return status;
    }
    const auto value_status = std::visit(
        [&](const auto value) -> core::Status {
            using Value = std::remove_cvref_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, renderer::ConstantFloatTexture>) {
                if (auto status = write_string_member(writer, "type", "float"); !status) {
                    return status;
                }
                if (auto status = write_key(writer, "value"); !status) {
                    return status;
                }
                return writer.write_float(value.value());
            } else if constexpr (std::is_same_v<Value, renderer::ConstantColorTexture>) {
                if (auto status = write_string_member(writer, "type", "color"); !status) {
                    return status;
                }
                if (auto status = write_key(writer, "value"); !status) {
                    return status;
                }
                const auto color = value.value();
                if (auto status = writer.begin_array(); !status) {
                    return status;
                }
                for (const auto component : {color.red, color.green, color.blue}) {
                    if (auto status = writer.write_float(component); !status) {
                        return status;
                    }
                }
                return writer.end_array();
            } else if constexpr (std::is_same_v<Value, renderer::ConstantSpectrumTexture>) {
                if (auto status = write_string_member(writer, "type", "spectrum"); !status) {
                    return status;
                }
                if (auto status = write_key(writer, "value"); !status) {
                    return status;
                }
                return write_spectrum(writer, value.value());
            } else {
                static_assert(AlwaysFalse<Value>, "Unhandled constant texture variant");
            }
        },
        texture.texture);
    if (!value_status) {
        return value_status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Result<std::string> path_utf8(const std::filesystem::path& path) {
    try {
        const auto encoded = path.generic_u8string();
        return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
    } catch (const std::filesystem::filesystem_error&) {
        return std::unexpected(writer_error(
            core::StatusCode::invalid_argument,
            "A scene JSON host-image source path cannot be represented as UTF-8 metadata."));
    }
}

[[nodiscard]] core::Status write_host_image_texture(Writer& writer,
                                                    const SceneHostImageTexture& texture) {
    if (!texture.image || !texture.image->source_image()) {
        return std::unexpected(
            writer_error(core::StatusCode::invalid_argument,
                         "A scene JSON host-image texture has no complete immutable snapshot."));
    }
    const auto image = texture.image->source_image();
    const auto source_path = path_utf8(image->source_path());
    const auto source_color_space = color_space_name(image->source_color_space());
    const auto storage_color_space = color_space_name(image->storage_color_space());
    if (!source_path) {
        return std::unexpected(source_path.error());
    }
    if (!source_color_space) {
        return std::unexpected(source_color_space.error());
    }
    if (!storage_color_space) {
        return std::unexpected(storage_color_space.error());
    }

    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", texture.id.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "image"); !status) {
        return status;
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "source_path", *source_path); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "format_name", image->format_name()); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "source_color_space", *source_color_space);
        !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "storage_color_space", *storage_color_space);
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "origin"); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    if (auto status = writer.write_i64(image->origin_x()); !status) {
        return status;
    }
    if (auto status = writer.write_i64(image->origin_y()); !status) {
        return status;
    }
    if (auto status = writer.end_array(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "extent"); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    if (auto status = writer.write_u64(image->width()); !status) {
        return status;
    }
    if (auto status = writer.write_u64(image->height()); !status) {
        return status;
    }
    if (auto status = writer.end_array(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "channel_names"); !status) {
        return status;
    }
    if (auto status = write_array(writer, image->channel_names(),
                                  [&](const auto& name) { return writer.write_string(name); });
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "pixels"); !status) {
        return status;
    }
    if (auto status = write_array(writer, image->pixels(),
                                  [&](const auto value) { return writer.write_float(value); });
        !status) {
        return status;
    }
    if (auto status = writer.end_object(); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_geometry(Writer& writer, const SceneGeometry& geometry) {
    if (!geometry.mesh) {
        return std::unexpected(writer_error(core::StatusCode::invalid_argument,
                                            "A scene JSON geometry has no mesh snapshot."));
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", geometry.id.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "mesh"); !status) {
        return status;
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "positions"); !status) {
        return status;
    }
    if (auto status = write_array(writer, geometry.mesh->positions(),
                                  [&](const auto value) { return write_xyz(writer, value); });
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "normals"); !status) {
        return status;
    }
    if (auto status = write_array(writer, geometry.mesh->normals(),
                                  [&](const auto value) { return write_xyz(writer, value); });
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "texture_coordinates"); !status) {
        return status;
    }
    if (auto status = write_array(writer, geometry.mesh->texture_coordinates(),
                                  [&](const auto value) { return write_xy(writer, value); });
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "triangles"); !status) {
        return status;
    }
    if (auto status = write_array(writer, geometry.mesh->triangles(),
                                  [&](const auto& triangle) {
                                      return write_array(writer, triangle.vertices,
                                                         [&](const auto value) {
                                                             return writer.write_u64(value);
                                                         });
                                  });
        !status) {
        return status;
    }
    if (auto status = writer.end_object(); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_ewa_limits(Writer& writer,
                                            const renderer::HostImageEwaLimits limits) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "maximum_anisotropy", limits.maximum_anisotropy);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "maximum_texel_visits", limits.maximum_texel_visits);
        !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_normal_map(Writer& writer, const SceneNormalMapBinding& binding) {
    const auto y_convention = normal_y_name(binding.y_convention);
    const auto u_wrap = wrap_mode_name(binding.u_wrap);
    const auto v_wrap = wrap_mode_name(binding.v_wrap);
    if (!y_convention) {
        return std::unexpected(y_convention.error());
    }
    if (!u_wrap) {
        return std::unexpected(u_wrap.error());
    }
    if (!v_wrap) {
        return std::unexpected(v_wrap.error());
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "texture", binding.texture.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "red_channel", binding.red_channel); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "green_channel", binding.green_channel); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "blue_channel", binding.blue_channel); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "y_convention", *y_convention); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "u_wrap", *u_wrap); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "v_wrap", *v_wrap); !status) {
        return status;
    }
    if (auto status = write_key(writer, "ewa"); !status) {
        return status;
    }
    if (auto status = write_ewa_limits(writer, binding.ewa_limits); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_bump_map(Writer& writer, const SceneBumpMapBinding& binding) {
    const auto u_wrap = wrap_mode_name(binding.u_wrap);
    const auto v_wrap = wrap_mode_name(binding.v_wrap);
    if (!u_wrap) {
        return std::unexpected(u_wrap.error());
    }
    if (!v_wrap) {
        return std::unexpected(v_wrap.error());
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "texture", binding.texture.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "channel", binding.channel); !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "scale", binding.scale); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "u_wrap", *u_wrap); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "v_wrap", *v_wrap); !status) {
        return status;
    }
    if (auto status = write_key(writer, "ewa"); !status) {
        return status;
    }
    if (auto status = write_ewa_limits(writer, binding.ewa_limits); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_parameter_spectrum(
    Writer& writer, const std::string_view key,
    const std::array<renderer::TransportScalar, renderer::ClosureParameterScalarCount>& parameters,
    const std::size_t offset) {
    if (auto status = write_key(writer, key); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        if (auto status = writer.write_float(parameters[offset + lane]); !status) {
            return status;
        }
    }
    return writer.end_array();
}

[[nodiscard]] core::Status write_closure_component(Writer& writer, const renderer::Closure& closure,
                                                   const renderer::TransportScalar probability) {
    const auto type = closure_kind_name(closure.kind);
    if (!type) {
        return std::unexpected(type.error());
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "probability", probability); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "type", *type); !status) {
        return status;
    }
    if (auto status = write_key(writer, "weight"); !status) {
        return status;
    }
    if (auto status = write_spectrum(writer, closure.weight); !status) {
        return status;
    }

    switch (closure.kind) {
    case renderer::ClosureKind::lambertian_reflection:
    case renderer::ClosureKind::specular_reflection:
        break;
    case renderer::ClosureKind::rough_diffuse_reflection:
        if (auto status = write_float_member(writer, "roughness", closure.parameters[0]); !status) {
            return status;
        }
        break;
    case renderer::ClosureKind::rough_conductor_reflection:
        if (auto status = write_parameter_spectrum(writer, "eta", closure.parameters, 0U);
            !status) {
            return status;
        }
        if (auto status = write_parameter_spectrum(writer, "k", closure.parameters,
                                                   renderer::TransportSpectrumSampleCount);
            !status) {
            return status;
        }
        if (auto status = write_float_member(
                writer, "alpha_x", closure.parameters[renderer::TransportSpectrumSampleCount * 2U]);
            !status) {
            return status;
        }
        if (auto status = write_float_member(
                writer, "alpha_y",
                closure.parameters[renderer::TransportSpectrumSampleCount * 2U + 1U]);
            !status) {
            return status;
        }
        break;
    case renderer::ClosureKind::rough_dielectric:
        if (auto status = write_float_member(writer, "exterior_eta", closure.parameters[0]);
            !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "interior_eta", closure.parameters[1]);
            !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "alpha_x", closure.parameters[2]); !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "alpha_y", closure.parameters[3]); !status) {
            return status;
        }
        break;
    case renderer::ClosureKind::specular_transmission:
        if (auto status = write_float_member(writer, "exterior_eta", closure.parameters[0]);
            !status) {
            return status;
        }
        if (auto status = write_float_member(writer, "interior_eta", closure.parameters[1]);
            !status) {
            return status;
        }
        break;
    case renderer::ClosureKind::none:
        return std::unexpected(writer_error(
            core::StatusCode::invalid_argument,
            "A scene JSON material contains an inactive closure in its active prefix."));
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_closure_mixture(Writer& writer,
                                                 const SceneClosureMixture& mixture) {
    const auto frame_mode = frame_mode_name(mixture.frame_mode);
    if (!frame_mode) {
        return std::unexpected(frame_mode.error());
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "frame_mode", *frame_mode); !status) {
        return status;
    }
    if (auto status = write_float_member(writer, "tangent_rotation_radians",
                                         mixture.tangent_rotation_radians);
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "components"); !status) {
        return status;
    }
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    const auto closures = mixture.closures.closures();
    for (auto index = std::size_t{}; index < closures.size(); ++index) {
        if (auto status = write_closure_component(writer, closures[index],
                                                  mixture.component_probabilities[index]);
            !status) {
            return status;
        }
    }
    if (auto status = writer.end_array(); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_material(Writer& writer, const SceneMaterial& material) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", material.id.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "spectral"); !status) {
        return status;
    }
    if (!material.spectral) {
        if (auto status = writer.write_null(); !status) {
            return status;
        }
        return writer.end_object();
    }

    const auto& spectral = *material.spectral;
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "wavelengths"); !status) {
        return status;
    }
    if (auto status = write_wavelengths(writer, spectral.wavelengths); !status) {
        return status;
    }
    if (auto status = write_key(writer, "closures"); !status) {
        return status;
    }
    if (auto status = write_closure_mixture(writer, spectral.closure_mixture); !status) {
        return status;
    }
    if (auto status = write_key(writer, "emission"); !status) {
        return status;
    }
    if (auto status = write_spectrum(writer, spectral.emitted_radiance); !status) {
        return status;
    }
    if (auto status = write_key(writer, "normal_map"); !status) {
        return status;
    }
    if (spectral.normal_map) {
        if (auto status = write_normal_map(writer, *spectral.normal_map); !status) {
            return status;
        }
    } else if (auto status = writer.write_null(); !status) {
        return status;
    }
    if (auto status = write_key(writer, "bump_map"); !status) {
        return status;
    }
    if (spectral.bump_map) {
        if (auto status = write_bump_map(writer, *spectral.bump_map); !status) {
            return status;
        }
    } else if (auto status = writer.write_null(); !status) {
        return status;
    }
    if (auto status = writer.end_object(); !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_matrix(Writer& writer, const renderer::Matrix4& matrix) {
    if (auto status = writer.begin_array(); !status) {
        return status;
    }
    for (auto row = std::size_t{}; row < 4U; ++row) {
        for (auto column = std::size_t{}; column < 4U; ++column) {
            if (auto status = writer.write_float(matrix(row, column)); !status) {
                return status;
            }
        }
    }
    return writer.end_array();
}

[[nodiscard]] core::Status write_instance(Writer& writer, const SceneInstance& instance) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", instance.id.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "parent"); !status) {
        return status;
    }
    if (instance.parent) {
        if (auto status = writer.write_u64(instance.parent->value); !status) {
            return status;
        }
    } else if (auto status = writer.write_null(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "object", instance.object.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "geometry", instance.geometry.value); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "material", instance.material.value); !status) {
        return status;
    }
    if (auto status = write_key(writer, "local_to_parent"); !status) {
        return status;
    }
    if (auto status = write_matrix(writer, instance.local_to_parent); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "visibility_mask", instance.visibility_mask);
        !status) {
        return status;
    }
    return writer.end_object();
}

[[nodiscard]] core::Status write_point_light(Writer& writer, const ScenePointLight& light) {
    if (auto status = write_key(writer, "position"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.position); !status) {
        return status;
    }
    if (auto status = write_key(writer, "absolute_position_error"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.absolute_position_error); !status) {
        return status;
    }
    if (auto status = write_key(writer, "spectral_radiant_intensity"); !status) {
        return status;
    }
    return write_spectrum(writer, light.spectral_radiant_intensity);
}

[[nodiscard]] core::Status write_directional_light(Writer& writer,
                                                   const SceneDirectionalLight& light) {
    if (auto status = write_key(writer, "propagation_direction"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.propagation_direction); !status) {
        return status;
    }
    if (auto status = write_key(writer, "spectral_irradiance"); !status) {
        return status;
    }
    return write_spectrum(writer, light.spectral_irradiance);
}

[[nodiscard]] core::Status write_spot_light(Writer& writer, const SceneSpotLight& light) {
    if (auto status = write_key(writer, "position"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.position); !status) {
        return status;
    }
    if (auto status = write_key(writer, "absolute_position_error"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.absolute_position_error); !status) {
        return status;
    }
    if (auto status = write_key(writer, "emission_direction"); !status) {
        return status;
    }
    if (auto status = write_xyz(writer, light.emission_direction); !status) {
        return status;
    }
    if (auto status =
            write_float_member(writer, "inner_half_angle_radians", light.inner_half_angle_radians);
        !status) {
        return status;
    }
    if (auto status =
            write_float_member(writer, "outer_half_angle_radians", light.outer_half_angle_radians);
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "on_axis_spectral_radiant_intensity"); !status) {
        return status;
    }
    return write_spectrum(writer, light.on_axis_spectral_radiant_intensity);
}

[[nodiscard]] core::Status write_environment_light(Writer& writer,
                                                   const SceneSpectralEnvironment& light) {
    if (auto status = write_key(writer, "wavelengths"); !status) {
        return status;
    }
    if (auto status = write_wavelengths(writer, light.wavelengths); !status) {
        return status;
    }
    if (auto status = write_key(writer, "radiance"); !status) {
        return status;
    }
    return write_spectrum(writer, light.radiance);
}

[[nodiscard]] core::Status write_light(Writer& writer, const SceneLightDescription& description) {
    if (description.light.valueless_by_exception()) {
        return std::unexpected(
            writer_error(core::StatusCode::invalid_argument, "A scene JSON light has no value."));
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "id", description.id.value); !status) {
        return status;
    }
    const auto light_status = std::visit(
        [&](const auto& light) -> core::Status {
            using Light = std::remove_cvref_t<decltype(light)>;
            if constexpr (std::is_same_v<Light, ScenePointLight>) {
                if (auto status = write_string_member(writer, "type", "point"); !status) {
                    return status;
                }
                return write_point_light(writer, light);
            } else if constexpr (std::is_same_v<Light, SceneDirectionalLight>) {
                if (auto status = write_string_member(writer, "type", "directional"); !status) {
                    return status;
                }
                return write_directional_light(writer, light);
            } else if constexpr (std::is_same_v<Light, SceneSpotLight>) {
                if (auto status = write_string_member(writer, "type", "spot"); !status) {
                    return status;
                }
                return write_spot_light(writer, light);
            } else if constexpr (std::is_same_v<Light, SceneSpectralEnvironment>) {
                if (auto status = write_string_member(writer, "type", "environment"); !status) {
                    return status;
                }
                return write_environment_light(writer, light);
            } else {
                static_assert(AlwaysFalse<Light>, "Unhandled scene light variant");
            }
        },
        description.light);
    if (!light_status) {
        return light_status;
    }
    return writer.end_object();
}

template <typename Records, typename RecordWriter>
[[nodiscard]] core::Status write_registry_member(Writer& writer, const std::string_view key,
                                                 const Records records,
                                                 RecordWriter&& write_record) {
    if (auto status = write_key(writer, key); !status) {
        return status;
    }
    return write_array(writer, records, std::forward<RecordWriter>(write_record));
}

[[nodiscard]] core::Status write_scene(Writer& writer, const SceneDescription& description) {
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_string_member(writer, "schema", SceneDescriptionJsonSchemaName);
        !status) {
        return status;
    }
    if (auto status =
            write_u64_member(writer, "schema_version", CurrentSceneDescriptionJsonSchemaVersion);
        !status) {
        return status;
    }
    if (auto status = write_key(writer, "active"); !status) {
        return status;
    }
    if (auto status = writer.begin_object(); !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "film", description.active_film_id().value);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "camera", description.active_camera_id().value);
        !status) {
        return status;
    }
    if (auto status = write_u64_member(writer, "render_options",
                                       description.active_render_options_id().value);
        !status) {
        return status;
    }
    if (auto status = writer.end_object(); !status) {
        return status;
    }

    if (auto status =
            write_registry_member(writer, "films", description.films(),
                                  [&](const auto& film) { return write_film(writer, film); });
        !status) {
        return status;
    }
    if (auto status =
            write_registry_member(writer, "cameras", description.cameras(),
                                  [&](const auto& camera) { return write_camera(writer, camera); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "render_options", description.render_options(),
            [&](const auto& options) { return write_render_options(writer, options); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "constant_textures", description.constant_textures(),
            [&](const auto& texture) { return write_constant_texture(writer, texture); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "host_image_textures", description.host_image_textures(),
            [&](const auto& texture) { return write_host_image_texture(writer, texture); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "objects", description.objects(),
            [&](const auto& object) -> core::Status {
                if (auto result = writer.begin_object(); !result) {
                    return result;
                }
                if (auto result = write_u64_member(writer, "id", object.id.value); !result) {
                    return result;
                }
                return writer.end_object();
            });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "geometries", description.geometries(),
            [&](const auto& geometry) { return write_geometry(writer, geometry); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "materials", description.materials(),
            [&](const auto& material) { return write_material(writer, material); });
        !status) {
        return status;
    }
    if (auto status = write_registry_member(
            writer, "instances", description.instances(),
            [&](const auto& instance) { return write_instance(writer, instance); });
        !status) {
        return status;
    }
    if (auto status =
            write_registry_member(writer, "lights", description.lights(),
                                  [&](const auto& light) { return write_light(writer, light); });
        !status) {
        return status;
    }
    return writer.end_object();
}

} // namespace

core::Result<std::string>
serialize_scene_description_json(const SceneDescription& description,
                                 const SceneDescriptionJsonLimits limits) {
    try {
        if (auto status = preflight_description(description, limits); !status) {
            return std::unexpected(status.error());
        }
        auto created_writer = Writer::create(limits);
        if (!created_writer) {
            return std::unexpected(created_writer.error());
        }
        auto writer = std::move(*created_writer);
        if (auto status = write_scene(writer, description); !status) {
            return std::unexpected(status.error());
        }
        return writer.finish();
    } catch (const std::bad_alloc&) {
        return std::unexpected(writer_error(core::StatusCode::resource_exhausted,
                                            "Scene JSON serialization exhausted memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            writer_error(core::StatusCode::resource_exhausted,
                         "A scene JSON serialization allocation size is not representable."));
    }
}

} // namespace blackframe::engine
