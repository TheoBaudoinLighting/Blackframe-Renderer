#include "SceneJsonSyntax.hpp"

#include <Blackframe/Engine/SceneDescriptionJson.hpp>
#include <Blackframe/Renderer/ClosureSet.hpp>
#include <Blackframe/Renderer/HostImage.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#include <Blackframe/Renderer/MatrixTypes.hpp>
#include <Blackframe/Renderer/TextureColorSpace.hpp>
#include <Blackframe/Renderer/TextureWrap.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

using scene_json_syntax::Reader;

[[nodiscard]] core::Error json_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Status unknown_field(const std::string_view context,
                                         const std::string_view key) {
    return std::unexpected(
        json_error(core::StatusCode::invalid_argument,
                   "Unknown " + std::string{context} + " key '" + std::string{key} + "'."));
}

[[nodiscard]] core::Status require_fields(const std::uint64_t seen, const std::uint64_t required,
                                          const std::string_view context) {
    if ((seen & required) != required) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       std::string{context} + " is missing one or more required keys."));
    }
    return {};
}

template <typename Callback>
[[nodiscard]] core::Status read_closed_object(Reader& reader, const std::string_view context,
                                              const std::uint64_t required, Callback&& callback) {
    auto seen = std::uint64_t{};
    auto status = reader.read_object(
        [&](const std::string_view key) { return std::invoke(callback, key, seen); });
    if (!status) {
        return status;
    }
    return require_fields(seen, required, context);
}

class DecodedBudget final {
  public:
    explicit DecodedBudget(const SceneDescriptionJsonLimits limits) noexcept : limits_{limits} {}

    [[nodiscard]] core::Status add_bytes(const std::uint64_t bytes,
                                         const std::string_view description) {
        if (bytes > limits_.maximum_decoded_bytes - used_bytes_) {
            return std::unexpected(
                json_error(core::StatusCode::resource_exhausted,
                           std::string{description} + " exceeds the decoded scene byte budget."));
        }
        used_bytes_ += bytes;
        return {};
    }

    template <typename Value>
    [[nodiscard]] core::Status add_value(const std::string_view description) {
        return add_bytes(sizeof(Value), description);
    }

    [[nodiscard]] core::Status add_mesh_vertices(const std::uint64_t count) {
        if (count > limits_.maximum_mesh_vertices - mesh_vertices_) {
            return std::unexpected(
                json_error(core::StatusCode::resource_exhausted,
                           "Scene geometry exceeds the configured mesh-vertex limit."));
        }
        mesh_vertices_ += count;
        return {};
    }

    [[nodiscard]] core::Status add_mesh_triangles(const std::uint64_t count) {
        if (count > limits_.maximum_mesh_triangles - mesh_triangles_) {
            return std::unexpected(
                json_error(core::StatusCode::resource_exhausted,
                           "Scene geometry exceeds the configured mesh-triangle limit."));
        }
        mesh_triangles_ += count;
        return {};
    }

    [[nodiscard]] core::Status add_image_scalars(const std::uint64_t count) {
        if (count > limits_.maximum_image_scalar_values - image_scalars_) {
            return std::unexpected(
                json_error(core::StatusCode::resource_exhausted,
                           "Scene images exceed the configured scalar-value limit."));
        }
        image_scalars_ += count;
        return add_bytes(count * sizeof(renderer::TransportScalar), "Decoded host-image pixels");
    }

    [[nodiscard]] std::uint64_t remaining_bytes() const noexcept {
        return limits_.maximum_decoded_bytes - used_bytes_;
    }

    [[nodiscard]] std::uint32_t maximum_records() const noexcept {
        return limits_.maximum_records_per_registry;
    }

    [[nodiscard]] std::uint64_t maximum_mesh_vertices() const noexcept {
        return limits_.maximum_mesh_vertices;
    }

    [[nodiscard]] std::uint64_t maximum_image_scalar_values() const noexcept {
        return limits_.maximum_image_scalar_values;
    }

  private:
    SceneDescriptionJsonLimits limits_{};
    std::uint64_t used_bytes_{};
    std::uint64_t mesh_vertices_{};
    std::uint64_t mesh_triangles_{};
    std::uint64_t image_scalars_{};
};

template <typename Identifier>
[[nodiscard]] core::Result<Identifier> read_identifier(Reader& reader,
                                                       const std::string_view description) {
    auto value = reader.read_u64();
    if (!value) {
        return std::unexpected(std::move(value.error()));
    }
    if (*value > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       std::string{description} + " exceeds the 32-bit identifier domain."));
    }
    return Identifier{.value = static_cast<std::uint32_t>(*value)};
}

[[nodiscard]] core::Result<std::uint32_t> read_u32(Reader& reader,
                                                   const std::string_view description) {
    auto value = reader.read_u64();
    if (!value) {
        return std::unexpected(std::move(value.error()));
    }
    if (*value > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       std::string{description} + " exceeds the 32-bit unsigned domain."));
    }
    return static_cast<std::uint32_t>(*value);
}

[[nodiscard]] core::Result<std::int32_t> read_i32(Reader& reader,
                                                  const std::string_view description) {
    auto value = reader.read_i64();
    if (!value) {
        return std::unexpected(std::move(value.error()));
    }
    if (*value < std::numeric_limits<std::int32_t>::min() ||
        *value > std::numeric_limits<std::int32_t>::max()) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       std::string{description} + " exceeds the 32-bit signed domain."));
    }
    return static_cast<std::int32_t>(*value);
}

template <typename Value, std::size_t Size, typename ElementReader>
[[nodiscard]] core::Result<std::array<Value, Size>>
read_fixed_array(Reader& reader, const std::string_view description,
                 ElementReader&& element_reader) {
    auto values = std::array<Value, Size>{};
    auto count = std::uint64_t{};
    auto status = reader.read_array([&](const std::uint64_t index) -> core::Status {
        if (index >= Size) {
            return std::unexpected(
                json_error(core::StatusCode::invalid_argument,
                           std::string{description} + " contains too many elements."));
        }
        auto value = std::invoke(element_reader);
        if (!value) {
            return std::unexpected(std::move(value.error()));
        }
        values[static_cast<std::size_t>(index)] = std::move(*value);
        count = index + 1U;
        return {};
    });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if (count != Size) {
        return std::unexpected(json_error(core::StatusCode::invalid_argument,
                                          std::string{description} + " must contain exactly " +
                                              std::to_string(Size) + " elements."));
    }
    return values;
}

[[nodiscard]] core::Result<renderer::Vector3> read_vector3(Reader& reader,
                                                           const std::string_view description) {
    auto values =
        read_fixed_array<float, 3U>(reader, description, [&] { return reader.read_float(); });
    if (!values) {
        return std::unexpected(std::move(values.error()));
    }
    return renderer::Vector3{.x = (*values)[0], .y = (*values)[1], .z = (*values)[2]};
}

[[nodiscard]] core::Result<renderer::Point3> read_point3(Reader& reader,
                                                         const std::string_view description) {
    auto vector = read_vector3(reader, description);
    if (!vector) {
        return std::unexpected(std::move(vector.error()));
    }
    return renderer::Point3{.x = vector->x, .y = vector->y, .z = vector->z};
}

[[nodiscard]] core::Result<renderer::Normal3> read_normal3(Reader& reader,
                                                           const std::string_view description) {
    auto vector = read_vector3(reader, description);
    if (!vector) {
        return std::unexpected(std::move(vector.error()));
    }
    return renderer::Normal3{.x = vector->x, .y = vector->y, .z = vector->z};
}

[[nodiscard]] core::Result<renderer::Point2> read_point2(Reader& reader,
                                                         const std::string_view description) {
    auto values =
        read_fixed_array<float, 2U>(reader, description, [&] { return reader.read_float(); });
    if (!values) {
        return std::unexpected(std::move(values.error()));
    }
    return renderer::Point2{.x = (*values)[0], .y = (*values)[1]};
}

[[nodiscard]] core::Result<renderer::TransportSpectrum>
read_spectrum(Reader& reader, const std::string_view description) {
    auto values =
        read_fixed_array<renderer::TransportScalar, renderer::TransportSpectrumSampleCount>(
            reader, description, [&] { return reader.read_float(); });
    if (!values) {
        return std::unexpected(std::move(values.error()));
    }
    return renderer::TransportSpectrum{.values = *values};
}

template <typename Enum, std::size_t Extent>
[[nodiscard]] core::Result<Enum>
read_named_enum(Reader& reader, const std::string_view description,
                const std::span<const std::pair<std::string_view, Enum>, Extent> names) {
    auto encoded = reader.read_string();
    if (!encoded) {
        return std::unexpected(std::move(encoded.error()));
    }
    const auto match =
        std::ranges::find(names, *encoded, &std::pair<std::string_view, Enum>::first);
    if (match == names.end()) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       std::string{description} + " has unsupported value '" + *encoded + "'."));
    }
    return match->second;
}

template <typename Value, typename Parser>
[[nodiscard]] core::Status read_registry(Reader& reader, std::vector<Value>& output,
                                         DecodedBudget& budget, const std::string_view description,
                                         Parser&& parser) {
    return reader.read_array([&](const std::uint64_t index) -> core::Status {
        if (index >= budget.maximum_records()) {
            return std::unexpected(json_error(
                core::StatusCode::resource_exhausted,
                std::string{description} + " exceeds the configured registry-record limit."));
        }
        if (auto status = budget.add_value<Value>(description); !status) {
            return status;
        }
        auto value = std::invoke(parser);
        if (!value) {
            return std::unexpected(std::move(value.error()));
        }
        output.push_back(std::move(*value));
        return {};
    });
}

[[nodiscard]] core::Result<renderer::AccumulationPrecision>
read_accumulation_precision(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"float32"}, renderer::AccumulationPrecision::float32},
        std::pair{std::string_view{"float64"}, renderer::AccumulationPrecision::float64},
    };
    return read_named_enum(reader, "Film accumulation precision", std::span{names});
}

[[nodiscard]] core::Result<renderer::PixelJitterMode> read_pixel_jitter(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"center"}, renderer::PixelJitterMode::center},
        std::pair{std::string_view{"uniform"}, renderer::PixelJitterMode::uniform},
    };
    return read_named_enum(reader, "Pixel-jitter mode", std::span{names});
}

[[nodiscard]] core::Result<renderer::MisHeuristic> read_mis_heuristic(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"balance"}, renderer::MisHeuristic::balance},
        std::pair{std::string_view{"power"}, renderer::MisHeuristic::power},
    };
    return read_named_enum(reader, "MIS heuristic", std::span{names});
}

[[nodiscard]] core::Result<renderer::LightSamplingStrategy>
read_light_sampling_strategy(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"uniform"}, renderer::LightSamplingStrategy::uniform},
        std::pair{std::string_view{"power_weighted"},
                  renderer::LightSamplingStrategy::power_weighted},
        std::pair{std::string_view{"spatial_tree"}, renderer::LightSamplingStrategy::spatial_tree},
    };
    return read_named_enum(reader, "Light-sampling strategy", std::span{names});
}

[[nodiscard]] core::Result<renderer::TextureColorSpace> read_texture_color_space(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"data"}, renderer::TextureColorSpace::data},
        std::pair{std::string_view{"srgb"}, renderer::TextureColorSpace::srgb},
        std::pair{std::string_view{"scene_linear_srgb"},
                  renderer::TextureColorSpace::scene_linear_srgb},
    };
    return read_named_enum(reader, "Texture color space", std::span{names});
}

[[nodiscard]] core::Result<renderer::TextureWrapMode> read_texture_wrap(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"repeat"}, renderer::TextureWrapMode::repeat},
        std::pair{std::string_view{"clamp"}, renderer::TextureWrapMode::clamp},
        std::pair{std::string_view{"mirror"}, renderer::TextureWrapMode::mirror},
        std::pair{std::string_view{"black"}, renderer::TextureWrapMode::black},
    };
    return read_named_enum(reader, "Texture wrap mode", std::span{names});
}

[[nodiscard]] core::Result<renderer::TangentSpaceNormalYConvention>
read_normal_y_convention(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"positive_v"},
                  renderer::TangentSpaceNormalYConvention::positive_v},
        std::pair{std::string_view{"negative_v"},
                  renderer::TangentSpaceNormalYConvention::negative_v},
    };
    return read_named_enum(reader, "Normal-map Y convention", std::span{names});
}

[[nodiscard]] core::Result<SceneClosureFrameMode> read_closure_frame_mode(Reader& reader) {
    constexpr auto names = std::array{
        std::pair{std::string_view{"shading_normal"}, SceneClosureFrameMode::shading_normal},
        std::pair{std::string_view{"surface_tangent"}, SceneClosureFrameMode::surface_tangent},
    };
    return read_named_enum(reader, "Closure frame mode", std::span{names});
}

[[nodiscard]] core::Result<SceneFilmDescription> read_film(Reader& reader) {
    auto result = SceneFilmDescription{};
    constexpr auto required = std::uint64_t{0x0FU};
    auto status = read_closed_object(
        reader, "Scene film", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value = read_identifier<renderer::FilmId>(reader, "Film identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "extent") {
                auto value = read_fixed_array<std::uint32_t, 2U>(
                    reader, "Film extent", [&] { return read_u32(reader, "Film extent"); });
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.extent = {.width = (*value)[0], .height = (*value)[1]};
                seen |= 1U << 1U;
                return {};
            }
            if (key == "crop") {
                auto value = read_fixed_array<std::uint32_t, 4U>(
                    reader, "Film crop", [&] { return read_u32(reader, "Film crop"); });
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.crop = {.minimum_x = (*value)[0],
                               .minimum_y = (*value)[1],
                               .maximum_x = (*value)[2],
                               .maximum_y = (*value)[3]};
                seen |= 1U << 2U;
                return {};
            }
            if (key == "accumulation") {
                auto value = read_accumulation_precision(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.accumulation_precision = *value;
                seen |= 1U << 3U;
                return {};
            }
            return unknown_field("scene film", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<ScenePinholeCameraDescription> read_pinhole_camera(Reader& reader) {
    auto type = std::string{};
    auto origin = renderer::Point3{};
    auto normal = renderer::Normal3{};
    auto tangent = renderer::Vector3{};
    auto bitangent = renderer::Vector3{};
    auto vertical_fov = renderer::TransportScalar{};
    auto t_min = renderer::TransportScalar{};
    auto t_max = renderer::TransportScalar{};
    auto visibility_mask = renderer::RayMask{};
    auto current_medium = renderer::MediumId{};
    constexpr auto required = std::uint64_t{0x3FFU};
    auto status =
        read_closed_object(reader, "Scene camera model", required,
                           [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
                               if (key == "type") {
                                   auto value = reader.read_string();
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   type = std::move(*value);
                                   seen |= 1U << 0U;
                                   return {};
                               }
                               if (key == "origin") {
                                   auto value = read_point3(reader, "Camera origin");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   origin = *value;
                                   seen |= 1U << 1U;
                                   return {};
                               }
                               if (key == "normal") {
                                   auto value = read_normal3(reader, "Camera normal");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   normal = *value;
                                   seen |= 1U << 2U;
                                   return {};
                               }
                               if (key == "tangent") {
                                   auto value = read_vector3(reader, "Camera tangent");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   tangent = *value;
                                   seen |= 1U << 3U;
                                   return {};
                               }
                               if (key == "bitangent") {
                                   auto value = read_vector3(reader, "Camera bitangent");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   bitangent = *value;
                                   seen |= 1U << 4U;
                                   return {};
                               }
                               if (key == "vertical_fov_radians") {
                                   auto value = reader.read_float();
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   vertical_fov = *value;
                                   seen |= 1U << 5U;
                                   return {};
                               }
                               if (key == "t_min") {
                                   auto value = reader.read_float();
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   t_min = *value;
                                   seen |= 1U << 6U;
                                   return {};
                               }
                               if (key == "t_max") {
                                   auto value = reader.read_float();
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   t_max = *value;
                                   seen |= 1U << 7U;
                                   return {};
                               }
                               if (key == "visibility_mask") {
                                   auto value = read_u32(reader, "Camera visibility mask");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   visibility_mask = *value;
                                   seen |= 1U << 8U;
                                   return {};
                               }
                               if (key == "current_medium") {
                                   auto value =
                                       read_identifier<renderer::MediumId>(reader, "Camera medium");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   current_medium = *value;
                                   seen |= 1U << 9U;
                                   return {};
                               }
                               return unknown_field("scene camera model", key);
                           });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if (type != "pinhole") {
        return std::unexpected(json_error(core::StatusCode::invalid_argument,
                                          "Scene camera model type is unsupported."));
    }
    auto orientation =
        renderer::OrthonormalFrame::from_orthonormal_axes(tangent, bitangent, normal);
    if (!orientation) {
        return std::unexpected(std::move(orientation.error()));
    }
    return ScenePinholeCameraDescription{
        .origin = origin,
        .orientation = *orientation,
        .vertical_field_of_view_radians = vertical_fov,
        .t_min = t_min,
        .t_max = t_max,
        .visibility_mask = visibility_mask,
        .current_medium = current_medium,
    };
}

[[nodiscard]] core::Result<SceneCameraDescription> read_camera(Reader& reader) {
    auto result = std::optional<SceneCameraDescription>{};
    auto id = renderer::CameraId{};
    auto film = renderer::FilmId{};
    auto model = std::optional<ScenePinholeCameraDescription>{};
    constexpr auto required = std::uint64_t{0x07U};
    auto status = read_closed_object(
        reader, "Scene camera", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value = read_identifier<renderer::CameraId>(reader, "Camera identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "film") {
                auto value = read_identifier<renderer::FilmId>(reader, "Camera film identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                film = *value;
                seen |= 1U << 1U;
                return {};
            }
            if (key == "model") {
                auto value = read_pinhole_camera(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                model = std::move(*value);
                seen |= 1U << 2U;
                return {};
            }
            return unknown_field("scene camera", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    result = SceneCameraDescription{.id = id, .film = film, .model = *model};
    return std::move(*result);
}

[[nodiscard]] core::Result<renderer::PathDepthLimits> read_depth_limits(Reader& reader) {
    auto result = renderer::PathDepthLimits{};
    constexpr auto required = std::uint64_t{0x1FU};
    auto status = read_closed_object(
        reader, "Path depth limits", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            auto assign = [&](std::uint32_t& target, const std::uint64_t bit) -> core::Status {
                auto value = read_u32(reader, "Path depth limit");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                target = *value;
                seen |= bit;
                return {};
            };
            if (key == "diffuse") {
                return assign(result.diffuse, 1U << 0U);
            }
            if (key == "glossy") {
                return assign(result.glossy, 1U << 1U);
            }
            if (key == "specular") {
                return assign(result.specular, 1U << 2U);
            }
            if (key == "transmission") {
                return assign(result.transmission, 1U << 3U);
            }
            if (key == "volume") {
                return assign(result.volume, 1U << 4U);
            }
            return unknown_field("path depth limits", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<renderer::RussianRoulettePolicy> read_russian_roulette(Reader& reader) {
    auto mode = std::string{};
    auto first_depth = std::optional<std::uint32_t>{};
    auto minimum = std::optional<renderer::TransportScalar>{};
    auto maximum = std::optional<renderer::TransportScalar>{};
    auto seen = std::uint64_t{};
    auto status = reader.read_object([&](const std::string_view key) -> core::Status {
        if (key == "mode") {
            auto value = reader.read_string();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            mode = std::move(*value);
            seen |= 1U << 0U;
            return {};
        }
        if (key == "first_eligible_depth") {
            auto value = read_u32(reader, "Russian roulette first eligible depth");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            first_depth = *value;
            seen |= 1U << 1U;
            return {};
        }
        if (key == "minimum_survival_probability") {
            auto value = reader.read_float();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            minimum = *value;
            seen |= 1U << 2U;
            return {};
        }
        if (key == "maximum_survival_probability") {
            auto value = reader.read_float();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            maximum = *value;
            seen |= 1U << 3U;
            return {};
        }
        return unknown_field("Russian roulette", key);
    });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if ((seen & 1U) == 0U) {
        return std::unexpected(json_error(core::StatusCode::invalid_argument,
                                          "Russian roulette is missing its mode."));
    }
    if (mode == "disabled") {
        if (seen != 1U) {
            return std::unexpected(
                json_error(core::StatusCode::invalid_argument,
                           "Disabled Russian roulette cannot contain enabled-mode parameters."));
        }
        return renderer::RussianRoulettePolicy::disabled();
    }
    if (mode != "enabled") {
        return std::unexpected(json_error(core::StatusCode::invalid_argument,
                                          "Russian roulette mode is unsupported."));
    }
    if (seen != 0x0FU || !first_depth || !minimum || !maximum) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       "Enabled Russian roulette requires every survival parameter."));
    }
    return renderer::RussianRoulettePolicy::create_enabled(*first_depth, *minimum, *maximum);
}

[[nodiscard]] core::Result<SceneRenderOptionsDescription> read_render_options(Reader& reader) {
    auto id = renderer::RenderOptionsId{};
    auto film = renderer::FilmId{};
    auto options = SceneRenderOptions{};
    constexpr auto required = std::uint64_t{0x7FFU};
    auto status = read_closed_object(
        reader, "Scene render options", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value =
                    read_identifier<renderer::RenderOptionsId>(reader, "Render-options identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "film") {
                auto value =
                    read_identifier<renderer::FilmId>(reader, "Render-options film identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                film = *value;
                seen |= 1U << 1U;
                return {};
            }
            auto assign_u32 = [&](std::uint32_t& target, const std::uint64_t bit,
                                  const std::string_view label) -> core::Status {
                auto value = read_u32(reader, label);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                target = *value;
                seen |= bit;
                return {};
            };
            if (key == "samples_per_pixel") {
                return assign_u32(options.samples_per_pixel, 1U << 2U, "Samples per pixel");
            }
            if (key == "maximum_path_depth") {
                return assign_u32(options.maximum_path_depth, 1U << 3U, "Maximum path depth");
            }
            if (key == "tile_edge_length") {
                return assign_u32(options.tile_edge_length, 1U << 4U, "Tile edge length");
            }
            if (key == "seed") {
                auto value = reader.read_u64();
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.seed = *value;
                seen |= 1U << 5U;
                return {};
            }
            if (key == "pixel_jitter") {
                auto value = read_pixel_jitter(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.pixel_jitter = *value;
                seen |= 1U << 6U;
                return {};
            }
            if (key == "mis_heuristic") {
                auto value = read_mis_heuristic(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.mis_heuristic = *value;
                seen |= 1U << 7U;
                return {};
            }
            if (key == "light_sampling_strategy") {
                auto value = read_light_sampling_strategy(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.light_sampling_strategy = *value;
                seen |= 1U << 8U;
                return {};
            }
            if (key == "depth_limits") {
                auto value = read_depth_limits(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.depth_limits = *value;
                seen |= 1U << 9U;
                return {};
            }
            if (key == "russian_roulette") {
                auto value = read_russian_roulette(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                options.roulette_policy = *value;
                seen |= 1U << 10U;
                return {};
            }
            return unknown_field("scene render options", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return SceneRenderOptionsDescription{.id = id, .film = film, .options = options};
}

struct ConstantTextureValue final {
    enum class Shape : std::uint8_t {
        scalar,
        color,
        spectrum,
    };

    Shape shape{};
    renderer::TransportScalar scalar{};
    std::array<renderer::TransportScalar, 4U> values{};
};

[[nodiscard]] core::Result<SceneConstantTexture> read_constant_texture(Reader& reader) {
    auto id = renderer::TextureId{};
    auto type = std::string{};
    auto value = std::optional<ConstantTextureValue>{};
    constexpr auto required = std::uint64_t{0x07U};
    auto status = read_closed_object(
        reader, "Scene constant texture", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto parsed = read_identifier<renderer::TextureId>(reader, "Texture identifier");
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                id = *parsed;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "type") {
                auto parsed = reader.read_string();
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                type = std::move(*parsed);
                seen |= 1U << 1U;
                return {};
            }
            if (key == "value") {
                // The discriminant may appear after the value. A scalar or exact 3/4-element
                // array is retained until the closed variant is validated below.
                if (!reader.peek_array()) {
                    auto parsed = reader.read_float();
                    if (!parsed) {
                        return std::unexpected(std::move(parsed.error()));
                    }
                    value = ConstantTextureValue{.shape = ConstantTextureValue::Shape::scalar,
                                                 .scalar = *parsed};
                } else {
                    auto elements = std::array<renderer::TransportScalar, 4U>{};
                    auto count = std::uint64_t{};
                    auto array_status =
                        reader.read_array([&](const std::uint64_t index) -> core::Status {
                            if (index >= elements.size()) {
                                return std::unexpected(json_error(
                                    core::StatusCode::invalid_argument,
                                    "A constant texture value contains too many elements."));
                            }
                            auto parsed = reader.read_float();
                            if (!parsed) {
                                return std::unexpected(std::move(parsed.error()));
                            }
                            elements[static_cast<std::size_t>(index)] = *parsed;
                            count = index + 1U;
                            return {};
                        });
                    if (!array_status) {
                        return array_status;
                    }
                    if (count != 3U && count != 4U) {
                        return std::unexpected(json_error(
                            core::StatusCode::invalid_argument,
                            "A constant texture array must contain exactly three or four values."));
                    }
                    value = ConstantTextureValue{
                        .shape = count == 3U ? ConstantTextureValue::Shape::color
                                             : ConstantTextureValue::Shape::spectrum,
                        .values = elements,
                    };
                }
                seen |= 1U << 2U;
                return {};
            }
            return unknown_field("scene constant texture", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if (type == "float" && value->shape == ConstantTextureValue::Shape::scalar) {
        auto texture = renderer::ConstantFloatTexture::create(value->scalar);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        return SceneConstantTexture{.id = id, .texture = *texture};
    }
    if (type == "color" && value->shape == ConstantTextureValue::Shape::color) {
        auto texture = renderer::ConstantColorTexture::create(
            {.red = value->values[0], .green = value->values[1], .blue = value->values[2]});
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        return SceneConstantTexture{.id = id, .texture = *texture};
    }
    if (type == "spectrum" && value->shape == ConstantTextureValue::Shape::spectrum) {
        auto texture = renderer::ConstantSpectrumTexture::create({.values = value->values});
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        return SceneConstantTexture{.id = id, .texture = *texture};
    }
    return std::unexpected(
        json_error(core::StatusCode::invalid_argument,
                   "A scene constant texture has an unsupported type or mismatched value shape."));
}

[[nodiscard]] core::Result<std::filesystem::path> path_from_utf8(const std::string& encoded) {
    try {
        auto value = std::u8string{};
        value.resize(encoded.size());
        std::ranges::transform(encoded, value.begin(),
                               [](const char byte) { return static_cast<char8_t>(byte); });
        return std::filesystem::path{value};
    } catch (const std::filesystem::filesystem_error&) {
        return std::unexpected(json_error(
            core::StatusCode::invalid_argument,
            "A scene host-image source path cannot be represented by the host filesystem."));
    }
}

[[nodiscard]] core::Result<renderer::HostImageMipChainHandle>
read_host_image(Reader& reader, DecodedBudget& budget) {
    auto source_path = std::string{};
    auto format_name = std::string{};
    auto source_color_space = renderer::TextureColorSpace::data;
    auto storage_color_space = renderer::TextureColorSpace::data;
    auto origin = std::array<std::int32_t, 2U>{};
    auto extent = std::array<std::uint32_t, 2U>{};
    auto channel_names = std::vector<std::string>{};
    auto pixels = std::vector<renderer::TransportScalar>{};
    constexpr auto required = std::uint64_t{0xFFU};
    auto status = read_closed_object(
        reader, "Scene host image", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "source_path") {
                auto parsed = reader.read_string();
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                source_path = std::move(*parsed);
                seen |= 1U << 0U;
                return {};
            }
            if (key == "format_name") {
                auto parsed = reader.read_string();
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                format_name = std::move(*parsed);
                seen |= 1U << 1U;
                return {};
            }
            if (key == "source_color_space") {
                auto parsed = read_texture_color_space(reader);
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                source_color_space = *parsed;
                seen |= 1U << 2U;
                return {};
            }
            if (key == "storage_color_space") {
                auto parsed = read_texture_color_space(reader);
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                storage_color_space = *parsed;
                seen |= 1U << 3U;
                return {};
            }
            if (key == "origin") {
                auto parsed = read_fixed_array<std::int32_t, 2U>(reader, "Host-image origin", [&] {
                    return read_i32(reader, "Host-image origin");
                });
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                origin = *parsed;
                seen |= 1U << 4U;
                return {};
            }
            if (key == "extent") {
                auto parsed = read_fixed_array<std::uint32_t, 2U>(reader, "Host-image extent", [&] {
                    return read_u32(reader, "Host-image extent");
                });
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                extent = *parsed;
                seen |= 1U << 5U;
                return {};
            }
            if (key == "channel_names") {
                auto array_status =
                    reader.read_array([&](const std::uint64_t index) -> core::Status {
                        if (index >= budget.maximum_image_scalar_values()) {
                            return std::unexpected(
                                json_error(core::StatusCode::resource_exhausted,
                                           "A host image exceeds the configured channel-name "
                                           "limit."));
                        }
                        auto parsed = reader.read_string();
                        if (!parsed) {
                            return std::unexpected(std::move(parsed.error()));
                        }
                        if (auto byte_status =
                                budget.add_bytes(sizeof(std::string) + parsed->size(),
                                                 "Decoded host-image channel names");
                            !byte_status) {
                            return byte_status;
                        }
                        channel_names.push_back(std::move(*parsed));
                        return {};
                    });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 6U;
                return {};
            }
            if (key == "pixels") {
                auto array_status = reader.read_array([&](const std::uint64_t) -> core::Status {
                    if (auto value_status = budget.add_image_scalars(1U); !value_status) {
                        return value_status;
                    }
                    auto parsed = reader.read_float();
                    if (!parsed) {
                        return std::unexpected(std::move(parsed.error()));
                    }
                    pixels.push_back(*parsed);
                    return {};
                });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 7U;
                return {};
            }
            return unknown_field("scene host image", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto byte_status = budget.add_bytes(source_path.size(), "Host-image source path metadata");
        !byte_status) {
        return std::unexpected(std::move(byte_status.error()));
    }
    if (auto byte_status = budget.add_bytes(format_name.size(), "Host-image format metadata");
        !byte_status) {
        return std::unexpected(std::move(byte_status.error()));
    }
    auto generated_metadata_bytes = std::uint64_t{};
    const auto add_metadata_bytes = [&](const std::uint64_t bytes) -> core::Status {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - generated_metadata_bytes) {
            return std::unexpected(json_error(core::StatusCode::resource_exhausted,
                                              "Host-image metadata size is not representable."));
        }
        generated_metadata_bytes += bytes;
        return {};
    };
    if (auto byte_status = add_metadata_bytes(source_path.size()); !byte_status) {
        return std::unexpected(std::move(byte_status.error()));
    }
    if (auto byte_status = add_metadata_bytes(format_name.size()); !byte_status) {
        return std::unexpected(std::move(byte_status.error()));
    }
    for (const auto& channel_name : channel_names) {
        if (auto byte_status = add_metadata_bytes(sizeof(std::string) + channel_name.size());
            !byte_status) {
            return std::unexpected(std::move(byte_status.error()));
        }
    }
    auto mip_width = extent[0];
    auto mip_height = extent[1];
    while (mip_width > 1U || mip_height > 1U) {
        mip_width = std::max(1U, mip_width / 2U);
        mip_height = std::max(1U, mip_height / 2U);
        if (auto byte_status =
                budget.add_bytes(generated_metadata_bytes, "Generated host-image metadata");
            !byte_status) {
            return std::unexpected(std::move(byte_status.error()));
        }
    }
    auto path = path_from_utf8(source_path);
    if (!path) {
        return std::unexpected(std::move(path.error()));
    }
    const auto pixel_bytes =
        static_cast<std::uint64_t>(pixels.size()) * sizeof(renderer::TransportScalar);
    auto snapshot = renderer::HostImage::create_snapshot(
        renderer::HostImageSnapshotDescription{
            .source_path = std::move(*path),
            .format_name = std::move(format_name),
            .source_color_space = source_color_space,
            .storage_color_space = storage_color_space,
            .origin_x = origin[0],
            .origin_y = origin[1],
            .width = extent[0],
            .height = extent[1],
            .channel_names = std::move(channel_names),
            .pixels = std::move(pixels),
        },
        {.maximum_pixel_bytes = std::max<std::uint64_t>(1U, pixel_bytes)});
    if (!snapshot) {
        return std::unexpected(std::move(snapshot.error()));
    }
    auto mip_chain = renderer::HostImageMipChain::generate(
        *snapshot,
        {.maximum_generated_pixel_bytes = std::max<std::uint64_t>(1U, budget.remaining_bytes())});
    if (!mip_chain) {
        return std::unexpected(std::move(mip_chain.error()));
    }
    if (auto byte_status = budget.add_bytes((*mip_chain)->generated_pixel_bytes(),
                                            "Generated host-image mip levels");
        !byte_status) {
        return std::unexpected(std::move(byte_status.error()));
    }
    return *mip_chain;
}

[[nodiscard]] core::Result<SceneHostImageTexture> read_host_image_texture(Reader& reader,
                                                                          DecodedBudget& budget) {
    auto id = renderer::TextureId{};
    auto image = renderer::HostImageMipChainHandle{};
    constexpr auto required = std::uint64_t{0x03U};
    auto status = read_closed_object(
        reader, "Scene host-image texture", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto parsed = read_identifier<renderer::TextureId>(reader, "Texture identifier");
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                id = *parsed;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "image") {
                auto parsed = read_host_image(reader, budget);
                if (!parsed) {
                    return std::unexpected(std::move(parsed.error()));
                }
                image = std::move(*parsed);
                seen |= 1U << 1U;
                return {};
            }
            return unknown_field("scene host-image texture", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return SceneHostImageTexture{.id = id, .image = std::move(image)};
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
read_triangle_mesh(Reader& reader, DecodedBudget& budget) {
    auto positions = std::vector<renderer::Point3>{};
    auto normals = std::vector<renderer::Normal3>{};
    auto texture_coordinates = std::vector<renderer::Point2>{};
    auto triangles = std::vector<TriangleVertexIndices>{};
    constexpr auto required = std::uint64_t{0x0FU};
    auto status = read_closed_object(
        reader, "Scene triangle mesh", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "positions") {
                auto array_status = reader.read_array([&](const std::uint64_t) -> core::Status {
                    if (auto limit = budget.add_mesh_vertices(1U); !limit) {
                        return limit;
                    }
                    if (auto bytes = budget.add_value<renderer::Point3>("Mesh positions"); !bytes) {
                        return bytes;
                    }
                    auto value = read_point3(reader, "Mesh position");
                    if (!value) {
                        return std::unexpected(std::move(value.error()));
                    }
                    positions.push_back(*value);
                    return {};
                });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 0U;
                return {};
            }
            if (key == "normals") {
                auto array_status =
                    reader.read_array([&](const std::uint64_t index) -> core::Status {
                        if (index >= budget.maximum_mesh_vertices()) {
                            return std::unexpected(json_error(
                                core::StatusCode::resource_exhausted,
                                "Mesh normals exceed the configured mesh-vertex limit."));
                        }
                        if (index >= positions.size() && !positions.empty()) {
                            return std::unexpected(
                                json_error(core::StatusCode::invalid_argument,
                                           "A mesh contains more normals than positions."));
                        }
                        if (auto bytes = budget.add_value<renderer::Normal3>("Mesh normals");
                            !bytes) {
                            return bytes;
                        }
                        auto value = read_normal3(reader, "Mesh normal");
                        if (!value) {
                            return std::unexpected(std::move(value.error()));
                        }
                        normals.push_back(*value);
                        return {};
                    });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 1U;
                return {};
            }
            if (key == "texture_coordinates") {
                auto array_status =
                    reader.read_array([&](const std::uint64_t index) -> core::Status {
                        if (index >= budget.maximum_mesh_vertices()) {
                            return std::unexpected(json_error(
                                core::StatusCode::resource_exhausted,
                                "Mesh texture coordinates exceed the configured mesh-vertex "
                                "limit."));
                        }
                        if (index >= positions.size() && !positions.empty()) {
                            return std::unexpected(json_error(
                                core::StatusCode::invalid_argument,
                                "A mesh contains more texture coordinates than positions."));
                        }
                        if (auto bytes =
                                budget.add_value<renderer::Point2>("Mesh texture coordinates");
                            !bytes) {
                            return bytes;
                        }
                        auto value = read_point2(reader, "Mesh texture coordinate");
                        if (!value) {
                            return std::unexpected(std::move(value.error()));
                        }
                        texture_coordinates.push_back(*value);
                        return {};
                    });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 2U;
                return {};
            }
            if (key == "triangles") {
                auto array_status = reader.read_array([&](const std::uint64_t) -> core::Status {
                    if (auto limit = budget.add_mesh_triangles(1U); !limit) {
                        return limit;
                    }
                    if (auto bytes = budget.add_value<TriangleVertexIndices>("Mesh triangles");
                        !bytes) {
                        return bytes;
                    }
                    auto indices =
                        read_fixed_array<std::uint32_t, 3U>(reader, "Triangle indices", [&] {
                            return read_u32(reader, "Triangle vertex index");
                        });
                    if (!indices) {
                        return std::unexpected(std::move(indices.error()));
                    }
                    triangles.push_back({.vertices = *indices});
                    return {};
                });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 3U;
                return {};
            }
            return unknown_field("scene triangle mesh", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    auto mesh = TriangleMesh::create(std::move(positions), std::move(normals),
                                     std::move(texture_coordinates), std::move(triangles));
    if (!mesh) {
        return std::unexpected(std::move(mesh.error()));
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<SceneGeometry> read_geometry(Reader& reader, DecodedBudget& budget) {
    auto id = renderer::GeometryId{};
    auto mesh = std::shared_ptr<const TriangleMesh>{};
    constexpr auto required = std::uint64_t{0x03U};
    auto status = read_closed_object(
        reader, "Scene geometry", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value = read_identifier<renderer::GeometryId>(reader, "Geometry identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "mesh") {
                auto value = read_triangle_mesh(reader, budget);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                mesh = std::move(*value);
                seen |= 1U << 1U;
                return {};
            }
            return unknown_field("scene geometry", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return SceneGeometry{.id = id, .mesh = std::move(mesh)};
}

[[nodiscard]] core::Result<SceneObject> read_object_record(Reader& reader) {
    auto result = SceneObject{};
    auto status =
        read_closed_object(reader, "Scene object", 1U,
                           [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
                               if (key != "id") {
                                   return unknown_field("scene object", key);
                               }
                               auto value =
                                   read_identifier<renderer::ObjectId>(reader, "Object identifier");
                               if (!value) {
                                   return std::unexpected(std::move(value.error()));
                               }
                               result.id = *value;
                               seen |= 1U;
                               return {};
                           });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<renderer::SampledWavelengths> read_wavelengths(Reader& reader) {
    auto nanometers =
        std::array<renderer::TransportScalar, renderer::TransportSpectrumSampleCount>{};
    auto pdf = std::array<renderer::TransportScalar, renderer::TransportSpectrumSampleCount>{};
    constexpr auto required = std::uint64_t{0x03U};
    auto status = read_closed_object(
        reader, "Sampled wavelengths", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "nanometers") {
                auto values = read_fixed_array<renderer::TransportScalar,
                                               renderer::TransportSpectrumSampleCount>(
                    reader, "Wavelength nanometers", [&] { return reader.read_float(); });
                if (!values) {
                    return std::unexpected(std::move(values.error()));
                }
                nanometers = *values;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "pdf") {
                auto values = read_fixed_array<renderer::TransportScalar,
                                               renderer::TransportSpectrumSampleCount>(
                    reader, "Wavelength PDFs", [&] { return reader.read_float(); });
                if (!values) {
                    return std::unexpected(std::move(values.error()));
                }
                pdf = *values;
                seen |= 1U << 1U;
                return {};
            }
            return unknown_field("sampled wavelengths", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    auto result = renderer::SampledWavelengths{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        result.samples[lane] = {
            .nanometers = nanometers[lane],
            .probability = {.value = pdf[lane],
                            .measure = renderer::ProbabilityMeasure::wavelength},
        };
    }
    return result;
}

[[nodiscard]] core::Result<renderer::HostImageEwaLimits> read_ewa_limits(Reader& reader) {
    auto result = renderer::HostImageEwaLimits{};
    constexpr auto required = std::uint64_t{0x03U};
    auto status =
        read_closed_object(reader, "Surface-map EWA limits", required,
                           [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
                               if (key == "maximum_anisotropy") {
                                   auto value = read_u32(reader, "Maximum EWA anisotropy");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   result.maximum_anisotropy = *value;
                                   seen |= 1U << 0U;
                                   return {};
                               }
                               if (key == "maximum_texel_visits") {
                                   auto value = read_u32(reader, "Maximum EWA texel visits");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   result.maximum_texel_visits = *value;
                                   seen |= 1U << 1U;
                                   return {};
                               }
                               return unknown_field("surface-map EWA limits", key);
                           });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<SceneNormalMapBinding> read_normal_map(Reader& reader) {
    auto result = SceneNormalMapBinding{};
    constexpr auto required = std::uint64_t{0xFFU};
    auto status = read_closed_object(
        reader, "Scene normal map", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "texture") {
                auto value = read_identifier<renderer::TextureId>(reader, "Normal-map texture");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.texture = *value;
                seen |= 1U << 0U;
                return {};
            }
            auto assign_channel = [&](std::uint32_t& channel,
                                      const std::uint64_t bit) -> core::Status {
                auto value = read_u32(reader, "Normal-map channel");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                channel = *value;
                seen |= bit;
                return {};
            };
            if (key == "red_channel") {
                return assign_channel(result.red_channel, 1U << 1U);
            }
            if (key == "green_channel") {
                return assign_channel(result.green_channel, 1U << 2U);
            }
            if (key == "blue_channel") {
                return assign_channel(result.blue_channel, 1U << 3U);
            }
            if (key == "y_convention") {
                auto value = read_normal_y_convention(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.y_convention = *value;
                seen |= 1U << 4U;
                return {};
            }
            if (key == "u_wrap" || key == "v_wrap") {
                auto value = read_texture_wrap(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                if (key == "u_wrap") {
                    result.u_wrap = *value;
                    seen |= 1U << 5U;
                } else {
                    result.v_wrap = *value;
                    seen |= 1U << 6U;
                }
                return {};
            }
            if (key == "ewa") {
                auto value = read_ewa_limits(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.ewa_limits = *value;
                seen |= 1U << 7U;
                return {};
            }
            return unknown_field("scene normal map", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<SceneBumpMapBinding> read_bump_map(Reader& reader) {
    auto result = SceneBumpMapBinding{};
    constexpr auto required = std::uint64_t{0x3FU};
    auto status = read_closed_object(
        reader, "Scene bump map", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "texture") {
                auto value = read_identifier<renderer::TextureId>(reader, "Bump-map texture");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.texture = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "channel") {
                auto value = read_u32(reader, "Bump-map channel");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.channel = *value;
                seen |= 1U << 1U;
                return {};
            }
            if (key == "scale") {
                auto value = reader.read_float();
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.scale = *value;
                seen |= 1U << 2U;
                return {};
            }
            if (key == "u_wrap" || key == "v_wrap") {
                auto value = read_texture_wrap(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                if (key == "u_wrap") {
                    result.u_wrap = *value;
                    seen |= 1U << 3U;
                } else {
                    result.v_wrap = *value;
                    seen |= 1U << 4U;
                }
                return {};
            }
            if (key == "ewa") {
                auto value = read_ewa_limits(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.ewa_limits = *value;
                seen |= 1U << 5U;
                return {};
            }
            return unknown_field("scene bump map", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

struct ParsedClosureComponent final {
    std::string type;
    renderer::TransportScalar probability{};
    renderer::TransportSpectrum weight{};
    std::optional<renderer::TransportScalar> roughness;
    std::optional<renderer::TransportSpectrum> eta;
    std::optional<renderer::TransportSpectrum> k;
    std::optional<renderer::TransportScalar> exterior_eta;
    std::optional<renderer::TransportScalar> interior_eta;
    std::optional<renderer::TransportScalar> alpha_x;
    std::optional<renderer::TransportScalar> alpha_y;
    std::uint64_t seen{};
};

[[nodiscard]] core::Result<ParsedClosureComponent> read_closure_component(Reader& reader) {
    auto result = ParsedClosureComponent{};
    auto status = reader.read_object([&](const std::string_view key) -> core::Status {
        if (key == "probability") {
            auto value = reader.read_float();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.probability = *value;
            result.seen |= 1U << 0U;
            return {};
        }
        if (key == "type") {
            auto value = reader.read_string();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.type = std::move(*value);
            result.seen |= 1U << 1U;
            return {};
        }
        if (key == "weight") {
            auto value = read_spectrum(reader, "Closure weight");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.weight = *value;
            result.seen |= 1U << 2U;
            return {};
        }
        auto assign_scalar = [&](std::optional<renderer::TransportScalar>& target,
                                 const std::uint64_t bit) -> core::Status {
            auto value = reader.read_float();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            target = *value;
            result.seen |= bit;
            return {};
        };
        if (key == "roughness") {
            return assign_scalar(result.roughness, 1U << 3U);
        }
        if (key == "eta") {
            auto value = read_spectrum(reader, "Closure eta");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.eta = *value;
            result.seen |= 1U << 4U;
            return {};
        }
        if (key == "k") {
            auto value = read_spectrum(reader, "Closure k");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.k = *value;
            result.seen |= 1U << 5U;
            return {};
        }
        if (key == "exterior_eta") {
            return assign_scalar(result.exterior_eta, 1U << 6U);
        }
        if (key == "interior_eta") {
            return assign_scalar(result.interior_eta, 1U << 7U);
        }
        if (key == "alpha_x") {
            return assign_scalar(result.alpha_x, 1U << 8U);
        }
        if (key == "alpha_y") {
            return assign_scalar(result.alpha_y, 1U << 9U);
        }
        return unknown_field("closure component", key);
    });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if ((result.seen & 0x07U) != 0x07U) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       "A closure component is missing probability, type, or weight."));
    }
    return result;
}

[[nodiscard]] core::Status append_closure(renderer::ClosureSet& closures,
                                          const ParsedClosureComponent& component) {
    auto append_status = renderer::ClosureAppendStatus::invalid_payload;
    auto expected_fields = std::uint64_t{0x07U};
    if (component.type == "lambertian_reflection") {
        append_status = closures.append_lambertian_reflection(component.weight);
    } else if (component.type == "rough_diffuse_reflection") {
        expected_fields |= 1U << 3U;
        if (component.roughness) {
            append_status =
                closures.append_rough_diffuse_reflection(component.weight, *component.roughness);
        }
    } else if (component.type == "rough_conductor_reflection") {
        expected_fields |= (1U << 4U) | (1U << 5U) | (1U << 8U) | (1U << 9U);
        if (component.eta && component.k && component.alpha_x && component.alpha_y) {
            append_status = closures.append_rough_conductor_reflection(
                component.weight, *component.eta, *component.k, *component.alpha_x,
                *component.alpha_y);
        }
    } else if (component.type == "rough_dielectric") {
        expected_fields |= (1U << 6U) | (1U << 7U) | (1U << 8U) | (1U << 9U);
        if (component.exterior_eta && component.interior_eta && component.alpha_x &&
            component.alpha_y) {
            append_status = closures.append_rough_dielectric(
                component.weight, *component.exterior_eta, *component.interior_eta,
                *component.alpha_x, *component.alpha_y);
        }
    } else if (component.type == "specular_reflection") {
        append_status = closures.append_specular_reflection(component.weight);
    } else if (component.type == "specular_transmission") {
        expected_fields |= (1U << 6U) | (1U << 7U);
        if (component.exterior_eta && component.interior_eta) {
            append_status = closures.append_specular_transmission(
                component.weight, *component.exterior_eta, *component.interior_eta);
        }
    } else {
        return std::unexpected(json_error(core::StatusCode::invalid_argument,
                                          "A closure component type is unsupported."));
    }
    if (component.seen != expected_fields) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       "A closure component has missing or type-incompatible parameters."));
    }
    if (append_status != renderer::ClosureAppendStatus::appended) {
        return std::unexpected(
            json_error(append_status == renderer::ClosureAppendStatus::capacity_exhausted
                           ? core::StatusCode::resource_exhausted
                           : core::StatusCode::invalid_argument,
                       "A closure component payload cannot be represented without repair."));
    }
    return {};
}

[[nodiscard]] core::Result<SceneClosureMixture> read_closure_mixture(Reader& reader) {
    auto frame_mode = SceneClosureFrameMode::shading_normal;
    auto tangent_rotation = renderer::TransportScalar{};
    auto closures = renderer::ClosureSet{};
    auto probabilities = std::array<renderer::TransportScalar, renderer::MaximumClosureCount>{};
    constexpr auto required = std::uint64_t{0x07U};
    auto status = read_closed_object(
        reader, "Scene closure mixture", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "frame_mode") {
                auto value = read_closure_frame_mode(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                frame_mode = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "tangent_rotation_radians") {
                auto value = reader.read_float();
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                tangent_rotation = *value;
                seen |= 1U << 1U;
                return {};
            }
            if (key == "components") {
                auto array_status =
                    reader.read_array([&](const std::uint64_t index) -> core::Status {
                        if (index >= renderer::MaximumClosureCount) {
                            return std::unexpected(json_error(
                                core::StatusCode::resource_exhausted,
                                "A closure mixture exceeds its fixed component capacity."));
                        }
                        auto component = read_closure_component(reader);
                        if (!component) {
                            return std::unexpected(std::move(component.error()));
                        }
                        probabilities[static_cast<std::size_t>(index)] = component->probability;
                        return append_closure(closures, *component);
                    });
                if (!array_status) {
                    return array_status;
                }
                seen |= 1U << 2U;
                return {};
            }
            return unknown_field("scene closure mixture", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return SceneClosureMixture::create(
        closures, std::span<const renderer::TransportScalar>{probabilities.data(), closures.size()},
        frame_mode, tangent_rotation);
}

[[nodiscard]] core::Result<SceneSpectralMaterial> read_spectral_material(Reader& reader) {
    auto result = SceneSpectralMaterial{};
    constexpr auto required = std::uint64_t{0x1FU};
    auto status =
        read_closed_object(reader, "Scene spectral material", required,
                           [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
                               if (key == "wavelengths") {
                                   auto value = read_wavelengths(reader);
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   result.wavelengths = *value;
                                   seen |= 1U << 0U;
                                   return {};
                               }
                               if (key == "closures") {
                                   auto value = read_closure_mixture(reader);
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   result.closure_mixture = *value;
                                   seen |= 1U << 1U;
                                   return {};
                               }
                               if (key == "emission") {
                                   auto value = read_spectrum(reader, "Material emission");
                                   if (!value) {
                                       return std::unexpected(std::move(value.error()));
                                   }
                                   result.emitted_radiance = *value;
                                   seen |= 1U << 2U;
                                   return {};
                               }
                               if (key == "normal_map") {
                                   if (reader.peek_null()) {
                                       if (auto null_status = reader.read_null(); !null_status) {
                                           return null_status;
                                       }
                                       result.normal_map.reset();
                                   } else {
                                       auto value = read_normal_map(reader);
                                       if (!value) {
                                           return std::unexpected(std::move(value.error()));
                                       }
                                       result.normal_map = *value;
                                   }
                                   seen |= 1U << 3U;
                                   return {};
                               }
                               if (key == "bump_map") {
                                   if (reader.peek_null()) {
                                       if (auto null_status = reader.read_null(); !null_status) {
                                           return null_status;
                                       }
                                       result.bump_map.reset();
                                   } else {
                                       auto value = read_bump_map(reader);
                                       if (!value) {
                                           return std::unexpected(std::move(value.error()));
                                       }
                                       result.bump_map = *value;
                                   }
                                   seen |= 1U << 4U;
                                   return {};
                               }
                               return unknown_field("scene spectral material", key);
                           });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<SceneMaterial> read_material(Reader& reader) {
    auto result = SceneMaterial{};
    constexpr auto required = std::uint64_t{0x03U};
    auto status = read_closed_object(
        reader, "Scene material", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value = read_identifier<renderer::MaterialId>(reader, "Material identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "spectral") {
                if (reader.peek_null()) {
                    if (auto null_status = reader.read_null(); !null_status) {
                        return null_status;
                    }
                    result.spectral.reset();
                } else {
                    auto value = read_spectral_material(reader);
                    if (!value) {
                        return std::unexpected(std::move(value.error()));
                    }
                    result.spectral = *value;
                }
                seen |= 1U << 1U;
                return {};
            }
            return unknown_field("scene material", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<renderer::Matrix4> read_matrix(Reader& reader) {
    auto values = read_fixed_array<renderer::TransportScalar, 16U>(
        reader, "Instance local-to-parent matrix", [&] { return reader.read_float(); });
    if (!values) {
        return std::unexpected(std::move(values.error()));
    }
    auto result = renderer::Matrix4{};
    for (auto row = std::size_t{}; row < 4U; ++row) {
        for (auto column = std::size_t{}; column < 4U; ++column) {
            result(row, column) = (*values)[row * 4U + column];
        }
    }
    return result;
}

[[nodiscard]] core::Result<SceneInstance> read_instance(Reader& reader) {
    auto result = SceneInstance{};
    constexpr auto required = std::uint64_t{0x7FU};
    auto status = read_closed_object(
        reader, "Scene instance", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "id") {
                auto value = read_identifier<renderer::InstanceId>(reader, "Instance identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.id = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "parent") {
                if (reader.peek_null()) {
                    if (auto null_status = reader.read_null(); !null_status) {
                        return null_status;
                    }
                    result.parent.reset();
                } else {
                    auto value =
                        read_identifier<renderer::InstanceId>(reader, "Parent instance identifier");
                    if (!value) {
                        return std::unexpected(std::move(value.error()));
                    }
                    result.parent = *value;
                }
                seen |= 1U << 1U;
                return {};
            }
            if (key == "object") {
                auto value = read_identifier<renderer::ObjectId>(reader, "Object identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.object = *value;
                seen |= 1U << 2U;
                return {};
            }
            if (key == "geometry") {
                auto value = read_identifier<renderer::GeometryId>(reader, "Geometry identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.geometry = *value;
                seen |= 1U << 3U;
                return {};
            }
            if (key == "material") {
                auto value = read_identifier<renderer::MaterialId>(reader, "Material identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.material = *value;
                seen |= 1U << 4U;
                return {};
            }
            if (key == "local_to_parent") {
                auto value = read_matrix(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.local_to_parent = *value;
                seen |= 1U << 5U;
                return {};
            }
            if (key == "visibility_mask") {
                auto value = read_u32(reader, "Instance visibility mask");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.visibility_mask = *value;
                seen |= 1U << 6U;
                return {};
            }
            return unknown_field("scene instance", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

struct ParsedLight final {
    renderer::LightId id{};
    std::string type;
    std::optional<renderer::Point3> position;
    std::optional<renderer::Vector3> absolute_position_error;
    std::optional<renderer::TransportSpectrum> spectral_radiant_intensity;
    std::optional<renderer::Vector3> propagation_direction;
    std::optional<renderer::TransportSpectrum> spectral_irradiance;
    std::optional<renderer::Vector3> emission_direction;
    std::optional<renderer::TransportScalar> inner_half_angle_radians;
    std::optional<renderer::TransportScalar> outer_half_angle_radians;
    std::optional<renderer::TransportSpectrum> on_axis_spectral_radiant_intensity;
    std::optional<renderer::SampledWavelengths> wavelengths;
    std::optional<renderer::TransportSpectrum> radiance;
    std::uint64_t seen{};
};

[[nodiscard]] core::Result<ParsedLight> read_light_fields(Reader& reader) {
    auto result = ParsedLight{};
    auto status = reader.read_object([&](const std::string_view key) -> core::Status {
        if (key == "id") {
            auto value = read_identifier<renderer::LightId>(reader, "Light identifier");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.id = *value;
            result.seen |= 1U << 0U;
            return {};
        }
        if (key == "type") {
            auto value = reader.read_string();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.type = std::move(*value);
            result.seen |= 1U << 1U;
            return {};
        }
        if (key == "position") {
            auto value = read_point3(reader, "Light position");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.position = *value;
            result.seen |= 1U << 2U;
            return {};
        }
        if (key == "absolute_position_error") {
            auto value = read_vector3(reader, "Light absolute position error");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.absolute_position_error = *value;
            result.seen |= 1U << 3U;
            return {};
        }
        if (key == "spectral_radiant_intensity") {
            auto value = read_spectrum(reader, "Point-light spectral radiant intensity");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.spectral_radiant_intensity = *value;
            result.seen |= 1U << 4U;
            return {};
        }
        if (key == "propagation_direction") {
            auto value = read_vector3(reader, "Directional-light propagation direction");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.propagation_direction = *value;
            result.seen |= 1U << 5U;
            return {};
        }
        if (key == "spectral_irradiance") {
            auto value = read_spectrum(reader, "Directional-light spectral irradiance");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.spectral_irradiance = *value;
            result.seen |= 1U << 6U;
            return {};
        }
        if (key == "emission_direction") {
            auto value = read_vector3(reader, "Spot-light emission direction");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.emission_direction = *value;
            result.seen |= 1U << 7U;
            return {};
        }
        auto assign_scalar = [&](std::optional<renderer::TransportScalar>& target,
                                 const std::uint64_t bit) -> core::Status {
            auto value = reader.read_float();
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            target = *value;
            result.seen |= bit;
            return {};
        };
        if (key == "inner_half_angle_radians") {
            return assign_scalar(result.inner_half_angle_radians, 1U << 8U);
        }
        if (key == "outer_half_angle_radians") {
            return assign_scalar(result.outer_half_angle_radians, 1U << 9U);
        }
        if (key == "on_axis_spectral_radiant_intensity") {
            auto value = read_spectrum(reader, "Spot-light on-axis spectral radiant intensity");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.on_axis_spectral_radiant_intensity = *value;
            result.seen |= 1U << 10U;
            return {};
        }
        if (key == "wavelengths") {
            auto value = read_wavelengths(reader);
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.wavelengths = *value;
            result.seen |= 1U << 11U;
            return {};
        }
        if (key == "radiance") {
            auto value = read_spectrum(reader, "Environment radiance");
            if (!value) {
                return std::unexpected(std::move(value.error()));
            }
            result.radiance = *value;
            result.seen |= 1U << 12U;
            return {};
        }
        return unknown_field("scene light", key);
    });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    if ((result.seen & 0x03U) != 0x03U) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       "A scene light is missing its identifier or explicit type."));
    }
    return result;
}

[[nodiscard]] core::Result<SceneLightDescription> read_light(Reader& reader) {
    auto parsed = read_light_fields(reader);
    if (!parsed) {
        return std::unexpected(std::move(parsed.error()));
    }

    auto expected = std::uint64_t{0x03U};
    auto value = SceneLightValue{};
    if (parsed->type == "point") {
        expected |= (1U << 2U) | (1U << 3U) | (1U << 4U);
        if (parsed->position && parsed->absolute_position_error &&
            parsed->spectral_radiant_intensity) {
            value = ScenePointLight{
                .position = *parsed->position,
                .absolute_position_error = *parsed->absolute_position_error,
                .spectral_radiant_intensity = *parsed->spectral_radiant_intensity,
            };
        }
    } else if (parsed->type == "directional") {
        expected |= (1U << 5U) | (1U << 6U);
        if (parsed->propagation_direction && parsed->spectral_irradiance) {
            value = SceneDirectionalLight{
                .propagation_direction = *parsed->propagation_direction,
                .spectral_irradiance = *parsed->spectral_irradiance,
            };
        }
    } else if (parsed->type == "spot") {
        expected |= (1U << 2U) | (1U << 3U) | (1U << 7U) | (1U << 8U) | (1U << 9U) | (1U << 10U);
        if (parsed->position && parsed->absolute_position_error && parsed->emission_direction &&
            parsed->inner_half_angle_radians && parsed->outer_half_angle_radians &&
            parsed->on_axis_spectral_radiant_intensity) {
            value = SceneSpotLight{
                .position = *parsed->position,
                .absolute_position_error = *parsed->absolute_position_error,
                .emission_direction = *parsed->emission_direction,
                .inner_half_angle_radians = *parsed->inner_half_angle_radians,
                .outer_half_angle_radians = *parsed->outer_half_angle_radians,
                .on_axis_spectral_radiant_intensity = *parsed->on_axis_spectral_radiant_intensity,
            };
        }
    } else if (parsed->type == "environment") {
        expected |= (1U << 11U) | (1U << 12U);
        if (parsed->wavelengths && parsed->radiance) {
            value = SceneSpectralEnvironment{
                .wavelengths = *parsed->wavelengths,
                .radiance = *parsed->radiance,
            };
        }
    } else {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument, "A scene light type is unsupported."));
    }
    if (parsed->seen != expected) {
        return std::unexpected(
            json_error(core::StatusCode::invalid_argument,
                       "A scene light has missing or type-incompatible parameters."));
    }
    return SceneLightDescription{.id = parsed->id, .light = std::move(value)};
}

struct ActiveSceneSelection final {
    renderer::FilmId film{};
    renderer::CameraId camera{};
    renderer::RenderOptionsId render_options{};
};

[[nodiscard]] core::Result<ActiveSceneSelection> read_active_selection(Reader& reader) {
    auto result = ActiveSceneSelection{};
    constexpr auto required = std::uint64_t{0x07U};
    auto status = read_closed_object(
        reader, "Active scene selection", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "film") {
                auto value = read_identifier<renderer::FilmId>(reader, "Active film identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.film = *value;
                seen |= 1U << 0U;
                return {};
            }
            if (key == "camera") {
                auto value =
                    read_identifier<renderer::CameraId>(reader, "Active camera identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.camera = *value;
                seen |= 1U << 1U;
                return {};
            }
            if (key == "render_options") {
                auto value = read_identifier<renderer::RenderOptionsId>(
                    reader, "Active render-options identifier");
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                result.render_options = *value;
                seen |= 1U << 2U;
                return {};
            }
            return unknown_field("active scene selection", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    return result;
}

[[nodiscard]] core::Result<SceneDescriptionInput> read_scene(Reader& reader,
                                                             DecodedBudget& budget) {
    auto input = SceneDescriptionInput{};
    auto active = ActiveSceneSelection{};
    constexpr auto required = std::uint64_t{0x1FFFU};
    auto status = read_closed_object(
        reader, "Scene JSON root", required,
        [&](const std::string_view key, std::uint64_t& seen) -> core::Status {
            if (key == "schema") {
                auto value = reader.read_string();
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                if (*value != SceneDescriptionJsonSchemaName) {
                    return std::unexpected(json_error(
                        core::StatusCode::incompatible,
                        "The native scene schema identifier is incompatible with Blackframe."));
                }
                seen |= 1U << 0U;
                return {};
            }
            if (key == "schema_version") {
                auto value = reader.read_u64();
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                if (*value != CurrentSceneDescriptionJsonSchemaVersion) {
                    return std::unexpected(json_error(
                        core::StatusCode::incompatible,
                        "The native scene schema version is incompatible with Blackframe."));
                }
                seen |= 1U << 1U;
                return {};
            }
            if (key == "active") {
                auto value = read_active_selection(reader);
                if (!value) {
                    return std::unexpected(std::move(value.error()));
                }
                active = *value;
                seen |= 1U << 2U;
                return {};
            }
            if (key == "films") {
                auto registry_status = read_registry(reader, input.films, budget, "Scene films",
                                                     [&] { return read_film(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 3U;
                return {};
            }
            if (key == "cameras") {
                auto registry_status = read_registry(reader, input.cameras, budget, "Scene cameras",
                                                     [&] { return read_camera(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 4U;
                return {};
            }
            if (key == "render_options") {
                auto registry_status =
                    read_registry(reader, input.render_options, budget, "Scene render options",
                                  [&] { return read_render_options(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 5U;
                return {};
            }
            if (key == "constant_textures") {
                auto registry_status = read_registry(reader, input.constant_textures, budget,
                                                     "Scene constant textures",
                                                     [&] { return read_constant_texture(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 6U;
                return {};
            }
            if (key == "host_image_textures") {
                auto registry_status = read_registry(
                    reader, input.host_image_textures, budget, "Scene host-image textures",
                    [&] { return read_host_image_texture(reader, budget); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 7U;
                return {};
            }
            if (key == "objects") {
                auto registry_status = read_registry(reader, input.objects, budget, "Scene objects",
                                                     [&] { return read_object_record(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 8U;
                return {};
            }
            if (key == "geometries") {
                auto registry_status =
                    read_registry(reader, input.geometries, budget, "Scene geometries",
                                  [&] { return read_geometry(reader, budget); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 9U;
                return {};
            }
            if (key == "materials") {
                auto registry_status =
                    read_registry(reader, input.materials, budget, "Scene materials",
                                  [&] { return read_material(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 10U;
                return {};
            }
            if (key == "instances") {
                auto registry_status =
                    read_registry(reader, input.instances, budget, "Scene instances",
                                  [&] { return read_instance(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 11U;
                return {};
            }
            if (key == "lights") {
                auto registry_status = read_registry(reader, input.lights, budget, "Scene lights",
                                                     [&] { return read_light(reader); });
                if (!registry_status) {
                    return registry_status;
                }
                seen |= 1U << 12U;
                return {};
            }
            return unknown_field("scene JSON root", key);
        });
    if (!status) {
        return std::unexpected(std::move(status.error()));
    }
    input.active_film = active.film;
    input.active_camera = active.camera;
    input.active_render_options = active.render_options;
    return input;
}

} // namespace

core::Result<SceneDescription>
deserialize_scene_description_json(const std::string_view encoded_scene,
                                   const SceneDescriptionJsonLimits limits) {
    try {
        auto created_reader = Reader::create(encoded_scene, limits);
        if (!created_reader) {
            return std::unexpected(std::move(created_reader.error()));
        }
        auto reader = std::move(*created_reader);
        auto budget = DecodedBudget{limits};
        auto input = read_scene(reader, budget);
        if (!input) {
            return std::unexpected(std::move(input.error()));
        }
        if (auto status = reader.finish(); !status) {
            return std::unexpected(std::move(status.error()));
        }
        return SceneDescription::create(std::move(*input));
    } catch (const std::bad_alloc&) {
        return std::unexpected(json_error(core::StatusCode::resource_exhausted,
                                          "Scene JSON decoding exhausted memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            json_error(core::StatusCode::resource_exhausted,
                       "A scene JSON decoded allocation size is not representable."));
    }
}

} // namespace blackframe::engine
