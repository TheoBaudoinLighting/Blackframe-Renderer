#include <Blackframe/Renderer/ExrWriter.hpp>
#include <IexBaseExc.h>
#include <ImathBox.h>
#include <ImathVec.h>
#include <ImfChannelList.h>
#include <ImfChromaticities.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <ImfStringAttribute.h>
#include <array>
#include <charconv>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

inline constexpr auto maximum_metadata_value_size = std::size_t{64U * 1024U};

[[nodiscard]] core::Error make_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Status validate_metadata_value(const std::string_view field,
                                                   const std::string_view value) {
    if (value.empty()) {
        return std::unexpected(
            make_error(core::StatusCode::invalid_argument,
                       "EXR run metadata field '" + std::string{field} + "' must not be empty."));
    }
    if (value.size() > maximum_metadata_value_size) {
        return std::unexpected(
            make_error(core::StatusCode::resource_exhausted,
                       "EXR run metadata field '" + std::string{field} + "' exceeds 64 KiB."));
    }
    if (value.find('\0') != std::string_view::npos) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "EXR run metadata field '" + std::string{field} +
                                              "' contains an embedded NUL."));
    }
    return {};
}

[[nodiscard]] core::Status validate_run_metadata(const ExrRunMetadata& metadata) {
    const auto values = std::array{
        std::pair{"scene", std::string_view{metadata.scene}},
        std::pair{"commit", std::string_view{metadata.commit}},
        std::pair{"options", std::string_view{metadata.options}},
        std::pair{"backend", std::string_view{metadata.backend}},
        std::pair{"capabilities", std::string_view{metadata.capabilities}},
        std::pair{"asset_hashes", std::string_view{metadata.asset_hashes}},
    };
    for (const auto& [field, value] : values) {
        const auto status = validate_metadata_value(field, value);
        if (!status.has_value()) {
            return status;
        }
    }
    return {};
}

[[nodiscard]] core::Status validate_output_path(const std::filesystem::path& output_path) {
    if (output_path.empty() || !output_path.is_absolute()) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "The EXR output path must be explicit and absolute."));
    }

    auto extension = output_path.extension().string();
    for (auto& character : extension) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    if (extension != ".exr") {
        return std::unexpected(
            make_error(core::StatusCode::invalid_argument,
                       "The scene-linear image output path must use the .exr extension."));
    }

    std::error_code error;
    const auto parent = output_path.parent_path();
    if (!std::filesystem::is_directory(parent, error) || error) {
        return std::unexpected(
            make_error(core::StatusCode::not_found,
                       "The EXR output directory does not exist or cannot be inspected."));
    }
    error.clear();
    const auto output_exists = std::filesystem::exists(output_path, error);
    if (error) {
        return std::unexpected(
            make_error(core::StatusCode::platform_error,
                       "The EXR output path cannot be inspected: " + error.message()));
    }
    if (output_exists && std::filesystem::is_directory(output_path, error)) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "The EXR output path names a directory."));
    }
    if (error) {
        return std::unexpected(
            make_error(core::StatusCode::platform_error,
                       "The EXR output path cannot be inspected: " + error.message()));
    }
    return {};
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

void insert_string_attribute(OPENEXR_IMF_NAMESPACE::Header& header, const char* const name,
                             const std::string_view value) {
    header.insert(name,
                  OPENEXR_IMF_NAMESPACE::StringAttribute{std::string{value.data(), value.size()}});
}

[[nodiscard]] core::Status attach_run_metadata(OPENEXR_IMF_NAMESPACE::Header& header,
                                               const ExrRunMetadata& metadata) {
    auto seed = std::array<char, std::numeric_limits<std::uint64_t>::digits10 + 3>{};
    const auto [seed_end, seed_error] =
        std::to_chars(seed.data(), seed.data() + seed.size() - 1, metadata.seed);
    if (seed_error != std::errc{}) {
        return std::unexpected(
            make_error(core::StatusCode::internal_error, "The EXR seed could not be serialized."));
    }

    insert_string_attribute(header, "blackframe.run_metadata_version", "1");
    insert_string_attribute(header, "blackframe.color_space", "scene-linear-srgb");
    insert_string_attribute(header, "blackframe.scene", metadata.scene);
    insert_string_attribute(
        header, "blackframe.seed",
        std::string_view{seed.data(), static_cast<std::size_t>(seed_end - seed.data())});
    insert_string_attribute(header, "blackframe.commit", metadata.commit);
    insert_string_attribute(header, "blackframe.options", metadata.options);
    insert_string_attribute(header, "blackframe.backend", metadata.backend);
    insert_string_attribute(header, "blackframe.capabilities", metadata.capabilities);
    insert_string_attribute(header, "blackframe.asset_hashes", metadata.asset_hashes);
    return {};
}

struct ExrPlanes final {
    std::vector<TransportScalar> blue;
    std::vector<TransportScalar> green;
    std::vector<TransportScalar> red;
};

[[nodiscard]] core::Result<ExrPlanes> resolve_film(const Film& film) {
    auto planes = ExrPlanes{
        .blue = std::vector<TransportScalar>(film.pixel_count()),
        .green = std::vector<TransportScalar>(film.pixel_count()),
        .red = std::vector<TransportScalar>(film.pixel_count()),
    };

    const auto crop = film.crop();
    auto pixel_index = std::size_t{};
    for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
            const auto resolved = film.resolved_pixel(x, y);
            if (!resolved.has_value()) {
                return std::unexpected(make_error(
                    core::StatusCode::invalid_argument,
                    "Every pixel in the active film crop must be resolved before EXR output."));
            }
            planes.red[pixel_index] = resolved->red;
            planes.green[pixel_index] = resolved->green;
            planes.blue[pixel_index] = resolved->blue;
            ++pixel_index;
        }
    }
    return planes;
}

[[nodiscard]] OPENEXR_IMF_NAMESPACE::Header make_header(const Film& film) {
    const auto crop = film.crop();
    const auto extent = film.extent();
    const auto display_window = IMATH_NAMESPACE::Box2i{
        IMATH_NAMESPACE::V2i{0, 0}, IMATH_NAMESPACE::V2i{static_cast<int>(extent.width - 1U),
                                                         static_cast<int>(extent.height - 1U)}};
    const auto data_window = IMATH_NAMESPACE::Box2i{
        IMATH_NAMESPACE::V2i{static_cast<int>(crop.minimum_x), static_cast<int>(crop.minimum_y)},
        IMATH_NAMESPACE::V2i{static_cast<int>(crop.maximum_x - 1U),
                             static_cast<int>(crop.maximum_y - 1U)}};

    auto header = OPENEXR_IMF_NAMESPACE::Header{display_window,
                                                data_window,
                                                1.0F,
                                                IMATH_NAMESPACE::V2f{0.0F, 0.0F},
                                                1.0F,
                                                OPENEXR_IMF_NAMESPACE::INCREASING_Y,
                                                OPENEXR_IMF_NAMESPACE::ZIP_COMPRESSION};
    header.channels().insert("B", OPENEXR_IMF_NAMESPACE::Channel{OPENEXR_IMF_NAMESPACE::FLOAT});
    header.channels().insert("G", OPENEXR_IMF_NAMESPACE::Channel{OPENEXR_IMF_NAMESPACE::FLOAT});
    header.channels().insert("R", OPENEXR_IMF_NAMESPACE::Channel{OPENEXR_IMF_NAMESPACE::FLOAT});
    header.insert("chromaticities", OPENEXR_IMF_NAMESPACE::ChromaticitiesAttribute{
                                        OPENEXR_IMF_NAMESPACE::Chromaticities{}});
    return header;
}

[[nodiscard]] OPENEXR_IMF_NAMESPACE::FrameBuffer make_frame_buffer(ExrPlanes& planes,
                                                                   const FilmCrop crop) {
    const auto data_window = IMATH_NAMESPACE::Box2i{
        IMATH_NAMESPACE::V2i{static_cast<int>(crop.minimum_x), static_cast<int>(crop.minimum_y)},
        IMATH_NAMESPACE::V2i{static_cast<int>(crop.maximum_x - 1U),
                             static_cast<int>(crop.maximum_y - 1U)}};
    auto frame_buffer = OPENEXR_IMF_NAMESPACE::FrameBuffer{};
    frame_buffer.insert("B", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                planes.blue.data(), data_window));
    frame_buffer.insert("G", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                planes.green.data(), data_window));
    frame_buffer.insert("R", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                planes.red.data(), data_window));
    return frame_buffer;
}

[[nodiscard]] core::Status write_scene_linear_exr_impl(const Film& film,
                                                       const std::filesystem::path& output_path,
                                                       const ExrRunMetadata& metadata) {
    const auto metadata_status = validate_run_metadata(metadata);
    if (!metadata_status.has_value()) {
        return metadata_status;
    }
    const auto path_status = validate_output_path(output_path);
    if (!path_status.has_value()) {
        return path_status;
    }

    auto planes = resolve_film(film);
    if (!planes.has_value()) {
        return std::unexpected(planes.error());
    }
    auto header = make_header(film);
    const auto metadata_attachment_status = attach_run_metadata(header, metadata);
    if (!metadata_attachment_status.has_value()) {
        return metadata_attachment_status;
    }
    auto frame_buffer = make_frame_buffer(*planes, film.crop());

    auto output = OPENEXR_IMF_NAMESPACE::OutputFile{utf8_path(output_path).c_str(), header, 1};
    output.setFrameBuffer(frame_buffer);
    output.writePixels(static_cast<int>(film.crop().height()));
    return {};
}

} // namespace

core::Status write_scene_linear_exr(const Film& film, const std::filesystem::path& output_path,
                                    const ExrRunMetadata& metadata) {
    try {
        return write_scene_linear_exr_impl(film, output_path, metadata);
    } catch (const std::bad_alloc&) {
        return std::unexpected(
            make_error(core::StatusCode::resource_exhausted, "EXR output exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "EXR output exceeds a host container limit."));
    } catch (const IEX_NAMESPACE::BaseExc& error) {
        return std::unexpected(make_error(core::StatusCode::platform_error,
                                          "OpenEXR output failed: " + std::string{error.what()}));
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(make_error(core::StatusCode::platform_error,
                                          "EXR output path failure: " + std::string{error.what()}));
    } catch (const std::exception& error) {
        return std::unexpected(
            make_error(core::StatusCode::internal_error,
                       "EXR output failed unexpectedly: " + std::string{error.what()}));
    }
}

} // namespace blackframe::renderer
