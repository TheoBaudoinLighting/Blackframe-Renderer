#include <Blackframe/Renderer/DisplayTransform.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

inline constexpr auto channel_count = std::size_t{3};

struct EncodedPng final {
    std::vector<std::uint8_t> bytes;
    bool allocation_failed{};
};

[[nodiscard]] core::Error make_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Status validate_output_path(const std::filesystem::path& output_path) {
    if (output_path.empty() || !output_path.is_absolute()) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "The PNG preview path must be explicit and absolute."));
    }

    auto extension = output_path.extension().string();
    for (auto& character : extension) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    if (extension != ".png") {
        return std::unexpected(
            make_error(core::StatusCode::invalid_argument,
                       "The fixed-display preview output path must use the .png extension."));
    }

    std::error_code error;
    const auto parent = output_path.parent_path();
    if (!std::filesystem::is_directory(parent, error) || error) {
        return std::unexpected(
            make_error(core::StatusCode::not_found,
                       "The PNG preview output directory does not exist or cannot be inspected."));
    }

    error.clear();
    const auto output_exists = std::filesystem::exists(output_path, error);
    if (error) {
        return std::unexpected(
            make_error(core::StatusCode::platform_error,
                       "The PNG preview output path cannot be inspected: " + error.message()));
    }
    if (output_exists && std::filesystem::is_directory(output_path, error)) {
        return std::unexpected(make_error(core::StatusCode::invalid_argument,
                                          "The PNG preview output path names a directory."));
    }
    if (error) {
        return std::unexpected(
            make_error(core::StatusCode::platform_error,
                       "The PNG preview output path cannot be inspected: " + error.message()));
    }
    return {};
}

[[nodiscard]] std::uint8_t quantize_display_channel(const ReferenceScalar display_value) {
    const auto quantized = std::floor(display_value * 255.0 + 0.5);
    return static_cast<std::uint8_t>(std::clamp(quantized, 0.0, 255.0));
}

void append_encoded_bytes(void* const context, void* const data, const int size) noexcept {
    auto& encoded = *static_cast<EncodedPng*>(context);
    if (encoded.allocation_failed || size <= 0) {
        return;
    }

    const auto* const first = static_cast<const std::uint8_t*>(data);
    try {
        encoded.bytes.insert(encoded.bytes.end(), first, first + size);
    } catch (const std::bad_alloc&) {
        encoded.allocation_failed = true;
    } catch (const std::length_error&) {
        encoded.allocation_failed = true;
    }
}

[[nodiscard]] core::Result<std::vector<std::uint8_t>> resolve_preview_pixels(const Film& film) {
    const auto crop = film.crop();
    if (crop.width() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        crop.height() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        crop.width() >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) / channel_count) {
        return std::unexpected(
            make_error(core::StatusCode::resource_exhausted,
                       "The film crop exceeds the PNG encoder dimension or stride range."));
    }

    const auto pixel_count =
        static_cast<std::size_t>(crop.width()) * static_cast<std::size_t>(crop.height());
    if (pixel_count > std::numeric_limits<std::size_t>::max() / channel_count) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The PNG preview byte count is not representable."));
    }

    try {
        auto pixels = std::vector<std::uint8_t>(pixel_count * channel_count);
        auto destination = std::size_t{};
        for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
            for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
                const auto color = film.resolved_pixel(x, y);
                if (!color.has_value()) {
                    return std::unexpected(color.error());
                }
                const auto display = apply_fixed_display_transform(*color);
                if (!display.has_value()) {
                    return std::unexpected(display.error());
                }
                pixels[destination++] = quantize_display_channel(display->red);
                pixels[destination++] = quantize_display_channel(display->green);
                pixels[destination++] = quantize_display_channel(display->blue);
            }
        }
        return pixels;
    } catch (const std::bad_alloc&) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The PNG preview pixel buffer cannot be allocated."));
    } catch (const std::length_error&) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The PNG preview pixel buffer cannot be represented."));
    }
}

[[nodiscard]] core::Result<std::vector<std::uint8_t>>
encode_png(const FilmCrop crop, const std::vector<std::uint8_t>& pixels) {
    auto encoded = EncodedPng{};
    const auto width = static_cast<int>(crop.width());
    const auto height = static_cast<int>(crop.height());
    const auto stride = static_cast<int>(static_cast<std::size_t>(crop.width()) * channel_count);
    const auto result =
        stbi_write_png_to_func(append_encoded_bytes, &encoded, width, height,
                               static_cast<int>(channel_count), pixels.data(), stride);
    if (encoded.allocation_failed) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The encoded PNG buffer cannot be allocated."));
    }
    if (result == 0 || encoded.bytes.empty()) {
        return std::unexpected(make_error(core::StatusCode::internal_error,
                                          "The PNG encoder rejected the preview image."));
    }
    return std::move(encoded.bytes);
}

[[nodiscard]] core::Status write_encoded_png(const std::filesystem::path& output_path,
                                             const std::vector<std::uint8_t>& encoded) {
    if (encoded.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The encoded PNG size exceeds the file API range."));
    }

    auto output = std::ofstream{output_path, std::ios::binary | std::ios::trunc};
    if (!output.is_open()) {
        return std::unexpected(make_error(core::StatusCode::platform_error,
                                          "The PNG preview output file cannot be opened."));
    }
    output.write(reinterpret_cast<const char*>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    output.close();
    if (!output) {
        std::error_code cleanup_error;
        std::filesystem::remove(output_path, cleanup_error);
        return std::unexpected(make_error(core::StatusCode::platform_error,
                                          "The encoded PNG preview cannot be written."));
    }
    return {};
}

} // namespace

core::Status write_png_preview(const Film& film, const std::filesystem::path& output_path) {
    try {
        const auto path_status = validate_output_path(output_path);
        if (!path_status.has_value()) {
            return path_status;
        }

        const auto pixels = resolve_preview_pixels(film);
        if (!pixels.has_value()) {
            return std::unexpected(pixels.error());
        }
        const auto encoded = encode_png(film.crop(), *pixels);
        if (!encoded.has_value()) {
            return std::unexpected(encoded.error());
        }
        return write_encoded_png(output_path, *encoded);
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(
            make_error(core::StatusCode::platform_error,
                       "The PNG preview path failed: " + std::string{error.what()}));
    } catch (const std::bad_alloc&) {
        return std::unexpected(make_error(core::StatusCode::resource_exhausted,
                                          "The PNG preview writer exhausted memory."));
    } catch (const std::exception& error) {
        return std::unexpected(
            make_error(core::StatusCode::internal_error,
                       "The PNG preview writer failed: " + std::string{error.what()}));
    } catch (...) {
        return std::unexpected(make_error(core::StatusCode::internal_error,
                                          "The PNG preview writer failed unexpectedly."));
    }
}

} // namespace blackframe::renderer
