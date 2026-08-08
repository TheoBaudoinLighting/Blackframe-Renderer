#include <Blackframe/Renderer/HostImage.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace blackframe::renderer {
namespace {

[[nodiscard]] core::Error snapshot_error(const core::StatusCode code,
                                         const std::string_view message) {
    return core::Error{
        .code = code,
        .message = std::string{message},
    };
}

[[nodiscard]] core::Result<std::uint64_t> checked_product(const std::uint64_t left,
                                                          const std::uint64_t right,
                                                          const std::string_view diagnostic) {
    if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
        return std::unexpected(snapshot_error(core::StatusCode::resource_exhausted, diagnostic));
    }
    return left * right;
}

[[nodiscard]] bool compatible_color_spaces(const TextureColorSpace source,
                                           const TextureColorSpace storage) noexcept {
    switch (source) {
    case TextureColorSpace::data:
        return storage == TextureColorSpace::data;
    case TextureColorSpace::srgb:
    case TextureColorSpace::scene_linear_srgb:
        return storage == TextureWorkingColorSpace;
    }
    return false;
}

[[nodiscard]] bool
has_exact_color_channels(const std::span<const std::string> channel_names) noexcept {
    auto red_count = std::size_t{};
    auto green_count = std::size_t{};
    auto blue_count = std::size_t{};
    for (const auto& name : channel_names) {
        red_count += static_cast<std::size_t>(name == "R");
        green_count += static_cast<std::size_t>(name == "G");
        blue_count += static_cast<std::size_t>(name == "B");
    }
    return red_count == 1U && green_count == 1U && blue_count == 1U;
}

} // namespace

core::Result<HostImageHandle> HostImage::create_snapshot(HostImageSnapshotDescription description,
                                                         const HostImageSnapshotLimits limits) {
    try {
        if (limits.maximum_pixel_bytes == 0U) {
            return std::unexpected(
                snapshot_error(core::StatusCode::invalid_argument,
                               "A host image snapshot pixel-byte limit must be non-zero."));
        }
        if (description.source_path.empty()) {
            return std::unexpected(
                snapshot_error(core::StatusCode::invalid_argument,
                               "A host image snapshot source path metadata must be non-empty."));
        }
        if (description.format_name.empty()) {
            return std::unexpected(snapshot_error(core::StatusCode::invalid_argument,
                                                  "A host image snapshot format must be named."));
        }
        if (!is_valid_texture_color_space(description.source_color_space) ||
            !is_valid_texture_color_space(description.storage_color_space) ||
            !compatible_color_spaces(description.source_color_space,
                                     description.storage_color_space)) {
            return std::unexpected(snapshot_error(
                core::StatusCode::invalid_argument,
                "A host image snapshot requires compatible explicit source and storage color "
                "spaces."));
        }
        if (description.width == 0U || description.height == 0U ||
            description.channel_names.empty()) {
            return std::unexpected(
                snapshot_error(core::StatusCode::invalid_argument,
                               "A host image snapshot requires non-zero dimensions and channels."));
        }
        if (description.channel_names.size() > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(snapshot_error(
                core::StatusCode::resource_exhausted,
                "A host image snapshot channel count exceeds the 32-bit channel domain."));
        }
        constexpr auto maximum_channel_name_bytes = std::size_t{255U};
        if (!std::ranges::all_of(description.channel_names, [](const auto& name) {
                return !name.empty() && name.size() <= maximum_channel_name_bytes;
            })) {
            return std::unexpected(snapshot_error(
                core::StatusCode::invalid_argument,
                "A host image snapshot channel name must contain between 1 and 255 bytes."));
        }
        if (is_color_texture_space(description.source_color_space) &&
            !has_exact_color_channels(description.channel_names)) {
            return std::unexpected(snapshot_error(
                core::StatusCode::invalid_argument,
                "A color host image snapshot requires exactly one R, G, and B channel."));
        }

        const auto pixel_count =
            checked_product(description.width, description.height,
                            "A host image snapshot pixel count is not representable.");
        if (!pixel_count) {
            return std::unexpected(pixel_count.error());
        }
        const auto element_count =
            checked_product(*pixel_count, description.channel_names.size(),
                            "A host image snapshot element count is not representable.");
        if (!element_count) {
            return std::unexpected(element_count.error());
        }
        const auto pixel_bytes =
            checked_product(*element_count, sizeof(TransportScalar),
                            "A host image snapshot pixel byte count is not representable.");
        if (!pixel_bytes) {
            return std::unexpected(pixel_bytes.error());
        }
        if (*element_count > std::numeric_limits<std::size_t>::max()) {
            return std::unexpected(snapshot_error(
                core::StatusCode::resource_exhausted,
                "A host image snapshot element count exceeds the host address space."));
        }
        if (*pixel_bytes > limits.maximum_pixel_bytes) {
            return std::unexpected(
                snapshot_error(core::StatusCode::resource_exhausted,
                               "A host image snapshot exceeds its configured pixel-byte limit."));
        }
        if (description.pixels.size() != static_cast<std::size_t>(*element_count)) {
            return std::unexpected(
                snapshot_error(core::StatusCode::invalid_argument,
                               "A host image snapshot requires exact pixel storage."));
        }
        if (!std::ranges::all_of(description.pixels,
                                 [](const auto value) { return std::isfinite(value); })) {
            return std::unexpected(
                snapshot_error(core::StatusCode::invalid_argument,
                               "A host image snapshot requires finite pixel values."));
        }

        auto pixels =
            std::make_shared<const std::vector<TransportScalar>>(std::move(description.pixels));
        return HostImageHandle{new HostImage{
            std::move(description.source_path), std::move(description.format_name),
            description.source_color_space, description.storage_color_space, description.origin_x,
            description.origin_y, description.width, description.height,
            std::move(description.channel_names), std::move(pixels)}};
    } catch (const std::bad_alloc&) {
        return std::unexpected(snapshot_error(core::StatusCode::resource_exhausted,
                                              "A host image snapshot exhausted memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            snapshot_error(core::StatusCode::resource_exhausted,
                           "A host image snapshot allocation size is not representable."));
    }
}

HostImage::HostImage(std::filesystem::path source_path, std::string format_name,
                     const TextureColorSpace source_color_space,
                     const TextureColorSpace storage_color_space, const std::int32_t origin_x,
                     const std::int32_t origin_y, const std::uint32_t width,
                     const std::uint32_t height, std::vector<std::string> channel_names,
                     std::shared_ptr<const std::vector<TransportScalar>> pixels) noexcept
    : source_path_{std::move(source_path)}, format_name_{std::move(format_name)},
      source_color_space_{source_color_space}, storage_color_space_{storage_color_space},
      origin_x_{origin_x}, origin_y_{origin_y}, width_{width}, height_{height},
      channel_names_{std::move(channel_names)}, pixels_{std::move(pixels)} {}

const std::filesystem::path& HostImage::source_path() const noexcept {
    return source_path_;
}

std::string_view HostImage::format_name() const noexcept {
    return format_name_;
}

TextureColorSpace HostImage::source_color_space() const noexcept {
    return source_color_space_;
}

TextureColorSpace HostImage::storage_color_space() const noexcept {
    return storage_color_space_;
}

std::int32_t HostImage::origin_x() const noexcept {
    return origin_x_;
}

std::int32_t HostImage::origin_y() const noexcept {
    return origin_y_;
}

std::uint32_t HostImage::width() const noexcept {
    return width_;
}

std::uint32_t HostImage::height() const noexcept {
    return height_;
}

std::uint32_t HostImage::channel_count() const noexcept {
    return static_cast<std::uint32_t>(channel_names_.size());
}

std::span<const std::string> HostImage::channel_names() const noexcept {
    return channel_names_;
}

std::span<const TransportScalar> HostImage::pixels() const noexcept {
    return *pixels_;
}

std::uint64_t HostImage::pixel_byte_size() const noexcept {
    return static_cast<std::uint64_t>(pixels_->size()) * sizeof(TransportScalar);
}

} // namespace blackframe::renderer
