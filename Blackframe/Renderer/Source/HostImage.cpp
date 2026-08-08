#include <Blackframe/Renderer/HostImage.hpp>
#include <utility>

namespace blackframe::renderer {

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
