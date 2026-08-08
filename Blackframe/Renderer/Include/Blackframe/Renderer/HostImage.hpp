#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <Blackframe/Renderer/TextureColorSpace.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

class HostImageCache;
class HostImageMipChain;
class HostImage;

using HostImageHandle = std::shared_ptr<const HostImage>;

struct HostImageSnapshotDescription final {
    std::filesystem::path source_path;
    std::string format_name;
    TextureColorSpace source_color_space{TextureColorSpace::data};
    TextureColorSpace storage_color_space{TextureColorSpace::data};
    std::int32_t origin_x{};
    std::int32_t origin_y{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::string> channel_names;
    std::vector<TransportScalar> pixels;
};

struct HostImageSnapshotLimits final {
    std::uint64_t maximum_pixel_bytes{1ULL << 30U};

    [[nodiscard]] constexpr bool
    operator==(const HostImageSnapshotLimits&) const noexcept = default;
};

static_assert(std::is_standard_layout_v<HostImageSnapshotLimits>);
static_assert(std::is_trivially_copyable_v<HostImageSnapshotLimits>);

// Immutable, pixel-major interleaved float image data. The value type is independent of the image
// decoder: renderer filtering and shading consume this snapshot without depending on OpenImageIO.
class HostImage final {
  public:
    // Reconstructs an already color-converted immutable snapshot. Pixels are interpreted directly
    // in storage_color_space; source_path is preserved as opaque provenance and is never resolved.
    // This function performs no color conversion, path lookup, or I/O.
    [[nodiscard]] static core::Result<HostImageHandle>
    create_snapshot(HostImageSnapshotDescription description, HostImageSnapshotLimits limits = {});

    [[nodiscard]] const std::filesystem::path& source_path() const noexcept;
    [[nodiscard]] std::string_view format_name() const noexcept;
    [[nodiscard]] TextureColorSpace source_color_space() const noexcept;
    [[nodiscard]] TextureColorSpace storage_color_space() const noexcept;
    [[nodiscard]] std::int32_t origin_x() const noexcept;
    [[nodiscard]] std::int32_t origin_y() const noexcept;
    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;
    [[nodiscard]] std::uint32_t channel_count() const noexcept;
    [[nodiscard]] std::span<const std::string> channel_names() const noexcept;
    [[nodiscard]] std::span<const TransportScalar> pixels() const noexcept;
    [[nodiscard]] std::uint64_t pixel_byte_size() const noexcept;

  private:
    friend class HostImageCache;
    friend class HostImageMipChain;

    HostImage(std::filesystem::path source_path, std::string format_name,
              TextureColorSpace source_color_space, TextureColorSpace storage_color_space,
              std::int32_t origin_x, std::int32_t origin_y, std::uint32_t width,
              std::uint32_t height, std::vector<std::string> channel_names,
              std::shared_ptr<const std::vector<TransportScalar>> pixels) noexcept;

    std::filesystem::path source_path_;
    std::string format_name_;
    TextureColorSpace source_color_space_{TextureColorSpace::data};
    TextureColorSpace storage_color_space_{TextureColorSpace::data};
    std::int32_t origin_x_{};
    std::int32_t origin_y_{};
    std::uint32_t width_{};
    std::uint32_t height_{};
    std::vector<std::string> channel_names_;
    std::shared_ptr<const std::vector<TransportScalar>> pixels_;
};

} // namespace blackframe::renderer
