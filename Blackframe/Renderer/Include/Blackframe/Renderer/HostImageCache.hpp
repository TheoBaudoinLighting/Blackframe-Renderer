#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace blackframe::renderer {

struct HostImageCacheLimits final {
    std::uint32_t maximum_width{32'768U};
    std::uint32_t maximum_height{32'768U};
    std::uint32_t maximum_channels{64U};
    std::uint64_t maximum_image_pixel_bytes{1ULL << 30U};
    std::uint64_t maximum_resident_pixel_bytes{4ULL << 30U};
    std::size_t maximum_entries{4'096U};

    [[nodiscard]] constexpr bool operator==(const HostImageCacheLimits&) const noexcept = default;
};

class HostImage final {
  public:
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept;
    [[nodiscard]] std::string_view format_name() const noexcept;
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

    HostImage(std::filesystem::path source_path, std::string format_name, std::int32_t origin_x,
              std::int32_t origin_y, std::uint32_t width, std::uint32_t height,
              std::vector<std::string> channel_names, std::vector<TransportScalar> pixels) noexcept;

    std::filesystem::path source_path_;
    std::string format_name_;
    std::int32_t origin_x_{};
    std::int32_t origin_y_{};
    std::uint32_t width_{};
    std::uint32_t height_{};
    std::vector<std::string> channel_names_;
    std::vector<TransportScalar> pixels_;
};

using HostImageHandle = std::shared_ptr<const HostImage>;

// The cache owns immutable, pixel-major interleaved float snapshots of the
// primary subimage at its highest-resolution mip. A canonical absolute path is
// one cache key; file changes become visible only through a new cache instance.
// Loading performs numeric conversion only. It never applies a color transform,
// an additional channel remap, alpha association, clamp, or substitute image.
// The selected reader's canonical channel order and names are exposed. The
// accepted suffixes are .exr, .png, .pbm, .pgm, .ppm, .pnm, and .pfm; a suffix
// selects exactly one embedded reader and content is never tried as another
// format. Concurrent loads and observers are supported. Moving or destroying
// the cache requires external synchronization.
class HostImageCache final {
  public:
    [[nodiscard]] static core::Result<HostImageCache> create(HostImageCacheLimits limits = {});

    ~HostImageCache();
    HostImageCache(HostImageCache&&) noexcept;
    HostImageCache& operator=(HostImageCache&&) noexcept;
    HostImageCache(const HostImageCache&) = delete;
    HostImageCache& operator=(const HostImageCache&) = delete;

    [[nodiscard]] core::Result<HostImageHandle> load(const std::filesystem::path& absolute_path);

    [[nodiscard]] core::Result<std::size_t> entry_count() const;
    [[nodiscard]] core::Result<std::uint64_t> resident_pixel_bytes() const;
    [[nodiscard]] core::Result<HostImageCacheLimits> limits() const;

  private:
    struct Impl;

    explicit HostImageCache(std::unique_ptr<Impl> implementation) noexcept;

    std::unique_ptr<Impl> implementation_;
};

} // namespace blackframe::renderer
