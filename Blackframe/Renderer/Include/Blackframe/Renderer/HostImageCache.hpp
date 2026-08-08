#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/HostImage.hpp>
#include <Blackframe/Renderer/TextureColorSpace.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

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

// The cache owns immutable, pixel-major interleaved float snapshots of the
// primary subimage at its highest-resolution mip. A canonical absolute path and
// an explicit source color-space tag form one cache key; file changes become
// visible only through a new cache instance. Data and scene-linear sRGB values
// remain bitwise unchanged. sRGB color channels are decoded into Blackframe's
// scene-linear sRGB working space with the fixed IEC transfer function. Loading
// never infers a tag, remaps channels, associates alpha, clamps, or substitutes
// an image. A color tag requires exactly one channel named R, G, and B; every
// other channel remains unchanged.
//
// The selected reader's canonical channel order and names are exposed. The accepted suffixes are
// .exr, .png, .pbm, .pgm, .ppm, .pnm, and .pfm; a suffix selects exactly one embedded reader and
// content is never tried as another format. Concurrent loads and observers are supported. Moving
// or destroying the cache requires external synchronization.
class HostImageCache final {
  public:
    [[nodiscard]] static core::Result<HostImageCache> create(HostImageCacheLimits limits = {});

    ~HostImageCache();
    HostImageCache(HostImageCache&&) noexcept;
    HostImageCache& operator=(HostImageCache&&) noexcept;
    HostImageCache(const HostImageCache&) = delete;
    HostImageCache& operator=(const HostImageCache&) = delete;

    [[nodiscard]] core::Result<HostImageHandle> load(const std::filesystem::path& absolute_path,
                                                     TextureColorSpace source_color_space);

    [[nodiscard]] core::Result<std::size_t> entry_count() const;
    [[nodiscard]] core::Result<std::uint64_t> resident_pixel_bytes() const;
    [[nodiscard]] core::Result<HostImageCacheLimits> limits() const;

  private:
    struct Impl;

    explicit HostImageCache(std::unique_ptr<Impl> implementation) noexcept;

    std::unique_ptr<Impl> implementation_;
};

} // namespace blackframe::renderer
