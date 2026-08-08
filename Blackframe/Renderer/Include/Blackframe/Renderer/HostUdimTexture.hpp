#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/TextureColorSpace.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <type_traits>

namespace blackframe::renderer {

inline constexpr std::uint32_t UdimFirstTileNumber = 1001U;
inline constexpr std::uint32_t UdimColumnsPerRow = 10U;
inline constexpr std::uint32_t UdimMaximumColumn = UdimColumnsPerRow - 1U;

struct UdimTile final {
    std::uint32_t number{};
    std::uint32_t column{};
    std::uint32_t row{};

    [[nodiscard]] constexpr bool operator==(const UdimTile&) const noexcept = default;
};

template <GeometryScalar Scalar> struct UdimAddressT final {
    UdimTile tile{};
    Point2T<Scalar> local_uv{};

    [[nodiscard]] constexpr bool operator==(const UdimAddressT&) const noexcept = default;
};

using UdimAddress = UdimAddressT<TransportScalar>;
using ReferenceUdimAddress = UdimAddressT<ReferenceScalar>;

// Each non-negative unit UV square identifies one tile. Columns are deliberately limited to
// [0, 9], because allowing column 10 would alias the first column of the next row under the
// standard 1001 + column + 10 * row numbering. Exact positive integer boundaries select the next
// tile and produce a zero local coordinate. Coordinates and the resulting tile number must be
// finite and representable; invalid input is never clamped to tile 1001.
[[nodiscard]] core::Result<UdimAddress> resolve_udim_address(Point2 uv);

[[nodiscard]] core::Result<ReferenceUdimAddress> resolve_udim_address(ReferencePoint2 uv);

template <GeometryScalar Scalar> struct HostUdimResolvedTileT final {
    UdimAddressT<Scalar> address{};
    HostImageHandle image;
};

using HostUdimResolvedTile = HostUdimResolvedTileT<TransportScalar>;
using ReferenceHostUdimResolvedTile = HostUdimResolvedTileT<ReferenceScalar>;

// A host UDIM texture is an immutable path contract, not a second image cache. Creation
// accepts one absolute path containing exactly one case-sensitive <UDIM> token and one explicit
// source color-space tag. Resolution substitutes the computed decimal tile number and asks the
// supplied HostImageCache to load that concrete file. The directory is never scanned and missing,
// corrupt, unsupported, or over-budget tiles remain explicit errors; no neighboring tile, black
// value, or literal pattern file is substituted.
class HostUdimTexture final {
  public:
    [[nodiscard]] static core::Result<HostUdimTexture>
    create(const std::filesystem::path& absolute_pattern, TextureColorSpace source_color_space);

    [[nodiscard]] const std::filesystem::path& pattern() const noexcept;
    [[nodiscard]] TextureColorSpace source_color_space() const noexcept;

    [[nodiscard]] core::Result<HostUdimResolvedTile> resolve(HostImageCache& cache,
                                                             Point2 uv) const;
    [[nodiscard]] core::Result<ReferenceHostUdimResolvedTile> resolve(HostImageCache& cache,
                                                                      ReferencePoint2 uv) const;

  private:
    HostUdimTexture(std::filesystem::path pattern, std::u8string encoded_pattern,
                    std::size_t token_offset, TextureColorSpace source_color_space) noexcept;

    std::filesystem::path pattern_;
    std::u8string encoded_pattern_;
    std::size_t token_offset_{};
    TextureColorSpace source_color_space_{TextureColorSpace::data};
};

static_assert(sizeof(UdimTile) == 3U * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<UdimTile>);
static_assert(std::is_trivially_copyable_v<UdimTile>);
static_assert(std::is_standard_layout_v<UdimAddress>);
static_assert(std::is_trivially_copyable_v<UdimAddress>);
static_assert(std::is_standard_layout_v<ReferenceUdimAddress>);
static_assert(std::is_trivially_copyable_v<ReferenceUdimAddress>);

} // namespace blackframe::renderer
