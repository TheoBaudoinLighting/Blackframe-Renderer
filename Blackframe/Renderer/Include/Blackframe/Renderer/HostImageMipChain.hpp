#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {

struct HostImageMipChainLimits final {
    std::uint64_t maximum_generated_pixel_bytes{1ULL << 30U};

    [[nodiscard]] constexpr bool
    operator==(const HostImageMipChainLimits&) const noexcept = default;
};

class HostImageMipChain;
using HostImageMipChainHandle = std::shared_ptr<const HostImageMipChain>;

// An immutable host pyramid generated from a color-converted HostImage snapshot. Level zero
// shares the exact source handle. Every following level has local origin (0, 0), retains channel
// order and color-space tags, and is generated until both dimensions reach one. Each destination
// texel is the exact-area box average of its source footprint; odd dimensions use floor-halving
// without dropping or duplicating their edge area. Channels, including alpha, are averaged
// independently without association, color conversion, wrapping, or clamping.
class HostImageMipChain final {
  public:
    [[nodiscard]] static core::Result<HostImageMipChainHandle>
    generate(HostImageHandle source_image, HostImageMipChainLimits limits = {});

    [[nodiscard]] HostImageHandle source_image() const noexcept;
    [[nodiscard]] std::uint32_t level_count() const noexcept;
    [[nodiscard]] core::Result<HostImageHandle> level(std::uint32_t index) const;
    [[nodiscard]] std::uint64_t generated_pixel_bytes() const noexcept;
    [[nodiscard]] std::uint64_t total_pixel_bytes() const noexcept;

    HostImageMipChain(const HostImageMipChain&) = delete;
    HostImageMipChain& operator=(const HostImageMipChain&) = delete;
    HostImageMipChain(HostImageMipChain&&) = delete;
    HostImageMipChain& operator=(HostImageMipChain&&) = delete;

  private:
    HostImageMipChain(std::vector<HostImageHandle> levels, std::uint64_t generated_pixel_bytes,
                      std::uint64_t total_pixel_bytes) noexcept;

    std::vector<HostImageHandle> levels_;
    std::uint64_t generated_pixel_bytes_{};
    std::uint64_t total_pixel_bytes_{};
};

// Trilinear filtering always performs bilinear reconstruction within each selected mip level.
// LOD is explicit and uses the same precision as UV: finite values below zero select level zero,
// and finite values above the last level select the last level. Integer and terminal LODs evaluate
// exactly one level. Any level, coordinate, channel, or wrap failure is returned unchanged; no
// level-zero or lower-level substitute is selected. Black taps remain zero without normalization.
[[nodiscard]] core::Result<TransportScalar>
filter_host_image_trilinear_channel(const HostImageMipChain& mip_chain, Point2 uv,
                                    TransportScalar lod, std::uint32_t channel,
                                    TextureWrapMode u_wrap, TextureWrapMode v_wrap);

[[nodiscard]] core::Result<ReferenceScalar>
filter_host_image_trilinear_channel(const HostImageMipChain& mip_chain, ReferencePoint2 uv,
                                    ReferenceScalar lod, std::uint32_t channel,
                                    TextureWrapMode u_wrap, TextureWrapMode v_wrap);

static_assert(sizeof(HostImageMipChainLimits) == sizeof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<HostImageMipChainLimits>);

} // namespace blackframe::renderer
