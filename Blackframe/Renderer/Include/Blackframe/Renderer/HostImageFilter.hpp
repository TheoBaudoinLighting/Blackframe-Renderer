#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/TextureWrap.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

class HostImageMipChain;

enum class TextureFilterMode : std::uint32_t {
    nearest = 0U,
    bilinear = 1U,
    bicubic = 2U,
};

template <GeometryScalar Scalar> struct TextureCoordinateDifferentialsT final {
    Scalar dudx{};
    Scalar dvdx{};
    Scalar dudy{};
    Scalar dvdy{};

    [[nodiscard]] constexpr bool
    operator==(const TextureCoordinateDifferentialsT&) const noexcept = default;
};

using TextureCoordinateDifferentials = TextureCoordinateDifferentialsT<TransportScalar>;
using ReferenceTextureCoordinateDifferentials = TextureCoordinateDifferentialsT<ReferenceScalar>;

inline constexpr std::uint32_t HostImageEwaMaximumAnisotropy = 64U;

struct HostImageEwaLimits final {
    std::uint32_t maximum_anisotropy{16U};
    std::uint32_t maximum_texel_visits{8'192U};

    [[nodiscard]] constexpr bool operator==(const HostImageEwaLimits&) const noexcept = default;
};

[[nodiscard]] constexpr bool is_valid_texture_filter_mode(const TextureFilterMode mode) noexcept {
    switch (mode) {
    case TextureFilterMode::nearest:
    case TextureFilterMode::bilinear:
    case TextureFilterMode::bicubic:
        return true;
    }
    return false;
}

// Samples one channel from an immutable host snapshot already converted to its storage color
// space. U follows stored X and V follows stored scanline Y; no implicit vertical flip, color
// conversion, alpha association, or value clamp is applied. The center of absolute texel (x, y)
// lies at ((x - origin_x + 0.5) / width, (y - origin_y + 0.5) / height). Nearest cells are
// half-open, so an exact boundary selects the higher local index. Bicubic is the separable
// interpolating Catmull-Rom kernel (Keys a = -0.5), whose signed weights may legitimately
// overshoot the source range. Every footprint tap is wrapped independently, and black taps
// contribute zero without weight renormalization. Point2 evaluates in float transport precision;
// ReferencePoint2 evaluates in double reference precision.
[[nodiscard]] core::Result<TransportScalar>
filter_host_image_channel(const HostImage& image, Point2 uv, std::uint32_t channel,
                          TextureFilterMode filter, TextureWrapMode u_wrap, TextureWrapMode v_wrap);

[[nodiscard]] core::Result<ReferenceScalar>
filter_host_image_channel(const HostImage& image, ReferencePoint2 uv, std::uint32_t channel,
                          TextureFilterMode filter, TextureWrapMode u_wrap, TextureWrapMode v_wrap);

// Elliptically weighted average filtering consumes normalized UV derivatives per raster pixel.
// The derivatives are mandatory and remain unwrapped. Every real mip extent converts them to
// texel space independently; the minor singular length brackets the two levels around one texel,
// including for odd dimensions. A footprint coarser than the complete pyramid selects the last
// level with its bounded one-texel EWA reconstruction. The minor eigenvalue is widened as needed to
// enforce the requested maximum_anisotropy in [1, 64]. Each level integrates the fixed
// exp(-2*r^2) - exp(-2) Gaussian over the unit ellipse, with a one-texel reconstruction covariance.
// A zero footprint is therefore a valid level-zero reconstruction, not an inferred derivative.
//
// Every ellipse tap applies its U/V wrap independently. Black taps contribute zero while retaining
// their weight in the denominator. The conservative bounding-box visits across both selected
// levels must fit maximum_texel_visits; oversized or unrepresentable footprints are errors and are
// never replaced by trilinear filtering or another mip. Point2 and float differentials evaluate in
// transport precision; their Reference counterparts evaluate the same contract in double.
[[nodiscard]] core::Result<TransportScalar>
filter_host_image_ewa_channel(const HostImageMipChain& mip_chain, Point2 uv,
                              TextureCoordinateDifferentials differentials, std::uint32_t channel,
                              TextureWrapMode u_wrap, TextureWrapMode v_wrap,
                              HostImageEwaLimits limits = {});

[[nodiscard]] core::Result<ReferenceScalar>
filter_host_image_ewa_channel(const HostImageMipChain& mip_chain, ReferencePoint2 uv,
                              ReferenceTextureCoordinateDifferentials differentials,
                              std::uint32_t channel, TextureWrapMode u_wrap, TextureWrapMode v_wrap,
                              HostImageEwaLimits limits = {});

static_assert(sizeof(TextureFilterMode) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<TextureFilterMode>);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::nearest) == 0U);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::bilinear) == 1U);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::bicubic) == 2U);
static_assert(sizeof(TextureCoordinateDifferentials) == 4U * sizeof(TransportScalar));
static_assert(sizeof(ReferenceTextureCoordinateDifferentials) == 4U * sizeof(ReferenceScalar));
static_assert(std::is_standard_layout_v<TextureCoordinateDifferentials>);
static_assert(std::is_standard_layout_v<ReferenceTextureCoordinateDifferentials>);
static_assert(std::is_trivially_copyable_v<TextureCoordinateDifferentials>);
static_assert(std::is_trivially_copyable_v<ReferenceTextureCoordinateDifferentials>);
static_assert(sizeof(HostImageEwaLimits) == 2U * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<HostImageEwaLimits>);
static_assert(std::is_trivially_copyable_v<HostImageEwaLimits>);

} // namespace blackframe::renderer
