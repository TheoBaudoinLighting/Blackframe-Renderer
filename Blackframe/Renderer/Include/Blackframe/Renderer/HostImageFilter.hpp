#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/TextureWrap.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

enum class TextureFilterMode : std::uint32_t {
    nearest = 0U,
    bilinear = 1U,
    bicubic = 2U,
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

static_assert(sizeof(TextureFilterMode) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<TextureFilterMode>);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::nearest) == 0U);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::bilinear) == 1U);
static_assert(static_cast<std::uint32_t>(TextureFilterMode::bicubic) == 2U);

} // namespace blackframe::renderer
