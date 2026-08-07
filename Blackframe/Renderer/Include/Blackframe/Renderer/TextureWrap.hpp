#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {

enum class TextureWrapMode : std::uint32_t {
    repeat = 0U,
    clamp = 1U,
    mirror = 2U,
    black = 3U,
};

[[nodiscard]] constexpr bool is_valid_texture_wrap_mode(const TextureWrapMode mode) noexcept {
    switch (mode) {
    case TextureWrapMode::repeat:
    case TextureWrapMode::clamp:
    case TextureWrapMode::mirror:
    case TextureWrapMode::black:
        return true;
    }
    return false;
}

namespace texture_wrap_detail {

[[nodiscard]] inline core::Error invalid_wrap_request(const char* const message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = message,
    };
}

[[nodiscard]] constexpr std::uint64_t unsigned_magnitude(const std::int64_t value) noexcept {
    if (value >= 0) {
        return static_cast<std::uint64_t>(value);
    }
    return static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

[[nodiscard]] constexpr std::uint64_t positive_modulo(const std::int64_t value,
                                                      const std::uint64_t modulus) noexcept {
    const auto remainder = unsigned_magnitude(value) % modulus;
    return value >= 0 || remainder == 0U ? remainder : modulus - remainder;
}

[[nodiscard]] constexpr std::uint64_t relative_modulo(const std::int64_t index,
                                                      const std::int64_t origin,
                                                      const std::uint64_t modulus) noexcept {
    const auto index_remainder = positive_modulo(index, modulus);
    const auto origin_remainder = positive_modulo(origin, modulus);
    return index_remainder >= origin_remainder ? index_remainder - origin_remainder
                                               : modulus - (origin_remainder - index_remainder);
}

[[nodiscard]] inline core::Result<std::int64_t> maximum_texture_index(const std::int64_t origin,
                                                                      const std::uint32_t extent) {
    if (extent == 0U) {
        return std::unexpected(invalid_wrap_request("A wrapped texture extent must be non-zero."));
    }
    const auto extent_offset = static_cast<std::int64_t>(extent - 1U);
    if (origin > std::numeric_limits<std::int64_t>::max() - extent_offset) {
        return std::unexpected(
            invalid_wrap_request("The wrapped texture index interval is not representable."));
    }
    return origin + extent_offset;
}

} // namespace texture_wrap_detail

// Filtering converts normalized U and V to texel space first, then resolves every integer tap on
// each axis independently. Coordinates must not be pre-wrapped: per-tap black addressing preserves
// the in-domain portion of a filter footprint that crosses an image edge. mirror duplicates edge
// texels (for extent 4: -1 -> 0 and 4 -> 3), while black returns an empty optional for an
// out-of-domain tap. Invalid modes and intervals are errors rather than implicit defaults.
[[nodiscard]] inline core::Result<std::optional<std::int64_t>>
wrap_texture_index(const std::int64_t index, const std::int64_t origin, const std::uint32_t extent,
                   const TextureWrapMode mode) {
    if (!is_valid_texture_wrap_mode(mode)) {
        return std::unexpected(texture_wrap_detail::invalid_wrap_request(
            "A texture wrap mode must be a supported explicit value."));
    }
    const auto maximum = texture_wrap_detail::maximum_texture_index(origin, extent);
    if (!maximum) {
        return std::unexpected(maximum.error());
    }

    switch (mode) {
    case TextureWrapMode::repeat: {
        const auto offset = texture_wrap_detail::relative_modulo(index, origin, extent);
        return std::optional<std::int64_t>{origin + static_cast<std::int64_t>(offset)};
    }
    case TextureWrapMode::clamp:
        if (index < origin) {
            return std::optional<std::int64_t>{origin};
        }
        if (index > *maximum) {
            return std::optional<std::int64_t>{*maximum};
        }
        return std::optional<std::int64_t>{index};
    case TextureWrapMode::mirror: {
        const auto period = static_cast<std::uint64_t>(extent) * 2U;
        const auto phase = texture_wrap_detail::relative_modulo(index, origin, period);
        const auto offset = phase < extent ? phase : period - 1U - phase;
        return std::optional<std::int64_t>{origin + static_cast<std::int64_t>(offset)};
    }
    case TextureWrapMode::black:
        if (index < origin || index > *maximum) {
            return std::optional<std::int64_t>{};
        }
        return std::optional<std::int64_t>{index};
    }
    return std::unexpected(texture_wrap_detail::invalid_wrap_request(
        "A texture wrap mode must be a supported explicit value."));
}

static_assert(sizeof(TextureWrapMode) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<TextureWrapMode>);
static_assert(static_cast<std::uint32_t>(TextureWrapMode::repeat) == 0U);
static_assert(static_cast<std::uint32_t>(TextureWrapMode::clamp) == 1U);
static_assert(static_cast<std::uint32_t>(TextureWrapMode::mirror) == 2U);
static_assert(static_cast<std::uint32_t>(TextureWrapMode::black) == 3U);

} // namespace blackframe::renderer
