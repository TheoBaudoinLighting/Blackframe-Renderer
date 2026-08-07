#pragma once

#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// Texture color interpretation is always explicit. Color texels are stored in the
// D65-referred, scene-linear sRGB working space used by LinearRGB; data texels are never
// transformed.
enum class TextureColorSpace : std::uint32_t {
    data = 0U,
    srgb = 1U,
    scene_linear_srgb = 2U,
};

inline constexpr auto TextureWorkingColorSpace = TextureColorSpace::scene_linear_srgb;

[[nodiscard]] constexpr bool
is_valid_texture_color_space(const TextureColorSpace color_space) noexcept {
    switch (color_space) {
    case TextureColorSpace::data:
    case TextureColorSpace::srgb:
    case TextureColorSpace::scene_linear_srgb:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool is_color_texture_space(const TextureColorSpace color_space) noexcept {
    return color_space == TextureColorSpace::srgb ||
           color_space == TextureColorSpace::scene_linear_srgb;
}

static_assert(sizeof(TextureColorSpace) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<TextureColorSpace>);
static_assert(static_cast<std::uint32_t>(TextureColorSpace::data) == 0U);
static_assert(static_cast<std::uint32_t>(TextureColorSpace::srgb) == 1U);
static_assert(static_cast<std::uint32_t>(TextureColorSpace::scene_linear_srgb) == 2U);

} // namespace blackframe::renderer
