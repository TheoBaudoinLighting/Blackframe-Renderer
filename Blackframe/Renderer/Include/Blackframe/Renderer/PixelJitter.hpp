#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <concepts>
#include <cstdint>

namespace blackframe::renderer {

// Center mode explicitly disables jitter and always selects the exact pixel center.
enum class PixelJitterMode : std::uint8_t {
    center = 0,
    uniform = 1,
};

struct PixelSampleIndex final {
    std::uint32_t pixel_x{};
    std::uint32_t pixel_y{};
    std::uint64_t sample_index{};
    std::uint64_t seed{};

    [[nodiscard]] constexpr bool operator==(const PixelSampleIndex&) const noexcept = default;
};

// Offsets are positions inside [0, 1) in the addressed pixel. Integer pixel
// coordinates remain separate so float transport cannot round a valid sample
// in the last pixel up to the image extent.
template <GeometryScalar Scalar> struct PixelSampleT final {
    std::uint32_t pixel_x{};
    std::uint32_t pixel_y{};
    Scalar offset_x{};
    Scalar offset_y{};

    [[nodiscard]] constexpr bool operator==(const PixelSampleT&) const noexcept = default;
};

using PixelSample = PixelSampleT<TransportScalar>;
using ReferencePixelSample = PixelSampleT<ReferenceScalar>;

namespace pixel_jitter_detail {

[[nodiscard]] constexpr std::uint64_t mix_bits(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] constexpr std::uint64_t indexed_bits(const PixelSampleIndex index,
                                                   const std::uint64_t dimension_tag) noexcept {
    const auto packed_pixel = (static_cast<std::uint64_t>(index.pixel_x) << 32U) |
                              static_cast<std::uint64_t>(index.pixel_y);
    auto state = mix_bits(index.seed ^ 0x9E3779B97F4A7C15ULL);
    state = mix_bits(state ^ packed_pixel ^ 0xD1B54A32D192ED03ULL);
    state = mix_bits(state ^ index.sample_index ^ 0x8CB92BA72F3D8DD7ULL);
    return mix_bits(state ^ dimension_tag);
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar unit_interval(const std::uint64_t bits) noexcept {
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        return static_cast<Scalar>(bits >> 40U) * Scalar{0x1p-24F};
    } else {
        return static_cast<Scalar>(bits >> 11U) * Scalar{0x1p-53};
    }
}

} // namespace pixel_jitter_detail

// The uniform mode is a stateless, indexed mapping. Replaying a seed, pixel,
// and sample index produces the same two values independently of call order.
template <GeometryScalar Scalar>
[[nodiscard]] core::Result<PixelSampleT<Scalar>> generate_pixel_sample(const PixelSampleIndex index,
                                                                       const PixelJitterMode mode) {
    switch (mode) {
    case PixelJitterMode::center:
        return PixelSampleT<Scalar>{
            .pixel_x = index.pixel_x,
            .pixel_y = index.pixel_y,
            .offset_x = Scalar{0.5},
            .offset_y = Scalar{0.5},
        };
    case PixelJitterMode::uniform:
        return PixelSampleT<Scalar>{
            .pixel_x = index.pixel_x,
            .pixel_y = index.pixel_y,
            .offset_x = pixel_jitter_detail::unit_interval<Scalar>(
                pixel_jitter_detail::indexed_bits(index, 0xA24BAED4963EE407ULL)),
            .offset_y = pixel_jitter_detail::unit_interval<Scalar>(
                pixel_jitter_detail::indexed_bits(index, 0x9FB21C651E98DF25ULL)),
        };
    }

    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = "Unsupported pixel jitter mode.",
    });
}

} // namespace blackframe::renderer
