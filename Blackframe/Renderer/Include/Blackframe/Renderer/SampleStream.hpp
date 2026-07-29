#pragma once

#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// A stream address contains every coordinate except the independently addressed dimension.
// Every bit pattern is valid; raster extent validation belongs to the camera and renderer.
struct SampleStreamIndex final {
    std::uint32_t pixel_x{};
    std::uint32_t pixel_y{};
    std::uint64_t sample_index{};
    std::uint64_t seed{};

    [[nodiscard]] constexpr bool operator==(const SampleStreamIndex&) const noexcept = default;
};

namespace sample_stream_detail {

[[nodiscard]] constexpr std::uint64_t mix_bits(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] constexpr std::uint64_t indexed_bits(const SampleStreamIndex index,
                                                   const std::uint64_t dimension) noexcept {
    const auto packed_pixel = (static_cast<std::uint64_t>(index.pixel_x) << 32U) |
                              static_cast<std::uint64_t>(index.pixel_y);
    auto state = mix_bits(index.seed ^ 0x9E3779B97F4A7C15ULL);
    state = mix_bits(state ^ packed_pixel ^ 0xD1B54A32D192ED03ULL);
    state = mix_bits(state ^ index.sample_index ^ 0x8CB92BA72F3D8DD7ULL);
    return mix_bits(state ^ dimension);
}

template <GeometryScalar Scalar>
[[nodiscard]] constexpr Scalar unit_interval(const std::uint64_t bits) noexcept {
    if constexpr (std::same_as<Scalar, TransportScalar>) {
        return static_cast<Scalar>(bits >> 40U) * Scalar{0x1p-24F};
    } else {
        return static_cast<Scalar>(bits >> 11U) * Scalar{0x1p-53};
    }
}

} // namespace sample_stream_detail

// SampleStream is a pure random-access mapping. It has no cursor or mutable generator state:
// replaying an address and dimension produces the same value independently of call order.
template <GeometryScalar Scalar> class SampleStreamT final {
  public:
    using value_type = Scalar;

    explicit constexpr SampleStreamT(const SampleStreamIndex index) noexcept : index_{index} {}

    [[nodiscard]] constexpr SampleStreamIndex index() const noexcept {
        return index_;
    }

    [[nodiscard]] constexpr Scalar sample_1d(const std::uint64_t dimension) const noexcept {
        return sample_stream_detail::unit_interval<Scalar>(
            sample_stream_detail::indexed_bits(index_, dimension));
    }

  private:
    SampleStreamIndex index_;
};

using SampleStream = SampleStreamT<TransportScalar>;
using ReferenceSampleStream = SampleStreamT<ReferenceScalar>;

static_assert(!std::is_same_v<SampleStream, ReferenceSampleStream>);
static_assert(std::is_standard_layout_v<SampleStreamIndex>);
static_assert(std::is_trivially_copyable_v<SampleStreamIndex>);
static_assert(std::is_standard_layout_v<SampleStream>);
static_assert(std::is_trivially_copyable_v<SampleStream>);
static_assert(std::is_standard_layout_v<ReferenceSampleStream>);
static_assert(std::is_trivially_copyable_v<ReferenceSampleStream>);

} // namespace blackframe::renderer
