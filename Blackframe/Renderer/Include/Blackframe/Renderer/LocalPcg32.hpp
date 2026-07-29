#pragma once

#include <Blackframe/Renderer/SampleStream.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

namespace local_pcg32_detail {

inline constexpr std::uint64_t Multiplier = 0x5851F42D4C957F2DULL;
inline constexpr std::uint64_t SequenceDomainSalt = 0x9E3779B97F4A7C15ULL;

} // namespace local_pcg32_detail

// LocalPcg32 is a mutable generator for one variable-consumption loop on one path. The mandatory
// anchor dimension seeds the loop without consuming neighboring dimensions from the frozen map.
class LocalPcg32 final {
  public:
    explicit constexpr LocalPcg32(const SampleStreamIndex path,
                                  const std::uint64_t anchor_dimension) noexcept
        : state_{0},
          // PCG selects one of 2^63 streams with an odd increment. The independently hashed path
          // state still incorporates every address bit.
          increment_{(sample_stream_detail::indexed_bits(
                          path, anchor_dimension ^ local_pcg32_detail::SequenceDomainSalt)
                      << 1U) |
                     1U} {
        static_cast<void>(next_u32());
        state_ += sample_stream_detail::indexed_bits(path, anchor_dimension);
        static_cast<void>(next_u32());
    }

    [[nodiscard]] constexpr std::uint32_t next_u32() noexcept {
        const auto previous_state = state_;
        state_ = previous_state * local_pcg32_detail::Multiplier + increment_;

        const auto xorshifted =
            static_cast<std::uint32_t>(((previous_state >> 18U) ^ previous_state) >> 27U);
        const auto rotation = static_cast<std::uint32_t>(previous_state >> 59U);
        return (xorshifted >> rotation) | (xorshifted << ((0U - rotation) & 31U));
    }

    [[nodiscard]] constexpr std::uint64_t next_u64() noexcept {
        const auto high = static_cast<std::uint64_t>(next_u32());
        const auto low = static_cast<std::uint64_t>(next_u32());
        return (high << 32U) | low;
    }

    template <GeometryScalar Scalar> [[nodiscard]] constexpr Scalar next_1d() noexcept {
        return sample_stream_detail::unit_interval<Scalar>(next_u64());
    }

  private:
    std::uint64_t state_;
    std::uint64_t increment_;
};

static_assert(!std::is_default_constructible_v<LocalPcg32>);
static_assert(std::is_standard_layout_v<LocalPcg32>);
static_assert(std::is_trivially_copyable_v<LocalPcg32>);
static_assert(sizeof(LocalPcg32) == 2 * sizeof(std::uint64_t));

} // namespace blackframe::renderer
