#pragma once

#include <Blackframe/Renderer/SampleStream.hpp>
#include <cstdint>
#include <type_traits>

namespace blackframe::renderer {

// The independent sampler is the simple hashed oracle. It binds an explicit seed and creates the
// canonical random-access stream for a pixel sample without maintaining a cursor or global state.
template <GeometryScalar Scalar> class IndependentSamplerT final {
  public:
    using value_type = Scalar;

    explicit constexpr IndependentSamplerT(const std::uint64_t seed) noexcept : seed_{seed} {}

    [[nodiscard]] constexpr std::uint64_t seed() const noexcept {
        return seed_;
    }

    [[nodiscard]] constexpr SampleStreamT<Scalar>
    make_stream(const std::uint32_t pixel_x, const std::uint32_t pixel_y,
                const std::uint64_t sample_index) const noexcept {
        return SampleStreamT<Scalar>{SampleStreamIndex{
            .pixel_x = pixel_x,
            .pixel_y = pixel_y,
            .sample_index = sample_index,
            .seed = seed_,
        }};
    }

  private:
    std::uint64_t seed_;
};

using IndependentSampler = IndependentSamplerT<TransportScalar>;
using ReferenceIndependentSampler = IndependentSamplerT<ReferenceScalar>;

static_assert(!std::is_same_v<IndependentSampler, ReferenceIndependentSampler>);
static_assert(!std::is_same_v<IndependentSampler, SampleStream>);
static_assert(!std::is_same_v<ReferenceIndependentSampler, ReferenceSampleStream>);
static_assert(!std::is_default_constructible_v<IndependentSampler>);
static_assert(!std::is_default_constructible_v<ReferenceIndependentSampler>);
static_assert(std::is_standard_layout_v<IndependentSampler>);
static_assert(std::is_trivially_copyable_v<IndependentSampler>);
static_assert(std::is_standard_layout_v<ReferenceIndependentSampler>);
static_assert(std::is_trivially_copyable_v<ReferenceIndependentSampler>);

} // namespace blackframe::renderer
