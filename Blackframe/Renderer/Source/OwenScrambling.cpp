#include <Blackframe/Renderer/OwenScrambling.hpp>
#include <cstdint>
#include <utility>

namespace blackframe::renderer {
namespace {

[[nodiscard]] constexpr std::uint64_t owen_tree_key(const SampleStreamIndex index,
                                                    const std::uint64_t sobol_dimension) noexcept {
    const auto packed_pixel = (static_cast<std::uint64_t>(index.pixel_x) << 32U) |
                              static_cast<std::uint64_t>(index.pixel_y);
    const auto dimension_key = sample_stream_detail::mix_bits(
        index.seed ^ sample_stream_detail::mix_bits(sobol_dimension + 1U));
    return dimension_key ^ sample_stream_detail::mix_bits(packed_pixel);
}

[[nodiscard]] constexpr std::uint64_t
nested_binary_scramble(const std::uint64_t input, const std::uint64_t tree_key) noexcept {
    auto input_prefix = std::uint64_t{0};
    auto output = std::uint64_t{0};

    for (auto depth = std::uint32_t{0}; depth < SobolDirectionBitCount; ++depth) {
        const auto shift = 63U - depth;
        const auto input_bit = (input >> shift) & 1U;
        // The sentinel bit makes every tree node's depth and input prefix unambiguous.
        const auto node_id = (std::uint64_t{1} << depth) | input_prefix;
        const auto flip = sample_stream_detail::mix_bits(tree_key ^ node_id) >> 63U;
        output |= (input_bit ^ flip) << shift;
        input_prefix = (input_prefix << 1U) | input_bit;
    }
    return output;
}

} // namespace

core::Result<std::uint64_t> owen_scrambled_sobol_bits(const SampleStreamIndex index,
                                                      const std::uint64_t sobol_dimension) {
    auto sobol = sobol_sample_bits(index.sample_index, sobol_dimension);
    if (!sobol.has_value()) {
        return std::unexpected(std::move(sobol.error()));
    }
    return nested_binary_scramble(*sobol, owen_tree_key(index, sobol_dimension));
}

} // namespace blackframe::renderer
