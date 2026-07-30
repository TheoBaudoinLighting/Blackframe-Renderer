#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <cstdint>
#include <limits>
#include <utility>

namespace blackframe::renderer {

inline constexpr std::uint64_t SobolDimensionCount = 21'201;
inline constexpr std::uint32_t SobolDirectionBitCount = 64;

// Returns the random-access Sobol coordinate as an unsigned fixed-point fraction. Bit 63 has
// weight 2^-1 and bit 0 has weight 2^-64. Dimensions are zero-based and are not semantic
// SampleDimension tags; an unsupported dimension is an explicit error.
[[nodiscard]] core::Result<std::uint64_t> sobol_sample_bits(std::uint64_t sample_index,
                                                            std::uint64_t sobol_dimension);

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> sobol_sample_1d(const std::uint64_t sample_index,
                                                   const std::uint64_t sobol_dimension) {
    auto bits = sobol_sample_bits(sample_index, sobol_dimension);
    if (!bits.has_value()) {
        return std::unexpected(std::move(bits.error()));
    }
    return sample_stream_detail::unit_interval<Scalar>(*bits);
}

static_assert(SobolDirectionBitCount == std::numeric_limits<std::uint64_t>::digits);

} // namespace blackframe::renderer
