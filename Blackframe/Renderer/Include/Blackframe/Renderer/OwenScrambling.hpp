#pragma once

#include <Blackframe/Renderer/SobolSampling.hpp>
#include <cstdint>
#include <utility>

namespace blackframe::renderer {

// sample_index selects the Sobol point. Pixel coordinates, seed, and the zero-based Sobol
// dimension select one reproducible nested Owen permutation shared by the full sequence.
[[nodiscard]] core::Result<std::uint64_t> owen_scrambled_sobol_bits(SampleStreamIndex index,
                                                                    std::uint64_t sobol_dimension);

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar> owen_scrambled_sobol_1d(const SampleStreamIndex index,
                                                           const std::uint64_t sobol_dimension) {
    auto bits = owen_scrambled_sobol_bits(index, sobol_dimension);
    if (!bits.has_value()) {
        return std::unexpected(std::move(bits.error()));
    }
    return sample_stream_detail::unit_interval<Scalar>(*bits);
}

} // namespace blackframe::renderer
