#pragma once

#include <Blackframe/XPU/CUDA/SampleStreamKernel.hpp>
#include <cstdint>

#if !defined(__CUDACC__)
#error "The CUDA SampleStream device helpers require the CUDA compiler."
#endif

namespace blackframe::xpu::cuda::sample_stream {

[[nodiscard]] __device__ inline std::uint64_t mix_bits(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] __device__ inline std::uint64_t indexed_bits(const shared::SampleStreamIndex& index,
                                                           const std::uint64_t dimension) noexcept {
    const auto packed_pixel = (static_cast<std::uint64_t>(index.pixel_x) << 32U) |
                              static_cast<std::uint64_t>(index.pixel_y);
    auto state = mix_bits(index.seed ^ 0x9E3779B97F4A7C15ULL);
    state = mix_bits(state ^ packed_pixel ^ 0xD1B54A32D192ED03ULL);
    state = mix_bits(state ^ index.sample_index ^ 0x8CB92BA72F3D8DD7ULL);
    return mix_bits(state ^ dimension);
}

[[nodiscard]] __device__ inline float sample_1d(const shared::SampleStreamIndex& index,
                                                const std::uint64_t dimension) noexcept {
    return static_cast<float>(indexed_bits(index, dimension) >> 40U) * 0x1p-24F;
}

[[nodiscard]] __device__ inline SampleStreamDumpStatus
dimensions_for_bounce(const std::uint64_t bounce_index,
                      SampleStreamBounceDimensions& dimensions) noexcept {
    dimensions = SampleStreamBounceDimensions{};
    if (bounce_index > MaximumBounceIndex) {
        return SampleStreamDumpStatus::bounce_out_of_range;
    }

    const auto first = FirstBounceDimension + bounce_index * DimensionsPerBounce;
    dimensions = SampleStreamBounceDimensions{
        .light_selection = first + BounceOffsets.light_selection,
        .light_u = first + BounceOffsets.light_u,
        .light_v = first + BounceOffsets.light_v,
        .bsdf_component = first + BounceOffsets.bsdf_component,
        .bsdf_u = first + BounceOffsets.bsdf_u,
        .bsdf_v = first + BounceOffsets.bsdf_v,
        .medium_distance = first + BounceOffsets.medium_distance,
        .medium_phase_u = first + BounceOffsets.medium_phase_u,
        .medium_phase_v = first + BounceOffsets.medium_phase_v,
        .russian_roulette = first + BounceOffsets.russian_roulette,
    };
    return SampleStreamDumpStatus::success;
}

[[nodiscard]] __device__ inline SampleStreamDimensionManifest dimension_manifest() noexcept {
    return SampleStreamDimensionManifest{
        .schema_version = DimensionMapSchemaVersion,
        .reserved = 0U,
        .primary_dimensions = PrimaryDimensions,
        .first_bounce_dimension = FirstBounceDimension,
        .dimensions_per_bounce = DimensionsPerBounce,
        .maximum_bounce_index = MaximumBounceIndex,
        .bounce_offsets = BounceOffsets,
    };
}

} // namespace blackframe::xpu::cuda::sample_stream
