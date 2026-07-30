#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace blackframe::renderer {

// The continuous uniform support is 360–830 nm. A half-open canonical input can round to either
// representable endpoint, both of which retain the same non-zero wavelength density.
inline constexpr ReferenceScalar VisibleWavelengthMinimumNanometers = 360.0;
inline constexpr ReferenceScalar VisibleWavelengthMaximumNanometers = 830.0;
inline constexpr ReferenceScalar VisibleWavelengthRangeNanometers =
    VisibleWavelengthMaximumNanometers - VisibleWavelengthMinimumNanometers;

template <SpectrumScalar Scalar> struct SampledWavelengthsT final {
    using value_type = std::conditional_t<std::is_same_v<Scalar, TransportScalar>, WavelengthSample,
                                          ReferenceWavelengthSample>;
    using probability_type = std::conditional_t<std::is_same_v<Scalar, TransportScalar>,
                                                ProbabilityDensity, ReferenceProbabilityDensity>;

    std::array<value_type, TransportSpectrumSampleCount> samples{};

    [[nodiscard]] constexpr value_type& operator[](const std::size_t index) noexcept {
        return samples[index];
    }

    [[nodiscard]] constexpr const value_type& operator[](const std::size_t index) const noexcept {
        return samples[index];
    }

    [[nodiscard]] constexpr bool operator==(const SampledWavelengthsT& other) const noexcept {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            if (samples[lane].nanometers != other.samples[lane].nanometers ||
                samples[lane].probability.value != other.samples[lane].probability.value ||
                samples[lane].probability.measure != other.samples[lane].probability.measure) {
                return false;
            }
        }
        return true;
    }
};

using SampledWavelengths = SampledWavelengthsT<TransportScalar>;
using ReferenceSampledWavelengths = SampledWavelengthsT<ReferenceScalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr Scalar uniform_visible_wavelength_pdf() noexcept {
    return Scalar{1} / Scalar{VisibleWavelengthRangeNanometers};
}

namespace wavelength_sampling_detail {

template <SpectrumScalar Scalar>
[[nodiscard]] inline SampledWavelengthsT<Scalar>
sample_visible_wavelengths_unchecked(const Scalar unit_sample) noexcept {
    auto result = SampledWavelengthsT<Scalar>{};
    constexpr auto lane_count = static_cast<Scalar>(TransportSpectrumSampleCount);
    constexpr auto probability = typename SampledWavelengthsT<Scalar>::probability_type{
        .value = uniform_visible_wavelength_pdf<Scalar>(),
        .measure = ProbabilityMeasure::wavelength,
    };

    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto offset = static_cast<Scalar>(lane) / lane_count;
        const auto wrap_threshold = Scalar{1} - offset;
        const auto shifted_sample = lane != 0 && unit_sample >= wrap_threshold
                                        ? unit_sample - wrap_threshold
                                        : unit_sample + offset;

        result[lane] = typename SampledWavelengthsT<Scalar>::value_type{
            .nanometers = std::fma(Scalar{VisibleWavelengthRangeNanometers}, shifted_sample,
                                   Scalar{VisibleWavelengthMinimumNanometers}),
            .probability = probability,
        };
    }
    return result;
}

} // namespace wavelength_sampling_detail

// Maps one canonical sample to four circularly stratified wavelengths. Every lane retains the same
// uniform marginal PDF, stored in inverse nanometers with the wavelength probability measure.
template <SpectrumScalar Scalar>
[[nodiscard]] inline core::Result<SampledWavelengthsT<Scalar>>
sample_uniform_visible_wavelengths(const Scalar unit_sample) {
    if (!std::isfinite(unit_sample) || unit_sample < Scalar{0} || unit_sample >= Scalar{1}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "Visible wavelength sampling requires a finite value in [0, 1).",
        });
    }
    return wavelength_sampling_detail::sample_visible_wavelengths_unchecked(unit_sample);
}

// The path adapter consumes only the wavelength dimension fixed by the versioned dimension map.
template <SpectrumScalar Scalar>
[[nodiscard]] inline SampledWavelengthsT<Scalar>
sample_visible_wavelengths(const SampleStreamT<Scalar>& stream) noexcept {
    return wavelength_sampling_detail::sample_visible_wavelengths_unchecked(
        stream.sample_1d(PrimarySampleDimensionMap.wavelength));
}

static_assert(TransportSpectrumSampleCount == 4);
static_assert(std::is_standard_layout_v<SampledWavelengths>);
static_assert(std::is_trivially_copyable_v<SampledWavelengths>);
static_assert(std::is_standard_layout_v<ReferenceSampledWavelengths>);
static_assert(std::is_trivially_copyable_v<ReferenceSampledWavelengths>);
static_assert(sizeof(SampledWavelengths) ==
              TransportSpectrumSampleCount * sizeof(WavelengthSample));
static_assert(sizeof(ReferenceSampledWavelengths) ==
              TransportSpectrumSampleCount * sizeof(ReferenceWavelengthSample));

} // namespace blackframe::renderer
