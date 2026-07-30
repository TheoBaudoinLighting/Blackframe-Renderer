#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
inline constexpr auto WavelengthTolerance =
    std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{2.0e-5} : ReferenceScalar{2.0e-12};

template <SpectrumScalar Scalar>
void expect_wavelength_near(const Scalar actual, const ReferenceScalar expected) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected, WavelengthTolerance<Scalar>);
}

template <SpectrumScalar Scalar>
void expect_packet_contract(const SampledWavelengthsT<Scalar>& packet) {
    constexpr auto pdf_tolerance = std::is_same_v<Scalar, TransportScalar> ? 2.0e-6 : 2.0e-12;
    for (const auto& sample : packet.samples) {
        EXPECT_TRUE(std::isfinite(sample.nanometers));
        EXPECT_GE(sample.nanometers, Scalar{VisibleWavelengthMinimumNanometers});
        EXPECT_LE(sample.nanometers, Scalar{VisibleWavelengthMaximumNanometers});
        EXPECT_TRUE(std::isfinite(sample.probability.value));
        EXPECT_GT(sample.probability.value, Scalar{0});
        EXPECT_EQ(sample.probability.value, uniform_visible_wavelength_pdf<Scalar>());
        EXPECT_EQ(sample.probability.measure, ProbabilityMeasure::wavelength);
        EXPECT_NEAR(static_cast<ReferenceScalar>(sample.probability.value) *
                        VisibleWavelengthRangeNanometers,
                    1.0, pdf_tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_canonical_packets() {
    const auto zero = sample_uniform_visible_wavelengths(Scalar{0});
    ASSERT_TRUE(zero.has_value());
    expect_wavelength_near((*zero)[0].nanometers, 360.0);
    expect_wavelength_near((*zero)[1].nanometers, 477.5);
    expect_wavelength_near((*zero)[2].nanometers, 595.0);
    expect_wavelength_near((*zero)[3].nanometers, 712.5);
    expect_packet_contract(*zero);

    const auto quarter = sample_uniform_visible_wavelengths(Scalar{0.25});
    ASSERT_TRUE(quarter.has_value());
    expect_wavelength_near((*quarter)[0].nanometers, 477.5);
    expect_wavelength_near((*quarter)[1].nanometers, 595.0);
    expect_wavelength_near((*quarter)[2].nanometers, 712.5);
    expect_wavelength_near((*quarter)[3].nanometers, 360.0);
    expect_packet_contract(*quarter);

    const auto three_quarters = sample_uniform_visible_wavelengths(Scalar{0.75});
    ASSERT_TRUE(three_quarters.has_value());
    expect_wavelength_near((*three_quarters)[0].nanometers, 712.5);
    expect_wavelength_near((*three_quarters)[1].nanometers, 360.0);
    expect_wavelength_near((*three_quarters)[2].nanometers, 477.5);
    expect_wavelength_near((*three_quarters)[3].nanometers, 595.0);
    expect_packet_contract(*three_quarters);
}

TEST(WavelengthSamplingTest, StoresFourWavelengthsAndTheirMarginalPdfs) {
    static_assert(TransportSpectrumSampleCount == 4);
    static_assert(std::is_same_v<SampledWavelengths::value_type, WavelengthSample>);
    static_assert(
        std::is_same_v<ReferenceSampledWavelengths::value_type, ReferenceWavelengthSample>);
    static_assert(std::is_same_v<SampledWavelengths::probability_type, ProbabilityDensity>);
    static_assert(
        std::is_same_v<ReferenceSampledWavelengths::probability_type, ReferenceProbabilityDensity>);
    static_assert(!std::is_same_v<SampledWavelengths, ReferenceSampledWavelengths>);
    static_assert(std::is_standard_layout_v<SampledWavelengths>);
    static_assert(std::is_trivially_copyable_v<SampledWavelengths>);
    static_assert(std::is_standard_layout_v<ReferenceSampledWavelengths>);
    static_assert(std::is_trivially_copyable_v<ReferenceSampledWavelengths>);
    static_assert(sizeof(SampledWavelengths::samples) / sizeof(WavelengthSample) == 4);
    static_assert(
        sizeof(ReferenceSampledWavelengths::samples) / sizeof(ReferenceWavelengthSample) == 4);

    expect_canonical_packets<TransportScalar>();
    expect_canonical_packets<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_circular_wrap_preserves_adjacent_samples() {
    constexpr auto cases = std::array{
        std::pair{Scalar{0.25}, std::size_t{3}},
        std::pair{Scalar{0.5}, std::size_t{2}},
        std::pair{Scalar{0.75}, std::size_t{1}},
    };

    for (auto case_index = std::size_t{0}; case_index < cases.size(); ++case_index) {
        const auto [threshold, lane] = cases[case_index];
        const auto at_threshold = sample_uniform_visible_wavelengths(threshold);
        ASSERT_TRUE(at_threshold.has_value());
        EXPECT_EQ((*at_threshold)[lane].nanometers, Scalar{VisibleWavelengthMinimumNanometers});

        auto adjacent = std::nextafter(threshold, Scalar{1});
        if (case_index == 0) {
            adjacent = std::nextafter(adjacent, Scalar{1});
        }
        const auto expected_shift = adjacent - threshold;
        const auto expected_wavelength =
            std::fma(Scalar{VisibleWavelengthRangeNanometers}, expected_shift,
                     Scalar{VisibleWavelengthMinimumNanometers});
        ASSERT_GT(expected_wavelength, Scalar{VisibleWavelengthMinimumNanometers});

        const auto after_threshold = sample_uniform_visible_wavelengths(adjacent);
        ASSERT_TRUE(after_threshold.has_value());
        EXPECT_EQ((*after_threshold)[lane].nanometers, expected_wavelength);
    }
}

TEST(WavelengthSamplingTest, PreservesAdjacentSamplesAcrossCircularWrap) {
    expect_circular_wrap_preserves_adjacent_samples<TransportScalar>();
    expect_circular_wrap_preserves_adjacent_samples<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_samples_rejected() {
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    for (const auto invalid : std::array{
             std::nextafter(Scalar{0}, -infinity),
             Scalar{-0.25},
             Scalar{1},
             std::numeric_limits<Scalar>::quiet_NaN(),
             infinity,
             -infinity,
         }) {
        const auto packet = sample_uniform_visible_wavelengths(invalid);
        ASSERT_FALSE(packet.has_value());
        EXPECT_EQ(packet.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(packet.error().message,
                  "Visible wavelength sampling requires a finite value in [0, 1).");
    }

    const auto signed_zero = sample_uniform_visible_wavelengths(-Scalar{0});
    ASSERT_TRUE(signed_zero.has_value());
    expect_packet_contract(*signed_zero);

    const auto maximum =
        sample_uniform_visible_wavelengths(std::nextafter(Scalar{1}, static_cast<Scalar>(0)));
    ASSERT_TRUE(maximum.has_value());
    expect_packet_contract(*maximum);
}

TEST(WavelengthSamplingTest, RejectsMalformedCanonicalSamplesWithoutSubstitution) {
    expect_invalid_samples_rejected<TransportScalar>();
    expect_invalid_samples_rejected<ReferenceScalar>();
}

TEST(WavelengthSamplingTest, UsesOnlyTheVersionedPathWavelengthDimension) {
    constexpr auto index = SampleStreamIndex{};
    constexpr auto transport_stream = SampleStream{index};
    constexpr auto reference_stream = ReferenceSampleStream{index};

    const auto transport = sample_visible_wavelengths(transport_stream);
    const auto transport_unit_sample =
        transport_stream.sample_1d(PrimarySampleDimensionMap.wavelength);
    const auto expected_transport = sample_uniform_visible_wavelengths(transport_unit_sample);
    ASSERT_TRUE(expected_transport.has_value());
    EXPECT_EQ(transport, *expected_transport);
    EXPECT_EQ(transport_unit_sample, 0.23662316799163818F);
    EXPECT_EQ(transport[0].nanometers, 471.212890625F);
    EXPECT_EQ(transport[1].nanometers, 588.712890625F);
    EXPECT_EQ(transport[2].nanometers, 706.212890625F);
    EXPECT_EQ(transport[3].nanometers, 823.712890625F);

    const auto reference = sample_visible_wavelengths(reference_stream);
    const auto reference_unit_sample =
        reference_stream.sample_1d(PrimarySampleDimensionMap.wavelength);
    const auto expected_reference = sample_uniform_visible_wavelengths(reference_unit_sample);
    ASSERT_TRUE(expected_reference.has_value());
    EXPECT_EQ(reference, *expected_reference);
    EXPECT_EQ(reference_unit_sample, 0.23662320061992437);
    EXPECT_EQ(reference[0].nanometers, 471.21290429136445);
    EXPECT_EQ(reference[1].nanometers, 588.7129042913645);
    EXPECT_EQ(reference[2].nanometers, 706.2129042913645);
    EXPECT_EQ(reference[3].nanometers, 823.7129042913645);
}

TEST(WavelengthSamplingTest, ReplaysPathsIndependentlyOfOrderAndInterleaving) {
    constexpr auto sample_count = std::size_t{2'048};
    constexpr auto base_index = SampleStreamIndex{
        .pixel_x = 113,
        .pixel_y = 47,
        .seed = 0xA5A5F00D12345678ULL,
    };
    std::vector<SampledWavelengths> recorded(sample_count);

    for (auto sample = std::size_t{0}; sample < sample_count; ++sample) {
        auto index = base_index;
        index.sample_index = sample;
        recorded[sample] = sample_visible_wavelengths(SampleStream{index});

        auto interleaved = index;
        interleaved.pixel_x ^= 0x80000000U;
        static_cast<void>(sample_visible_wavelengths(ReferenceSampleStream{interleaved}));
    }

    for (auto reverse = sample_count; reverse > 0; --reverse) {
        const auto sample = reverse - 1;
        auto index = base_index;
        index.sample_index = sample;
        EXPECT_EQ(sample_visible_wavelengths(SampleStream{index}), recorded[sample]);
    }
}

template <SpectrumScalar Scalar> void expect_conformant_spectral_histogram() {
    constexpr auto path_count = std::size_t{65'536};
    constexpr auto bin_count = std::size_t{64};
    constexpr auto expected_per_bin =
        static_cast<ReferenceScalar>(path_count) / static_cast<ReferenceScalar>(bin_count);
    constexpr auto base_index = SampleStreamIndex{
        .pixel_x = 113,
        .pixel_y = 47,
        .seed = 0xA5A5F00D12345678ULL,
    };
    std::array<std::array<std::uint32_t, bin_count>, TransportSpectrumSampleCount> histograms{};
    std::array<ReferenceScalar, TransportSpectrumSampleCount> sums{};
    std::array<ReferenceScalar, TransportSpectrumSampleCount> squared_sums{};
    auto every_packet_covers_all_quarters = true;

    for (auto path = std::size_t{0}; path < path_count; ++path) {
        auto index = base_index;
        index.sample_index = path;
        const auto packet = sample_visible_wavelengths(SampleStreamT<Scalar>{index});
        std::array<std::uint8_t, TransportSpectrumSampleCount> quarter_counts{};

        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto& sample = packet[lane];
            const auto normalized = (static_cast<ReferenceScalar>(sample.nanometers) -
                                     VisibleWavelengthMinimumNanometers) /
                                    VisibleWavelengthRangeNanometers;
            if (!(normalized >= 0.0 && normalized <= 1.0) ||
                sample.probability.value != uniform_visible_wavelength_pdf<Scalar>() ||
                sample.probability.measure != ProbabilityMeasure::wavelength) {
                ADD_FAILURE() << "A wavelength lane violated its domain or PDF contract.";
                return;
            }

            const auto bin = std::min(
                static_cast<std::size_t>(normalized * static_cast<ReferenceScalar>(bin_count)),
                bin_count - 1);
            const auto quarter =
                std::min(static_cast<std::size_t>(normalized * static_cast<ReferenceScalar>(
                                                                   TransportSpectrumSampleCount)),
                         TransportSpectrumSampleCount - 1);
            ++histograms[lane][bin];
            ++quarter_counts[quarter];
            sums[lane] += normalized;
            squared_sums[lane] += normalized * normalized;
        }
        every_packet_covers_all_quarters =
            every_packet_covers_all_quarters &&
            std::ranges::all_of(quarter_counts, [](const auto count) { return count == 1; });
    }

    EXPECT_TRUE(every_packet_covers_all_quarters);
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        auto chi_squared = ReferenceScalar{0};
        for (const auto observed : histograms[lane]) {
            const auto difference = static_cast<ReferenceScalar>(observed) - expected_per_bin;
            chi_squared += difference * difference / expected_per_bin;
        }
        const auto inverse_count = ReferenceScalar{1} / static_cast<ReferenceScalar>(path_count);
        EXPECT_NEAR(sums[lane] * inverse_count, 0.5, 0.005);
        EXPECT_NEAR(squared_sums[lane] * inverse_count, 1.0 / 3.0, 0.006);
        EXPECT_LT(chi_squared, 150.0);
    }
}

TEST(WavelengthSamplingTest, HasAConformantTransportSpectralHistogram) {
    expect_conformant_spectral_histogram<TransportScalar>();
}

TEST(WavelengthSamplingTest, HasAConformantReferenceSpectralHistogram) {
    expect_conformant_spectral_histogram<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
