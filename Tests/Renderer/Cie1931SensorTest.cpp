#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = std::conditional_t<std::is_same_v<Scalar, TransportScalar>, TransportSpectrum,
                                       ReferenceSpectrum>;

template <SpectrumScalar Scalar>
using WavelengthsFor = std::conditional_t<std::is_same_v<Scalar, TransportScalar>,
                                          SampledWavelengths, ReferenceSampledWavelengths>;

inline constexpr std::string_view SensorConversionError =
    "CIE 1931 sensor conversion requires finite contributions and results, wavelengths in [360, "
    "830] nm, and positive wavelength PDFs with wavelength measure.";

template <SpectrumScalar Scalar>
[[nodiscard]] WavelengthsFor<Scalar>
make_wavelengths(const std::array<Scalar, TransportSpectrumSampleCount>& nanometers,
                 const std::array<Scalar, TransportSpectrumSampleCount>& probabilities) {
    auto result = WavelengthsFor<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane] = typename WavelengthsFor<Scalar>::value_type{
            .nanometers = nanometers[lane],
            .probability =
                typename WavelengthsFor<Scalar>::probability_type{
                    .value = probabilities[lane],
                    .measure = ProbabilityMeasure::wavelength,
                },
        };
    }
    return result;
}

template <SpectrumScalar Scalar> [[nodiscard]] WavelengthsFor<Scalar> make_observer_row_packet() {
    return make_wavelengths<Scalar>(
        {Scalar{360}, Scalar{479}, Scalar{555}, Scalar{830}},
        {uniform_visible_wavelength_pdf<Scalar>(), uniform_visible_wavelength_pdf<Scalar>(),
         uniform_visible_wavelength_pdf<Scalar>(), uniform_visible_wavelength_pdf<Scalar>()});
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<XyzT<Scalar>> convert(const SpectrumFor<Scalar>& spectrum,
                                                 const WavelengthsFor<Scalar>& wavelengths) {
    return cie_1931_spectrum_to_xyz(spectrum, wavelengths);
}

template <SpectrumScalar Scalar>
void expect_xyz_near(const XyzT<Scalar> actual, const ReferenceXYZ expected,
                     const ReferenceScalar tolerance) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.x), expected.x, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.y), expected.y, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.z), expected.z, tolerance);
}

template <SpectrumScalar Scalar>
void expect_rgb_near(const LinearRgbT<Scalar> actual, const ReferenceLinearRGB expected,
                     const ReferenceScalar tolerance) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.red), expected.red, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.green), expected.green, tolerance);
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual.blue), expected.blue, tolerance);
}

template <SpectrumScalar Scalar> void expect_zero_spectrum_is_black() {
    const auto wavelengths = sample_uniform_visible_wavelengths(Scalar{0.125});
    ASSERT_TRUE(wavelengths.has_value());
    const auto xyz = convert<Scalar>(SpectrumFor<Scalar>{}, *wavelengths);
    ASSERT_TRUE(xyz.has_value());
    EXPECT_EQ(*xyz, XyzT<Scalar>{});

    const auto rgb = xyz_to_linear_rgb(*xyz);
    ASSERT_TRUE(rgb.has_value());
    EXPECT_EQ(*rgb, LinearRgbT<Scalar>{});
}

TEST(Cie1931SensorTest, KeepsTransportAndReferenceResultsDistinct) {
    static_assert(
        std::same_as<decltype(cie_1931_spectrum_to_xyz(std::declval<const TransportSpectrum&>(),
                                                       std::declval<const SampledWavelengths&>())),
                     core::Result<XYZ>>);
    static_assert(std::same_as<decltype(cie_1931_spectrum_to_xyz(
                                   std::declval<const ReferenceSpectrum&>(),
                                   std::declval<const ReferenceSampledWavelengths&>())),
                               core::Result<ReferenceXYZ>>);
    static_assert(!std::same_as<XYZ, ReferenceXYZ>);

    expect_zero_spectrum_is_black<TransportScalar>();
    expect_zero_spectrum_is_black<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_official_rows_and_interpolation() {
    constexpr auto official_row_results = std::array{
        ReferenceXYZ{
            .x = 0.00014283820576224636,
            .y = 0.0000043071381983889064,
            .z = 0.00066646833342954204,
        },
        ReferenceXYZ{
            .x = 0.11468610393202611,
            .y = 0.14674486917588844,
            .z = 0.94193967539115797,
        },
        ReferenceXYZ{
            .x = 0.56305094337474060,
            .y = 1.0996012760757994,
            .z = 0.0063227062378345706,
        },
        ReferenceXYZ{
            .x = 0.0000013757562401507516,
            .y = 0.00000049681085254380694,
            .z = 0.0,
        },
    };
    constexpr auto tolerance = std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{8.0e-7}
                                                                       : ReferenceScalar{2.0e-12};
    const auto wavelengths = make_observer_row_packet<Scalar>();

    for (auto active_lane = std::size_t{0}; active_lane < TransportSpectrumSampleCount;
         ++active_lane) {
        auto spectrum = SpectrumFor<Scalar>{};
        spectrum[active_lane] = Scalar{1};
        const auto xyz = convert<Scalar>(spectrum, wavelengths);
        ASSERT_TRUE(xyz.has_value());
        expect_xyz_near(*xyz, official_row_results[active_lane], tolerance);
    }

    auto midpoint_wavelengths = wavelengths;
    midpoint_wavelengths[0].nanometers = Scalar{479.5};
    midpoint_wavelengths[0].probability.value = Scalar{0.5};
    auto midpoint_spectrum = SpectrumFor<Scalar>{};
    midpoint_spectrum[0] = Scalar{1};
    const auto midpoint = convert<Scalar>(midpoint_spectrum, midpoint_wavelengths);
    ASSERT_TRUE(midpoint.has_value());
    expect_xyz_near(*midpoint,
                    ReferenceXYZ{
                        .x = 0.00046777014888492674,
                        .y = 0.00063747114590626821,
                        .z = 0.0039060864739087374,
                    },
                    tolerance);
}

TEST(Cie1931SensorTest, MatchesOfficialObserverRowsAndMidpointInterpolation) {
    expect_official_rows_and_interpolation<TransportScalar>();
    expect_official_rows_and_interpolation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_synthetic_spectrum_conversion() {
    const auto wavelengths =
        make_wavelengths<Scalar>({Scalar{360}, Scalar{479}, Scalar{555}, Scalar{830}},
                                 {Scalar{0.5}, Scalar{0.25}, Scalar{0.125}, Scalar{0.0625}});
    const auto spectrum =
        SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{2}, Scalar{0.5}, Scalar{4}}};
    const auto xyz = convert<Scalar>(spectrum, wavelengths);
    ASSERT_TRUE(xyz.has_value());

    constexpr auto expected_xyz = ReferenceXYZ{
        .x = 0.0067448219782256704,
        .y = 0.011856179718896307,
        .z = 0.016089662052654174,
    };
    constexpr auto expected_rgb = ReferenceLinearRGB{
        .red = -0.0043895053498775680,
        .green = 0.016373416576103646,
        .blue = 0.014966733271103356,
    };
    constexpr auto tolerance = std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{3.0e-7}
                                                                       : ReferenceScalar{2.0e-12};
    expect_xyz_near(*xyz, expected_xyz, tolerance);

    const auto rgb = xyz_to_linear_rgb(*xyz);
    ASSERT_TRUE(rgb.has_value());
    expect_rgb_near(*rgb, expected_rgb, tolerance);
    EXPECT_LT(rgb->red, Scalar{0});
}

TEST(Cie1931SensorTest, ConvertsSyntheticSpectrumToXyzThenSignedLinearRgb) {
    expect_synthetic_spectrum_conversion<TransportScalar>();
    expect_synthetic_spectrum_conversion<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_linearity_and_permutation_invariance() {
    const auto wavelengths =
        make_wavelengths<Scalar>({Scalar{360}, Scalar{479}, Scalar{555}, Scalar{830}},
                                 {Scalar{0.5}, Scalar{0.25}, Scalar{0.125}, Scalar{0.0625}});
    const auto spectrum =
        SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{2}, Scalar{0.5}, Scalar{4}}};
    const auto baseline = convert<Scalar>(spectrum, wavelengths);
    ASSERT_TRUE(baseline.has_value());

    const auto negative = convert<Scalar>(-spectrum, wavelengths);
    ASSERT_TRUE(negative.has_value());
    const auto tolerance = std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{3.0e-7}
                                                                   : ReferenceScalar{2.0e-12};
    expect_xyz_near(*negative,
                    ReferenceXYZ{
                        .x = -static_cast<ReferenceScalar>(baseline->x),
                        .y = -static_cast<ReferenceScalar>(baseline->y),
                        .z = -static_cast<ReferenceScalar>(baseline->z),
                    },
                    tolerance);

    auto permutation = std::array<std::size_t, TransportSpectrumSampleCount>{0, 1, 2, 3};
    do {
        auto permuted_spectrum = SpectrumFor<Scalar>{};
        auto permuted_wavelengths = WavelengthsFor<Scalar>{};
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            permuted_spectrum[lane] = spectrum[permutation[lane]];
            permuted_wavelengths[lane] = wavelengths[permutation[lane]];
        }
        const auto permuted = convert<Scalar>(permuted_spectrum, permuted_wavelengths);
        ASSERT_TRUE(permuted.has_value());
        expect_xyz_near(*permuted,
                        ReferenceXYZ{
                            .x = static_cast<ReferenceScalar>(baseline->x),
                            .y = static_cast<ReferenceScalar>(baseline->y),
                            .z = static_cast<ReferenceScalar>(baseline->z),
                        },
                        tolerance);
    } while (std::ranges::next_permutation(permutation).found);
}

TEST(Cie1931SensorTest, IsLinearAndInvariantUnderJointLanePermutation) {
    expect_linearity_and_permutation_invariance<TransportScalar>();
    expect_linearity_and_permutation_invariance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_constant_spectrum_integral() {
    constexpr auto packet_count = std::size_t{470};
    constexpr auto wavelength_grid_size =
        static_cast<Scalar>(packet_count * TransportSpectrumSampleCount);
    constexpr auto spectrum =
        SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}}};
    auto sum = ReferenceXYZ{};

    for (auto packet_index = std::size_t{0}; packet_index < packet_count; ++packet_index) {
        const auto unit_sample =
            (static_cast<Scalar>(packet_index) + Scalar{0.5}) / wavelength_grid_size;
        const auto wavelengths = sample_uniform_visible_wavelengths(unit_sample);
        ASSERT_TRUE(wavelengths.has_value());
        const auto xyz = convert<Scalar>(spectrum, *wavelengths);
        ASSERT_TRUE(xyz.has_value());
        sum.x += static_cast<ReferenceScalar>(xyz->x);
        sum.y += static_cast<ReferenceScalar>(xyz->y);
        sum.z += static_cast<ReferenceScalar>(xyz->z);
    }

    const auto inverse_packet_count =
        ReferenceScalar{1} / static_cast<ReferenceScalar>(packet_count);
    const auto average = ReferenceXYZ{
        .x = sum.x * inverse_packet_count,
        .y = sum.y * inverse_packet_count,
        .z = sum.z * inverse_packet_count,
    };
    constexpr auto expected = ReferenceXYZ{
        .x = 1.0000794426571664,
        .y = 1.0,
        .z = 1.0003278525483952,
    };
    constexpr auto tolerance = std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{2.0e-5}
                                                                       : ReferenceScalar{2.0e-11};
    expect_xyz_near(average, expected, tolerance);

    const auto rgb = xyz_to_linear_rgb(average);
    ASSERT_TRUE(rgb.has_value());
    expect_rgb_near(*rgb,
                    ReferenceLinearRGB{
                        .red = 1.2048782855021287,
                        .green = 0.94823742317396009,
                        .blue = 0.90919373443559726,
                    },
                    tolerance);
}

TEST(Cie1931SensorTest, IntegratesAConstantSpectrumToEqualEnergyWhite) {
    expect_constant_spectrum_integral<TransportScalar>();
    expect_constant_spectrum_integral<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_invalid(const SpectrumFor<Scalar>& spectrum,
                    const WavelengthsFor<Scalar>& wavelengths) {
    const auto result = convert<Scalar>(spectrum, wavelengths);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, SensorConversionError);
}

template <SpectrumScalar Scalar> void expect_malformed_inputs_rejected() {
    const auto valid_spectrum =
        SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{2}, Scalar{0.5}, Scalar{4}}};
    const auto valid_wavelengths = make_observer_row_packet<Scalar>();
    const auto infinity = std::numeric_limits<Scalar>::infinity();

    for (const auto invalid : std::array{
             std::numeric_limits<Scalar>::quiet_NaN(),
             infinity,
             -infinity,
         }) {
        auto spectrum = valid_spectrum;
        spectrum[0] = invalid;
        expect_invalid<Scalar>(spectrum, valid_wavelengths);
    }

    for (const auto invalid : std::array{
             std::nextafter(Scalar{VisibleWavelengthMinimumNanometers}, -infinity),
             std::nextafter(Scalar{VisibleWavelengthMaximumNanometers}, infinity),
             std::numeric_limits<Scalar>::quiet_NaN(),
             infinity,
         }) {
        auto wavelengths = valid_wavelengths;
        wavelengths[0].nanometers = invalid;
        expect_invalid<Scalar>(valid_spectrum, wavelengths);
    }

    for (const auto invalid : std::array{
             Scalar{0},
             Scalar{-1},
             std::numeric_limits<Scalar>::quiet_NaN(),
             infinity,
         }) {
        auto wavelengths = valid_wavelengths;
        wavelengths[0].probability.value = invalid;
        expect_invalid<Scalar>(valid_spectrum, wavelengths);
    }

    auto wrong_measure = valid_wavelengths;
    wrong_measure[0].probability.measure = ProbabilityMeasure::solid_angle;
    expect_invalid<Scalar>(valid_spectrum, wrong_measure);

    auto overflowing_spectrum = valid_spectrum;
    auto overflowing_wavelengths = valid_wavelengths;
    overflowing_spectrum[0] = std::numeric_limits<Scalar>::max();
    overflowing_wavelengths[0].probability.value = std::numeric_limits<Scalar>::denorm_min();
    expect_invalid<Scalar>(overflowing_spectrum, overflowing_wavelengths);

    const auto sum_overflow_spectrum = SpectrumFor<Scalar>{
        .values =
            {
                std::numeric_limits<Scalar>::max() * Scalar{0.5},
                std::numeric_limits<Scalar>::max() * Scalar{0.5},
                std::numeric_limits<Scalar>::max() * Scalar{0.5},
                std::numeric_limits<Scalar>::max() * Scalar{0.5},
            },
    };
    const auto sum_overflow_wavelengths = make_wavelengths<Scalar>(
        {Scalar{555}, Scalar{555}, Scalar{555}, Scalar{555}},
        {uniform_visible_wavelength_pdf<Scalar>(), uniform_visible_wavelength_pdf<Scalar>(),
         uniform_visible_wavelength_pdf<Scalar>(), uniform_visible_wavelength_pdf<Scalar>()});
    expect_invalid<Scalar>(sum_overflow_spectrum, sum_overflow_wavelengths);
}

TEST(Cie1931SensorTest, RejectsMalformedInputsWithoutSubstitution) {
    expect_malformed_inputs_rejected<TransportScalar>();
    expect_malformed_inputs_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_representable_extremes_accepted() {
    auto spectrum = SpectrumFor<Scalar>{};
    auto wavelengths = make_observer_row_packet<Scalar>();
    spectrum[0] = std::numeric_limits<Scalar>::denorm_min();
    wavelengths[0].probability.value = std::numeric_limits<Scalar>::denorm_min();

    const auto xyz = convert<Scalar>(spectrum, wavelengths);
    ASSERT_TRUE(xyz.has_value());
    EXPECT_TRUE(color_detail::finite(*xyz));
    EXPECT_GT(xyz->x, Scalar{0});
    EXPECT_GT(xyz->y, Scalar{0});
    EXPECT_GT(xyz->z, Scalar{0});

    spectrum = SpectrumFor<Scalar>{};
    wavelengths = make_observer_row_packet<Scalar>();
    spectrum[0] = std::numeric_limits<Scalar>::max();
    wavelengths[0].probability.value = Scalar{0.5};
    const auto large_but_representable = convert<Scalar>(spectrum, wavelengths);
    ASSERT_TRUE(large_but_representable.has_value());
    EXPECT_TRUE(color_detail::finite(*large_but_representable));
    EXPECT_GT(large_but_representable->x, Scalar{0});
    EXPECT_GT(large_but_representable->y, Scalar{0});
    EXPECT_GT(large_but_representable->z, Scalar{0});
}

TEST(Cie1931SensorTest, AcceptsEveryPositiveRepresentablePdfWhenResultIsFinite) {
    expect_representable_extremes_accepted<TransportScalar>();
    expect_representable_extremes_accepted<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
