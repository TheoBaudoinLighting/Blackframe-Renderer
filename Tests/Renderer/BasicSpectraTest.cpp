#include <Blackframe/Renderer/BasicSpectra.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <span>
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

template <SpectrumScalar Scalar> using KnotFor = TabulatedSpectrumKnotT<Scalar>;

inline constexpr std::string_view ConstantError = "A constant spectrum requires a finite value.";
inline constexpr std::string_view TableSizeError =
    "A tabulated spectrum requires at least two knots.";
inline constexpr std::string_view TableValueError =
    "Tabulated spectrum knots require finite positive wavelengths and finite values.";
inline constexpr std::string_view TableOrderError =
    "Tabulated spectrum wavelengths must be strictly increasing.";
inline constexpr std::string_view EvaluationWavelengthError =
    "Spectrum evaluation requires a finite positive wavelength in nanometers.";
inline constexpr std::string_view EvaluationDomainError =
    "Tabulated spectrum evaluation requires every wavelength to lie inside the inclusive table "
    "domain.";

template <SpectrumScalar Scalar>
[[nodiscard]] WavelengthsFor<Scalar>
make_wavelengths(const std::array<Scalar, TransportSpectrumSampleCount>& nanometers) {
    auto result = WavelengthsFor<Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        result[lane].nanometers = nanometers[lane];
        result[lane].probability.value = uniform_visible_wavelength_pdf<Scalar>();
        result[lane].probability.measure = ProbabilityMeasure::wavelength;
    }
    return result;
}

template <typename Result>
void expect_invalid(const Result& result, const std::string_view message) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, message);
}

template <SpectrumScalar Scalar> void expect_black_and_constant_spectra() {
    constexpr auto black = black_spectrum<Scalar>();
    static_assert(black == SpectrumFor<Scalar>{});

    const auto constant = constant_spectrum(Scalar{2.5});
    ASSERT_TRUE(constant.has_value());
    EXPECT_EQ(*constant, (SpectrumFor<Scalar>{.values = {
                                                  Scalar{2.5},
                                                  Scalar{2.5},
                                                  Scalar{2.5},
                                                  Scalar{2.5},
                                              }}));

    const auto signed_constant = constant_spectrum(Scalar{-4});
    ASSERT_TRUE(signed_constant.has_value());
    EXPECT_EQ(*signed_constant, (SpectrumFor<Scalar>{.values = {
                                                         Scalar{-4},
                                                         Scalar{-4},
                                                         Scalar{-4},
                                                         Scalar{-4},
                                                     }}));
}

TEST(BasicSpectraTest, EvaluatesBlackAndConstantSpectraExactlyInEachPrecision) {
    static_assert(std::same_as<decltype(black_spectrum<TransportScalar>()), TransportSpectrum>);
    static_assert(std::same_as<decltype(black_spectrum<ReferenceScalar>()), ReferenceSpectrum>);
    static_assert(std::same_as<decltype(constant_spectrum(TransportScalar{})),
                               core::Result<TransportSpectrum>>);
    static_assert(std::same_as<decltype(constant_spectrum(ReferenceScalar{})),
                               core::Result<ReferenceSpectrum>>);
    static_assert(std::same_as<decltype(evaluate_tabulated_spectrum(
                                   std::declval<std::span<const TabulatedSpectrumKnot>>(),
                                   std::declval<const SampledWavelengths&>())),
                               core::Result<TransportSpectrum>>);
    static_assert(std::same_as<decltype(evaluate_tabulated_spectrum(
                                   std::declval<std::span<const ReferenceTabulatedSpectrumKnot>>(),
                                   std::declval<const ReferenceSampledWavelengths&>())),
                               core::Result<ReferenceSpectrum>>);
    static_assert(!std::same_as<TabulatedSpectrumKnot, ReferenceTabulatedSpectrumKnot>);

    expect_black_and_constant_spectra<TransportScalar>();
    expect_black_and_constant_spectra<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_knots_bounds_and_interpolation() {
    const auto knots = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{360}, .value = Scalar{2}},
        KnotFor<Scalar>{.nanometers = Scalar{600}, .value = Scalar{10}},
        KnotFor<Scalar>{.nanometers = Scalar{830}, .value = Scalar{16}},
    };
    const auto wavelengths =
        make_wavelengths<Scalar>({Scalar{830}, Scalar{420}, Scalar{360}, Scalar{600}});
    const auto evaluated = evaluate_tabulated_spectrum(std::span{knots}, wavelengths);
    ASSERT_TRUE(evaluated.has_value());
    EXPECT_EQ(*evaluated, (SpectrumFor<Scalar>{.values = {
                                                   Scalar{16},
                                                   Scalar{4},
                                                   Scalar{2},
                                                   Scalar{10},
                                               }}));

    auto mutable_knots = knots;
    const auto mutable_evaluated =
        evaluate_tabulated_spectrum(std::span{mutable_knots}, wavelengths);
    ASSERT_TRUE(mutable_evaluated.has_value());
    EXPECT_EQ(*mutable_evaluated, *evaluated);

    auto different_pdfs = wavelengths;
    different_pdfs[0].probability.value = Scalar{0.5};
    different_pdfs[1].probability.value = Scalar{0.25};
    different_pdfs[2].probability.value = Scalar{0.125};
    different_pdfs[3].probability.value = Scalar{0.0625};
    const auto pdf_independent = evaluate_tabulated_spectrum(std::span{knots}, different_pdfs);
    ASSERT_TRUE(pdf_independent.has_value());
    EXPECT_EQ(*pdf_independent, *evaluated);

    const auto minimum_table = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{-2}},
        KnotFor<Scalar>{.nanometers = Scalar{700}, .value = Scalar{4}},
    };
    const auto minimum_wavelengths =
        make_wavelengths<Scalar>({Scalar{400}, Scalar{550}, Scalar{700}, Scalar{550}});
    const auto minimum_evaluated =
        evaluate_tabulated_spectrum(std::span{minimum_table}, minimum_wavelengths);
    ASSERT_TRUE(minimum_evaluated.has_value());
    EXPECT_EQ(*minimum_evaluated, (SpectrumFor<Scalar>{.values = {
                                                           Scalar{-2},
                                                           Scalar{1},
                                                           Scalar{4},
                                                           Scalar{1},
                                                       }}));

    const auto non_dyadic_table = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
        KnotFor<Scalar>{.nanometers = Scalar{700}, .value = Scalar{2}},
        KnotFor<Scalar>{.nanometers = Scalar{830}, .value = Scalar{-1}},
    };
    const auto non_dyadic_wavelengths =
        make_wavelengths<Scalar>({Scalar{500}, Scalar{650}, Scalar{750}, Scalar{830}});
    const auto non_dyadic =
        evaluate_tabulated_spectrum(std::span{non_dyadic_table}, non_dyadic_wavelengths);
    ASSERT_TRUE(non_dyadic.has_value());
    const auto tolerance = Scalar{16} * std::numeric_limits<Scalar>::epsilon();
    EXPECT_NEAR((*non_dyadic)[0], Scalar{4} / Scalar{3}, tolerance);
    EXPECT_NEAR((*non_dyadic)[1], Scalar{11} / Scalar{6}, tolerance);
    EXPECT_NEAR((*non_dyadic)[2], Scalar{11} / Scalar{13}, tolerance);
    EXPECT_EQ((*non_dyadic)[3], Scalar{-1});
}

TEST(BasicSpectraTest, EvaluatesKnotsInclusiveBoundsAndLinearInterpolation) {
    expect_knots_bounds_and_interpolation<TransportScalar>();
    expect_knots_bounds_and_interpolation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_extreme_interpolation_without_overflow() {
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto lowest = std::numeric_limits<Scalar>::lowest();
    const auto extreme_knots = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = maximum},
        KnotFor<Scalar>{.nanometers = Scalar{600}, .value = lowest},
    };
    const auto wavelengths =
        make_wavelengths<Scalar>({Scalar{400}, Scalar{500}, Scalar{600}, Scalar{450}});
    const auto evaluated = evaluate_tabulated_spectrum(std::span{extreme_knots}, wavelengths);
    ASSERT_TRUE(evaluated.has_value());
    EXPECT_EQ((*evaluated)[0], maximum);
    EXPECT_EQ((*evaluated)[1], Scalar{0});
    EXPECT_EQ((*evaluated)[2], lowest);
    EXPECT_TRUE(std::isfinite((*evaluated)[3]));
    EXPECT_GT((*evaluated)[3], Scalar{0});

    const auto adjacent = std::nextafter(Scalar{400}, std::numeric_limits<Scalar>::infinity());
    const auto two_ulps = std::nextafter(adjacent, std::numeric_limits<Scalar>::infinity());
    const auto adjacent_knots = std::array{
        KnotFor<Scalar>{
            .nanometers = Scalar{400},
            .value = Scalar{2},
        },
        KnotFor<Scalar>{
            .nanometers = two_ulps,
            .value = Scalar{4},
        },
    };
    const auto adjacent_wavelengths =
        make_wavelengths<Scalar>({Scalar{400}, adjacent, two_ulps, adjacent});
    const auto adjacent_evaluated =
        evaluate_tabulated_spectrum(std::span{adjacent_knots}, adjacent_wavelengths);
    ASSERT_TRUE(adjacent_evaluated.has_value());
    EXPECT_EQ((*adjacent_evaluated)[0], Scalar{2});
    EXPECT_EQ((*adjacent_evaluated)[1], Scalar{3});
    EXPECT_EQ((*adjacent_evaluated)[2], Scalar{4});
    EXPECT_EQ((*adjacent_evaluated)[3], Scalar{3});
}

TEST(BasicSpectraTest, InterpolatesExtremeFiniteValuesWithoutIntermediateOverflow) {
    expect_extreme_interpolation_without_overflow<TransportScalar>();
    expect_extreme_interpolation_without_overflow<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_malformed_inputs_rejected() {
    const auto wavelengths =
        make_wavelengths<Scalar>({Scalar{400}, Scalar{450}, Scalar{500}, Scalar{550}});
    const auto empty = std::span<const KnotFor<Scalar>>{};
    expect_invalid(evaluate_tabulated_spectrum(empty, wavelengths), TableSizeError);

    const auto singleton = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
    };
    expect_invalid(evaluate_tabulated_spectrum(std::span{singleton}, wavelengths), TableSizeError);

    const auto duplicate = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{2}},
    };
    expect_invalid(evaluate_tabulated_spectrum(std::span{duplicate}, wavelengths), TableOrderError);

    const auto decreasing = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{500}, .value = Scalar{1}},
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{2}},
    };
    expect_invalid(evaluate_tabulated_spectrum(std::span{decreasing}, wavelengths),
                   TableOrderError);

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    for (const auto invalid_wavelength :
         std::array{Scalar{0}, Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(), infinity,
                    -infinity}) {
        const auto invalid_first = std::array{
            KnotFor<Scalar>{.nanometers = invalid_wavelength, .value = Scalar{1}},
            KnotFor<Scalar>{.nanometers = Scalar{600}, .value = Scalar{2}},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_first}, wavelengths),
                       TableValueError);
        const auto invalid_middle = std::array{
            KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
            KnotFor<Scalar>{.nanometers = invalid_wavelength, .value = Scalar{2}},
            KnotFor<Scalar>{.nanometers = Scalar{600}, .value = Scalar{3}},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_middle}, wavelengths),
                       TableValueError);
        const auto invalid_last = std::array{
            KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
            KnotFor<Scalar>{.nanometers = Scalar{500}, .value = Scalar{2}},
            KnotFor<Scalar>{.nanometers = invalid_wavelength, .value = Scalar{3}},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_last}, wavelengths),
                       TableValueError);
    }

    for (const auto invalid_value :
         std::array{std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        const auto invalid_first = std::array{
            KnotFor<Scalar>{.nanometers = Scalar{400}, .value = invalid_value},
            KnotFor<Scalar>{.nanometers = Scalar{600}, .value = Scalar{2}},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_first}, wavelengths),
                       TableValueError);
        const auto invalid_middle = std::array{
            KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
            KnotFor<Scalar>{.nanometers = Scalar{500}, .value = invalid_value},
            KnotFor<Scalar>{.nanometers = Scalar{600}, .value = Scalar{3}},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_middle}, wavelengths),
                       TableValueError);
        const auto invalid_last = std::array{
            KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{1}},
            KnotFor<Scalar>{.nanometers = Scalar{500}, .value = Scalar{2}},
            KnotFor<Scalar>{.nanometers = Scalar{600}, .value = invalid_value},
        };
        expect_invalid(evaluate_tabulated_spectrum(std::span{invalid_last}, wavelengths),
                       TableValueError);
    }

    for (const auto invalid_constant :
         std::array{std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(constant_spectrum(invalid_constant), ConstantError);
    }
}

TEST(BasicSpectraTest, RejectsMalformedTablesAndConstantsWithoutRepair) {
    expect_malformed_inputs_rejected<TransportScalar>();
    expect_malformed_inputs_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_evaluation_bounds_enforced() {
    const auto knots = std::array{
        KnotFor<Scalar>{.nanometers = Scalar{400}, .value = Scalar{2}},
        KnotFor<Scalar>{.nanometers = Scalar{700}, .value = Scalar{8}},
    };
    const auto valid =
        make_wavelengths<Scalar>({Scalar{400}, Scalar{500}, Scalar{600}, Scalar{700}});
    ASSERT_TRUE(evaluate_tabulated_spectrum(std::span{knots}, valid).has_value());

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    for (const auto invalid :
         std::array{Scalar{0}, Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(), infinity,
                    -infinity}) {
        auto wavelengths = valid;
        wavelengths[1].nanometers = invalid;
        expect_invalid(evaluate_tabulated_spectrum(std::span{knots}, wavelengths),
                       EvaluationWavelengthError);
    }

    for (const auto outside : std::array{std::nextafter(Scalar{400}, -infinity),
                                         std::nextafter(Scalar{700}, infinity)}) {
        auto wavelengths = valid;
        wavelengths[1].nanometers = outside;
        expect_invalid(evaluate_tabulated_spectrum(std::span{knots}, wavelengths),
                       EvaluationDomainError);
    }

    auto interior_bounds = valid;
    interior_bounds[0].nanometers = Scalar{401};
    interior_bounds[3].nanometers = Scalar{699};
    const auto interior_evaluated = evaluate_tabulated_spectrum(std::span{knots}, interior_bounds);
    ASSERT_TRUE(interior_evaluated.has_value());
    const auto tolerance = Scalar{64} * std::numeric_limits<Scalar>::epsilon();
    EXPECT_NEAR((*interior_evaluated)[0], Scalar{2} + Scalar{6} / Scalar{300}, tolerance);
    EXPECT_NEAR((*interior_evaluated)[3], Scalar{8} - Scalar{6} / Scalar{300}, tolerance);
}

TEST(BasicSpectraTest, IncludesTableBoundsAndRejectsEveryOutsideWavelength) {
    expect_evaluation_bounds_enforced<TransportScalar>();
    expect_evaluation_bounds_enforced<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
