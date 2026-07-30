#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace blackframe::renderer {
namespace {

inline constexpr std::size_t Cie1931SampleCount = 471;
inline constexpr auto Cie1931ColorMatchingFunctions = std::array<ReferenceXYZ, Cie1931SampleCount>{
#include "Cie1931ColorMatchingFunctions.inc"
};

[[nodiscard]] constexpr ReferenceXYZ sum_color_matching_functions() noexcept {
    auto sum = ReferenceXYZ{};
    for (const auto response : Cie1931ColorMatchingFunctions) {
        sum.x += response.x;
        sum.y += response.y;
        sum.z += response.z;
    }
    return sum;
}

[[nodiscard]] constexpr ReferenceXYZ integrate_color_matching_functions() noexcept {
    auto integral = ReferenceXYZ{};
    for (auto index = std::size_t{0}; index + 1 < Cie1931SampleCount; ++index) {
        integral.x +=
            (Cie1931ColorMatchingFunctions[index].x + Cie1931ColorMatchingFunctions[index + 1].x) /
            2.0;
        integral.y +=
            (Cie1931ColorMatchingFunctions[index].y + Cie1931ColorMatchingFunctions[index + 1].y) /
            2.0;
        integral.z +=
            (Cie1931ColorMatchingFunctions[index].z + Cie1931ColorMatchingFunctions[index + 1].z) /
            2.0;
    }
    return integral;
}

[[nodiscard]] constexpr std::uint64_t mix_fingerprint_word(std::uint64_t hash,
                                                           const std::uint64_t word) noexcept {
    constexpr auto prime = std::uint64_t{1'099'511'628'211};
    for (auto shift = std::uint32_t{0}; shift < 64; shift += 8) {
        hash ^= (word >> shift) & std::uint64_t{0xff};
        hash *= prime;
    }
    return hash;
}

[[nodiscard]] constexpr std::uint64_t color_matching_function_fingerprint() noexcept {
    auto hash = std::uint64_t{14'695'981'039'346'656'037ULL};
    for (auto index = std::size_t{0}; index < Cie1931SampleCount; ++index) {
        const auto response = Cie1931ColorMatchingFunctions[index];
        hash = mix_fingerprint_word(hash, static_cast<std::uint64_t>(index));
        hash = mix_fingerprint_word(hash, std::bit_cast<std::uint64_t>(response.x));
        hash = mix_fingerprint_word(hash, std::bit_cast<std::uint64_t>(response.y));
        hash = mix_fingerprint_word(hash, std::bit_cast<std::uint64_t>(response.z));
    }
    return hash;
}

inline constexpr auto Cie1931TableSum = sum_color_matching_functions();
inline constexpr auto Cie1931Integral = integrate_color_matching_functions();
inline constexpr ReferenceScalar Cie1931YIntegralNanometers = Cie1931Integral.y;
inline constexpr auto Cie1931TableFingerprint = color_matching_function_fingerprint();

template <SpectrumScalar Scalar>
[[nodiscard]] XyzT<Scalar> color_matching_functions(const Scalar wavelength) noexcept {
    const auto coordinate = wavelength - Scalar{VisibleWavelengthMinimumNanometers};
    const auto lower_index = static_cast<std::size_t>(coordinate);
    if (lower_index + 1 >= Cie1931SampleCount) {
        const auto response = Cie1931ColorMatchingFunctions.back();
        return XyzT<Scalar>{
            .x = static_cast<Scalar>(response.x),
            .y = static_cast<Scalar>(response.y),
            .z = static_cast<Scalar>(response.z),
        };
    }

    const auto fraction = coordinate - static_cast<Scalar>(lower_index);
    const auto lower = Cie1931ColorMatchingFunctions[lower_index];
    const auto upper = Cie1931ColorMatchingFunctions[lower_index + 1];
    const auto lower_x = static_cast<Scalar>(lower.x);
    const auto lower_y = static_cast<Scalar>(lower.y);
    const auto lower_z = static_cast<Scalar>(lower.z);
    return XyzT<Scalar>{
        .x = std::fma(static_cast<Scalar>(upper.x) - lower_x, fraction, lower_x),
        .y = std::fma(static_cast<Scalar>(upper.y) - lower_y, fraction, lower_y),
        .z = std::fma(static_cast<Scalar>(upper.z) - lower_z, fraction, lower_z),
    };
}

template <typename Color> [[nodiscard]] core::Result<Color> sensor_conversion_error() {
    return std::unexpected(core::Error{
        .code = core::StatusCode::invalid_argument,
        .message =
            "CIE 1931 sensor conversion requires finite contributions and results, wavelengths "
            "in [360, 830] nm, and positive wavelength PDFs with wavelength measure.",
    });
}

template <SpectrumScalar Scalar>
[[nodiscard]] Scalar scaled_sensor_term(const Scalar contribution, const Scalar response,
                                        const Scalar normalization,
                                        const Scalar probability) noexcept {
    if (contribution == Scalar{0} || response == Scalar{0}) {
        return Scalar{0};
    }

    auto contribution_exponent = 0;
    auto response_exponent = 0;
    auto normalization_exponent = 0;
    auto probability_exponent = 0;
    const auto contribution_mantissa = std::frexp(contribution, &contribution_exponent);
    const auto response_mantissa = std::frexp(response, &response_exponent);
    const auto normalization_mantissa = std::frexp(normalization, &normalization_exponent);
    const auto probability_mantissa = std::frexp(probability, &probability_exponent);
    const auto mantissa =
        contribution_mantissa * response_mantissa * normalization_mantissa / probability_mantissa;
    return std::ldexp(mantissa, contribution_exponent + response_exponent + normalization_exponent -
                                    probability_exponent);
}

template <SpectrumScalar Scalar, typename Spectrum, typename Wavelengths>
[[nodiscard]] core::Result<XyzT<Scalar>>
cie_1931_spectrum_to_xyz_impl(const Spectrum& spectrum, const Wavelengths& wavelengths) {
    auto sum = XyzT<Scalar>{};
    constexpr auto inverse_lane_count = Scalar{1} / Scalar{TransportSpectrumSampleCount};
    const auto normalization = inverse_lane_count / static_cast<Scalar>(Cie1931YIntegralNanometers);
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto contribution = spectrum[lane];
        const auto& wavelength = wavelengths[lane];
        if (!std::isfinite(contribution) || !std::isfinite(wavelength.nanometers) ||
            wavelength.nanometers < Scalar{VisibleWavelengthMinimumNanometers} ||
            wavelength.nanometers > Scalar{VisibleWavelengthMaximumNanometers} ||
            !std::isfinite(wavelength.probability.value) ||
            wavelength.probability.value <= Scalar{0} ||
            wavelength.probability.measure != ProbabilityMeasure::wavelength) {
            return sensor_conversion_error<XyzT<Scalar>>();
        }

        const auto response = color_matching_functions(wavelength.nanometers);
        const auto term = XyzT<Scalar>{
            .x = scaled_sensor_term(contribution, response.x, normalization,
                                    wavelength.probability.value),
            .y = scaled_sensor_term(contribution, response.y, normalization,
                                    wavelength.probability.value),
            .z = scaled_sensor_term(contribution, response.z, normalization,
                                    wavelength.probability.value),
        };
        if (!color_detail::finite(term)) {
            return sensor_conversion_error<XyzT<Scalar>>();
        }

        sum.x += term.x;
        sum.y += term.y;
        sum.z += term.z;
        if (!color_detail::finite(sum)) {
            return sensor_conversion_error<XyzT<Scalar>>();
        }
    }
    return sum;
}

static_assert(Cie1931ColorMatchingFunctions.size() == Cie1931SampleCount);
static_assert(Cie1931ColorMatchingFunctions[119] ==
              ReferenceXYZ{.x = 0.1042979, .y = 0.1334528, .z = 0.8566193});
static_assert(Cie1931TableSum.x > 106.8654694895 && Cie1931TableSum.x < 106.8654694897);
static_assert(Cie1931TableSum.y > 106.8569171011 && Cie1931TableSum.y < 106.8569171013);
static_assert(Cie1931TableSum.z > 106.8922512785 && Cie1931TableSum.z < 106.8922512788);
static_assert(Cie1931Integral.x > 106.8654039139 && Cie1931Integral.x < 106.8654039142);
static_assert(Cie1931Integral.y > 106.8569149166 && Cie1931Integral.y < 106.8569149170);
static_assert(Cie1931Integral.z > 106.8919482285 && Cie1931Integral.z < 106.8919482288);
static_assert(Cie1931TableFingerprint == std::uint64_t{0x504f4b16e8279104});

} // namespace

core::Result<XYZ> cie_1931_spectrum_to_xyz(const TransportSpectrum& spectrum,
                                           const SampledWavelengths& wavelengths) {
    return cie_1931_spectrum_to_xyz_impl<TransportScalar>(spectrum, wavelengths);
}

core::Result<ReferenceXYZ>
cie_1931_spectrum_to_xyz(const ReferenceSpectrum& spectrum,
                         const ReferenceSampledWavelengths& wavelengths) {
    return cie_1931_spectrum_to_xyz_impl<ReferenceScalar>(spectrum, wavelengths);
}

} // namespace blackframe::renderer
