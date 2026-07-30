#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {

template <SpectrumScalar Scalar> struct TabulatedSpectrumKnotT final {
    Scalar nanometers{};
    Scalar value{};

    [[nodiscard]] constexpr bool operator==(const TabulatedSpectrumKnotT&) const noexcept = default;
};

using TabulatedSpectrumKnot = TabulatedSpectrumKnotT<TransportScalar>;
using ReferenceTabulatedSpectrumKnot = TabulatedSpectrumKnotT<ReferenceScalar>;

namespace basic_spectra_detail {

[[nodiscard]] inline core::Error invalid_spectrum(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
validate_tabulated_knots(const std::span<const TabulatedSpectrumKnotT<Scalar>> knots) {
    if (knots.size() < 2) {
        return std::unexpected(
            invalid_spectrum("A tabulated spectrum requires at least two knots."));
    }

    for (auto index = std::size_t{0}; index < knots.size(); ++index) {
        const auto& knot = knots[index];
        if (!std::isfinite(knot.nanometers) || knot.nanometers <= Scalar{0} ||
            !std::isfinite(knot.value)) {
            return std::unexpected(invalid_spectrum(
                "Tabulated spectrum knots require finite positive wavelengths and finite values."));
        }
        if (index != 0 && knot.nanometers <= knots[index - 1].nanometers) {
            return std::unexpected(
                invalid_spectrum("Tabulated spectrum wavelengths must be strictly increasing."));
        }
    }
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<Scalar>
evaluate_validated_tabulation(const std::span<const TabulatedSpectrumKnotT<Scalar>> knots,
                              const Scalar wavelength_nanometers) {
    if (!std::isfinite(wavelength_nanometers) || wavelength_nanometers <= Scalar{0}) {
        return std::unexpected(invalid_spectrum(
            "Spectrum evaluation requires a finite positive wavelength in nanometers."));
    }
    if (wavelength_nanometers < knots.front().nanometers ||
        wavelength_nanometers > knots.back().nanometers) {
        return std::unexpected(invalid_spectrum(
            "Tabulated spectrum evaluation requires every wavelength to lie inside the inclusive "
            "table domain."));
    }

    const auto upper =
        std::lower_bound(knots.begin(), knots.end(), wavelength_nanometers,
                         [](const TabulatedSpectrumKnotT<Scalar>& knot, const Scalar wavelength) {
                             return knot.nanometers < wavelength;
                         });
    if (upper != knots.end() && upper->nanometers == wavelength_nanometers) {
        return upper->value;
    }
    if (upper == knots.begin() || upper == knots.end()) {
        return std::unexpected(invalid_spectrum(
            "Tabulated spectrum interpolation could not bracket the requested wavelength."));
    }

    const auto& lower = *(upper - 1);
    const auto interval = upper->nanometers - lower.nanometers;
    const auto offset = wavelength_nanometers - lower.nanometers;
    const auto interpolation = offset / interval;
    if (!std::isfinite(interpolation) || interpolation < Scalar{0} || interpolation > Scalar{1}) {
        return std::unexpected(
            invalid_spectrum("Tabulated spectrum interpolation produced an invalid coordinate."));
    }

    const auto value = std::lerp(lower.value, upper->value, interpolation);
    if (!std::isfinite(value)) {
        return std::unexpected(
            invalid_spectrum("Tabulated spectrum interpolation produced a non-finite value."));
    }
    return value;
}

} // namespace basic_spectra_detail

template <SpectrumScalar Scalar>
[[nodiscard]] constexpr SampledSpectrum<TransportSpectrumSampleCount, Scalar>
black_spectrum() noexcept {
    return {};
}

template <SpectrumScalar Scalar>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
constant_spectrum(const Scalar value) {
    if (!std::isfinite(value)) {
        return std::unexpected(
            basic_spectra_detail::invalid_spectrum("A constant spectrum requires a finite value."));
    }

    auto result = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    result.values.fill(value);
    return result;
}

// The wavelength PDFs are deliberately not consumed here. They belong to the estimator that uses
// the evaluated spectrum. The table support is inclusive, and values outside it are rejected
// instead of being clamped, extrapolated, or silently replaced with zero.
template <SpectrumScalar Scalar, typename Knot, std::size_t Extent>
    requires std::same_as<std::remove_const_t<Knot>, TabulatedSpectrumKnotT<Scalar>>
[[nodiscard]] core::Result<SampledSpectrum<TransportSpectrumSampleCount, Scalar>>
evaluate_tabulated_spectrum(const std::span<Knot, Extent> knots,
                            const SampledWavelengthsT<Scalar>& wavelengths) {
    const auto knot_view = std::span<const TabulatedSpectrumKnotT<Scalar>>{knots};
    const auto table_status = basic_spectra_detail::validate_tabulated_knots(knot_view);
    if (!table_status.has_value()) {
        return std::unexpected(table_status.error());
    }

    auto result = SampledSpectrum<TransportSpectrumSampleCount, Scalar>{};
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto value = basic_spectra_detail::evaluate_validated_tabulation(
            knot_view, wavelengths[lane].nanometers);
        if (!value.has_value()) {
            return std::unexpected(value.error());
        }
        result[lane] = *value;
    }
    return result;
}

static_assert(std::is_standard_layout_v<TabulatedSpectrumKnot>);
static_assert(std::is_trivially_copyable_v<TabulatedSpectrumKnot>);
static_assert(std::is_standard_layout_v<ReferenceTabulatedSpectrumKnot>);
static_assert(std::is_trivially_copyable_v<ReferenceTabulatedSpectrumKnot>);
static_assert(sizeof(TabulatedSpectrumKnot) == 2 * sizeof(TransportScalar));
static_assert(sizeof(ReferenceTabulatedSpectrumKnot) == 2 * sizeof(ReferenceScalar));

} // namespace blackframe::renderer
