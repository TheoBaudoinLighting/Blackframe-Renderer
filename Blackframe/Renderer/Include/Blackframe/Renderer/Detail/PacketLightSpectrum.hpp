#pragma once

#include <Blackframe/Renderer/Light.hpp>
#include <array>
#include <cmath>
#include <cstddef>

namespace blackframe::renderer::light_detail {

template <SpectrumScalar Scalar>
[[nodiscard]] core::Status
validate_light_wavelength_packet(const SampledWavelengthsT<Scalar>& wavelengths) {
    for (const auto& sample : wavelengths.samples) {
        if (!std::isfinite(sample.nanometers) ||
            sample.nanometers < Scalar{VisibleWavelengthMinimumNanometers} ||
            sample.nanometers > Scalar{VisibleWavelengthMaximumNanometers} ||
            sample.probability.measure != ProbabilityMeasure::wavelength ||
            !std::isfinite(sample.probability.value) || !(sample.probability.value > Scalar{0})) {
            return std::unexpected(invalid_light(
                "Packet-bound lights require finite visible wavelengths with positive "
                "wavelength densities."));
        }
    }
    return {};
}

// Constant light data is bound to the exact wavelength locations supplied at
// construction. Probability densities may change when the same packet is
// replayed because they are sampling metadata rather than spectral values.
template <SpectrumScalar Scalar> class PacketLightSpectrumT final {
  public:
    [[nodiscard]] static core::Result<PacketLightSpectrumT>
    create(const SampledWavelengthsT<Scalar>& wavelengths, const LightSpectrumT<Scalar>& values) {
        const auto wavelength_status = validate_light_wavelength_packet(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        if (!finite_non_negative(values)) {
            return std::unexpected(
                invalid_light("Packet-bound light spectra require finite non-negative lanes."));
        }

        auto nanometers = std::array<Scalar, TransportSpectrumSampleCount>{};
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            nanometers[lane] = wavelengths[lane].nanometers;
        }
        return PacketLightSpectrumT{nanometers, values};
    }

    [[nodiscard]] core::Result<LightSpectrumT<Scalar>>
    evaluate(const SampledWavelengthsT<Scalar>& wavelengths) const {
        const auto wavelength_status = validate_light_wavelength_packet(wavelengths);
        if (!wavelength_status) {
            return std::unexpected(wavelength_status.error());
        }
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            if (wavelengths[lane].nanometers != nanometers_[lane]) {
                return std::unexpected(invalid_light(
                    "A packet-bound light cannot be evaluated at different wavelengths."));
            }
        }
        return values_;
    }

    [[nodiscard]] constexpr bool is_black() const noexcept {
        for (const auto value : values_.values) {
            if (value != Scalar{0}) {
                return false;
            }
        }
        return true;
    }

  private:
    constexpr PacketLightSpectrumT(
        const std::array<Scalar, TransportSpectrumSampleCount> nanometers,
        const LightSpectrumT<Scalar> values) noexcept
        : nanometers_{nanometers}, values_{values} {}

    std::array<Scalar, TransportSpectrumSampleCount> nanometers_;
    LightSpectrumT<Scalar> values_;
};

} // namespace blackframe::renderer::light_detail
