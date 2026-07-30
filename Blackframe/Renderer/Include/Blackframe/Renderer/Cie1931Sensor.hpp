#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>

namespace blackframe::renderer {

// Converts four unweighted spectral contributions to relative CIE XYZ with the 1931 2-degree
// standard observer. Every wavelength PDF is interpreted as a marginal density in inverse
// nanometers. A constant unit spectrum has Y=1 in expectation.
[[nodiscard]] core::Result<XYZ> cie_1931_spectrum_to_xyz(const TransportSpectrum& spectrum,
                                                         const SampledWavelengths& wavelengths);

[[nodiscard]] core::Result<ReferenceXYZ>
cie_1931_spectrum_to_xyz(const ReferenceSpectrum& spectrum,
                         const ReferenceSampledWavelengths& wavelengths);

} // namespace blackframe::renderer
