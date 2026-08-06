#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/SceneIdentifiers.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>

namespace blackframe::engine {

// Scalar-reference evaluation widens the exact transport payload after typed lookup. It does not
// synthesize spectral data from color values or apply material-domain range constraints.
[[nodiscard]] core::Result<renderer::ReferenceScalar>
evaluate_scalar_reference_constant_float_texture(const FrameScene& scene,
                                                 renderer::TextureId texture);
[[nodiscard]] core::Result<renderer::ReferenceLinearRGB>
evaluate_scalar_reference_constant_color_texture(const FrameScene& scene,
                                                 renderer::TextureId texture);
[[nodiscard]] core::Result<renderer::ReferenceSpectrum>
evaluate_scalar_reference_constant_spectrum_texture(const FrameScene& scene,
                                                    renderer::TextureId texture);

// CPU transport evaluation preserves the stored float payload exactly. A missing identifier or a
// value-kind mismatch is an explicit error and is never replaced with zero or another texture.
[[nodiscard]] core::Result<renderer::TransportScalar>
evaluate_cpu_transport_constant_float_texture(const FrameScene& scene, renderer::TextureId texture);
[[nodiscard]] core::Result<renderer::LinearRGB>
evaluate_cpu_transport_constant_color_texture(const FrameScene& scene, renderer::TextureId texture);
[[nodiscard]] core::Result<renderer::TransportSpectrum>
evaluate_cpu_transport_constant_spectrum_texture(const FrameScene& scene,
                                                 renderer::TextureId texture);

} // namespace blackframe::engine
