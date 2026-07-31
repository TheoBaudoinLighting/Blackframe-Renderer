#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>

namespace blackframe::engine {

// Traces one opaque visibility ray through the committed scene. Vacuum is the
// only supported medium until volumetric transmittance owns this operation.
// A non-vacuum ray is rejected instead of being treated as unattenuated.
[[nodiscard]] core::Result<renderer::TransportSpectrum>
trace_vacuum_visibility(const AccelBackend& acceleration, const renderer::Ray& ray);

} // namespace blackframe::engine
