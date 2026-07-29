#pragma once

#include <Blackframe/Renderer/Ray.hpp>
#include <cstdint>
#include <string>

namespace blackframe::renderer {

inline constexpr std::uint32_t CurrentRayDiagnosticSchemaVersion = 1;

// Scalars are serialized as fixed-width IEEE bit patterns so diagnostics remain exact and
// locale-independent, including signed zero and infinity.
[[nodiscard]] std::string serialize_ray_diagnostic(const Ray& ray);
[[nodiscard]] std::string serialize_ray_diagnostic(const ReferenceRay& ray);

} // namespace blackframe::renderer
