#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <optional>

namespace blackframe::engine {

// A closest hit resolved against the exact immutable scene snapshot that owns
// its geometry and material records. position_error is the conservative
// interpolation bound used by the robust continuation-ray offset.
struct ResolvedSceneSurface final {
    renderer::SurfaceInteraction interaction;
    renderer::Vector3 position_error;
    renderer::LambertianReflection reflection;
    renderer::OneSidedSurfaceEmission emission;
};

// Resolves already traversed closest-hit data without issuing a second acceleration query. The
// caller owns the provenance contract: hit must come from the supplied immutable scene snapshot
// and ray. This function validates the observable identifiers, barycentrics, geometry, ray
// parameter, and material lookup, but cannot prove which equivalent snapshot produced the hit.
[[nodiscard]] core::Result<ResolvedSceneSurface>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit, const renderer::Ray& ray);

[[nodiscard]] core::Result<std::optional<ResolvedSceneSurface>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::Ray& ray);

} // namespace blackframe::engine
