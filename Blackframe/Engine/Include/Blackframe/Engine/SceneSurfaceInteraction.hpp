#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/ClosureMixture.hpp>
#include <Blackframe/Renderer/Emission.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <optional>

namespace blackframe::engine {

// A closest hit resolved against the exact immutable scene snapshot that owns
// its geometry and material records. position_error is the conservative
// interpolation bound used by the robust continuation-ray offset.
struct ResolvedSceneSurface final {
    renderer::SurfaceInteraction interaction;
    renderer::Vector3 position_error;
    renderer::ClosureMixture closures;
    renderer::OrthonormalFrame closure_frame;
    renderer::OneSidedSurfaceEmission emission;
};

// The scalar reference path evaluates its two analytic neighbors on the extension of the central
// triangle. UVs and smooth normals are interpolated from that exact triangle rather than inferred
// from a replacement frame, so singular or out-of-range parameterizations are preserved.
struct ResolvedSceneSurfaceDifferentials final {
    renderer::SurfacePointDifferentials positions;
    renderer::TextureCoordinateDifferentials texture_coordinates;
    renderer::Normal3 rx_shading_normal;
    renderer::Normal3 ry_shading_normal;
};

struct ResolvedSceneSurfaceWithDifferentials final {
    ResolvedSceneSurface surface;
    ResolvedSceneSurfaceDifferentials differentials;
};

// Resolves already traversed closest-hit data without issuing a second acceleration query. The
// caller owns the provenance contract: hit must come from the supplied immutable scene snapshot
// and ray. This function validates the observable identifiers, barycentrics, geometry, ray
// parameter, and material lookup, but cannot prove which equivalent snapshot produced the hit.
[[nodiscard]] core::Result<ResolvedSceneSurface>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit, const renderer::Ray& ray);

[[nodiscard]] core::Result<ResolvedSceneSurfaceWithDifferentials>
resolve_scene_surface_hit(const FrameScene& scene, const AccelHit& hit,
                          const renderer::RayDifferential& ray);

[[nodiscard]] core::Result<std::optional<ResolvedSceneSurface>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::Ray& ray);

[[nodiscard]] core::Result<std::optional<ResolvedSceneSurfaceWithDifferentials>>
resolve_scene_surface(const AccelBackend& acceleration, const renderer::RayDifferential& ray);

} // namespace blackframe::engine
