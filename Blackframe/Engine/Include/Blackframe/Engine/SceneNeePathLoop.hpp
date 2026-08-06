#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>

namespace blackframe::engine {

// Traces the scalar bounded-closure path loop and samples exactly one
// punctual-light registry slot at every accepted non-delta surface vertex.
// Point, directional, and spot lights are delta emitters that cannot also be
// reached as emissive geometry, so this pre-MIS estimator remains disjoint
// from the historical hit-surface and environment contributions. Visibility
// always goes through AccelBackend::occluded(); only vacuum transmittance is
// currently available. The supplied sampler must expose one slot per committed
// punctual-light record and positive support for every non-black record at the
// evaluated context.
[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_scene_nee(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy);

} // namespace blackframe::engine
