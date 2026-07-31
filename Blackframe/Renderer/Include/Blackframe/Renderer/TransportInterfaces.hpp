#pragma once

#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>

namespace blackframe::renderer {

// SampleStream, PathState, and these three reserved contracts are the only five central transport
// interfaces. LightSampler is now defined; the remaining operations stay undefined until their
// owning features are implemented.
struct ClosureSet;
struct MediumTracker;

} // namespace blackframe::renderer
