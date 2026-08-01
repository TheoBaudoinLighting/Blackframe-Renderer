#pragma once

#include <Blackframe/Renderer/ClosureSet.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>

namespace blackframe::renderer {

// SampleStream, PathState, ClosureSet, LightSampler, and MediumTracker are the only five central
// transport interfaces. ClosureSet and LightSampler are defined; MediumTracker remains reserved
// until its owning feature is implemented.
struct MediumTracker;

} // namespace blackframe::renderer
