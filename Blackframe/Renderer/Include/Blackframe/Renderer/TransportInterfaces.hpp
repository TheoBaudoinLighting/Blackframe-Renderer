#pragma once

#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>

namespace blackframe::renderer {

// SampleStream and these four reserved contracts are the only five central transport interfaces.
// The remaining operations stay undefined until their owning features are implemented.
struct ClosureSet;
struct LightSampler;
struct MediumTracker;
struct PathState;

} // namespace blackframe::renderer
