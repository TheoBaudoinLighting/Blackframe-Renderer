#pragma once

#include <Blackframe/Renderer/TransportConventions.hpp>

namespace blackframe::renderer {

// These are the only five central interfaces in the transport domain. Their operations remain
// intentionally undefined until the complete renderer feature inventory is approved.
struct SampleStream;
struct ClosureSet;
struct LightSampler;
struct MediumTracker;
struct PathState;

} // namespace blackframe::renderer
