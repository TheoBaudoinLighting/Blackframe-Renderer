#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <vector>

namespace blackframe::engine::scene_hierarchy_validation {

struct ResolvedTransforms final {
    std::vector<renderer::AffineTransform> local;
    std::vector<renderer::AffineTransform> world;
};

// Instances must already be sorted by identifier. The result is index-aligned with that canonical
// order and validates local transforms, parent references, cycles, and every world composition.
[[nodiscard]] core::Result<ResolvedTransforms>
resolve_instance_transforms(const std::vector<SceneInstance>& instances);

} // namespace blackframe::engine::scene_hierarchy_validation
