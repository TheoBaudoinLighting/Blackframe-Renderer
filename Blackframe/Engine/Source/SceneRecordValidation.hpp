#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <vector>

namespace blackframe::engine::scene_record_validation {

[[nodiscard]] core::Status validate_constant_texture(const SceneConstantTexture& record);
[[nodiscard]] core::Status validate_host_image_texture(const SceneHostImageTexture& record);
[[nodiscard]] core::Status
reject_shared_texture_identifiers(const std::vector<SceneConstantTexture>& constants,
                                  const std::vector<SceneHostImageTexture>& host_images);
[[nodiscard]] core::Status
validate_normal_map_binding(const std::vector<SceneHostImageTexture>& textures,
                            const SceneNormalMapBinding& binding);
[[nodiscard]] core::Status
validate_bump_map_binding(const std::vector<SceneHostImageTexture>& textures,
                          const SceneBumpMapBinding& binding);
[[nodiscard]] core::Status validate_closure_mixture(const SceneClosureMixture& closure_mixture);

} // namespace blackframe::engine::scene_record_validation
