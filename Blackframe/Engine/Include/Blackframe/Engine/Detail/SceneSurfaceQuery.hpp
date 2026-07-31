#pragma once

#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <optional>
#include <utility>

namespace blackframe::engine::detail {

// Compile-time adapter shared by scalar scene transport loops. It resolves
// every hit through the selected AccelBackend and does not create another
// runtime transport or acceleration interface.
struct ScenePathSurface final {
    ResolvedSceneSurface surface;

    [[nodiscard]] const renderer::SurfaceInteraction& interaction() const noexcept {
        return surface.interaction;
    }

    [[nodiscard]] const renderer::Normal3& geometric_normal() const noexcept {
        return surface.interaction.geometric_normal();
    }

    [[nodiscard]] const renderer::LambertianReflection& reflection() const noexcept {
        return surface.reflection;
    }

    [[nodiscard]] const renderer::OneSidedSurfaceEmission& emission() const noexcept {
        return surface.emission;
    }

    [[nodiscard]] const renderer::Point3& position() const noexcept {
        return surface.interaction.position();
    }

    [[nodiscard]] const renderer::Vector3& position_error() const noexcept {
        return surface.position_error;
    }
};

class SceneSurfaceQuery final {
  public:
    explicit SceneSurfaceQuery(const AccelBackend& acceleration) noexcept
        : acceleration_{acceleration} {}

    [[nodiscard]] core::Status validate(const renderer::SampledWavelengths&) const noexcept {
        // FrameScene construction closes every spectral material against its
        // environment packet before any acceleration backend can commit it.
        return {};
    }

    [[nodiscard]] core::Result<std::optional<ScenePathSurface>>
    closest_hit(const renderer::Ray& ray) const {
        auto resolved = resolve_scene_surface(acceleration_, ray);
        if (!resolved) {
            return std::unexpected(resolved.error());
        }
        if (!*resolved) {
            return std::optional<ScenePathSurface>{};
        }
        return std::optional<ScenePathSurface>{ScenePathSurface{.surface = std::move(**resolved)}};
    }

  private:
    const AccelBackend& acceleration_;
};

} // namespace blackframe::engine::detail
