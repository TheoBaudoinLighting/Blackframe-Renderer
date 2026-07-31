#include <Blackframe/Engine/SceneBsdfOnlyPathLoop.hpp>
#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <optional>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_path_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

struct ScenePathSurface final {
    ResolvedSceneSurface surface;

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
        // FrameScene construction already enforces one packet for its
        // environment and every material. The shared transport kernel checks
        // that packet against PathState before the first query.
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

} // namespace

core::Result<renderer::BsdfOnlyPathResult>
trace_scene_bsdf_only(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                      const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                      const renderer::PathDepthLimits& depth_limits,
                      const renderer::RussianRoulettePolicy& roulette_policy) {
    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            scene_path_error(core::StatusCode::internal_error,
                             "The acceleration backend has no committed frame scene."));
    }
    if (!scene->spectral_environment()) {
        return std::unexpected(
            scene_path_error(core::StatusCode::unavailable,
                             "The committed frame scene has no spectral environment."));
    }
    const auto& environment_record = *scene->spectral_environment();
    if (environment_record.wavelengths != initial_state.wavelengths()) {
        return std::unexpected(scene_path_error(
            core::StatusCode::incompatible,
            "The committed frame scene was not resolved at the path wavelengths."));
    }
    const auto environment = renderer::ConstantEnvironment::create(environment_record.radiance);
    if (!environment) {
        return std::unexpected(
            scene_path_error(core::StatusCode::internal_error,
                             "The committed frame scene lost its validated spectral environment."));
    }

    auto query = SceneSurfaceQuery{acceleration};
    const auto resolved_environment =
        renderer::BsdfOnlyEnvironment{*environment, environment_record.wavelengths};
    return renderer::bsdf_only_path_loop_detail::trace_bsdf_only_with_query(
        initial_ray, initial_state, sample_stream, query, resolved_environment, depth_limits,
        roulette_policy);
}

} // namespace blackframe::engine
