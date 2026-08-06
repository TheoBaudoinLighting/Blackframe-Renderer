#include <Blackframe/Engine/Detail/SceneSurfaceQuery.hpp>
#include <Blackframe/Engine/SceneBsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_path_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

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

    auto query = detail::SceneSurfaceQuery{acceleration};
    const auto resolved_environment =
        renderer::BsdfOnlyEnvironment{*environment, environment_record.wavelengths};
    return renderer::bsdf_only_path_loop_detail::trace_closure_bsdf_only_with_query(
        initial_ray, initial_state, sample_stream, query, resolved_environment, depth_limits,
        roulette_policy);
}

} // namespace blackframe::engine
