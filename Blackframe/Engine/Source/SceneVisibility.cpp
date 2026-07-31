#include <Blackframe/Engine/SceneVisibility.hpp>

namespace blackframe::engine {

core::Result<renderer::TransportSpectrum> trace_vacuum_visibility(const AccelBackend& acceleration,
                                                                  const renderer::Ray& ray) {
    if (ray.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::unavailable,
            .message = "Scene visibility supports vacuum transmittance only.",
        });
    }

    const auto occluded = acceleration.occluded(ray);
    if (!occluded) {
        return std::unexpected(occluded.error());
    }
    if (*occluded) {
        return renderer::TransportSpectrum{};
    }
    return renderer::TransportSpectrum{.values = {1.0F, 1.0F, 1.0F, 1.0F}};
}

} // namespace blackframe::engine
