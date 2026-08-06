#include <Blackframe/Engine/Detail/SceneSurfaceQuery.hpp>
#include <Blackframe/Engine/SceneNeePathLoop.hpp>
#include <Blackframe/Engine/SceneVisibility.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/DirectLighting.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <Blackframe/Renderer/ShadingNormalCorrection.hpp>
#include <Blackframe/Renderer/ShadowRay.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_nee_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] core::Result<std::optional<renderer::IncidentLightSample>>
sample_punctual_light(const ScenePunctualLight& light, const renderer::LightSampleContext& context,
                      const renderer::Point2 canonical_sample,
                      const renderer::SampledWavelengths& wavelengths) {
    return std::visit(
        [&](const auto& record) -> core::Result<std::optional<renderer::IncidentLightSample>> {
            using Record = std::remove_cvref_t<decltype(record)>;
            if constexpr (std::same_as<Record, ScenePointLight>) {
                const auto model =
                    renderer::PointLight::create(record.position, record.absolute_position_error,
                                                 wavelengths, record.spectral_radiant_intensity);
                if (!model) {
                    return std::unexpected(model.error());
                }
                return model->sample_li(context, canonical_sample, wavelengths);
            } else if constexpr (std::same_as<Record, SceneDirectionalLight>) {
                const auto model = renderer::DirectionalLight::create(
                    record.propagation_direction, wavelengths, record.spectral_irradiance);
                if (!model) {
                    return std::unexpected(model.error());
                }
                return model->sample_li(context, canonical_sample, wavelengths);
            } else if constexpr (std::same_as<Record, SceneSpotLight>) {
                const auto model = renderer::SpotLight::create(
                    record.position, record.absolute_position_error, record.emission_direction,
                    record.inner_half_angle_radians, record.outer_half_angle_radians, wavelengths,
                    record.on_axis_spectral_radiant_intensity);
                if (!model) {
                    return std::unexpected(model.error());
                }
                return model->sample_li(context, canonical_sample, wavelengths);
            } else {
                static_assert(!std::same_as<Record, Record>,
                              "Every scene punctual-light record must be sampled explicitly.");
            }
        },
        light);
}

[[nodiscard]] bool is_black(const ScenePunctualLight& light) noexcept {
    return std::visit(
        [](const auto& record) {
            const auto& spectrum = [&]() -> const renderer::TransportSpectrum& {
                using Record = std::remove_cvref_t<decltype(record)>;
                if constexpr (std::same_as<Record, ScenePointLight>) {
                    return record.spectral_radiant_intensity;
                } else if constexpr (std::same_as<Record, SceneDirectionalLight>) {
                    return record.spectral_irradiance;
                } else {
                    return record.on_axis_spectral_radiant_intensity;
                }
            }();
            for (const auto value : spectrum.values) {
                if (value != 0.0F) {
                    return false;
                }
            }
            return true;
        },
        light);
}

[[nodiscard]] bool is_black(const renderer::TransportSpectrum& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (value != 0.0F) {
            return false;
        }
    }
    return true;
}

class ScenePunctualDirectLighting final {
  public:
    static constexpr bool enabled = true;

    ScenePunctualDirectLighting(const AccelBackend& acceleration,
                                const renderer::LightSampler& sampler,
                                FrameSceneHandle scene) noexcept
        : acceleration_{acceleration}, sampler_{sampler}, scene_{std::move(scene)} {}

    [[nodiscard]] core::Result<renderer::TransportSpectrum>
    estimate(const detail::ScenePathSurface& surface,
             const renderer::detail::DepthFilteredClosureMixture& closures,
             const renderer::TransportSpectrum& beta, const renderer::OrthonormalFrame& frame,
             const renderer::Vector3 outgoing_world,
             const renderer::BounceSampleDimensions& dimensions,
             const renderer::SampleStream& sample_stream, const renderer::Ray& path_ray) const {
        auto all_black = true;
        for (auto index = std::size_t{}; index < closures.size(); ++index) {
            all_black = all_black && is_black(closures.active_closure(index).weight);
        }
        if (all_black) {
            return renderer::TransportSpectrum{};
        }
        const auto context = renderer::LightSampleContext::create(surface.interaction().position(),
                                                                  surface.interaction().time());
        if (!context) {
            return std::unexpected(context.error());
        }

        const auto lights = scene_->punctual_lights();
        for (auto index = std::size_t{}; index < lights.size(); ++index) {
            const auto probability =
                sampler_.probability(*context, static_cast<std::uint32_t>(index));
            if (!probability) {
                return std::unexpected(probability.error());
            }
            if (!is_black(lights[index]) && !(probability->value() > 0.0F)) {
                return std::unexpected(scene_nee_error(
                    core::StatusCode::incompatible,
                    "The light sampler gives zero support to a non-black punctual light."));
            }
        }

        const auto selection =
            sampler_.sample(*context, sample_stream.sample_1d(dimensions.light_selection));
        if (!selection) {
            return std::unexpected(selection.error());
        }

        const auto light_index = static_cast<std::size_t>(selection->light_index());
        if (light_index >= lights.size()) {
            return std::unexpected(scene_nee_error(
                core::StatusCode::incompatible,
                "The light sampler selected a slot outside the committed punctual registry."));
        }
        const auto incident =
            sample_punctual_light(lights[light_index], *context,
                                  renderer::Point2{
                                      .x = sample_stream.sample_1d(dimensions.light_u),
                                      .y = sample_stream.sample_1d(dimensions.light_v),
                                  },
                                  scene_->spectral_environment()->wavelengths);
        if (!incident) {
            return std::unexpected(incident.error());
        }
        if (!*incident) {
            return renderer::TransportSpectrum{};
        }
        if ((**incident).probability().measure != renderer::ProbabilityMeasure::discrete) {
            return std::unexpected(scene_nee_error(
                core::StatusCode::incompatible,
                "A scene punctual light returned a non-discrete conditional probability."));
        }

        const auto outgoing_local = frame.to_local(outgoing_world);
        const auto incoming_local = frame.to_local((**incident).direction_to_light());
        if (outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
            return renderer::TransportSpectrum{};
        }
        auto singleton_lambertian = std::optional<renderer::LambertianReflection>{};
        auto bsdf_value = renderer::TransportSpectrum{};
        if (closures.size() == 1U &&
            closures.active_closure(0U).kind == renderer::ClosureKind::lambertian_reflection) {
            const auto model =
                renderer::LambertianReflection::create(closures.active_closure(0U).weight);
            if (!model) {
                return std::unexpected(model.error());
            }
            if (is_black(model->reflectance())) {
                return renderer::TransportSpectrum{};
            }
            singleton_lambertian = *model;
        } else {
            const auto evaluated =
                closures.eval(outgoing_local, incoming_local, renderer::TransportMode::radiance);
            if (!evaluated) {
                return std::unexpected(evaluated.error());
            }
            if (is_black(*evaluated)) {
                return renderer::TransportSpectrum{};
            }
            bsdf_value = *evaluated;
        }
        const auto correction = renderer::shading_normal_correction(
            surface.geometric_normal(), surface.shading_normal(), outgoing_world,
            (**incident).direction_to_light(), renderer::TransportMode::radiance);
        if (!correction) {
            return std::unexpected(correction.error());
        }
        if (*correction == 0.0F) {
            return renderer::TransportSpectrum{};
        }

        const auto shadow_ray =
            renderer::make_shadow_ray(surface.interaction(), surface.position_error(), **incident,
                                      path_ray.mask(), path_ray.current_medium());
        if (!shadow_ray) {
            return std::unexpected(shadow_ray.error());
        }
        const auto transmittance = trace_vacuum_visibility(acceleration_, *shadow_ray);
        if (!transmittance) {
            return std::unexpected(transmittance.error());
        }
        const auto evaluated = singleton_lambertian
                                   ? renderer::evaluate_lambertian_direct_lighting(
                                         beta, *singleton_lambertian, frame, outgoing_world,
                                         selection->probability(), **incident, *transmittance)
                                   : renderer::evaluate_bsdf_direct_lighting(
                                         beta, bsdf_value, std::abs(incoming_local.z),
                                         selection->probability(), **incident, *transmittance);
        if (!evaluated) {
            return std::unexpected(evaluated.error());
        }
        return *evaluated * *correction;
    }

  private:
    const AccelBackend& acceleration_;
    const renderer::LightSampler& sampler_;
    FrameSceneHandle scene_;
};

} // namespace

core::Result<renderer::BsdfOnlyPathResult>
trace_scene_nee(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy) {
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::invalid_argument,
                            "The NEE ray and path state must carry the same current medium."));
    }
    if (initial_state.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::unavailable,
                            "The NEE path currently supports vacuum transmittance only."));
    }

    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::internal_error,
                            "The acceleration backend has no committed frame scene."));
    }
    if (!scene->spectral_environment()) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::unavailable,
                            "The committed frame scene has no spectral environment."));
    }
    const auto& environment_record = *scene->spectral_environment();
    if (environment_record.wavelengths != initial_state.wavelengths()) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::incompatible,
                            "The committed frame scene was not resolved at the path wavelengths."));
    }
    if (scene->punctual_lights().empty()) {
        return std::unexpected(scene_nee_error(
            core::StatusCode::unavailable,
            "NEE requires at least one punctual light in the committed frame scene."));
    }
    if (light_sampler.light_count() != scene->punctual_lights().size()) {
        return std::unexpected(scene_nee_error(
            core::StatusCode::incompatible,
            "The light sampler does not match the committed punctual-light registry."));
    }

    const auto environment = renderer::ConstantEnvironment::create(environment_record.radiance);
    if (!environment) {
        return std::unexpected(
            scene_nee_error(core::StatusCode::internal_error,
                            "The committed frame scene lost its validated spectral environment."));
    }

    auto query = detail::SceneSurfaceQuery{acceleration};
    auto direct_lighting = ScenePunctualDirectLighting{acceleration, light_sampler, scene};
    const auto resolved_environment =
        renderer::BsdfOnlyEnvironment{*environment, environment_record.wavelengths};
    return renderer::bsdf_only_path_loop_detail::trace_closure_mixture_with_query(
        initial_ray, initial_state, sample_stream, query, resolved_environment, depth_limits,
        roulette_policy, direct_lighting);
}

} // namespace blackframe::engine
