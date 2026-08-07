#include <Blackframe/Engine/Detail/SceneSurfaceQuery.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/SceneVisibility.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/DirectLighting.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <Blackframe/Renderer/ShadingNormalCorrection.hpp>
#include <Blackframe/Renderer/ShadowRay.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error scene_mis_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] bool is_black(const renderer::TransportSpectrum& spectrum) noexcept {
    for (const auto value : spectrum.values) {
        if (value != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] renderer::PathDeltaFlags
next_delta_flags(const renderer::PathDeltaFlags current,
                 const renderer::ScatteringLobe event) noexcept {
    auto flags =
        renderer::has_path_delta_flag(current, renderer::PathDeltaFlags::any_non_delta_bounces)
            ? renderer::PathDeltaFlags::any_non_delta_bounces
            : renderer::PathDeltaFlags::none;
    if (renderer::is_delta_surface_scattering_event(event)) {
        flags = flags | renderer::PathDeltaFlags::previous_bounce_was_delta;
    } else {
        flags = flags | renderer::PathDeltaFlags::any_non_delta_bounces;
    }
    return flags;
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
            return is_black(spectrum);
        },
        light);
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

[[nodiscard]] core::Result<std::size_t> registered_light_count(const FrameScene& scene) {
    const auto punctual_count = scene.punctual_lights().size();
    const auto area_count = scene.mesh_area_lights().size();
    if (area_count > std::numeric_limits<std::size_t>::max() - punctual_count) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::resource_exhausted,
                            "The combined scene light-registry size is not representable."));
    }
    const auto count = punctual_count + area_count;
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::resource_exhausted,
                            "The combined scene light registry exceeds its 32-bit slot domain."));
    }
    return count;
}

class SceneMisDirectLighting final {
  public:
    SceneMisDirectLighting(const AccelBackend& acceleration, const renderer::LightSampler& sampler,
                           FrameSceneHandle scene, const renderer::MisHeuristic heuristic) noexcept
        : acceleration_{acceleration}, sampler_{sampler}, scene_{std::move(scene)},
          heuristic_{heuristic} {}

    [[nodiscard]] core::Result<renderer::TransportSpectrum>
    estimate(const detail::ScenePathSurface& surface,
             const renderer::detail::DepthFilteredClosureMixture& closures,
             const renderer::TransportSpectrum& beta, const renderer::OrthonormalFrame& frame,
             const renderer::Vector3 outgoing_world,
             const renderer::BounceSampleDimensions& dimensions,
             const renderer::SampleStream& sample_stream, const renderer::Ray& path_ray) const {
        if (is_black(beta)) {
            return renderer::TransportSpectrum{};
        }
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
        const auto registry_status = validate_sampler_support(*context);
        if (!registry_status) {
            return std::unexpected(registry_status.error());
        }

        const auto selection =
            sampler_.sample(*context, sample_stream.sample_1d(dimensions.light_selection));
        if (!selection) {
            return std::unexpected(selection.error());
        }
        const auto visible = selected_area_light_visible(selection->light_index(), path_ray.mask());
        if (!visible) {
            return std::unexpected(visible.error());
        }
        if (!*visible) {
            return renderer::TransportSpectrum{};
        }
        const auto incident =
            sample_registered_light(*context, selection->light_index(),
                                    renderer::Point2{
                                        .x = sample_stream.sample_1d(dimensions.light_u),
                                        .y = sample_stream.sample_1d(dimensions.light_v),
                                    });
        if (!incident) {
            return std::unexpected(incident.error());
        }
        if (!*incident) {
            return renderer::TransportSpectrum{};
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

        auto weight = renderer::TransportScalar{1};
        const auto conditional_probability = (**incident).probability();
        if (conditional_probability.measure == renderer::ProbabilityMeasure::solid_angle) {
            const auto bsdf_probability =
                closures.pdf(outgoing_local, incoming_local, renderer::TransportMode::radiance);
            if (!bsdf_probability) {
                return std::unexpected(bsdf_probability.error());
            }
            const auto light_probability = renderer::joint_light_pdf<renderer::TransportScalar>(
                selection->probability().probability_density(), conditional_probability);
            if (!light_probability) {
                return std::unexpected(light_probability.error());
            }
            const auto mis = renderer::mis_weight<renderer::TransportScalar>(
                heuristic_, *light_probability, *bsdf_probability);
            if (!mis) {
                return std::unexpected(mis.error());
            }
            weight = *mis;
        } else if (conditional_probability.measure != renderer::ProbabilityMeasure::discrete) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "A registered direct light returned an unsupported conditional measure."));
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
        const auto evaluated =
            singleton_lambertian
                ? renderer::evaluate_lambertian_direct_lighting(
                      beta, *singleton_lambertian, frame, outgoing_world, selection->probability(),
                      **incident, *transmittance, weight)
                : renderer::evaluate_bsdf_direct_lighting(
                      beta, bsdf_value, std::abs(incoming_local.z), selection->probability(),
                      **incident, *transmittance, weight);
        if (!evaluated) {
            return std::unexpected(evaluated.error());
        }
        return *evaluated * *correction;
    }

    [[nodiscard]] core::Result<renderer::TransportScalar>
    surface_emission_weight(const detail::ScenePathSurface& surface, const renderer::Ray& path_ray,
                            const renderer::TransportSpectrum& emitted_radiance) const {
        if (is_black(emitted_radiance) || !previous_bsdf_sample_) {
            return renderer::TransportScalar{1};
        }
        if (path_ray.direction() != previous_bsdf_sample_->incoming_world) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "The emissive hit direction does not match the recorded BSDF sample."));
        }

        const auto area_lights = scene_->mesh_area_lights();
        const auto instance_ids = scene_->mesh_area_light_instance_ids();
        if (area_lights.size() != instance_ids.size()) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::internal_error,
                                "The committed mesh-light registry lost its instance alignment."));
        }

        const auto hit_instance = surface.interaction().identifiers().instance;
        auto area_index = area_lights.size();
        for (auto index = std::size_t{}; index < instance_ids.size(); ++index) {
            if (instance_ids[index] == hit_instance) {
                area_index = index;
                break;
            }
        }
        if (area_index == area_lights.size()) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "A non-black emissive hit is absent from the committed mesh-light registry."));
        }

        const auto punctual_count = scene_->punctual_lights().size();
        if (area_index > std::numeric_limits<std::size_t>::max() - punctual_count) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::resource_exhausted,
                                "The emissive-hit light slot is not representable."));
        }
        const auto global_slot = punctual_count + area_index;
        if (global_slot > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::resource_exhausted,
                                "The emissive-hit light slot exceeds its 32-bit domain."));
        }

        const auto selection_probability = sampler_.probability(
            previous_bsdf_sample_->context, static_cast<std::uint32_t>(global_slot));
        if (!selection_probability) {
            return std::unexpected(selection_probability.error());
        }
        if (!(selection_probability->value() > 0.0F)) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "The light sampler gives zero support to the emissive surface hit by the BSDF."));
        }

        // The acceleration hit is the path endpoint. Re-intersecting the light from
        // the unoffset previous vertex can disagree at a mesh edge because
        // continuation rays deliberately use a robustly offset origin. The
        // mesh sampler is uniform over total area, so evaluate its exact-hit
        // Jacobian from the already validated endpoint instead of creating a
        // second geometric answer for the same path.
        const auto conditional_probability = area_lights[area_index].pdf_li_at_surface(
            previous_bsdf_sample_->context, surface.interaction().position(),
            surface.geometric_normal(), scene_->spectral_environment()->wavelengths);
        if (!conditional_probability) {
            return std::unexpected(conditional_probability.error());
        }

        const auto light_probability = renderer::joint_light_pdf<renderer::TransportScalar>(
            selection_probability->probability_density(),
            conditional_probability->probability_density());
        if (!light_probability) {
            return std::unexpected(light_probability.error());
        }
        return renderer::mis_weight<renderer::TransportScalar>(
            heuristic_, previous_bsdf_sample_->probability, *light_probability);
    }

    [[nodiscard]] core::Status record_bsdf_sample(const detail::ScenePathSurface& surface,
                                                  const renderer::ProbabilityDensity probability,
                                                  const renderer::Vector3 incoming_world,
                                                  const renderer::ScatteringLobe lobes) {
        if (renderer::is_delta_surface_scattering_event(lobes)) {
            if (probability.measure != renderer::DeltaBsdfProbabilityMeasure ||
                !std::isfinite(probability.value) || !(probability.value > 0.0F) ||
                probability.value > 1.0F) {
                return std::unexpected(scene_mis_error(
                    core::StatusCode::incompatible,
                    "A delta MIS continuation requires a positive discrete BSDF mass."));
            }
            previous_bsdf_sample_.reset();
            return {};
        }
        if (probability.measure != renderer::ProbabilityMeasure::solid_angle ||
            !std::isfinite(probability.value) || !(probability.value > 0.0F)) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "A MIS continuation requires a positive solid-angle BSDF density."));
        }
        const auto context = renderer::LightSampleContext::create(surface.interaction().position(),
                                                                  surface.interaction().time());
        if (!context) {
            return std::unexpected(context.error());
        }
        previous_bsdf_sample_ = PreviousBsdfSample{
            .context = *context,
            .probability = probability,
            .incoming_world = incoming_world,
        };
        return {};
    }

  private:
    struct PreviousBsdfSample final {
        renderer::LightSampleContext context;
        renderer::ProbabilityDensity probability;
        renderer::Vector3 incoming_world;
    };

    [[nodiscard]] core::Status
    validate_sampler_support(const renderer::LightSampleContext& context) const {
        const auto punctual_lights = scene_->punctual_lights();
        const auto area_lights = scene_->mesh_area_lights();
        for (auto index = std::size_t{}; index < punctual_lights.size(); ++index) {
            const auto probability =
                sampler_.probability(context, static_cast<std::uint32_t>(index));
            if (!probability) {
                return std::unexpected(probability.error());
            }
            if (!is_black(punctual_lights[index]) && !(probability->value() > 0.0F)) {
                return std::unexpected(scene_mis_error(
                    core::StatusCode::incompatible,
                    "The light sampler gives zero support to a non-black punctual light."));
            }
        }
        for (auto index = std::size_t{}; index < area_lights.size(); ++index) {
            const auto global_index = punctual_lights.size() + index;
            const auto probability =
                sampler_.probability(context, static_cast<std::uint32_t>(global_index));
            if (!probability) {
                return std::unexpected(probability.error());
            }
            if (!(probability->value() > 0.0F)) {
                return std::unexpected(scene_mis_error(
                    core::StatusCode::incompatible,
                    "The light sampler gives zero support to a non-black mesh area light."));
            }
        }
        return {};
    }

    [[nodiscard]] core::Result<std::optional<renderer::IncidentLightSample>>
    sample_registered_light(const renderer::LightSampleContext& context,
                            const std::uint32_t light_index,
                            const renderer::Point2 canonical_sample) const {
        const auto punctual_lights = scene_->punctual_lights();
        const auto area_lights = scene_->mesh_area_lights();
        const auto index = static_cast<std::size_t>(light_index);
        const auto& wavelengths = scene_->spectral_environment()->wavelengths;
        if (index < punctual_lights.size()) {
            return sample_punctual_light(punctual_lights[index], context, canonical_sample,
                                         wavelengths);
        }
        const auto area_index = index - punctual_lights.size();
        if (area_index >= area_lights.size()) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "The light sampler selected a slot outside the committed light registry."));
        }
        return area_lights[area_index].sample_li(context, canonical_sample, wavelengths);
    }

    [[nodiscard]] core::Result<bool>
    selected_area_light_visible(const std::uint32_t light_index,
                                const renderer::RayMask path_mask) const {
        const auto punctual_count = scene_->punctual_lights().size();
        const auto index = static_cast<std::size_t>(light_index);
        if (index < punctual_count) {
            return true;
        }
        const auto area_index = index - punctual_count;
        const auto instance_ids = scene_->mesh_area_light_instance_ids();
        if (area_index >= instance_ids.size()) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::incompatible,
                "The light sampler selected a slot outside the committed light registry."));
        }
        const auto instance = scene_->instance(instance_ids[area_index]);
        if (!instance) {
            return std::unexpected(instance.error());
        }
        return (instance->get().visibility_mask & path_mask) != 0U;
    }

    const AccelBackend& acceleration_;
    const renderer::LightSampler& sampler_;
    FrameSceneHandle scene_;
    renderer::MisHeuristic heuristic_;
    std::optional<PreviousBsdfSample> previous_bsdf_sample_;
};

[[nodiscard]] core::Result<renderer::TransportSpectrum>
accumulate_weighted_emission(const renderer::TransportSpectrum& accumulated,
                             const renderer::TransportSpectrum& throughput,
                             const renderer::TransportSpectrum& emitted_radiance,
                             const renderer::TransportScalar estimator_weight) {
    if (!std::isfinite(estimator_weight) || estimator_weight < 0.0F || estimator_weight > 1.0F) {
        return std::unexpected(scene_mis_error(
            core::StatusCode::invalid_argument,
            "An emitted-radiance MIS weight must lie in the closed interval [0, 1]."));
    }
    if (estimator_weight == 1.0F) {
        return renderer::bsdf_only_path_loop_detail::accumulate_emission(accumulated, throughput,
                                                                         emitted_radiance);
    }

    auto result = renderer::TransportSpectrum{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        if (throughput[lane] == 0.0F || emitted_radiance[lane] == 0.0F ||
            estimator_weight == 0.0F) {
            result[lane] = accumulated[lane];
            continue;
        }

        auto throughput_exponent = 0;
        auto emission_exponent = 0;
        auto weight_exponent = 0;
        const auto significand = std::frexp(throughput[lane], &throughput_exponent) *
                                 std::frexp(emitted_radiance[lane], &emission_exponent) *
                                 std::frexp(estimator_weight, &weight_exponent);
        auto normalization_exponent = 0;
        const auto normalized = std::frexp(significand, &normalization_exponent);
        const auto contribution =
            std::scalbn(normalized, throughput_exponent + emission_exponent + weight_exponent +
                                        normalization_exponent);
        if (!std::isfinite(contribution) || !(contribution > 0.0F)) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::invalid_argument,
                                "A weighted emitted-radiance contribution is not representable."));
        }
        result[lane] = accumulated[lane] + contribution;
        if (!std::isfinite(result[lane])) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::invalid_argument,
                                "MIS radiance accumulation is not representable."));
        }
    }
    return result;
}

struct RayDifferentialTracker final {
    std::optional<renderer::RayDifferential> active;
    RayDifferentialLossReason loss_reason{RayDifferentialLossReason::none};
    std::uint32_t propagated_specular_bounces{};
};

[[nodiscard]] core::Status update_ray_differentials(RayDifferentialTracker& tracker,
                                                    const detail::ScenePathSurface& surface,
                                                    const renderer::ClosureMixtureSample& sample,
                                                    const renderer::Vector3 outgoing_local,
                                                    const renderer::Ray& next_ray) {
    if (!tracker.active) {
        return {};
    }
    if (!renderer::is_delta_surface_scattering_event(sample.lobes)) {
        tracker.active.reset();
        tracker.loss_reason = RayDifferentialLossReason::non_delta_scattering;
        return {};
    }
    if (!surface.differentials()) {
        return std::unexpected(scene_mis_error(
            core::StatusCode::internal_error,
            "An active scalar ray differential reached a surface without differential data."));
    }
    const auto closures = surface.closures().closure_set().closures();
    if (sample.selected_closure >= closures.size()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::internal_error,
                            "A sampled delta closure does not identify a source closure record."));
    }
    const auto& closure = closures[sample.selected_closure];
    const auto& differentials = *surface.differentials();
    const auto rx_position = surface.position() + differentials.positions.dpdx;
    const auto ry_position = surface.position() + differentials.positions.dpdy;

    switch (closure.kind) {
    case renderer::ClosureKind::specular_reflection: {
        const auto propagated = renderer::propagate_specular_reflection(
            *tracker.active, next_ray, surface.shading_normal(), rx_position,
            differentials.rx_shading_normal, ry_position, differentials.ry_shading_normal);
        if (!propagated) {
            return std::unexpected(propagated.error());
        }
        if (tracker.propagated_specular_bounces == std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::resource_exhausted,
                                "The scalar ray-differential bounce count is not representable."));
        }
        tracker.active = *propagated;
        ++tracker.propagated_specular_bounces;
        return {};
    }
    case renderer::ClosureKind::specular_transmission: {
        const auto propagated = renderer::propagate_specular_transmission(
            *tracker.active, next_ray, surface.shading_normal(), rx_position,
            differentials.rx_shading_normal, ry_position, differentials.ry_shading_normal,
            outgoing_local.z > 0.0F, closure.parameters[0], closure.parameters[1]);
        if (!propagated) {
            return std::unexpected(propagated.error());
        }
        if (!*propagated) {
            tracker.active.reset();
            tracker.loss_reason = RayDifferentialLossReason::specular_discontinuity;
            return {};
        }
        if (tracker.propagated_specular_bounces == std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::resource_exhausted,
                                "The scalar ray-differential bounce count is not representable."));
        }
        tracker.active = **propagated;
        ++tracker.propagated_specular_bounces;
        return {};
    }
    default:
        return std::unexpected(scene_mis_error(
            core::StatusCode::incompatible,
            "A delta scattering event selected a closure without differential propagation."));
    }
}

// The MIS continuation state stays in this dedicated loop so the frozen
// BSDF-only scalar oracle remains an independent transport implementation.
[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult> trace_lambertian_mis_with_query(
    const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
    const renderer::SampleStream& sample_stream, detail::SceneSurfaceQuery& surface_query,
    const renderer::BsdfOnlyEnvironment& environment, const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy, SceneMisDirectLighting& direct_lighting,
    RayDifferentialTracker* const differentials) {
    const auto roulette_status = renderer::validate_russian_roulette_policy(roulette_policy);
    if (!roulette_status) {
        return std::unexpected(roulette_status.error());
    }
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::invalid_argument,
                            "The MIS ray and path state must carry the same current medium."));
    }
    if (!renderer::bsdf_only_path_loop_detail::finite_non_negative(initial_state.beta())) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::invalid_argument,
                            "MIS path throughput must be finite and non-negative."));
    }
    const auto initial_depth_status = renderer::validate_path_depth_state(
        depth_limits, initial_state.depth_counters(), initial_state.depth());
    if (!initial_depth_status) {
        return std::unexpected(initial_depth_status.error());
    }
    if (environment.wavelengths() != initial_state.wavelengths()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::invalid_argument,
                            "The MIS environment was not resolved at the path wavelengths."));
    }
    const auto query_status = surface_query.validate(initial_state.wavelengths());
    if (!query_status) {
        return std::unexpected(query_status.error());
    }

    auto beta = initial_state.beta();
    auto radiance = initial_state.accumulated_radiance();
    auto depth = initial_state.depth();
    auto depth_counters = initial_state.depth_counters();
    auto eta_scale = initial_state.eta_scale();
    const auto wavelengths = initial_state.wavelengths();
    auto delta_flags = initial_state.delta_flags();
    const auto current_medium = initial_state.current_medium();
    auto ray = initial_ray;

    const auto finish = [&](const renderer::BsdfOnlyPathTermination termination,
                            const renderer::ScatteringLobe blocked_depth_limits)
        -> core::Result<renderer::BsdfOnlyPathResult> {
        return renderer::bsdf_only_path_loop_detail::make_result(
            beta, radiance, depth, eta_scale, wavelengths, delta_flags, current_medium,
            depth_limits, depth_counters, ray, termination, blocked_depth_limits);
    };

    while (true) {
        const auto resolved = differentials != nullptr && differentials->active
                                  ? surface_query.closest_hit(*differentials->active)
                                  : surface_query.closest_hit(ray);
        if (!resolved) {
            return std::unexpected(resolved.error());
        }
        if (!*resolved) {
            const auto emitted = environment.environment().eval(ray.direction());
            if (!emitted) {
                return std::unexpected(emitted.error());
            }
            const auto accumulated =
                renderer::bsdf_only_path_loop_detail::accumulate_emission(radiance, beta, *emitted);
            if (!accumulated) {
                return std::unexpected(accumulated.error());
            }
            radiance = *accumulated;
            return finish(renderer::BsdfOnlyPathTermination::escaped_environment,
                          renderer::ScatteringLobe::none);
        }

        const auto& surface = **resolved;
        const auto emitted = surface.emission().eval(surface.geometric_normal(), -ray.direction());
        if (!emitted) {
            return std::unexpected(emitted.error());
        }
        const auto emission_weight =
            direct_lighting.surface_emission_weight(surface, ray, *emitted);
        if (!emission_weight) {
            return std::unexpected(emission_weight.error());
        }
        const auto accumulated =
            accumulate_weighted_emission(radiance, beta, *emitted, *emission_weight);
        if (!accumulated) {
            return std::unexpected(accumulated.error());
        }
        radiance = *accumulated;

        if (renderer::bsdf_only_path_loop_detail::zero_spectrum(beta)) {
            return finish(renderer::BsdfOnlyPathTermination::zero_throughput,
                          renderer::ScatteringLobe::none);
        }

        const auto filtered_closures = renderer::detail::DepthFilteredClosureMixture::create(
            surface.closures(), depth_limits, depth_counters);
        if (!filtered_closures) {
            return std::unexpected(filtered_closures.error());
        }
        if (filtered_closures->empty()) {
            if (filtered_closures->source_empty()) {
                return finish(renderer::BsdfOnlyPathTermination::outside_bsdf_support,
                              renderer::ScatteringLobe::none);
            }
            return finish(renderer::BsdfOnlyPathTermination::depth_limit,
                          filtered_closures->blocked_lobes());
        }

        const auto& frame = surface.closure_frame();
        const auto outgoing_world =
            renderer::bsdf_only_path_loop_detail::robust_unit_direction(-ray.direction());
        if (!outgoing_world) {
            return std::unexpected(outgoing_world.error());
        }
        const auto outgoing_local = frame.to_local(*outgoing_world);

        const auto dimensions = renderer::sample_dimensions_for_bounce(depth);
        if (!dimensions) {
            return std::unexpected(dimensions.error());
        }
        const auto canonical_sample = renderer::Point2{
            .x = sample_stream.sample_1d(dimensions->bsdf_u),
            .y = sample_stream.sample_1d(dimensions->bsdf_v),
        };
        const auto sampled = filtered_closures->sample(
            outgoing_local, sample_stream.sample_1d(dimensions->bsdf_component), canonical_sample,
            renderer::TransportMode::radiance);
        if (!sampled) {
            return std::unexpected(sampled.error());
        }
        auto depth_event = renderer::PathDepthEventResult{};
        if (*sampled) {
            const auto& candidate = **sampled;
            if (!renderer::is_valid_surface_scattering_event(candidate.lobes) ||
                !renderer::bsdf_only_path_loop_detail::finite_non_negative(candidate.value) ||
                !std::isfinite(candidate.probability.value) ||
                !(candidate.probability.value > 0.0F) || candidate.incoming_local.z == 0.0F) {
                return std::unexpected(scene_mis_error(
                    core::StatusCode::invalid_argument,
                    "The closure mixture returned an invalid MIS continuation sample."));
            }
            const auto evaluated_depth =
                renderer::evaluate_path_depth_event(depth_limits, depth_counters, candidate.lobes);
            if (!evaluated_depth) {
                return std::unexpected(evaluated_depth.error());
            }
            if (!evaluated_depth->accepted()) {
                return std::unexpected(scene_mis_error(
                    core::StatusCode::internal_error,
                    "A depth-filtered closure selected a blocked MIS continuation event."));
            }
            depth_event = *evaluated_depth;
        }

        const auto direct =
            direct_lighting.estimate(surface, *filtered_closures, beta, frame, *outgoing_world,
                                     *dimensions, sample_stream, ray);
        if (!direct) {
            return std::unexpected(direct.error());
        }
        const auto accumulated_direct =
            renderer::bsdf_only_path_loop_detail::accumulate_direct_lighting(radiance, *direct);
        if (!accumulated_direct) {
            return std::unexpected(accumulated_direct.error());
        }
        radiance = *accumulated_direct;

        if (!*sampled) {
            return finish(renderer::BsdfOnlyPathTermination::outside_bsdf_support,
                          renderer::ScatteringLobe::none);
        }
        const auto& bsdf_sample = **sampled;

        const auto incoming_world = renderer::bsdf_only_path_loop_detail::robust_unit_direction(
            frame.to_world(bsdf_sample.incoming_local));
        if (!incoming_world) {
            return std::unexpected(incoming_world.error());
        }
        const auto correction = renderer::shading_normal_correction(
            surface.geometric_normal(), surface.shading_normal(), *outgoing_world, *incoming_world,
            renderer::TransportMode::radiance);
        if (!correction) {
            return std::unexpected(correction.error());
        }
        if (*correction == 0.0F) {
            return finish(renderer::BsdfOnlyPathTermination::outside_bsdf_support,
                          renderer::ScatteringLobe::none);
        }
        const auto updated_beta = renderer::bsdf_only_path_loop_detail::update_closure_throughput(
            beta, bsdf_sample, *correction, *filtered_closures);
        if (!updated_beta) {
            return std::unexpected(updated_beta.error());
        }

        const auto next_depth = renderer::path_depth_total(depth_event.counters);
        if (!next_depth) {
            return std::unexpected(next_depth.error());
        }
        const auto next_eta_scale = eta_scale * bsdf_sample.eta_scale_multiplier;
        if (!std::isfinite(next_eta_scale) || !(next_eta_scale > 0.0F)) {
            return std::unexpected(
                scene_mis_error(core::StatusCode::resource_exhausted,
                                "A closure eta-scale update is not representable."));
        }
        const auto origin =
            renderer::offset_ray_origin(surface.position(), surface.position_error(),
                                        surface.geometric_normal(), *incoming_world);
        if (!origin) {
            return std::unexpected(origin.error());
        }
        if (*origin == surface.position()) {
            return std::unexpected(scene_mis_error(
                core::StatusCode::invalid_argument,
                "The derived triangle error did not move the MIS continuation-ray origin."));
        }

        const auto next_ray = renderer::Ray::create(*origin, *incoming_world, 0.0F,
                                                    std::numeric_limits<float>::infinity(),
                                                    ray.time(), ray.mask(), ray.current_medium());
        if (!next_ray) {
            return std::unexpected(next_ray.error());
        }
        const auto recorded = direct_lighting.record_bsdf_sample(
            surface, bsdf_sample.probability, *incoming_world, bsdf_sample.lobes);
        if (!recorded) {
            return std::unexpected(recorded.error());
        }
        if (differentials != nullptr) {
            const auto differential_status = update_ray_differentials(
                *differentials, surface, bsdf_sample, outgoing_local, *next_ray);
            if (!differential_status) {
                return std::unexpected(differential_status.error());
            }
        }

        beta = *updated_beta;
        depth = *next_depth;
        depth_counters = depth_event.counters;
        eta_scale = next_eta_scale;
        delta_flags = next_delta_flags(delta_flags, bsdf_sample.lobes);
        ray = *next_ray;
        if (renderer::bsdf_only_path_loop_detail::zero_spectrum(beta)) {
            return finish(renderer::BsdfOnlyPathTermination::zero_throughput,
                          renderer::ScatteringLobe::none);
        }
        if (roulette_policy.is_enabled() && depth >= roulette_policy.first_eligible_depth()) {
            const auto roulette = renderer::evaluate_russian_roulette(
                beta, eta_scale, depth, sample_stream.sample_1d(dimensions->russian_roulette),
                roulette_policy);
            if (!roulette) {
                return std::unexpected(roulette.error());
            }
            switch (roulette->outcome) {
            case renderer::RussianRouletteOutcome::survived:
                beta = roulette->throughput;
                break;
            case renderer::RussianRouletteOutcome::terminated:
                beta = roulette->throughput;
                return finish(renderer::BsdfOnlyPathTermination::russian_roulette,
                              renderer::ScatteringLobe::none);
            case renderer::RussianRouletteOutcome::not_evaluated:
                return std::unexpected(
                    scene_mis_error(core::StatusCode::invalid_argument,
                                    "An eligible Russian roulette decision was not evaluated."));
            default:
                return std::unexpected(
                    scene_mis_error(core::StatusCode::invalid_argument,
                                    "Russian roulette returned an unsupported outcome."));
            }
        }
    }
}

[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_scene_mis_impl(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                     const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                     const renderer::LightSampler& light_sampler,
                     const renderer::MisHeuristic heuristic,
                     const renderer::PathDepthLimits& depth_limits,
                     const renderer::RussianRoulettePolicy& roulette_policy,
                     RayDifferentialTracker* const differentials) {
    if (initial_ray.current_medium() != initial_state.current_medium()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::invalid_argument,
                            "The MIS ray and path state must carry the same current medium."));
    }
    if (initial_state.current_medium() != renderer::VacuumMedium) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::unavailable,
                            "The MIS path currently supports vacuum transmittance only."));
    }
    if (initial_state.depth() != 0U) {
        return std::unexpected(scene_mis_error(
            core::StatusCode::incompatible,
            "MIS tracing must start at a primary path because PathState does not carry the prior "
            "directional PDFs."));
    }
    const auto heuristic_status = renderer::mis_weight<renderer::TransportScalar>(
        heuristic,
        renderer::ProbabilityDensity{
            .value = 1.0F,
            .measure = renderer::ProbabilityMeasure::solid_angle,
        },
        renderer::ProbabilityDensity{
            .value = 0.0F,
            .measure = renderer::ProbabilityMeasure::solid_angle,
        });
    if (!heuristic_status) {
        return std::unexpected(heuristic_status.error());
    }

    const auto scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::internal_error,
                            "The acceleration backend has no committed frame scene."));
    }
    if (!scene->spectral_environment()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::unavailable,
                            "The committed frame scene has no spectral environment."));
    }
    const auto& environment_record = *scene->spectral_environment();
    if (environment_record.wavelengths != initial_state.wavelengths()) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::incompatible,
                            "The committed frame scene was not resolved at the path wavelengths."));
    }

    const auto light_count = registered_light_count(*scene);
    if (!light_count) {
        return std::unexpected(light_count.error());
    }
    if (*light_count == 0U) {
        return std::unexpected(scene_mis_error(
            core::StatusCode::unavailable,
            "MIS requires at least one punctual or emissive mesh light in the committed scene."));
    }
    if (static_cast<std::size_t>(light_sampler.light_count()) != *light_count) {
        return std::unexpected(scene_mis_error(
            core::StatusCode::incompatible,
            "The light sampler does not match the committed combined light registry."));
    }

    const auto environment = renderer::ConstantEnvironment::create(environment_record.radiance);
    if (!environment) {
        return std::unexpected(
            scene_mis_error(core::StatusCode::internal_error,
                            "The committed frame scene lost its validated spectral environment."));
    }

    auto query = detail::SceneSurfaceQuery{acceleration};
    auto direct_lighting = SceneMisDirectLighting{acceleration, light_sampler, scene, heuristic};
    const auto resolved_environment =
        renderer::BsdfOnlyEnvironment{*environment, environment_record.wavelengths};
    return trace_lambertian_mis_with_query(initial_ray, initial_state, sample_stream, query,
                                           resolved_environment, depth_limits, roulette_policy,
                                           direct_lighting, differentials);
}

} // namespace

core::Result<renderer::BsdfOnlyPathResult>
trace_scene_mis(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler, const renderer::MisHeuristic heuristic,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy) {
    return trace_scene_mis_impl(initial_ray, initial_state, sample_stream, acceleration,
                                light_sampler, heuristic, depth_limits, roulette_policy, nullptr);
}

core::Result<SceneMisRayDifferentialResult> trace_scene_mis_with_ray_differentials(
    const renderer::RayDifferential& initial_ray, const renderer::PathState& initial_state,
    const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler, const renderer::MisHeuristic heuristic,
    const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy) {
    auto tracker = RayDifferentialTracker{.active = initial_ray};
    auto path =
        trace_scene_mis_impl(initial_ray.ray(), initial_state, sample_stream, acceleration,
                             light_sampler, heuristic, depth_limits, roulette_policy, &tracker);
    if (!path) {
        return std::unexpected(path.error());
    }
    return SceneMisRayDifferentialResult{
        .path = std::move(*path),
        .terminal_differential = std::move(tracker.active),
        .loss_reason = tracker.loss_reason,
        .propagated_specular_bounces = tracker.propagated_specular_bounces,
    };
}

} // namespace blackframe::engine
