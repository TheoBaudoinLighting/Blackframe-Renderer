#include <Blackframe/Backends/CPU/Embree/WavefrontMisTransport.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Renderer/ClosureMixture.hpp>
#include <Blackframe/Renderer/Detail/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/Detail/DepthFilteredClosureMixture.hpp>
#include <Blackframe/Renderer/DirectLighting.hpp>
#include <Blackframe/Renderer/PathStateSoA.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <Blackframe/Renderer/ShadingNormalCorrection.hpp>
#include <Blackframe/Renderer/ShadowRay.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace blackframe::engine {
namespace {

using StageClock = std::chrono::steady_clock;

static_assert(StageClock::is_steady);

[[nodiscard]] core::Error wavefront_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Error
queue_overflow_error(const renderer::WavefrontQueueKind kind, const std::size_t size,
                     const std::size_t requested_lanes, const std::size_t capacity,
                     const std::uint64_t overflow_attempts, const std::uint64_t rejected_lanes) {
    return wavefront_error(core::StatusCode::resource_exhausted,
                           "The " + std::string{renderer::wavefront_queue_kind_name(kind)} +
                               " CPU wavefront queue cannot append " +
                               std::to_string(requested_lanes) + " lane(s) at size " +
                               std::to_string(size) + " with capacity " + std::to_string(capacity) +
                               "; overflow_attempts=" + std::to_string(overflow_attempts) +
                               ", rejected_lanes=" + std::to_string(rejected_lanes) +
                               ". The batch was aborted without resize or fallback.");
}

[[nodiscard]] bool is_black(const renderer::TransportSpectrum& spectrum) noexcept {
    return std::ranges::all_of(spectrum.values, [](const auto value) { return value == 0.0F; });
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
        return std::unexpected(wavefront_error(
            core::StatusCode::resource_exhausted,
            "The combined CPU wavefront light-registry size is not representable."));
    }
    const auto count = punctual_count + area_count;
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(wavefront_error(
            core::StatusCode::resource_exhausted,
            "The combined CPU wavefront light registry exceeds its 32-bit slot domain."));
    }
    return count;
}

struct PreviousBsdfSample final {
    renderer::LightSampleContext context;
    renderer::ProbabilityDensity probability;
    renderer::Vector3 incoming_world;
};

struct PendingShadow final {
    renderer::Ray ray;
    renderer::TransportSpectrum beta;
    renderer::TransportSpectrum bsdf_value;
    renderer::TransportScalar absolute_incoming_cosine;
    bool singleton_lambertian;
    renderer::TransportSpectrum lambertian_reflectance;
    renderer::OrthonormalFrame frame;
    renderer::Vector3 outgoing_world;
    renderer::LightSelectionProbability selection_probability;
    renderer::IncidentLightSample incident_light;
    renderer::TransportScalar estimator_weight;
    renderer::TransportScalar shading_normal_correction;
};

struct PendingTermination final {
    renderer::BsdfOnlyPathTermination reason;
    renderer::ScatteringLobe blocked_depth_limits;
};

struct RuntimePath final {
    renderer::SampleStreamIndex sample{};
    std::optional<renderer::Ray> ray;
    std::optional<renderer::RayCone> cone;
    renderer::TransportSpectrum beta{};
    renderer::TransportSpectrum radiance{};
    renderer::PathDepthCounters depth_counters{};
    renderer::TransportScalar eta_scale{1.0F};
    renderer::SampledWavelengths wavelengths{};
    renderer::PathDeltaFlags delta_flags{renderer::PathDeltaFlags::none};
    renderer::MediumId current_medium{};
    bool initialized{};
    std::optional<AccelHit> hit;
    std::optional<ResolvedSceneSurface> surface;
    std::optional<PreviousBsdfSample> previous_bsdf_sample;
    std::optional<PendingShadow> pending_shadow;
    std::optional<PendingTermination> pending_termination;
    bool continuation_requested{};
    std::optional<renderer::BsdfOnlyPathResult> result;
    std::uint64_t closure_samples{};
    std::uint64_t light_samples{};
    std::uint64_t shadow_queries{};
};

[[nodiscard]] constexpr CpuWavefrontMisQueueStatistics
initial_queue_statistics(const renderer::WavefrontQueueKind kind,
                         const std::uint64_t capacity) noexcept {
    return CpuWavefrontMisQueueStatistics{
        .kind = kind,
        .capacity = capacity,
    };
}

[[nodiscard]] constexpr CpuWavefrontMisQueueStatisticsSet
initial_queue_statistics_set(const std::uint64_t capacity) noexcept {
    return CpuWavefrontMisQueueStatisticsSet{
        .camera = initial_queue_statistics(renderer::WavefrontQueueKind::camera, capacity),
        .ray = initial_queue_statistics(renderer::WavefrontQueueKind::ray, capacity),
        .hit = initial_queue_statistics(renderer::WavefrontQueueKind::hit, capacity),
        .miss = initial_queue_statistics(renderer::WavefrontQueueKind::miss, capacity),
        .shade = initial_queue_statistics(renderer::WavefrontQueueKind::shade, capacity),
        .shadow = initial_queue_statistics(renderer::WavefrontQueueKind::shadow, capacity),
        .continuation =
            initial_queue_statistics(renderer::WavefrontQueueKind::continuation, capacity),
    };
}

[[nodiscard]] core::Status add_statistics_counter(std::uint64_t& destination,
                                                  const std::uint64_t increment,
                                                  const char* const failure) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - destination) {
        return std::unexpected(wavefront_error(core::StatusCode::resource_exhausted, failure));
    }
    destination += increment;
    return {};
}

[[nodiscard]] core::Status record_queue_size(CpuWavefrontMisQueueStatistics& statistics,
                                             const std::size_t size) {
    const auto measured_size = static_cast<std::uint64_t>(size);
    if (measured_size > statistics.capacity) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "A CPU wavefront queue size exceeded its declared statistics capacity."));
    }
    statistics.peak_size = std::max(statistics.peak_size, measured_size);
    return {};
}

[[nodiscard]] core::Status record_queue_overflow(CpuWavefrontMisQueueStatistics& statistics,
                                                 const std::size_t rejected_lanes) {
    const auto rejected = static_cast<std::uint64_t>(rejected_lanes);
    if (statistics.overflow_attempts == std::numeric_limits<std::uint64_t>::max() ||
        rejected > std::numeric_limits<std::uint64_t>::max() - statistics.rejected_lanes) {
        return std::unexpected(
            wavefront_error(core::StatusCode::resource_exhausted,
                            "CPU wavefront queue-overflow accounting is not representable."));
    }
    ++statistics.overflow_attempts;
    statistics.rejected_lanes += rejected;
    return {};
}

[[nodiscard]] core::Result<std::uint64_t> stage_wall_nanoseconds(const StageClock::time_point begin,
                                                                 const StageClock::time_point end) {
    if (end < begin) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "The steady clock moved backwards while measuring a CPU wavefront stage."));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (elapsed < 0) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "A CPU wavefront stage produced a negative wall-clock duration."));
    }
    return static_cast<std::uint64_t>(elapsed);
}

[[nodiscard]] core::Status record_stage_dispatch(CpuWavefrontMisQueueStatistics& statistics,
                                                 const std::size_t input_lanes,
                                                 const std::uint64_t elapsed_nanoseconds) {
    if (auto status = record_queue_size(statistics, input_lanes); !status) {
        return status;
    }
    if (auto status =
            add_statistics_counter(statistics.dispatch_count, 1U,
                                   "CPU wavefront queue-dispatch accounting is not representable.");
        !status) {
        return status;
    }
    if (auto status =
            add_statistics_counter(statistics.input_lanes, static_cast<std::uint64_t>(input_lanes),
                                   "CPU wavefront queue-size accounting is not representable.");
        !status) {
        return status;
    }
    return add_statistics_counter(statistics.stage_wall_nanoseconds, elapsed_nanoseconds,
                                  "CPU wavefront stage-time accounting is not representable.");
}

[[nodiscard]] core::Status initialize_runtime_path(RuntimePath& runtime,
                                                   const CpuWavefrontMisPathInput& input,
                                                   const renderer::PathState& state) {
    if (runtime.initialized || runtime.result) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "A CPU wavefront camera lane attempted to initialize an occupied path slot."));
    }
    runtime.sample = input.sample;
    runtime.ray = input.primary_ray;
    runtime.cone = input.primary_cone;
    runtime.beta = state.beta();
    runtime.radiance = state.accumulated_radiance();
    runtime.depth_counters = state.depth_counters();
    runtime.eta_scale = state.eta_scale();
    runtime.wavelengths = state.wavelengths();
    runtime.delta_flags = state.delta_flags();
    runtime.current_medium = state.current_medium();
    runtime.initialized = true;
    return {};
}

[[nodiscard]] core::Status finish_path(RuntimePath& runtime,
                                       const renderer::BsdfOnlyPathTermination termination,
                                       const renderer::ScatteringLobe blocked_depth_limits) {
    if (!runtime.initialized || !runtime.ray || !runtime.cone || runtime.result) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "A CPU wavefront accumulation lane received incomplete or duplicate path state."));
    }
    const auto state = renderer::PathState::create(
        runtime.beta, runtime.radiance, runtime.depth_counters, runtime.eta_scale,
        runtime.wavelengths, runtime.delta_flags, runtime.current_medium);
    if (!state) {
        return std::unexpected(state.error());
    }
    runtime.result.emplace(renderer::BsdfOnlyPathResult{
        .state = *state,
        .terminal_ray = *runtime.ray,
        .termination = termination,
        .blocked_depth_limits = blocked_depth_limits,
    });
    runtime.pending_shadow.reset();
    runtime.pending_termination.reset();
    runtime.continuation_requested = false;
    return {};
}

class WavefrontDirectLighting final {
  public:
    struct Preparation final {
        std::optional<PendingShadow> shadow;
        bool sampled_light{};
    };

    WavefrontDirectLighting(const renderer::LightSampler& sampler, FrameSceneHandle scene,
                            const renderer::MisHeuristic heuristic) noexcept
        : sampler_{sampler}, scene_{std::move(scene)}, heuristic_{heuristic} {}

    [[nodiscard]] core::Result<Preparation>
    prepare(const ResolvedSceneSurface& surface,
            const renderer::detail::DepthFilteredClosureMixture& closures,
            const renderer::TransportSpectrum& beta, const renderer::OrthonormalFrame& frame,
            const renderer::Vector3 outgoing_world,
            const renderer::BounceSampleDimensions& dimensions,
            const renderer::SampleStream& sample_stream, const renderer::Ray& path_ray) const {
        if (is_black(beta)) {
            return Preparation{};
        }
        auto all_black = true;
        for (auto index = std::size_t{}; index < closures.size(); ++index) {
            all_black = all_black && is_black(closures.active_closure(index).weight);
        }
        if (all_black) {
            return Preparation{};
        }

        const auto context = renderer::LightSampleContext::create(surface.interaction.position(),
                                                                  surface.interaction.time());
        if (!context) {
            return std::unexpected(context.error());
        }
        if (auto status = validate_sampler_support(*context); !status) {
            return std::unexpected(status.error());
        }

        const auto selection =
            sampler_.sample(*context, sample_stream.sample_1d(dimensions.light_selection));
        if (!selection) {
            return std::unexpected(selection.error());
        }
        auto preparation = Preparation{
            .shadow = std::nullopt,
            .sampled_light = true,
        };
        const auto visible = selected_area_light_visible(selection->light_index(), path_ray.mask());
        if (!visible) {
            return std::unexpected(visible.error());
        }
        if (!*visible) {
            return preparation;
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
            return preparation;
        }

        const auto outgoing_local = frame.to_local(outgoing_world);
        const auto incoming_local = frame.to_local((**incident).direction_to_light());
        if (outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
            return preparation;
        }
        auto singleton_lambertian = false;
        auto lambertian_reflectance = renderer::TransportSpectrum{};
        auto bsdf_value = renderer::TransportSpectrum{};
        if (closures.size() == 1U &&
            closures.active_closure(0U).kind == renderer::ClosureKind::lambertian_reflection) {
            const auto model =
                renderer::LambertianReflection::create(closures.active_closure(0U).weight);
            if (!model) {
                return std::unexpected(model.error());
            }
            if (is_black(model->reflectance())) {
                return preparation;
            }
            singleton_lambertian = true;
            lambertian_reflectance = model->reflectance();
        } else {
            const auto evaluated =
                closures.eval(outgoing_local, incoming_local, renderer::TransportMode::radiance);
            if (!evaluated) {
                return std::unexpected(evaluated.error());
            }
            if (is_black(*evaluated)) {
                return preparation;
            }
            bsdf_value = *evaluated;
        }
        const auto correction = renderer::shading_normal_correction(
            surface.interaction.geometric_normal(), surface.interaction.shading_normal(),
            outgoing_world, (**incident).direction_to_light(), renderer::TransportMode::radiance);
        if (!correction) {
            return std::unexpected(correction.error());
        }
        if (*correction == 0.0F) {
            return preparation;
        }

        auto estimator_weight = renderer::TransportScalar{1};
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
            estimator_weight = *mis;
        } else if (conditional_probability.measure != renderer::ProbabilityMeasure::discrete) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "A registered CPU wavefront light returned an unsupported conditional measure."));
        }

        const auto shadow_ray =
            renderer::make_shadow_ray(surface.interaction, surface.position_error, **incident,
                                      path_ray.mask(), path_ray.current_medium());
        if (!shadow_ray) {
            return std::unexpected(shadow_ray.error());
        }
        preparation.shadow.emplace(PendingShadow{
            .ray = *shadow_ray,
            .beta = beta,
            .bsdf_value = bsdf_value,
            .absolute_incoming_cosine = std::abs(incoming_local.z),
            .singleton_lambertian = singleton_lambertian,
            .lambertian_reflectance = lambertian_reflectance,
            .frame = frame,
            .outgoing_world = outgoing_world,
            .selection_probability = selection->probability(),
            .incident_light = **incident,
            .estimator_weight = estimator_weight,
            .shading_normal_correction = *correction,
        });
        return preparation;
    }

    [[nodiscard]] core::Result<renderer::TransportScalar>
    surface_emission_weight(const ResolvedSceneSurface& surface, const renderer::Ray& path_ray,
                            const renderer::TransportSpectrum& emitted_radiance,
                            const std::optional<PreviousBsdfSample>& previous_bsdf_sample) const {
        if (is_black(emitted_radiance) || !previous_bsdf_sample) {
            return renderer::TransportScalar{1};
        }
        if (path_ray.direction() != previous_bsdf_sample->incoming_world) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "The emissive CPU wavefront hit direction does not match its BSDF sample."));
        }

        const auto area_lights = scene_->mesh_area_lights();
        const auto instance_ids = scene_->mesh_area_light_instance_ids();
        if (area_lights.size() != instance_ids.size()) {
            return std::unexpected(
                wavefront_error(core::StatusCode::internal_error,
                                "The committed mesh-light registry lost its instance alignment."));
        }
        const auto hit_instance = surface.interaction.identifiers().instance;
        auto area_index = area_lights.size();
        for (auto index = std::size_t{}; index < instance_ids.size(); ++index) {
            if (instance_ids[index] == hit_instance) {
                area_index = index;
                break;
            }
        }
        if (area_index == area_lights.size()) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "A non-black CPU wavefront emissive hit is absent from the light registry."));
        }

        const auto punctual_count = scene_->punctual_lights().size();
        if (area_index > std::numeric_limits<std::size_t>::max() - punctual_count) {
            return std::unexpected(
                wavefront_error(core::StatusCode::resource_exhausted,
                                "The CPU wavefront emissive-hit light slot is not representable."));
        }
        const auto global_slot = punctual_count + area_index;
        if (global_slot > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(
                wavefront_error(core::StatusCode::resource_exhausted,
                                "The CPU wavefront emissive-hit slot exceeds its 32-bit domain."));
        }
        const auto selection_probability = sampler_.probability(
            previous_bsdf_sample->context, static_cast<std::uint32_t>(global_slot));
        if (!selection_probability) {
            return std::unexpected(selection_probability.error());
        }
        if (!(selection_probability->value() > 0.0F)) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "The light sampler gives zero support to a CPU wavefront emissive hit."));
        }
        const auto conditional_probability = area_lights[area_index].pdf_li_at_surface(
            previous_bsdf_sample->context, surface.interaction.position(),
            surface.interaction.geometric_normal(), scene_->spectral_environment()->wavelengths);
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
            heuristic_, previous_bsdf_sample->probability, *light_probability);
    }

    [[nodiscard]] core::Result<std::optional<PreviousBsdfSample>> record_bsdf_sample(
        const ResolvedSceneSurface& surface, const renderer::ProbabilityDensity probability,
        const renderer::Vector3 incoming_world, const renderer::ScatteringLobe lobes) const {
        if (renderer::is_delta_surface_scattering_event(lobes)) {
            if (probability.measure != renderer::DeltaBsdfProbabilityMeasure ||
                !std::isfinite(probability.value) || !(probability.value > 0.0F) ||
                probability.value > 1.0F) {
                return std::unexpected(wavefront_error(
                    core::StatusCode::incompatible,
                    "A delta CPU wavefront continuation requires a positive discrete mass."));
            }
            return std::optional<PreviousBsdfSample>{};
        }
        if (probability.measure != renderer::ProbabilityMeasure::solid_angle ||
            !std::isfinite(probability.value) || !(probability.value > 0.0F)) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "A CPU wavefront MIS continuation requires a positive solid-angle BSDF PDF."));
        }
        const auto context = renderer::LightSampleContext::create(surface.interaction.position(),
                                                                  surface.interaction.time());
        if (!context) {
            return std::unexpected(context.error());
        }
        return std::optional<PreviousBsdfSample>{PreviousBsdfSample{
            .context = *context,
            .probability = probability,
            .incoming_world = incoming_world,
        }};
    }

  private:
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
                return std::unexpected(wavefront_error(
                    core::StatusCode::incompatible,
                    "The CPU light sampler has zero support for a non-black punctual light."));
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
                return std::unexpected(wavefront_error(
                    core::StatusCode::incompatible,
                    "The CPU light sampler has zero support for a non-black mesh area light."));
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
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "The CPU light sampler selected a slot outside the committed registry."));
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
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "The CPU light sampler selected a slot outside the committed registry."));
        }
        const auto instance = scene_->instance(instance_ids[area_index]);
        if (!instance) {
            return std::unexpected(instance.error());
        }
        return (instance->get().visibility_mask & path_mask) != 0U;
    }

    const renderer::LightSampler& sampler_;
    FrameSceneHandle scene_;
    renderer::MisHeuristic heuristic_;
};

[[nodiscard]] core::Result<renderer::TransportSpectrum>
accumulate_weighted_emission(const renderer::TransportSpectrum& accumulated,
                             const renderer::TransportSpectrum& throughput,
                             const renderer::TransportSpectrum& emitted_radiance,
                             const renderer::TransportScalar estimator_weight) {
    if (!std::isfinite(estimator_weight) || estimator_weight < 0.0F || estimator_weight > 1.0F) {
        return std::unexpected(
            wavefront_error(core::StatusCode::invalid_argument,
                            "A CPU wavefront emitted-radiance MIS weight must lie in [0, 1]."));
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
            return std::unexpected(wavefront_error(
                core::StatusCode::invalid_argument,
                "A CPU wavefront emitted-radiance contribution is not representable."));
        }
        result[lane] = accumulated[lane] + contribution;
        if (!std::isfinite(result[lane])) {
            return std::unexpected(
                wavefront_error(core::StatusCode::invalid_argument,
                                "CPU wavefront MIS radiance accumulation is not representable."));
        }
    }
    return result;
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

[[nodiscard]] core::Status shade_path(RuntimePath& runtime,
                                      const WavefrontDirectLighting& direct_lighting,
                                      const renderer::PathDepthLimits& depth_limits,
                                      const renderer::RussianRoulettePolicy& roulette_policy) {
    if (!runtime.initialized || !runtime.ray || !runtime.cone || !runtime.hit || !runtime.surface ||
        runtime.result) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "A CPU wavefront shade lane received incomplete path or surface state."));
    }
    runtime.pending_shadow.reset();
    runtime.pending_termination.reset();
    runtime.continuation_requested = false;
    const auto& surface = *runtime.surface;
    const auto emitted =
        surface.emission.eval(surface.interaction.geometric_normal(), -runtime.ray->direction());
    if (!emitted) {
        return std::unexpected(emitted.error());
    }
    const auto emission_weight = direct_lighting.surface_emission_weight(
        surface, *runtime.ray, *emitted, runtime.previous_bsdf_sample);
    if (!emission_weight) {
        return std::unexpected(emission_weight.error());
    }
    const auto accumulated =
        accumulate_weighted_emission(runtime.radiance, runtime.beta, *emitted, *emission_weight);
    if (!accumulated) {
        return std::unexpected(accumulated.error());
    }
    runtime.radiance = *accumulated;

    if (renderer::bsdf_only_path_loop_detail::zero_spectrum(runtime.beta)) {
        runtime.pending_termination = PendingTermination{
            .reason = renderer::BsdfOnlyPathTermination::zero_throughput,
            .blocked_depth_limits = renderer::ScatteringLobe::none,
        };
        return {};
    }

    const auto filtered_closures = renderer::detail::DepthFilteredClosureMixture::create(
        surface.closures, depth_limits, runtime.depth_counters);
    if (!filtered_closures) {
        return std::unexpected(filtered_closures.error());
    }
    if (filtered_closures->empty()) {
        runtime.pending_termination = PendingTermination{
            .reason = filtered_closures->source_empty()
                          ? renderer::BsdfOnlyPathTermination::outside_bsdf_support
                          : renderer::BsdfOnlyPathTermination::depth_limit,
            .blocked_depth_limits = filtered_closures->source_empty()
                                        ? renderer::ScatteringLobe::none
                                        : filtered_closures->blocked_lobes(),
        };
        return {};
    }

    const auto& frame = surface.closure_frame;
    const auto outgoing_world =
        renderer::bsdf_only_path_loop_detail::robust_unit_direction(-runtime.ray->direction());
    if (!outgoing_world) {
        return std::unexpected(outgoing_world.error());
    }
    const auto outgoing_local = frame.to_local(*outgoing_world);
    const auto current_depth = renderer::path_depth_total(runtime.depth_counters);
    if (!current_depth) {
        return std::unexpected(current_depth.error());
    }
    const auto dimensions = renderer::sample_dimensions_for_bounce(*current_depth);
    if (!dimensions) {
        return std::unexpected(dimensions.error());
    }
    const auto stream = renderer::SampleStream{runtime.sample};
    const auto sampled =
        filtered_closures->sample(outgoing_local, stream.sample_1d(dimensions->bsdf_component),
                                  renderer::Point2{
                                      .x = stream.sample_1d(dimensions->bsdf_u),
                                      .y = stream.sample_1d(dimensions->bsdf_v),
                                  },
                                  renderer::TransportMode::radiance);
    if (!sampled) {
        return std::unexpected(sampled.error());
    }
    ++runtime.closure_samples;
    auto depth_event = renderer::PathDepthEventResult{};
    if (*sampled) {
        const auto& candidate = **sampled;
        if (!renderer::is_valid_surface_scattering_event(candidate.lobes)) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible, "A CPU closure produced an invalid event mask."));
        }
        const auto evaluated_depth = renderer::evaluate_path_depth_event(
            depth_limits, runtime.depth_counters, candidate.lobes);
        if (!evaluated_depth) {
            return std::unexpected(evaluated_depth.error());
        }
        if (!evaluated_depth->accepted()) {
            return std::unexpected(wavefront_error(
                core::StatusCode::internal_error,
                "A depth-filtered CPU closure selected a blocked continuation event."));
        }
        depth_event = *evaluated_depth;
    }

    const auto direct = direct_lighting.prepare(surface, *filtered_closures, runtime.beta, frame,
                                                *outgoing_world, *dimensions, stream, *runtime.ray);
    if (!direct) {
        return std::unexpected(direct.error());
    }
    if (direct->sampled_light) {
        ++runtime.light_samples;
    }
    runtime.pending_shadow = direct->shadow;

    if (!*sampled) {
        runtime.pending_termination = PendingTermination{
            .reason = renderer::BsdfOnlyPathTermination::outside_bsdf_support,
            .blocked_depth_limits = renderer::ScatteringLobe::none,
        };
        return {};
    }
    const auto& closure_sample = **sampled;

    const auto incoming_world = renderer::bsdf_only_path_loop_detail::robust_unit_direction(
        frame.to_world(closure_sample.incoming_local));
    if (!incoming_world) {
        return std::unexpected(incoming_world.error());
    }
    const auto correction = renderer::shading_normal_correction(
        surface.interaction.geometric_normal(), surface.interaction.shading_normal(),
        *outgoing_world, *incoming_world, renderer::TransportMode::radiance);
    if (!correction) {
        return std::unexpected(correction.error());
    }
    if (*correction == 0.0F) {
        runtime.pending_termination = PendingTermination{
            .reason = renderer::BsdfOnlyPathTermination::outside_bsdf_support,
            .blocked_depth_limits = renderer::ScatteringLobe::none,
        };
        return {};
    }
    const auto updated_beta = renderer::bsdf_only_path_loop_detail::update_closure_throughput(
        runtime.beta, closure_sample, *correction, *filtered_closures);
    if (!updated_beta) {
        return std::unexpected(updated_beta.error());
    }
    const auto next_depth = renderer::path_depth_total(depth_event.counters);
    if (!next_depth) {
        return std::unexpected(next_depth.error());
    }
    const auto next_eta_scale = runtime.eta_scale * closure_sample.eta_scale_multiplier;
    if (!std::isfinite(next_eta_scale) || !(next_eta_scale > 0.0F)) {
        return std::unexpected(
            wavefront_error(core::StatusCode::invalid_argument,
                            "A CPU wavefront closure eta-scale update is not representable."));
    }
    const auto origin =
        renderer::offset_ray_origin(surface.interaction.position(), surface.position_error,
                                    surface.interaction.geometric_normal(), *incoming_world);
    if (!origin) {
        return std::unexpected(origin.error());
    }
    if (*origin == surface.interaction.position()) {
        return std::unexpected(wavefront_error(
            core::StatusCode::invalid_argument,
            "The CPU wavefront continuation-ray origin did not move from its surface."));
    }
    const auto next_ray = renderer::Ray::create(
        *origin, *incoming_world, 0.0F, std::numeric_limits<float>::infinity(), runtime.ray->time(),
        runtime.ray->mask(), runtime.ray->current_medium());
    if (!next_ray) {
        return std::unexpected(next_ray.error());
    }
    const auto previous = direct_lighting.record_bsdf_sample(surface, closure_sample.probability,
                                                             *incoming_world, closure_sample.lobes);
    if (!previous) {
        return std::unexpected(previous.error());
    }

    const auto surface_cone =
        renderer::advance_ray_cone(*runtime.cone, *runtime.ray, runtime.hit->triangle.parameter);
    if (!surface_cone) {
        return std::unexpected(surface_cone.error());
    }
    const auto closures = surface.closures.closure_set().closures();
    if (closure_sample.selected_closure >= closures.size()) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "A CPU wavefront closure sample does not identify a source closure record."));
    }
    const auto next_cone = renderer::propagate_ray_cone_scattering(
        *surface_cone, closures[closure_sample.selected_closure], closure_sample.lobes,
        outgoing_local, closure_sample.incoming_local);
    if (!next_cone) {
        return std::unexpected(next_cone.error());
    }

    runtime.beta = *updated_beta;
    runtime.depth_counters = depth_event.counters;
    runtime.eta_scale = next_eta_scale;
    runtime.delta_flags = next_delta_flags(runtime.delta_flags, closure_sample.lobes);
    runtime.ray = *next_ray;
    runtime.cone = *next_cone;
    runtime.previous_bsdf_sample = std::move(*previous);
    if (renderer::bsdf_only_path_loop_detail::zero_spectrum(runtime.beta)) {
        runtime.pending_termination = PendingTermination{
            .reason = renderer::BsdfOnlyPathTermination::zero_throughput,
            .blocked_depth_limits = renderer::ScatteringLobe::none,
        };
        return {};
    }
    if (roulette_policy.is_enabled() && *next_depth >= roulette_policy.first_eligible_depth()) {
        const auto roulette = renderer::evaluate_russian_roulette(
            runtime.beta, runtime.eta_scale, *next_depth,
            stream.sample_1d(dimensions->russian_roulette), roulette_policy);
        if (!roulette) {
            return std::unexpected(roulette.error());
        }
        switch (roulette->outcome) {
        case renderer::RussianRouletteOutcome::survived:
            runtime.beta = roulette->throughput;
            break;
        case renderer::RussianRouletteOutcome::terminated:
            runtime.beta = roulette->throughput;
            runtime.pending_termination = PendingTermination{
                .reason = renderer::BsdfOnlyPathTermination::russian_roulette,
                .blocked_depth_limits = renderer::ScatteringLobe::none,
            };
            return {};
        case renderer::RussianRouletteOutcome::not_evaluated:
            return std::unexpected(
                wavefront_error(core::StatusCode::invalid_argument,
                                "An eligible CPU wavefront roulette decision was not evaluated."));
        default:
            return std::unexpected(
                wavefront_error(core::StatusCode::invalid_argument,
                                "CPU wavefront roulette returned an unsupported outcome."));
        }
    }

    runtime.continuation_requested = true;
    return {};
}

[[nodiscard]] core::Status
validate_inputs(const std::span<const CpuWavefrontMisPathInput> inputs,
                const renderer::CpuWavefrontScheduler& scheduler, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler, const renderer::MisHeuristic heuristic,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy, FrameSceneHandle& scene) {
    if (scheduler.worker_count() == 0U) {
        return std::unexpected(
            wavefront_error(core::StatusCode::unavailable,
                            "A moved-from CPU wavefront scheduler cannot execute transport."));
    }
    if (acceleration.kind() != AccelBackendKind::embree) {
        return std::unexpected(wavefront_error(
            core::StatusCode::unavailable,
            "CPU wavefront transport requires an explicitly selected Embree backend."));
    }
    if (inputs.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return std::unexpected(
            wavefront_error(core::StatusCode::resource_exhausted,
                            "The CPU wavefront path batch exceeds its 32-bit stable-slot domain."));
    }
    if (auto status = renderer::validate_russian_roulette_policy(roulette_policy); !status) {
        return std::unexpected(status.error());
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
    scene = acceleration.frame_scene();
    if (!scene) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "The Embree CPU wavefront backend has no committed frame scene."));
    }
    if (!scene->spectral_environment()) {
        return std::unexpected(
            wavefront_error(core::StatusCode::unavailable,
                            "CPU wavefront MIS requires a committed spectral environment."));
    }
    const auto light_count = registered_light_count(*scene);
    if (!light_count) {
        return std::unexpected(light_count.error());
    }
    if (*light_count == 0U) {
        return std::unexpected(
            wavefront_error(core::StatusCode::unavailable,
                            "CPU wavefront MIS requires at least one registered scene light."));
    }
    if (static_cast<std::size_t>(light_sampler.light_count()) != *light_count) {
        return std::unexpected(wavefront_error(
            core::StatusCode::incompatible,
            "The CPU wavefront light sampler does not match the committed registry."));
    }
    const auto& wavelengths = scene->spectral_environment()->wavelengths;
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        if (input.primary_ray.current_medium() != input.initial_state.current_medium()) {
            return std::unexpected(
                wavefront_error(core::StatusCode::invalid_argument,
                                "CPU wavefront ray and path-state media differ at input lane " +
                                    std::to_string(index) + "."));
        }
        if (input.initial_state.current_medium() != renderer::VacuumMedium) {
            return std::unexpected(
                wavefront_error(core::StatusCode::unavailable,
                                "CPU wavefront MIS currently supports vacuum transport only."));
        }
        if (input.initial_state.depth() != 0U) {
            return std::unexpected(
                wavefront_error(core::StatusCode::incompatible,
                                "CPU wavefront MIS must begin at primary path depth zero."));
        }
        if (input.initial_state.wavelengths() != wavelengths) {
            return std::unexpected(wavefront_error(
                core::StatusCode::incompatible,
                "A CPU wavefront input was not resolved at the scene wavelength packet."));
        }
        if (!renderer::bsdf_only_path_loop_detail::finite_non_negative(
                input.initial_state.beta())) {
            return std::unexpected(
                wavefront_error(core::StatusCode::invalid_argument,
                                "CPU wavefront path throughput must be finite and non-negative."));
        }
        if (auto status = renderer::validate_path_depth_state(
                depth_limits, input.initial_state.depth_counters(), input.initial_state.depth());
            !status) {
            return std::unexpected(status.error());
        }
    }
    return {};
}

[[nodiscard]] core::Status add_stage_lanes(CpuWavefrontMisStageLaneCounts& counts,
                                           const renderer::WavefrontQueueKind kind,
                                           const std::size_t lane_count) {
    auto* destination = [&]() -> std::uint64_t* {
        switch (kind) {
        case renderer::WavefrontQueueKind::camera:
            return &counts.camera;
        case renderer::WavefrontQueueKind::ray:
            return &counts.ray;
        case renderer::WavefrontQueueKind::hit:
            return &counts.hit;
        case renderer::WavefrontQueueKind::miss:
            return &counts.miss;
        case renderer::WavefrontQueueKind::shade:
            return &counts.shade;
        case renderer::WavefrontQueueKind::shadow:
            return &counts.shadow;
        case renderer::WavefrontQueueKind::continuation:
            return &counts.continuation;
        }
        return nullptr;
    }();
    if (destination == nullptr ||
        lane_count > std::numeric_limits<std::uint64_t>::max() - *destination) {
        return std::unexpected(
            wavefront_error(core::StatusCode::resource_exhausted,
                            "CPU wavefront stage-lane accounting is not representable."));
    }
    *destination += static_cast<std::uint64_t>(lane_count);
    return {};
}

[[nodiscard]] core::Status validate_queue_statistics(const CpuWavefrontMisReport& report) {
    const auto expected_capacity = static_cast<std::uint64_t>(report.path_count);
    const auto entries = std::array{
        std::pair{&report.queue_statistics.camera, report.stage_lanes.camera},
        std::pair{&report.queue_statistics.ray, report.stage_lanes.ray},
        std::pair{&report.queue_statistics.hit, report.stage_lanes.hit},
        std::pair{&report.queue_statistics.miss, report.stage_lanes.miss},
        std::pair{&report.queue_statistics.shade, report.stage_lanes.shade},
        std::pair{&report.queue_statistics.shadow, report.stage_lanes.shadow},
        std::pair{&report.queue_statistics.continuation, report.stage_lanes.continuation},
    };
    constexpr auto expected_kinds = std::array{
        renderer::WavefrontQueueKind::camera,       renderer::WavefrontQueueKind::ray,
        renderer::WavefrontQueueKind::hit,          renderer::WavefrontQueueKind::miss,
        renderer::WavefrontQueueKind::shade,        renderer::WavefrontQueueKind::shadow,
        renderer::WavefrontQueueKind::continuation,
    };
    for (auto index = std::size_t{}; index < entries.size(); ++index) {
        const auto& statistics = *entries[index].first;
        const auto expected_lanes = entries[index].second;
        if (statistics.kind != expected_kinds[index] || statistics.capacity != expected_capacity ||
            statistics.peak_size > statistics.capacity ||
            statistics.input_lanes != expected_lanes || statistics.overflow_attempts != 0U ||
            statistics.rejected_lanes != 0U ||
            (statistics.dispatch_count == 0U && statistics.input_lanes != 0U) ||
            !std::isfinite(statistics.peak_occupancy()) ||
            !std::isfinite(statistics.mean_occupancy()) || statistics.peak_occupancy() < 0.0 ||
            statistics.peak_occupancy() > 1.0 || statistics.mean_occupancy() < 0.0 ||
            statistics.mean_occupancy() > 1.0) {
            return std::unexpected(wavefront_error(
                core::StatusCode::internal_error,
                "CPU wavefront queue statistics violate their successful-batch contract."));
        }
    }
    return {};
}

template <typename Queue>
[[nodiscard]] core::Status push_slot(Queue& queue, CpuWavefrontMisQueueStatistics& statistics,
                                     const renderer::WavefrontPathSlot slot) {
    if (statistics.kind != Queue::kind()) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "CPU wavefront queue statistics were bound to the wrong stage."));
    }
    if (queue.push(slot) != renderer::WavefrontQueuePushStatus::pushed) {
        if (auto status = record_queue_overflow(statistics, 1U); !status) {
            return status;
        }
        return std::unexpected(queue_overflow_error(Queue::kind(), queue.size(), 1U,
                                                    queue.capacity(), statistics.overflow_attempts,
                                                    statistics.rejected_lanes));
    }
    return record_queue_size(statistics, queue.size());
}

template <typename Queue>
[[nodiscard]] core::Status push_batch(Queue& queue, CpuWavefrontMisQueueStatistics& statistics,
                                      const std::span<const renderer::WavefrontPathSlot> slots) {
    if (statistics.kind != Queue::kind()) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "CPU wavefront queue statistics were bound to the wrong stage."));
    }
    if (queue.push_batch(slots) != renderer::WavefrontQueuePushStatus::pushed) {
        if (auto status = record_queue_overflow(statistics, slots.size()); !status) {
            return status;
        }
        return std::unexpected(queue_overflow_error(Queue::kind(), queue.size(), slots.size(),
                                                    queue.capacity(), statistics.overflow_attempts,
                                                    statistics.rejected_lanes));
    }
    return record_queue_size(statistics, queue.size());
}

template <typename Queue>
[[nodiscard]] core::Status
execute_queue(const renderer::CpuWavefrontScheduler& scheduler, const std::size_t path_count,
              const Queue& queue, const renderer::CpuWavefrontStageKernel& kernel,
              CpuWavefrontMisStageLaneCounts& counts, CpuWavefrontMisQueueStatistics& statistics) {
    if (statistics.kind != Queue::kind()) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "CPU wavefront dispatch statistics were bound to the wrong stage."));
    }
    const auto begin = StageClock::now();
    const auto report = scheduler.execute_stage(Queue::kind(), path_count, queue.entries(), kernel);
    const auto end = StageClock::now();
    if (!report) {
        return std::unexpected(report.error());
    }
    const auto elapsed = stage_wall_nanoseconds(begin, end);
    if (!elapsed) {
        return std::unexpected(elapsed.error());
    }
    if (auto status = add_stage_lanes(counts, Queue::kind(), report->input_lanes); !status) {
        return status;
    }
    return record_stage_dispatch(statistics, report->input_lanes, *elapsed);
}

[[nodiscard]] core::Result<CpuWavefrontMisBatch> trace_cpu_wavefront_mis_impl(
    const std::span<const CpuWavefrontMisPathInput> inputs,
    const renderer::CpuWavefrontScheduler& scheduler, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler, const renderer::MisHeuristic heuristic,
    const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy) {
    auto scene = FrameSceneHandle{};
    if (auto status = validate_inputs(inputs, scheduler, acceleration, light_sampler, heuristic,
                                      depth_limits, roulette_policy, scene);
        !status) {
        return std::unexpected(status.error());
    }

    auto report = CpuWavefrontMisReport{
        .schema_version = CurrentCpuWavefrontMisReportSchemaVersion,
        .configured_workers = scheduler.worker_count(),
        .path_count = inputs.size(),
        .queue_statistics = initial_queue_statistics_set(static_cast<std::uint64_t>(inputs.size())),
    };
    if (inputs.empty()) {
        if (auto status = validate_queue_statistics(report); !status) {
            return std::unexpected(status.error());
        }
        return CpuWavefrontMisBatch{.paths = {}, .terminal_cones = {}, .report = report};
    }

    auto initial_states = std::vector<renderer::PathState>{};
    initial_states.reserve(inputs.size());
    for (const auto& input : inputs) {
        initial_states.push_back(input.initial_state);
    }
    const auto initial_state_soa = renderer::PathStateSoA::from_aos(
        initial_states, renderer::CurrentPathStateSoASchemaVersion);
    if (!initial_state_soa) {
        return std::unexpected(initial_state_soa.error());
    }
    auto runtimes = std::vector<RuntimePath>(inputs.size());

    auto camera_created = renderer::CameraQueue::create(inputs.size());
    auto ray_created = renderer::RayQueue::create(inputs.size());
    auto hit_created = renderer::HitQueue::create(inputs.size());
    auto miss_created = renderer::MissQueue::create(inputs.size());
    auto shade_created = renderer::ShadeQueue::create(inputs.size());
    auto shadow_created = renderer::ShadowQueue::create(inputs.size());
    auto continuation_created = renderer::ContinuationQueue::create(inputs.size());
    if (!camera_created || !ray_created || !hit_created || !miss_created || !shade_created ||
        !shadow_created || !continuation_created) {
        const auto* error = [&]() -> const core::Error* {
            if (!camera_created) {
                return &camera_created.error();
            }
            if (!ray_created) {
                return &ray_created.error();
            }
            if (!hit_created) {
                return &hit_created.error();
            }
            if (!miss_created) {
                return &miss_created.error();
            }
            if (!shade_created) {
                return &shade_created.error();
            }
            if (!shadow_created) {
                return &shadow_created.error();
            }
            return &continuation_created.error();
        }();
        return std::unexpected(*error);
    }
    auto camera = std::move(*camera_created);
    auto ray = std::move(*ray_created);
    auto hit = std::move(*hit_created);
    auto miss = std::move(*miss_created);
    auto shade = std::move(*shade_created);
    auto shadow = std::move(*shadow_created);
    auto continuation = std::move(*continuation_created);
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        if (auto status =
                push_slot(camera, report.queue_statistics.camera,
                          renderer::WavefrontPathSlot{.value = static_cast<std::uint32_t>(index)});
            !status) {
            return std::unexpected(status.error());
        }
    }

    if (auto status = execute_queue(
            scheduler, inputs.size(), camera,
            renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                const auto slot = static_cast<std::size_t>(lane.path_slot.value);
                const auto state = initial_state_soa->at(slot);
                if (!state) {
                    return core::Status{std::unexpected(state.error())};
                }
                return initialize_runtime_path(runtimes[slot], inputs[slot], *state);
            }},
            report.stage_lanes, report.queue_statistics.camera);
        !status) {
        return std::unexpected(status.error());
    }
    if (auto status = push_batch(ray, report.queue_statistics.ray, camera.entries()); !status) {
        return std::unexpected(status.error());
    }
    camera.clear();

    const auto environment =
        renderer::ConstantEnvironment::create(scene->spectral_environment()->radiance);
    if (!environment) {
        return std::unexpected(
            wavefront_error(core::StatusCode::internal_error,
                            "The committed scene lost its validated CPU spectral environment."));
    }
    const auto direct_lighting = WavefrontDirectLighting{light_sampler, scene, heuristic};

    while (!ray.empty()) {
        hit.clear();
        miss.clear();
        if (auto status = execute_queue(
                scheduler, inputs.size(), ray,
                renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                    auto& runtime = runtimes[lane.path_slot.value];
                    if (!runtime.ray || !runtime.cone || runtime.result) {
                        return core::Status{std::unexpected(
                            wavefront_error(core::StatusCode::internal_error,
                                            "A CPU wavefront ray lane received no active ray."))};
                    }
                    const auto closest = acceleration.closest_hit(*runtime.ray);
                    if (!closest) {
                        return core::Status{std::unexpected(closest.error())};
                    }
                    runtime.hit = *closest;
                    runtime.surface.reset();
                    return core::Status{};
                }},
                report.stage_lanes, report.queue_statistics.ray);
            !status) {
            return std::unexpected(status.error());
        }
        for (const auto slot : ray.entries()) {
            if (runtimes[slot.value].hit) {
                if (auto status = push_slot(hit, report.queue_statistics.hit, slot); !status) {
                    return std::unexpected(status.error());
                }
            } else {
                if (auto status = push_slot(miss, report.queue_statistics.miss, slot); !status) {
                    return std::unexpected(status.error());
                }
            }
        }
        ray.clear();

        if (!miss.empty()) {
            if (auto status = execute_queue(
                    scheduler, inputs.size(), miss,
                    renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                        auto& runtime = runtimes[lane.path_slot.value];
                        if (!runtime.ray || !runtime.cone || runtime.hit || runtime.result) {
                            return core::Status{std::unexpected(wavefront_error(
                                core::StatusCode::internal_error,
                                "A CPU wavefront miss lane received contradictory state."))};
                        }
                        const auto emitted = environment->eval(runtime.ray->direction());
                        if (!emitted) {
                            return core::Status{std::unexpected(emitted.error())};
                        }
                        const auto accumulated =
                            renderer::bsdf_only_path_loop_detail::accumulate_emission(
                                runtime.radiance, runtime.beta, *emitted);
                        if (!accumulated) {
                            return core::Status{std::unexpected(accumulated.error())};
                        }
                        runtime.radiance = *accumulated;
                        return finish_path(runtime,
                                           renderer::BsdfOnlyPathTermination::escaped_environment,
                                           renderer::ScatteringLobe::none);
                    }},
                    report.stage_lanes, report.queue_statistics.miss);
                !status) {
                return std::unexpected(status.error());
            }
            miss.clear();
        }

        shade.clear();
        if (!hit.empty()) {
            if (auto status = execute_queue(
                    scheduler, inputs.size(), hit,
                    renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                        auto& runtime = runtimes[lane.path_slot.value];
                        if (!runtime.ray || !runtime.cone || !runtime.hit || runtime.result) {
                            return core::Status{
                                std::unexpected(wavefront_error(core::StatusCode::internal_error,
                                                                "A CPU wavefront hit lane received "
                                                                "incomplete intersection state."))};
                        }
                        const auto resolved = resolve_scene_surface_hit(
                            *scene, *runtime.hit, *runtime.ray, *runtime.cone);
                        if (!resolved) {
                            return core::Status{std::unexpected(resolved.error())};
                        }
                        runtime.surface = *resolved;
                        return core::Status{};
                    }},
                    report.stage_lanes, report.queue_statistics.hit);
                !status) {
                return std::unexpected(status.error());
            }
            if (auto status = push_batch(shade, report.queue_statistics.shade, hit.entries());
                !status) {
                return std::unexpected(status.error());
            }
            hit.clear();
        }

        shadow.clear();
        continuation.clear();
        if (!shade.empty()) {
            if (auto status = execute_queue(
                    scheduler, inputs.size(), shade,
                    renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                        return shade_path(runtimes[lane.path_slot.value], direct_lighting,
                                          depth_limits, roulette_policy);
                    }},
                    report.stage_lanes, report.queue_statistics.shade);
                !status) {
                return std::unexpected(status.error());
            }
            for (const auto slot : shade.entries()) {
                auto& runtime = runtimes[slot.value];
                if (runtime.pending_shadow) {
                    if (auto status = push_slot(shadow, report.queue_statistics.shadow, slot);
                        !status) {
                        return std::unexpected(status.error());
                    }
                } else if (runtime.continuation_requested) {
                    if (auto status =
                            push_slot(continuation, report.queue_statistics.continuation, slot);
                        !status) {
                        return std::unexpected(status.error());
                    }
                } else if (runtime.pending_termination) {
                    if (auto status =
                            finish_path(runtime, runtime.pending_termination->reason,
                                        runtime.pending_termination->blocked_depth_limits);
                        !status) {
                        return std::unexpected(status.error());
                    }
                } else {
                    return std::unexpected(
                        wavefront_error(core::StatusCode::internal_error,
                                        "A CPU wavefront shade lane produced no explicit route."));
                }
            }
            shade.clear();
        }

        if (!shadow.empty()) {
            if (auto status = execute_queue(
                    scheduler, inputs.size(), shadow,
                    renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                        auto& runtime = runtimes[lane.path_slot.value];
                        if (!runtime.pending_shadow || runtime.result) {
                            return core::Status{std::unexpected(wavefront_error(
                                core::StatusCode::internal_error,
                                "A CPU wavefront shadow lane received no pending query."))};
                        }
                        const auto occluded = acceleration.occluded(runtime.pending_shadow->ray);
                        if (!occluded) {
                            return core::Status{std::unexpected(occluded.error())};
                        }
                        ++runtime.shadow_queries;
                        if (!*occluded) {
                            auto visible_transmittance = renderer::TransportSpectrum{};
                            visible_transmittance.values.fill(1.0F);
                            const auto direct = [&]() -> core::Result<renderer::TransportSpectrum> {
                                if (!runtime.pending_shadow->singleton_lambertian) {
                                    return renderer::evaluate_bsdf_direct_lighting(
                                        runtime.pending_shadow->beta,
                                        runtime.pending_shadow->bsdf_value,
                                        runtime.pending_shadow->absolute_incoming_cosine,
                                        runtime.pending_shadow->selection_probability,
                                        runtime.pending_shadow->incident_light,
                                        visible_transmittance,
                                        runtime.pending_shadow->estimator_weight);
                                }
                                const auto model = renderer::LambertianReflection::create(
                                    runtime.pending_shadow->lambertian_reflectance);
                                if (!model) {
                                    return std::unexpected(model.error());
                                }
                                return renderer::evaluate_lambertian_direct_lighting(
                                    runtime.pending_shadow->beta, *model,
                                    runtime.pending_shadow->frame,
                                    runtime.pending_shadow->outgoing_world,
                                    runtime.pending_shadow->selection_probability,
                                    runtime.pending_shadow->incident_light, visible_transmittance,
                                    runtime.pending_shadow->estimator_weight);
                            }();
                            if (!direct) {
                                return core::Status{std::unexpected(direct.error())};
                            }
                            const auto contribution =
                                *direct * runtime.pending_shadow->shading_normal_correction;
                            const auto accumulated =
                                renderer::bsdf_only_path_loop_detail::accumulate_direct_lighting(
                                    runtime.radiance, contribution);
                            if (!accumulated) {
                                return core::Status{std::unexpected(accumulated.error())};
                            }
                            runtime.radiance = *accumulated;
                        }
                        runtime.pending_shadow.reset();
                        return core::Status{};
                    }},
                    report.stage_lanes, report.queue_statistics.shadow);
                !status) {
                return std::unexpected(status.error());
            }
            for (const auto slot : shadow.entries()) {
                auto& runtime = runtimes[slot.value];
                if (runtime.continuation_requested) {
                    if (auto status =
                            push_slot(continuation, report.queue_statistics.continuation, slot);
                        !status) {
                        return std::unexpected(status.error());
                    }
                } else if (runtime.pending_termination) {
                    if (auto status =
                            finish_path(runtime, runtime.pending_termination->reason,
                                        runtime.pending_termination->blocked_depth_limits);
                        !status) {
                        return std::unexpected(status.error());
                    }
                } else {
                    return std::unexpected(wavefront_error(
                        core::StatusCode::internal_error,
                        "A CPU wavefront shadow lane produced no explicit accumulation route."));
                }
            }
            shadow.clear();
        }

        if (!continuation.empty()) {
            if (auto status = execute_queue(
                    scheduler, inputs.size(), continuation,
                    renderer::CpuWavefrontStageKernel{[&](const renderer::CpuWavefrontLane lane) {
                        auto& runtime = runtimes[lane.path_slot.value];
                        if (!runtime.continuation_requested || !runtime.ray || !runtime.cone ||
                            runtime.result || runtime.pending_termination ||
                            runtime.pending_shadow) {
                            return core::Status{std::unexpected(wavefront_error(
                                core::StatusCode::internal_error,
                                "A CPU continuation lane received contradictory route state."))};
                        }
                        const auto state = renderer::PathState::create(
                            runtime.beta, runtime.radiance, runtime.depth_counters,
                            runtime.eta_scale, runtime.wavelengths, runtime.delta_flags,
                            runtime.current_medium);
                        if (!state) {
                            return core::Status{std::unexpected(state.error())};
                        }
                        runtime.continuation_requested = false;
                        runtime.hit.reset();
                        runtime.surface.reset();
                        return core::Status{};
                    }},
                    report.stage_lanes, report.queue_statistics.continuation);
                !status) {
                return std::unexpected(status.error());
            }
            if (auto status = push_batch(ray, report.queue_statistics.ray, continuation.entries());
                !status) {
                return std::unexpected(status.error());
            }
            continuation.clear();
        }
    }

    auto paths = std::vector<renderer::BsdfOnlyPathResult>{};
    auto terminal_cones = std::vector<renderer::RayCone>{};
    paths.reserve(runtimes.size());
    terminal_cones.reserve(runtimes.size());
    for (auto& runtime : runtimes) {
        if (!runtime.result || !runtime.cone) {
            return std::unexpected(
                wavefront_error(core::StatusCode::internal_error,
                                "CPU wavefront transport ended with an incomplete terminal path "
                                "slot."));
        }
        if (runtime.closure_samples >
                std::numeric_limits<std::uint64_t>::max() - report.closure_samples ||
            runtime.light_samples >
                std::numeric_limits<std::uint64_t>::max() - report.light_samples ||
            runtime.shadow_queries >
                std::numeric_limits<std::uint64_t>::max() - report.shadow_queries) {
            return std::unexpected(
                wavefront_error(core::StatusCode::resource_exhausted,
                                "CPU wavefront transport accounting is not representable."));
        }
        report.closure_samples += runtime.closure_samples;
        report.light_samples += runtime.light_samples;
        report.shadow_queries += runtime.shadow_queries;
        paths.push_back(std::move(*runtime.result));
        terminal_cones.push_back(*runtime.cone);
    }
    if (auto status = validate_queue_statistics(report); !status) {
        return std::unexpected(status.error());
    }
    return CpuWavefrontMisBatch{
        .paths = std::move(paths),
        .terminal_cones = std::move(terminal_cones),
        .report = report,
    };
}

} // namespace

core::Result<CpuWavefrontMisBatch> trace_cpu_wavefront_mis(
    const std::span<const CpuWavefrontMisPathInput> inputs,
    const renderer::CpuWavefrontScheduler& scheduler, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler, const renderer::MisHeuristic heuristic,
    const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy) {
    try {
        return trace_cpu_wavefront_mis_impl(inputs, scheduler, acceleration, light_sampler,
                                            heuristic, depth_limits, roulette_policy);
    } catch (const std::bad_alloc&) {
        return std::unexpected(wavefront_error(
            core::StatusCode::resource_exhausted,
            "CPU wavefront transport exhausted host memory; no fallback was selected."));
    } catch (const std::exception&) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "CPU wavefront transport failed unexpectedly; no fallback was selected."));
    } catch (...) {
        return std::unexpected(wavefront_error(
            core::StatusCode::internal_error,
            "CPU wavefront transport failed with an unknown error; no fallback was selected."));
    }
}

} // namespace blackframe::engine
